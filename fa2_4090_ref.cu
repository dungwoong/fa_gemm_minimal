#include "kittens.cuh"

constexpr int ATTN_B = 16;
constexpr int ATTN_H = 16;
constexpr int ATTN_N = 1024; 
constexpr int ATTN_D = 64;
constexpr int ITER   = 10;

using namespace kittens;

constexpr int NUM_WORKERS = 4;
constexpr int PIPE_STAGES = 3; 

template<int D> constexpr size_t ROWS = 16*(128/D); // height of each worker tile (rows)
template<int D, typename T=bf16, typename L=row_l> using qkvo_tile = rt<T, ROWS<D>, D, L>;
template<int D, typename T=float> using attn_tile = rt<T, ROWS<D>, ROWS<D>>;
template<int D> using shared_tile = st_bf<ROWS<D>, D>;
template<int D> using global_layout = gl<bf16, -1, -1, -1, D>; // B, N, H, specified at runtime, D known at compile time for this kernel
template<int D> struct globals { global_layout<D> Qg, Kg, Vg, Og; };

template<int D> __launch_bounds__(NUM_WORKERS*WARP_THREADS, 1)
__global__ void attend_ker(const __grid_constant__ globals<D> g) {
    
    using load_group = kittens::group<2>; // pairs of workers collaboratively load k, v tiles
    int loadid = load_group::groupid(), workerid = kittens::warpid(); // which worker am I?
    constexpr int LOAD_BLOCKS = NUM_WORKERS / load_group::GROUP_WARPS;
    const int batch = blockIdx.z, head = blockIdx.y, q_seq = blockIdx.x * NUM_WORKERS + workerid;

    extern __shared__ alignment_dummy __shm[]; 
    shared_allocator al((int*)&__shm[0]);
    
    shared_tile<D> (&k_smem)[LOAD_BLOCKS][PIPE_STAGES] = al.allocate<shared_tile<D>, LOAD_BLOCKS, PIPE_STAGES>();
    shared_tile<D> (&v_smem)[LOAD_BLOCKS][PIPE_STAGES] = al.allocate<shared_tile<D>, LOAD_BLOCKS, PIPE_STAGES>();
    
    shared_tile<D> (&qo_smem)[NUM_WORKERS] = reinterpret_cast<shared_tile<D>(&)[NUM_WORKERS]>(k_smem);
    // Initialize all of the register tiles.
    qkvo_tile<D, bf16> q_reg, k_reg; // Q and K are both row layout, as we use mma_ABt.
    qkvo_tile<D, bf16, col_l> v_reg; // V is column layout, as we use mma_AB.
    qkvo_tile<D, float> o_reg; // Output tile.
    attn_tile<D, float> att_block; // attention tile, in float. (We want to use float wherever possible.)
    attn_tile<D, bf16> att_block_mma; // bf16 attention tile for the second mma_AB. We cast right before that op.
    typename attn_tile<D, float>::col_vec max_vec_last, max_vec, norm_vec; // these are column vectors for the in-place softmax.
    // each warp loads its own Q tile of 16x64
    if (q_seq*ROWS<D> < g.Qg.depth()) {
        load<1, false>(qo_smem[workerid], g.Qg, {batch, q_seq, head, 0});  // going through shared memory improves coalescing of dram reads.
        __syncwarp();
        load(q_reg, qo_smem[workerid]);
    }
    __syncthreads();

    if constexpr(D == 64) q_reg *= __float2bfloat16(0.125f * 1.44269504089f);
    else if constexpr(D == 128) q_reg *= __float2bfloat16(0.08838834764f * 1.44269504089f);

    max_vec = base_types::constants<float>::neg_infty();
    norm_vec = 0.f;
    o_reg = 0.f;
    // launch the load of the first k, v tiles
    int kv_blocks = (g.Kg.depth() + LOAD_BLOCKS*ROWS<D>-1) / (LOAD_BLOCKS*ROWS<D>), tic = 0;
    load_group::load_async<1, false>(k_smem[loadid][0], g.Kg, {batch, loadid, head, 0});
    load_group::load_async<1, false>(v_smem[loadid][0], g.Vg, {batch, loadid, head, 0});
    // iterate over k, v for these q's that have been loaded
    for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++, tic=(tic+1)%3) {
        int next_load_idx = (kv_idx+1)*LOAD_BLOCKS + loadid;
        if(next_load_idx*ROWS<D> < g.Kg.depth()) {
            int next_tic = (tic+1)%3;
            load_group::load_async<1, false>(k_smem[loadid][next_tic], g.Kg, {batch, next_load_idx, head, 0});
            load_group::load_async<1, false>(v_smem[loadid][next_tic], g.Vg, {batch, next_load_idx, head, 0});
            load_async_wait<1>(); // next k, v can stay in flight.
        }
        else load_async_wait();
        __syncthreads();

        #pragma unroll LOAD_BLOCKS
        for(int subtile = 0; subtile < LOAD_BLOCKS && (kv_idx*LOAD_BLOCKS + subtile)*ROWS<D> < g.Kg.depth(); subtile++) {
            load(k_reg, k_smem[subtile][tic]); // load k from shared into registers
            att_block = 0.f; // zero 16x16 attention tile
            mma<transpose::N, transpose::T>(att_block, q_reg, k_reg, att_block); // Q@K.T
            int first_index = (kv_idx*LOAD_BLOCKS + subtile)*ROWS<D>; // one past the last KV index of this tile
            int start_fill = g.Kg.depth()-first_index < ROWS<D> ? g.Kg.depth()-first_index : ROWS<D>;
            right_fill(att_block, att_block, start_fill, base_types::constants<float>::neg_infty());
            max_vec_last = max_vec;
            max_vec = max<axis::COL>(att_block, max_vec); 
            att_block = exp2(att_block - max_vec); 
            max_vec_last = exp2(max_vec_last - max_vec); 
            norm_vec *= max_vec_last; 
            norm_vec = sum<axis::COL>(att_block, norm_vec); 
            att_block_mma = att_block; // copy to bf16 tile
            load(v_reg, v_smem[subtile][tic]); 
            o_reg *= max_vec_last; 
            mma<transpose::N, transpose::N>(o_reg, att_block_mma, v_reg, o_reg);
        }
    }

    o_reg /= norm_vec;
    __syncthreads();
    if (q_seq*ROWS<D> < g.Og.depth()) { // write out o.
        store(qo_smem[workerid], o_reg); // going through shared memory improves coalescing of dram writes.
        __syncwarp();
        store<1, false>(g.Og, qo_smem[workerid], {batch, q_seq, head, 0});
    }
}

#include <iostream>
#include <string>
#include <fstream>

#define BLOCK_SIZE (32*NUM_WORKERS)

#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError( const char *file, const int line ) {
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
}

// Compute FLOPs for forward attention
constexpr uint64_t ATTN_FLOPS = 
    2llu * ATTN_B * ATTN_H * ATTN_N * ATTN_N * ATTN_D + // Q * K^T: 2BHNND (multiply-add)
    4llu * ATTN_B * ATTN_H * ATTN_N * ATTN_N +          // Softmax: 2BHNN (exp and divide, plus flash-attn bookkeeping)
    2llu * ATTN_B * ATTN_H * ATTN_N * ATTN_N * ATTN_D;      // (Q * K^T) * V: 2BHNND (multiply-add)

int main(int argc, char **argv) {
    // TODO: consider doing sequential kernel launches to force batches dimension element to execute sequentially,
    // which may increase the probability of L2 cache hits on KV

    std::cout << "Entered main!" << std::endl;

    // create dummy variables that are the right size
    constexpr int TOTAL_ELEMENTS = ATTN_B*ATTN_H*ATTN_N*ATTN_D;
    constexpr int TOTAL_UNIQUE_ELEMENTS = ATTN_H*ATTN_N*ATTN_D;

    float *q = new float[TOTAL_ELEMENTS];
    float *k = new float[TOTAL_ELEMENTS];
    float *v = new float[TOTAL_ELEMENTS];
    float *o_ref = new float[TOTAL_ELEMENTS];

    bf16 *q_bf = new bf16[TOTAL_ELEMENTS];
    bf16 *k_bf = new bf16[TOTAL_ELEMENTS];
    bf16 *v_bf = new bf16[TOTAL_ELEMENTS];
    bf16 *o_bf = new bf16[TOTAL_ELEMENTS];
    float *o = new float[TOTAL_ELEMENTS];

    std::ifstream infile(argv[1]);

    std::cout << "Starting to enter!" << std::endl;

    for(int i = 0; i < TOTAL_ELEMENTS/ATTN_B; i++) infile >> q[i];
    std::cout << "Finished loading Q" << std::endl;
    for(int i = 0; i < TOTAL_ELEMENTS/ATTN_B; i++) infile >> k[i];
    std::cout << "Finished loading K" << std::endl;
    for(int i = 0; i < TOTAL_ELEMENTS/ATTN_B; i++) infile >> v[i];
    std::cout << "Finished loading V" << std::endl;
    for(int i = 0; i < TOTAL_ELEMENTS/ATTN_B; i++) infile >> o_ref[i];
    std::cout << "Finished loading O_REF" << std::endl;

    std::cout << "Finished loading file from " << argv[1] << "!" << std::endl;

    // replicate into batch elements
    for(int i = 0; i < TOTAL_ELEMENTS; i++) {
        q_bf[i] = __float2bfloat16(q[i % (TOTAL_ELEMENTS/ATTN_B)]);
        k_bf[i] = __float2bfloat16(k[i % (TOTAL_ELEMENTS/ATTN_B)]);
        v_bf[i] = __float2bfloat16(v[i % (TOTAL_ELEMENTS/ATTN_B)]);
    }

    bf16 *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, TOTAL_ELEMENTS * sizeof(bf16));
    cudaMalloc(&d_k, TOTAL_ELEMENTS * sizeof(bf16));
    cudaMalloc(&d_v, TOTAL_ELEMENTS * sizeof(bf16));
    cudaMalloc(&d_o, TOTAL_ELEMENTS * sizeof(bf16));

    cudaMemcpy(d_q, q_bf, TOTAL_ELEMENTS * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k_bf, TOTAL_ELEMENTS * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v_bf, TOTAL_ELEMENTS * sizeof(bf16), cudaMemcpyHostToDevice);

    global_layout<ATTN_D> Qg(d_q, ATTN_B, ATTN_N, ATTN_H, nullptr);
    global_layout<ATTN_D> Kg(d_k, ATTN_B, ATTN_N, ATTN_H, nullptr);
    global_layout<ATTN_D> Vg(d_v, ATTN_B, ATTN_N, ATTN_H, nullptr);
    global_layout<ATTN_D> Og(d_o, ATTN_B, ATTN_N, ATTN_H, nullptr);
    globals<ATTN_D> g(Qg, Kg, Vg, Og);
    
    unsigned long mem_size = (kittens::MAX_SHARED_MEMORY) / 2; // have the flag tell us
    std::cout << "Max shared memory size: " << mem_size << std::endl;
    
    cudaFuncSetAttribute(
        attend_ker<ATTN_D>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    cudaDeviceSynchronize();
    std::cout << "Starting kernel\n";
    dim3 grid((ATTN_N + qkvo_tile<ATTN_D>::rows*NUM_WORKERS - 1) / (qkvo_tile<ATTN_D>::rows*NUM_WORKERS), ATTN_H, ATTN_B);
    const auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < ITER; i++) {
        attend_ker<ATTN_D><<<grid, BLOCK_SIZE, mem_size>>>(g);
    }
    cudaDeviceSynchronize();
    const auto finish = std::chrono::high_resolution_clock::now();
    CudaCheckError();
    std::cout << "Finished kernel\n";

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // check correctness
    cudaMemcpy(o_bf, d_o, TOTAL_ELEMENTS * sizeof(bf16), cudaMemcpyDeviceToHost);
    for(int i = 0; i < TOTAL_ELEMENTS; i++) {
        o[i] = __bfloat162float(o_bf[i]);
    }

    bool good = true;
    std::ofstream o_ref_file("printouts/o_ref.txt");
    std::ofstream o_file("printouts/o.txt");
    std::ofstream diff_file("printouts/diff.txt");

    float total_diff = 0;
    float max_error  = 0;

    for(int i = 0; i < TOTAL_ELEMENTS; i++) {
        float diff = o[i] - o_ref[i % (TOTAL_ELEMENTS/ATTN_B)];
        if(i < TOTAL_UNIQUE_ELEMENTS) {
            o_ref_file << o_ref[i % (TOTAL_ELEMENTS/ATTN_B)] << ' ';
            o_file << o[i] << ' ';
            diff_file << diff << ' ';
        }
        if(i % ATTN_D == ATTN_D-1) {
            o_ref_file << '\n';
            o_file << '\n';
            diff_file << '\n';
        }
        if(abs(diff) > 0.01 || isnan(diff)) {
            good = false;
        }
        total_diff += abs(diff);
    }
    std::cout << "Average diff: " << total_diff / TOTAL_UNIQUE_ELEMENTS << std::endl;
    std::cout << "Average execution time: " << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() / ITER << " us" << std::endl;
    if(good) std::cout << "Correct :)\n";
    else std::cout << "Incorrect :(\n";
    // Compute and print average TFLOPs achieved
    double avg_time_s = (double)(std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()) / (ITER * 1e6);
    double avg_tflops = (ATTN_FLOPS / avg_time_s) / 1e12;
    std::cout << "Efficiency: " << avg_tflops << " TFLOPS\n\n\n" << std::endl;

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);

    delete[] q, k, v, o, o_ref;
    delete[] q_bf, k_bf, v_bf, o_bf;

    return 0;
}