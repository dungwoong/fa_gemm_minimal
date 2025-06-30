#include "cuTK.cuh"

// Let's rebuild the FA implementation from cuda only
using namespace cuTK;
constexpr int ROWS = 32;
constexpr int COLS = 64; // == ATTN_D
constexpr int NUM_WORKERS = 4;
constexpr int PIPE_STAGES = 2;

constexpr int ATTN_B = 16;
constexpr int ATTN_H = 16;
constexpr int ATTN_N = 1024;
constexpr int ATTN_D = COLS;
constexpr int ITER = 10;

// all matrices are B N H D
using q_nd_layout = global_layout4<ATTN_B, ATTN_N, ATTN_H, ATTN_D>;
using qkv_layout = st16<ROWS, COLS, SWIZZLE_MODE::SWIZZLE_128>;

__global__ void attention_cuda_4090(bf16 *Q, bf16 *K, bf16 *V, bf16 *O)
{

    // 4 warps per block, load in groups of 2 warps
    using lgroup = load_group<2>;
    int workerid = cuTK::warpid();
    int loadid = lgroup::groupid();
    int laneid = cuTK::laneid();
    constexpr int LOAD_BLOCKS = NUM_WORKERS / lgroup::GROUP_WARPS;

    const int batch = blockIdx.z, head = blockIdx.y;
    const int q_seq = blockIdx.x * NUM_WORKERS + workerid;
    const int incr = (batch * ATTN_H * ATTN_N * ATTN_D);
    Q += incr;
    K += incr;
    V += incr;
    O += incr;

    // declare stuff, allocate space
    extern __shared__ DEFAULT_ALIGN int __shm[];
    shared_allocator<> al((int *)&__shm[0]);
    bf16(&k_smem)[LOAD_BLOCKS][PIPE_STAGES][ROWS * COLS] = al.allocate<bf16, LOAD_BLOCKS, PIPE_STAGES, ROWS * COLS>();
    bf16(&v_smem)[LOAD_BLOCKS][PIPE_STAGES][ROWS * COLS] = al.allocate<bf16, LOAD_BLOCKS, PIPE_STAGES, ROWS * COLS>();

    bf16(&qo_smem)[NUM_WORKERS][ROWS * COLS] = reinterpret_cast<bf16(&)[NUM_WORKERS][ROWS * COLS]>(k_smem);

    // declare registers
    // this is from my tiling setup
    constexpr uint r2 = ROWS / 16;
    constexpr uint r4 = COLS / 16;
    constexpr uint r8 = COLS / 8;
    bf16 q_reg[r2][r4][8];  // 16x16 tiles
    bf16 k_reg[r4][r4][4];  // 8x16(16x8 if transpose) each time
    bf16 v_reg[r2][r8][4];  // 16x8 each time
    float o_reg[r2][r8][4]; // 16x8 each time

    float attn_block[r2][r4][4]; // 16x8, total 32x32
    bf16 attn_block_mma[r2][r4][4];

    // for max and norms, just keep track of 1 thing for each row maybe
    float max_vec[r2][2];
    float max_vec_last[r2][2];
    float norm_vec[r2][2];

    // load in Q for each warp, since 128B swizzle we gotta think about...something
    // each warp loads in their own 32x64 tile, going down seqlen
    // TODO need if statement here
    stridedLoadCoords coords = load_group<1>::getStridedLoadItems<8>(COLS);
    for (uint qLoadIdx = 0; qLoadIdx < ROWS; qLoadIdx += coords.strideRows)
    {
        int s_idx = qkv_layout::getIdx<8>(qLoadIdx + coords.innerRowA, coords.innerColA); // 32x64, correct

        // int g_idx = global_layout<ATTN_N, COLS>::getIdx<8>((q_seq * ROWS + qLoadIdx + coords.innerRowA) * ATTN_H + head, coords.innerColA);
        int g_idx = q_nd_layout::getIdx<8>(0, q_seq * ROWS + qLoadIdx + coords.innerRowA, head, coords.innerColA);
        reinterpret_cast<float4 *>(&qo_smem[workerid][s_idx])[0] = reinterpret_cast<float4 *>(&Q[g_idx])[0];
    }
    __syncwarp();
    // TODO load into registers from ROWS * COLS
    for (uint r = 0; r < r2; ++r)
    {
        for (uint c = 0; c < r4; ++c)
        {
            // Load in the (r, c) (16x16) region
            ldmatrix_x4_b16((uint32_t *)q_reg[r][c], __cvta_generic_to_shared(&qo_smem[workerid][qkv_layout::getMatrixIdx(r * 2, c * 2 + laneid / 16, laneid % 16)]));
        }
    }
    __syncthreads(); // all threads need to be done so we can start loading K into qo_smem

    // q_reg *= __float2bfloat16(0.125f * 1.44269504089f);
    bf16 scale_factor = __float2bfloat16(0.125f * 1.44269504089f);
#pragma unroll
    for (uint r = 0; r < r2; ++r)
    {
#pragma unroll
        for (uint c = 0; c < r4; ++c)
        {
#pragma unroll
            for (uint i = 0; i < 8; ++i)
            {
                q_reg[r][c][i] *= scale_factor;
            }
        }
    }

    // Initialize max_vec, norm_vec, o_reg
    for (uint r = 0; r < r2; ++r)
    {
        max_vec[r][0] = -INFINITY;
        max_vec[r][1] = -INFINITY;
    }

    memset(norm_vec, 0, r2 * 2 * sizeof(float));
    memset(o_reg, 0, r2 * r8 * 4 * sizeof(float));

    // load first K block async
    // load a 32x64 block into K[loadID][0] basically
    // come back to this...need to decrease bank conflicts potentially
    int kv_blocks = ceil_div(ATTN_N, LOAD_BLOCKS * ROWS), tic = 0;

    // load async k and v
    // TODO need to figure out why bank conflicts
    coords = lgroup::getStridedLoadItems<8>(COLS);
    for (uint kLoadIdx = 0; kLoadIdx < ROWS; kLoadIdx += coords.strideRows)
    {
        int s_idx = qkv_layout::getIdx<8>(kLoadIdx + coords.innerRowA, coords.innerColA);
        int g_idx = q_nd_layout::getIdx<8>(0, loadid * ROWS + kLoadIdx + coords.innerRowA, head, coords.innerColA);
        // reinterpret_cast<float4 *>(&k_smem[loadid][0][s_idx])[0] = reinterpret_cast<float4 *>(&K[g_idx])[0];
        cp_async_16(&k_smem[loadid][0][s_idx], &K[g_idx]);
    }
    for (uint vLoadIdx = 0; vLoadIdx < ROWS; vLoadIdx += coords.strideRows)
    {
        int s_idx = qkv_layout::getIdx<8>(vLoadIdx + coords.innerRowA, coords.innerColA);
        int g_idx = q_nd_layout::getIdx<8>(0, loadid * ROWS + vLoadIdx + coords.innerRowA, head, coords.innerColA);
        cp_async_16(&v_smem[loadid][0][s_idx], &V[g_idx]);
    }
    cp_async_commit_group();

    for (uint kv_idx = 0; kv_idx < kv_blocks; ++kv_idx, tic = (tic + 1) % PIPE_STAGES)
    {
        int next_load_idx = (kv_idx + 1) * LOAD_BLOCKS + loadid;

        // load next KV and wait
        if (next_load_idx * ROWS < ATTN_N)
        {
            int next_tic = (tic + 1) % PIPE_STAGES;
            for (uint kLoadIdx = 0; kLoadIdx < ROWS; kLoadIdx += coords.strideRows)
            {
                int s_idx = qkv_layout::getIdx<8>(kLoadIdx + coords.innerRowA, coords.innerColA);
                int g_idx = q_nd_layout::getIdx<8>(0, next_load_idx * ROWS + kLoadIdx + coords.innerRowA, head, coords.innerColA);
                cp_async_16(&k_smem[loadid][next_tic][s_idx], &K[g_idx]);
            }
            for (uint vLoadIdx = 0; vLoadIdx < ROWS; vLoadIdx += coords.strideRows)
            {
                int s_idx = qkv_layout::getIdx<8>(vLoadIdx + coords.innerRowA, coords.innerColA);
                int g_idx = q_nd_layout::getIdx<8>(0, next_load_idx * ROWS + vLoadIdx + coords.innerRowA, head, coords.innerColA);
                cp_async_16(&v_smem[loadid][next_tic][s_idx], &V[g_idx]);
            }
            cp_async_commit_group();
            cp_async_wait_group<1>();
        }
        else
        {
            cp_async_wait_group();
        }
        __syncthreads();

#pragma unroll LOAD_BLOCKS
        for (int subtile = 0; subtile < LOAD_BLOCKS && (kv_idx * LOAD_BLOCKS + subtile) * ROWS < ATTN_N; subtile++)
        {
            // load k_smem to registers
            for (uint r = 0; r < r4; ++r)
            {
                for (uint c = 0; c < r4; ++c)
                {
                    // use ldmatrix, no trans to get transposing when using mma
                    // load in the 8x16 at (r, c)
                    // in sm_86, threads 17-31 don't need valid smem addresses but I'm including it anyways
                    ldmatrix_x2_b16((uint32_t *)(k_reg[r][c]), __cvta_generic_to_shared(&k_smem[subtile][tic][qkv_layout::getMatrixIdx(r, (c * 2) + (laneid / 8) % 2, (laneid % 8))]));
                }
            }

            // zero the attn block tile
            memset(attn_block, 0, r2 * r4 * 4 * sizeof(float));

            // Q@K.T ========================================================================================================================
            // NOTE can change to e.g. loading all first, then doing matmul
            for (uint kIdx = 0; kIdx < r4; ++kIdx)
            {
                for (uint qRow = 0; qRow < r2; ++qRow)
                {
                    for (uint kCol = 0; kCol < r4; ++kCol)
                    {
                        hmma_16_8_16_bf16_float(attn_block[qRow][kCol], (uint32_t *)q_reg[qRow][kIdx], (uint32_t *)k_reg[kCol][kIdx], attn_block[qRow][kCol]);
                    }
                }
            }

// TODO right fill neg infty.

// copy max_vec to max_vec_last ==========================================================================================
#pragma unroll
            for (uint i = 0; i < r2; ++i)
            {
                max_vec_last[i][0] = max_vec[i][0];
                max_vec_last[i][1] = max_vec[i][1];
            }

// max_vec = max<axis::COL>(att_block, max_vec); ============================================================
// TODO why is the max over columns?? rightt we loop over the n dimension and not the h dimension
// float attn_block[r2][r4][4];
// float max_vec[r2][2];

// Take the maxes within each thread first
#pragma unroll
            for (uint i = 0; i < r2; ++i)
            {
                for (uint j = 0; j < r4; ++j)
                {
                    max_vec[i][0] = max(max_vec[i][0], max(attn_block[i][j][0], attn_block[i][j][1]));
                    max_vec[i][1] = max(max_vec[i][1], max(attn_block[i][j][2], attn_block[i][j][3]));
                }
            }

            // reduce max within each 4 threads
            for (int i = 2; i > 0; i /= 2)
            {
                for (uint j = 0; j < r2; ++j)
                {
                    max_vec[j][0] = max(max_vec[j][0], __shfl_down_sync(cuTK::FULL_SHFL_MASK, max_vec[j][0], i, 4));
                    max_vec[j][1] = max(max_vec[j][1], __shfl_down_sync(cuTK::FULL_SHFL_MASK, max_vec[j][1], i, 4));
                }
                __syncwarp();
            }
            for (uint j = 0; j < r2; ++j)
            {
                // save packed type
                max_vec[j][0] = __shfl_sync(cuTK::FULL_SHFL_MASK, max_vec[j][0], 0, 4);
                max_vec[j][1] = __shfl_sync(cuTK::FULL_SHFL_MASK, max_vec[j][1], 0, 4);
            }

// att_block = exp2(att_block - max_vec); ==========================================================================================
// float attn_block[r2][r4][4];
// float max_vec[r2][2];
#pragma unroll
            for (uint i = 0; i < r2; ++i)
            {
                for (uint j = 0; j < r4; ++j)
                {
                    attn_block[i][j][0] = exp2f(attn_block[i][j][0] - max_vec[i][0]);
                    attn_block[i][j][1] = exp2f(attn_block[i][j][1] - max_vec[i][0]);
                    attn_block[i][j][2] = exp2f(attn_block[i][j][2] - max_vec[i][1]);
                    attn_block[i][j][3] = exp2f(attn_block[i][j][3] - max_vec[i][1]);
                }
            }

// max_vec_last = exp2(max_vec_last - max_vec); ==========================================================================================
#pragma unroll
            for (uint i = 0; i < r2; ++i)
            {
                max_vec_last[i][0] = exp2f(max_vec_last[i][0] - max_vec[i][0]);
                max_vec_last[i][1] = exp2f(max_vec_last[i][1] - max_vec[i][1]);
            }

            // norm_vec *= max_vec_last; ==========================================================================================
            // float norm_vec[r2][2];
            for (uint i = 0; i < r2; ++i)
            {
                norm_vec[i][0] *= max_vec_last[i][0];
                norm_vec[i][1] *= max_vec_last[i][1];
            }

            // norm_vec = sum<axis::COL>(att_block, norm_vec); ==========================================================================================
            // float norm_vec[r2][2];
            // float attn_block[r2][r4][4];
            float norm_sum_accum[r2][2];
            memset(norm_sum_accum, 0, r2 * 2 * sizeof(float));

            // sum within own attn block
            for (uint i = 0; i < r2; ++i)
            {
                for (uint j = 0; j < r4; ++j)
                {
                    norm_sum_accum[i][0] += (attn_block[i][j][0] + attn_block[i][j][1]);
                    norm_sum_accum[i][1] += (attn_block[i][j][2] + attn_block[i][j][3]);
                }
            }

            // pass down the sums, also sum within every 4 threads
            for (int i = 2; i > 0; i /= 2)
            {
                for (uint j = 0; j < r2; ++j)
                {
                    norm_sum_accum[j][0] += __shfl_down_sync(cuTK::FULL_SHFL_MASK, norm_sum_accum[j][0], i, 4);
                    norm_sum_accum[j][1] += __shfl_down_sync(cuTK::FULL_SHFL_MASK, norm_sum_accum[j][1], i, 4);
                }
                __syncwarp(); // I think syncwarp is to make sure the shuffles don't get reordered by the compiler or smth
            }

#pragma unroll
            for (uint j = 0; j < r2; ++j)
            {
                norm_vec[j][0] += norm_sum_accum[j][0];
                norm_vec[j][1] += norm_sum_accum[j][1];
            }

// copy sums from leader
#pragma unroll
            for (uint j = 0; j < r2; ++j)
            {
                // save packed type
                norm_vec[j][0] = __shfl_sync(cuTK::FULL_SHFL_MASK, norm_vec[j][0], 0, 4);
                norm_vec[j][1] = __shfl_sync(cuTK::FULL_SHFL_MASK, norm_vec[j][1], 0, 4);
            }

// att_block_mma = att_block; ==========================================================================================
// float attn_block[r2][r4][4];
// bf16 attn_block_mma[r2][r4][4];
#pragma unroll
            for (uint i = 0; i < r2; ++i)
            {
                for (uint j = 0; j < r4; ++j)
                {
                    ((bf162 *)attn_block_mma[i][j])[0] = __float22bfloat162_rn(((float2 *)attn_block[i][j])[0]);
                    ((bf162 *)attn_block_mma[i][j])[1] = __float22bfloat162_rn(((float2 *)attn_block[i][j])[1]);
                }
            }

            // load(v_reg, v_smem[subtile][tic]); ==========================================================================================
            // bf16(&v_smem)[LOAD_BLOCKS][PIPE_STAGES][ROWS * COLS]
            // bf16 v_reg[r2][r8][4];
            for (uint i = 0; i < r2; ++i)
            {
                for (uint j = 0; j < r4; ++j)
                {
                    // load a 16x16 at a time. in total 32x64
                    int sIdx = qkv_layout::getMatrixIdx(i * 2, j * 2 + laneid / 16, laneid % 16);
                    ldmatrix_x4_trans_b16((uint32_t *)(v_reg[i][j * 2]), (uint32_t *)(v_reg[i][j * 2 + 1]), __cvta_generic_to_shared(&v_smem[subtile][tic][sIdx]));
                }
            }

// o_reg *= max_vec_last; ==========================================================================================
// float o_reg[r2][r8][4];
// float max_vec[r2][2];
#pragma unroll
            for (uint i = 0; i < r2; ++i)
            {
                for (uint j = 0; j < r8; ++j)
                {
                    for (uint k = 0; k < 4; ++k)
                    {
                        o_reg[i][j][k] *= max_vec_last[i][k / 2];
                    }
                }
            }

            // mma<transpose::N, transpose::N>(o_reg, att_block_mma, v_reg, o_reg); ===============================================================
            // float o_reg[r2][r8][4];
            // bf16 attn_block_mma[r2][r4][4]; // 32x32
            // bf16 v_reg[r2][r8][4]; // 32x64
            for (uint kIdx = 0; kIdx < r2; ++kIdx)
            {
                for (uint mIdx = 0; mIdx < r2; ++mIdx)
                {
                    for (uint nIdx = 0; nIdx < r8; ++nIdx)
                    {
                        // ((float2 *)(o_reg[mIdx][nIdx]))[0], ((float2 *)o_reg[mIdx][nIdx])[1] (C)
                        // ((bf162 *)(attn_block_mma[mIdx][kIdx * 2]))[0], ((bf162 *)(attn_block_mma[mIdx][kIdx * 2]))[1], ((bf162 *)(attn_block_mma[mIdx][kIdx * 2 + 1]))[0], ((bf162 *)(attn_block_mma[mIdx][kIdx * 2 + 1]))[1]
                        // ((bf162 *)(v_reg[kIdx][nIdx]))[0], ((bf162 *)(v_reg[kIdx][nIdx]))[1]
                        hmma16816_tk(((float2 *)(o_reg[mIdx][nIdx]))[0], ((float2 *)o_reg[mIdx][nIdx])[1],
                                     ((bf162 *)(attn_block_mma[mIdx][kIdx * 2]))[0], ((bf162 *)(attn_block_mma[mIdx][kIdx * 2]))[1], ((bf162 *)(attn_block_mma[mIdx][kIdx * 2 + 1]))[0], ((bf162 *)(attn_block_mma[mIdx][kIdx * 2 + 1]))[1],
                                     ((bf162 *)(v_reg[kIdx][nIdx]))[0], ((bf162 *)(v_reg[kIdx][nIdx]))[1],
                                     ((float2 *)(o_reg[mIdx][nIdx]))[0], ((float2 *)o_reg[mIdx][nIdx])[1]);
                    }
                }
            }
        }
    }

    // TODO epilogue
    // o_reg /= norm_vec; ===============================================================
    // float o_reg[r2][r8][4]; float norm_vec[r2][2];
    #pragma unroll
    for (uint i = 0; i < r2; ++i) {
        for (uint j = 0; j < r8; ++j) {
            for (uint k = 0; k < 4; ++k) {
                o_reg[i][j][k] /= norm_vec[i][k / 2];
            }
        }
    }
    __syncthreads();

    
    // storing to GMEM seems fine
    for (uint i = 0; i < r2; ++i) {
        for (uint j = 0; j < r4; ++j) {
            // left 16x8
            reinterpret_cast<bf162 *>(&qo_smem[workerid][qkv_layout::getMatrixIdx(i * 2, j * 2, laneid / 4)])[laneid % 4] = __float22bfloat162_rn(((float2 *)o_reg[i][j * 2])[0]);
            reinterpret_cast<bf162 *>(&qo_smem[workerid][qkv_layout::getMatrixIdx(i * 2, j * 2, laneid / 4 + 8)])[laneid % 4] = __float22bfloat162_rn(((float2 *)o_reg[i][j * 2])[1]);
            reinterpret_cast<bf162 *>(&qo_smem[workerid][qkv_layout::getMatrixIdx(i * 2, j * 2 + 1, laneid / 4)])[laneid % 4] = __float22bfloat162_rn(((float2 *)o_reg[i][j * 2 + 1])[0]);
            reinterpret_cast<bf162 *>(&qo_smem[workerid][qkv_layout::getMatrixIdx(i * 2, j * 2 + 1, laneid / 4 + 8)])[laneid % 4] = __float22bfloat162_rn(((float2 *)o_reg[i][j * 2 + 1])[1]);
        }
    }

    __syncwarp();
    coords = load_group<1>::getStridedLoadItems<8>(COLS);
    for (uint oLoadIdx = 0; oLoadIdx < ROWS; oLoadIdx += coords.strideRows)
    {
        int s_idx = qkv_layout::getIdx<8>(oLoadIdx + coords.innerRowA, coords.innerColA); // 32x64, correct

        // int g_idx = global_layout<ATTN_N, COLS>::getIdx<8>((q_seq * ROWS + qLoadIdx + coords.innerRowA) * ATTN_H + head, coords.innerColA);
        int g_idx = q_nd_layout::getIdx<8>(0, q_seq * ROWS + oLoadIdx + coords.innerRowA, head, coords.innerColA);
        reinterpret_cast<float4 *>(&O[g_idx])[0] = reinterpret_cast<float4 *>(&qo_smem[workerid][s_idx])[0];
    }
}

#include "fa2_cuda_test_harness.impl"