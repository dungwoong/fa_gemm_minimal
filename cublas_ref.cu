#pragma once
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <cublas_v2.h> // includes all the constants used, cublasGemmEx, etc.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

void runCublasFP32(cublasHandle_t handle, int M, int N, int K, half alpha,
    half *A, half *B, half beta, half *C)
{
// cuBLAS uses column-major order. So we change the order of our row-major A &
// B, since (B^T*A^T)^T = (A*B)
// This runs cuBLAS in full fp32 mode

// https://docs.nvidia.com/cuda/cublas/#cublasgemmex
// CUBLAS_OP_N means no change to matrices A and B
// m, n, k are dimensions
// alpha is ADDRESS of the alpha float
// then A, Atype, leading dimension, we have B^T so N*K, N is leading dim
// B Btype ldb
// beta, scaling factor
// C, Ctype, ldc
// computation type, algorithm
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F,
  N, A, CUDA_R_16F, K, &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_16F,
  CUBLAS_GEMM_DEFAULT_TENSOR_OP); // changed the algorithm type since deprecation https://docs.nvidia.com/cuda/cublas/#cublasgemmalgo-t
}

// Checks if error and prints it if we get one
void cudaCheck(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

void randomize_matrix(half *mat, int N)
{
    // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
    // precision is too low and the same random number is generated.
    struct timeval time{};        // struct of type timeval
    gettimeofday(&time, nullptr); // populate the struct, null timezone
    srand(time.tv_usec);          // initialize rng with seed
    for (int i = 0; i < N; i++)
    {
        float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i] = tmp;
    }
}

void deterministic_fill_matrix(half *mat, int N)
{
    for (int i = 0; i < N; i++)
    {
        // mat[i] = 0.1f * (i % 2);
        mat[i] = 0.001f * i;
    }
}

bool verify_matrix(half *ref, half *out, int N)
{
    double maxDiff = 0.0;
    double maxDiffRef = 0.0;
    double maxDiffOut = 0.0;
    double diff = 0.0;
    half maxElement = 0.0;
    int i;
    for (i = 0; i < N; i++)
    {
        diff = std::fabs(__half2float(ref[i] - out[i])); // float abs, does this work??
        if (diff > 10)
        {
            printf("Matrix divergence ref %f out %f at index %d\n", __half2float(ref[i]), __half2float(out[i]), i);
            return false;
        }
        else if (diff > maxDiff)
        {
            maxDiff = diff;
            maxDiffRef = ref[i];
            maxDiffOut = out[i];
        }

        if (ref[i] > maxElement)
        {
            maxElement = ref[i];
        }
    }
    // add maxref just to make sure ref isn't all zeroes or something
    printf("Matrix Verified, largest diff was %f (%f(ref), %f(out)) (maxref=%f)\n", __half2float(maxDiff), __half2float(maxDiffRef), __half2float(maxDiffOut), __half2float(maxElement));
    return true;
}

int main(int argc, char *argv[])
{
    cublasHandle_t handle; // cuBLAS context https://docs.nvidia.com/cuda/cublas/#cublashandle-t
    cublasStatus_t stat;   // cuBLAS functions status. Only for cuBLAS

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx)); // Sets device for future usage
    printf("Running cublas on device %d.\n", deviceIdx);

    if (cublasCreate(&handle))
    {
        // Make error
        std::cerr << "Create cublas handle error." << std::endl;
        exit(EXIT_FAILURE);
    };

    // stat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
    // publishing event tasks in the target stream
    // So it basically inserts into the GPU stream and records start/end time.
    // This is better than CPU which records when it launches the kernel, not the actual exec time
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    // Make the matrices
    // Let's try to multiply 2 4096 * 4096 matrices with cuBLAS
    half alpha = 0.5, beta = 0.5;
    long m = 4096;
    long k = 4096;
    long n = 4096;

    std::vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096};

    // Initialize matrices to nullptr. If you don't point at anything, the ptrs may reference some random mem address
    // Cr is reference
    half *A = nullptr, *B = nullptr, *C = nullptr, *Cr = nullptr;     // host
    half *dA = nullptr, *dB = nullptr, *dC = nullptr, *dCr = nullptr; // device

    // allocate memory for A, B and C. Cast to float * after
    A = (half *)malloc(sizeof(half) * m * k);
    B = (half *)malloc(sizeof(half) * k * n);
    C = (half *)malloc(sizeof(half) * m * n);
    Cr = (half *)malloc(sizeof(half) * m * n);

    // deterministic_fill_matrix(A, m * k);
    // deterministic_fill_matrix(B, k * n);
    randomize_matrix(A, m * k);
    randomize_matrix(B, k * n);
    // randomize_matrix(C, m * n);

    cudaCheck(cudaMalloc((void **)&dA, sizeof(half) * m * k));
    cudaCheck(cudaMalloc((void **)&dB, sizeof(half) * k * n));
    cudaCheck(cudaMalloc((void **)&dC, sizeof(half) * m * n));
    cudaCheck(cudaMalloc((void **)&dCr, sizeof(half) * m * n));

    // cudaMemcpy(cuda address, host address, amount, transfer type)
    cudaCheck(cudaMemcpy(dA, A, sizeof(half) * m * k,
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B, sizeof(half) * k * n,
                         cudaMemcpyHostToDevice));

    cublasCreate(&handle); // stat =

    for (int size : SIZE)
    {
        m = n = k = size;
        std::cout << "dimensions(m=n=k) " << m << ", alpha: " << __half2float(alpha)
                  << ", beta: " << __half2float(beta) << std::endl;

        cudaCheck(cudaMemcpy(dC, C, sizeof(half) * m * n,
                             cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dCr, C, sizeof(half) * m * n, cudaMemcpyHostToDevice));

        runCublasFP32(handle, m, n, k, alpha, dA, dB, beta, dCr);
        std::cout << "warmup done" << std::endl;

        int repeat_times = 50;
        float elapsed_time;
        cudaEventRecord(beg);
        for (int i = 0; i < repeat_times; i++)
        {
            // runCublasFP32(handle, m, n, k, alpha, dA, dB, beta, dC);
            runCublasFP32(handle, m, n, k, alpha, dA, dB, beta, dCr);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, beg, end);
        elapsed_time /= 1000.; // convert elapsed time to seconds

        long flops = 2 * m * n * k;

        // Copied from sgemm.cu
        printf(
            "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
            "(%ld).\n\n",
            elapsed_time / repeat_times,
            (repeat_times * flops * 1e-9) / elapsed_time, m);
        fflush(stdout);
    }

    free(A);
    free(B);
    free(C);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cublasDestroy(handle);

    return 0;
}