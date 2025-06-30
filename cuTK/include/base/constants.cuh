#pragma once
#include <cuda.h>
#include <cuda_bf16.h>

namespace cuTK {

enum SWIZZLE_MODE {
    SWIZZLE_NONE,
    SWIZZLE_128
};

enum LAYOUT {
    ROW_MAJOR
};

// Types
using bf16 = __nv_bfloat16;
using bf162 = __nv_bfloat162;

static constexpr uint32_t FULL_SHFL_MASK = 0xFFFFFFFF;

constexpr int WARP_THREADS{32};

struct stridedLoadCoords {
    int innerRowA;
    int innerColA;
    int strideRows;
};

__device__ __forceinline__ int warpid() {
    return threadIdx.x >> 5;
}

__device__ __forceinline__ int laneid() {
    return threadIdx.x % 32;
}

__device__ __forceinline__ bf162 pack16(bf16 x1, bf16 x2) {
    return {x1, x2};
}

__device__ __forceinline__ int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

}