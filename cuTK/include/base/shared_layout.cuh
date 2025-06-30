#pragma once
#include "constants.cuh"

#if defined(__CUDACC__)
#define ALIGN_AS(n) __align__(n)
#else
#define ALIGN_AS(n) alignas(n)
#endif

// #ifdef KITTENS_HOPPER
// #define KITTENS_DEFAULT_ALIGN KITTENS_ALIGN_AS(128)
// #else
#define DEFAULT_ALIGN ALIGN_AS(16)
// #endif

namespace cuTK {

template <int rows, int cols, SWIZZLE_MODE swizzle_bytes=SWIZZLE_MODE::SWIZZLE_128>
struct DEFAULT_ALIGN st16 {
    static_assert(rows % 8 == 0, "Rows must be divisible by 8");
    static_assert(cols % 8 == 0, "Cols must be divisible by 8");

    /**
     * Swizzles 16B, so col has to be 16Bytes
     * x --> row, y --> col
     */
    __device__ static inline int2 swizzle16B(int2 coords) {
        // assume swizzle 64
        return {coords.x, (coords.x % 8) ^ (coords.y / 8) };
    }

    /**
     * Convert matrix coords so that leading dim is 64
     * eg. dim(32, x) becomes (64, x/2)
     */
    __device__ static inline int2 convertTo64(int row, int col) {
        if constexpr (cols == 64) {
            return {row, col};
        } else if constexpr (cols < 64) {
            return {row / (64 / cols), (row % (64 / cols)) * cols + col};
        } else if constexpr (cols > 64) {
            return {row * (cols / 64) + (col / 64), col % 64};
        }
    }

    /**
     * Get pointer of the 1st element on row <row> for the (matrixR, matrixC) 8x8 matrix
     * Useful for ldmatrix ops, for example
     * 
     * call like shared_tile[st16<...>::getPtr(...)]
     */
    __device__ static inline int getMatrixIdx(int matrixR, int matrixC, int row) {
        return getIdx<8>(matrixR * 8 + row, matrixC);
    }

    /**
     * Get pointer to element on (row, col) of the tile, accounting for swizzling
     * 
     * For col, assume each item is <itemsize> size
     */
    template<int itemSize>
    __device__ static inline int getIdx(int row, int col) {
        if constexpr(swizzle_bytes == SWIZZLE_MODE::SWIZZLE_128) {
            // NOTE: not actually sure how well this would do with TMA since there's some finagling there
            int2 coords = swizzle16B(convertTo64(row, col * itemSize));
            return coords.x * 64 + (coords.y * 8) + ((col * itemSize) % 8);
        } else if constexpr(swizzle_bytes == SWIZZLE_MODE::SWIZZLE_NONE) {
            return (row * cols) + (col * itemSize);;
        }
    }

    /**
     * eg. __shared__ bf16 [st16::getSize()]
     */
    __device__ static inline int getSize() {
        return rows * cols;
    }
};

}