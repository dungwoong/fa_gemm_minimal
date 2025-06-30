#pragma once
#include "constants.cuh"

namespace cuTK {

/**
 * A matrix in GMEM that's meant to be broken down into blocks
 * 
 * may need to think about B and H dimensions in the future but dw for now
 */
template <int rows, int cols, LAYOUT l=LAYOUT::ROW_MAJOR>
struct global_layout {

    /**
     * Gets pointer for the item at block bR, bC (where each block is (bRows, bCols)) and coord r, c within the block
     */
    template <int bRows, int bCols, int itemSize>
    __device__ static inline int getBlockedIdx(int bR, int bC, int r, int c) {
        return (bR * bRows + r) * cols + (bC * bCols + (c * itemSize));
    }

    template <int itemSize>
    __device__ static inline int getIdx(int r, int c) {
        return (r * cols) + (c * itemSize);
    }
};

/**
 * dimension 4 is assumed to have stride 1
 */
template <int d1, int d2, int d3, int d4>
struct global_layout4 {
    template <int itemSize>
    __device__ static inline int getIdx(int coord1, int coord2, int coord3, int coord4) {
        return (((coord1 * d2 + coord2) * d3 + coord3) * d4) + coord4 * itemSize;
    }
};

}