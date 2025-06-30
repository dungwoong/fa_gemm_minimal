#pragma once
#include "constants.cuh"

namespace cuTK
{

    /**
     * for completing (non-TMA) async loads, copied from TK
     *
     * For cp.async, you wait for the thread to finish it's committed groups and syncwarp to finish the warp-wide load
     */
    template <int N = 0>
    __device__ static inline void cp_async_wait_group()
    {
        if constexpr (N == 0)
        {
            asm volatile("cp.async.wait_all;\n" ::);
        }
        else
        {
            asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
        }
    }

    __device__ static inline void cp_async_16(bf16 *dst, bf16 *src)
    {
        // static_cast<uint32_t> ??
        uint32_t dst_ptr = __cvta_generic_to_shared(dst);
        // uint32_t src_ptr = __cvta_generic_to_global(src);

        asm volatile(
            "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" 
            :
            :"r"(dst_ptr), "l"(src)
            : "memory");
    }

    __device__ static inline void cp_async_commit_group() {
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    template <int N_WARPS>
    struct load_group
    {
        static constexpr int GROUP_WARPS = N_WARPS;                        // This alias produces nice parallelism.
        static constexpr int GROUP_THREADS = N_WARPS * cuTK::WARP_THREADS; // This alias produces nice parallelism.
        __device__ static inline int grouplaneid() { return threadIdx.x % GROUP_THREADS; }
        __device__ static inline int warpid() { return grouplaneid() / cuTK::WARP_THREADS; }
        __device__ static inline int groupid() { return threadIdx.x / GROUP_THREADS; }

        /**
         * strided loads(assume you have a (rows, cols) region and each thread will load loadElements at a time)
         *
         * You can use this with cp async or some synchronous stuff.
         */
        template <int loadElements>
        __device__ static inline stridedLoadCoords getStridedLoadItems(int cols)
        {
            struct stridedLoadCoords coords;
            coords.innerRowA = grouplaneid() / (cols / loadElements);
            coords.innerColA = grouplaneid() % (cols / loadElements);
            coords.strideRows = (GROUP_THREADS * loadElements) / cols;
            return coords;
        }

        // NOTE TK has a sync and arrive func here too
        // NOTE then TK imports their other files so load_groups have load_async, load etc.
    };

}