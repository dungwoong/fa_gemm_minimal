// taken from kittens ops/warp/memory/util
#pragma once
#include "constants.cuh"

namespace cuTK
{

    /**
     * NOTE added .shared::cta compared to original .shared
     *
     * Loads in a 2x2 grid of 8x8 matrices
     */
    __device__ __forceinline__ void ldmatrix_x4_b16(uint32_t r[4], uint32_t addr)
    {
        // r0c0 r1c0 r0c1 r1c1
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                     : "r"(addr));
    }

    /**
     * r is a 16x8, r1 is another 16x8
     */
    __device__ __forceinline__ void ldmatrix_x4_trans_b16(uint32_t r[2], uint32_t r1[2], uint32_t addr)
    {
        // r0c0 r1c0 r0c1 r1c1
        asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r1[0]), "=r"(r1[1])
                     : "r"(addr));
    }

    /**
     * Loads in 2x1 grid of 8x8 matrices, col-major
     */
    __device__ __forceinline__ void ldmatrix_x2_trans_b16(uint32_t r[2], uint32_t addr)
    {
        // r0c0 r1c0
        asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(r[0]), "=r"(r[1])
                     : "r"(addr));
    }

    /**
     * Loads in 2x1 grid of 8x8 matrices, row-major
     * NOTE: can also treat this as a transposed col-major 1x2 grid of 8x8
     */
    __device__ __forceinline__ void ldmatrix_x2_b16(uint32_t r[2], uint32_t addr)
    {
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(r[0]), "=r"(r[1])
                     : "r"(addr));
    }

    /**
     * mma of 16x16 A with 16x8 B + 16x8 C
     * NOTE: you usually use ldmatrix_x4 for A, ldmatrix_x2_trans for b
     * if you want ABt instead of AB, just use ldmatrix for A and B
     */
    __device__ void hmma_16_8_16_f16(uint32_t rd[2], uint32_t ra[4], uint32_t rb[2], uint32_t rc[2])
    {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
                     : "=r"(rd[0]), "=r"(rd[1])
                     : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]), "r"(rb[1]), "r"(rc[0]), "r"(rc[1]));
    }


    // Taken from ThunderKittens mma_AB_base
    // d is 16x8
    __device__ static inline void hmma_16_8_16_bf16_float(float d[4],
                                                          uint32_t a[4], // packed, so actually 8 elements
                                                          uint32_t b[2],
                                                          float c[4])
    {
        asm volatile(
            // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#multiply-and-accumulate-instruction-mma
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};"

            // D matrix
            : "+f"(d[0]), "+f"(d[1]),
              "+f"(d[2]), "+f"(d[3])

            // A matrix
            : "r"(a[0]), "r"(a[1]),
              "r"(a[2]), "r"(a[3]),

              // B matrix
              "r"(b[0]), "r"(b[1]),

              // C matrix
              "f"(c[0]), "f"(c[1]),
              "f"(c[2]), "f"(c[3]));
    }

    __device__ static inline void hmma16816_tk(      float2 &d0,       float2 &d1,
                                            const bf162 &a0, const bf162 &a1, const bf162 &a2, const bf162 &a3,
                                            const bf162 &b0, const bf162 &b1,
                                            const float2 &c0, const float2 &c1                                    ) {
        asm volatile(
            // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#multiply-and-accumulate-instruction-mma
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
            "{%0, %1, %2, %3}, " \
            "{%4, %5, %6, %7}, " \
            "{%8, %9}, " \
            "{%10, %11, %12, %13};"

            // D matrix
        :   "+f"(d0.x), "+f"(d0.y),
            "+f"(d1.x), "+f"(d1.y)

            // A matrix
        :   "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),
            "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),

            // B matrix
            "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),

            // C matrix
            "f"(c0.x), "f"(c0.y),
            "f"(c1.x), "f"(c1.y)
        );
    }

    __device__ static inline void stsm4(uint32_t dst, bf162& src1, bf162& src2, bf162& src3, bf162& src4) {
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%4], {%0, %1, %2, %3};\n" ::
                     "r"(*(uint32_t*)&src1), "r"(*(uint32_t*)&src2), "r"(*(uint32_t*)&src3), "r"(*(uint32_t*)&src4), "r"(dst));
    }

}