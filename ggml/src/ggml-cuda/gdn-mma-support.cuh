// Isolated helpers from ggml-org/llama.cpp #29353/#29612, MIT.
// Copyright (c) 2023-2026 The ggml authors. See repository LICENSE.
#pragma once
#include "mma.cuh"
namespace ggml_cuda_gdn_support {
using namespace ggml_cuda_mma;
    template <int stride, typename T>
    static __device__ __forceinline__ uint32_t swizzle(const uint32_t offset, const uint32_t i) {
        static_assert(sizeof(T) <= 4, "unsupported type size");
        constexpr int stride_bytes = stride*sizeof(T);
        static_assert(stride_bytes % 16 == 0, "bad stride");
        constexpr uint32_t shift = sizeof(T) == 1 ? 4 : (sizeof(T) == 2 ? 3 : 2);
        if (stride_bytes % 32 != 0) {
            return offset; // Equivalent to padding with 16 bytes.
        }
        if (stride_bytes % 64 != 0) {
            return offset ^ (((i / 4) % 2) << shift);
        }
        if (stride_bytes % 128 != 0) {
            return offset ^ (((i / 2) % 4) << shift);
        }
        return offset ^ ((i % 8) << shift);
    }

    template <int stride, typename T>
    static __device__ __forceinline__ T * swizzle(T * ptr, const uint32_t offset, const uint32_t i) {
        return ptr + swizzle<stride, T>(offset, i);
    }

    template <int stride, int I, int J, typename T, data_layout dl>
    static __device__ __forceinline__ void load_ldmatrix_swizzled(
            tile<I, J, T, dl> & t, const T * __restrict__ xs0, const int offset) {
#if defined(TURING_MMA_AVAILABLE)
        static_assert(I == 8 || I == 16, "bad tile width");
        static_assert(J ==  8, "bad tile height");
        const int i = threadIdx.x % t.I;
        const int j = ((threadIdx.x / t.I) * (t.J / 2)) % t.J;
        int offset_ij = offset + i * stride + j;
        offset_ij = swizzle<stride, T>(offset_ij, i);
        int * xi = (int *) t.x;
        if constexpr (I == 16) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
                : "l"(xs0 + offset_ij));
        } else {
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
                : "=r"(xi[0]), "=r"(xi[1])
                : "l"(xs0 + offset_ij));
        }
#else
        GGML_UNUSED_VARS(t,xs0,offset); NO_DEVICE_CODE;
#endif
    }
    template <int stride, int I, typename T, data_layout dl>
    static __device__ __forceinline__ void load_ldmatrix_trans_swizzled(
            tile<I, 8, T, dl> & t, const T * __restrict__ xs0, const int offset) {
#if defined(TURING_MMA_AVAILABLE)
        static_assert(I == 8 || I == 16, "bad tile width");
        static_assert(dl == DATA_LAYOUT_I_MAJOR, "bad data layout");
        const int i = threadIdx.x % 16;
        const int j = I == 16 ? (threadIdx.x / 16) * (t.J / 2) : 0;
        int offset_ij = offset + i * stride + j;
        offset_ij = swizzle<stride, T>(offset_ij, i);
        int * xi = (int *) t.x;
        if constexpr (I == 16) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])
                : "l"(xs0 + offset_ij));
        } else {
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0, %1}, [%2];"
                : "=r"(xi[0]), "=r"(xi[1])
                : "l"(xs0 + offset_ij));
        }
#else
        GGML_UNUSED_VARS(t,xs0,offset); NO_DEVICE_CODE;
#endif
    }
}
