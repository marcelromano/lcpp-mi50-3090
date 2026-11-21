#include "allreduce.cuh"

#if !defined(GGML_USE_MUSA)

#include "convert.cuh"
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>

// ---------------------------------------------------------------------------
// AllReduce for tensor-parallel inference across two GPUs (CUDA or
// ROCm/HIP).
//
// Provides an in-place sum reduction over matching tensors on two GPUs
// in the same process.  Used by the tensor-split path alongside NCCL;
// targets setups without NVLink/xGMI, where data is exchanged between the
// GPUs by staging it through pinned host memory over PCIe.
//
// Two reduction strategies are selected per call by tensor size:
//
//   * Chunked kernel path (small reductions): a single CUDA kernel both
//     stages data through pinned host memory and performs the local sum.
//     Cross-GPU synchronization happens *inside the kernel* (busy-wait on
//     a host-memory flag), which keeps launch overhead low for the
//     latency-sensitive token-generation case.
//
//   * Copy-engine path (large reductions): the transfer is split into
//     D2H + H2D cudaMemcpyAsync chunks driven by the GPU's copy engine,
//     followed by a small device-side add kernel.  Cross-GPU
//     synchronization happens *outside the kernel*, via CUDA events
//     between streams.  This keeps the compute engine free while large
//     transfers are in flight, which matters for prefill-sized tensors.
//     Reductions larger than the per-call inner cap are processed by an
//     outer chunker that issues sequential inner calls.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cross-GPU signal mechanism
//
// One int per (slot, rank) pair in pinned host memory.  Each AR call writes a
// strictly increasing token (= the AR call number) into its own arrival int.
// The peer spins until its read of the other's arrival int equals the token
// it expects for this call -- a mismatch means the peer hasn't arrived yet.
// Tokens never repeat over realistic call rates (32-bit int wraps in tens of
// days at thousands of ARs/sec), so arrival ints don't need to be reset
// between calls; we initialize once at pipeline init and let the values
// accumulate.
//
// There is exactly one writer (the owning GPU) and one reader (the peer), so
// we don't need atomics.  A volatile store paired with __threadfence_system()
// provides the release ordering that makes the D2H writes visible system-wide
// before the arrival token is observed.
//
// atomicAdd_system() requires hostNativeAtomicSupported, which is unavailable
// on PCIe-attached consumer GPUs without NVLink, so the volatile path is the
// portable choice.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void ggml_cuda_ar_signal_set(int * p, int token) {
    *(volatile int *)p = token;
}
static __device__ __forceinline__ int ggml_cuda_ar_signal_get(const int * p) {
    return *(const volatile int *)p;
}

// Byte spacing between adjacent arrival ints.  64 bytes (one cache line)
// ensures each GPU/block's arrival slot lives on its own line, preventing
// false-sharing stalls on the polling GPU.
static constexpr size_t GGML_CUDA_AR_ARRIVAL_STRIDE = 64;

// Number of blocks the chunked kernel launches with.  Each block stripes a
// disjoint slice of the data and synchronizes through its own arrival-token
// slot so multiple SMs can pump PCIe stores in parallel.
static constexpr int GGML_CUDA_AR_KERNEL_BLOCKS = 8;

// ---------------------------------------------------------------------------
// Chunked kernel AllReduce -- 2 GPUs, supports float, half, and bfloat16.
//
// Both GPUs run this kernel simultaneously on independent streams.  sendbuf
// and recvbuf live in T_dst (the caller's tensor type); host_mine / host_other
// carry data in T_wire (the on-wire type, possibly narrower than T_dst -- e.g.
// T_dst=F32 with T_wire=BF16 halves the bytes pushed across PCIe).  When
// T_dst == T_wire the casts below are no-ops.
//
// Each GPU runs three phases:
//
//   Phase 1 (all threads): cast sendbuf (T_dst) -> T_wire and store as
//                          single-instruction-width vectors into host_mine.
//                          __threadfence_system() commits these writes to host
//                          memory.
//   Phase 2 (thread 0):    write token to arrival_mine; spin until
//                          arrival_other == token.
//   Phase 3 (all threads): read T_wire vectors from host_other, cast
//                          each element to T_dst, and sum with the local
//                          sendbuf value (also rounded through T_wire so that
//                          both GPUs truncate identically -- this guarantees
//                          bit-equivalent results across the two devices).
//
// Multi-block: blocks stripe vectors across (gridDim.x * blockDim.x) global
// threads to keep multiple SMs issuing PCIe stores in parallel.  Each block
// has its own arrival-token slot (offset by blockIdx.x * ARRIVAL_STRIDE);
// thread 0 of each block signals/spins on that slot independently of other
// blocks.  Tail elements (the leftover < ELEMS_PER_VEC at the end) are
// handled only by block 0 to avoid cross-block writes to the same slots.
// ---------------------------------------------------------------------------
template <typename T_dst, typename T_wire>
static __global__ void ggml_cuda_ar_kernel(
        const T_dst  *              sendbuf,
        T_dst        *              recvbuf,
        T_wire       * __restrict__ host_mine,
        const T_wire * __restrict__ host_other,
        int                         count,
        int *                       arrival_mine,
        int *                       arrival_other,
        int                         token) {

    // Vector unit for the wire type, sized to the arch's widest single-instruction
    // copy (16 B on Volta+).  Each phase-1 iter writes one vector to host memory;
    // each phase-3 iter reads one and produces ELEMS_PER_VEC sums.
    constexpr int ELEMS_PER_VEC = ggml_cuda_get_max_cpy_bytes() / sizeof(T_wire);
    constexpr int ARRIVAL_INTS  = (int)(GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(int));

    const int tid       = threadIdx.x;
    const int nt        = blockDim.x;
    const int bid       = blockIdx.x;
    const int gtid      = bid * nt + tid;
    const int gnt       = gridDim.x * nt;
    const int count_vec = count / ELEMS_PER_VEC;
    const int tail      = count_vec * ELEMS_PER_VEC;

    // Phase 1: cast sendbuf (T_dst) -> host_mine (T_wire) and store as vectors.
    {
        for (int i = gtid; i < count_vec; i += gnt) {
            const int off = i * ELEMS_PER_VEC;
            T_wire wire[ELEMS_PER_VEC];
            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                wire[k] = ggml_cuda_cast<T_wire>(sendbuf[off + k]);
            }
            ggml_cuda_memcpy_1<sizeof(wire)>(&host_mine[off], wire);
        }
        if (bid == 0 && tid < count - tail) {
            host_mine[tail + tid] = ggml_cuda_cast<T_wire>(sendbuf[tail + tid]);
        }
    }

    // Commit this block's host writes before signalling.
    __threadfence_system();
    __syncthreads();

    // Phase 2: thread 0 of each block signals on its own arrival slot, then
    // spins for the matching slot from peer.  Per-block tokens mean blocks
    // proceed independently -- no inter-block barrier needed.
    if (tid == 0) {
        int       * my_slot    = arrival_mine  + bid * ARRIVAL_INTS;
        const int * other_slot = arrival_other + bid * ARRIVAL_INTS;

        ggml_cuda_ar_signal_set(my_slot, token);
        __threadfence_system(); // make our signal visible system-wide

        while (ggml_cuda_ar_signal_get(other_slot) != token) {
#ifdef GGML_USE_HIP
            // Equals ~100ns at 2500 MHz (sleeps for n * [1,64] clock cycles)
            __builtin_amdgcn_s_sleep(4);
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
            __nanosleep(100);
#else
            NO_DEVICE_CODE;
#endif // GGML_USE_HIP
        }
    }

    __syncthreads();

    // Acquire peer's host_other writes (this block's stripe of them).
    __threadfence_system();

    // Phase 3: read peer's T_wire vector, cast both sides through T_wire for
    // bit-equivalence, sum in T_dst precision, and write back to recvbuf.
    {
        for (int i = gtid; i < count_vec; i += gnt) {
            const int off = i * ELEMS_PER_VEC;
            T_wire wire[ELEMS_PER_VEC];
            ggml_cuda_memcpy_1<sizeof(wire)>(wire, &host_other[off]);
            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                const T_wire d_low = ggml_cuda_cast<T_wire>(sendbuf[off + k]);
                recvbuf[off + k] = ggml_cuda_cast<T_dst>(
                    ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(wire[k]));
            }
        }
        if (bid == 0 && tid < count - tail) {
            const T_wire d_low = ggml_cuda_cast<T_wire>(sendbuf[tail + tid]);
            recvbuf[tail + tid] = ggml_cuda_cast<T_dst>(
                ggml_cuda_cast<float>(d_low) +
                ggml_cuda_cast<float>(host_other[tail + tid]));
        }
    }
}

// Combined load-convert-add kernel.  The peer's contribution arrives as T_src
// (which may be a lower-precision type than T_dst when the BF16 round-trip is
// active).  For bit-equivalence between the two GPUs, dst is first rounded
// through T_src's precision via ggml_cuda_cast -- peer already truncated its
// own value the same way before sending -- so both sides perform identical
// arithmetic.  When T_dst == T_src the round-trip cast is a no-op.
template <typename T_dst, typename T_src>
static __global__ void ggml_cuda_ar_add_kernel(
        T_dst       * __restrict__ dst,
        const T_src * __restrict__ src,
        int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt  = gridDim.x * blockDim.x;
    for (int i = tid; i < count; i += nt) {
        const T_src d_low = ggml_cuda_cast<T_src>(dst[i]);
        dst[i] = ggml_cuda_cast<T_dst>(
            ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(src[i]));
    }
}

// P2P direct-write kernels: Phase 1 converts/packs local shard (or 0 when inactive)
// directly into peer's device receive buffer over PCIe; Phase 2 performs the exact
// symmetric wire accumulation float(Q(dst)) + float(peer_wire) so rank 0 and rank 1
// remain bitwise identical with upstream #27825.
template <typename T_src, typename T_wire>
static __global__ void ggml_cuda_ar_p2p_write_kernel(
        const T_src * __restrict__ src,
        T_wire      * __restrict__ peer_wire,
        int count,
        bool active) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt  = gridDim.x * blockDim.x;
    if constexpr (std::is_same_v<T_src, float> && std::is_same_v<T_wire, nv_bfloat16>) {
        if ((count & 1) == 0 &&
            ((reinterpret_cast<uintptr_t>(src) & 7u) == 0) &&
            ((reinterpret_cast<uintptr_t>(peer_wire) & 3u) == 0)) {
            const int pairs = count >> 1;
            const float2 * src2 = reinterpret_cast<const float2 *>(src);
            uint32_t * dst2 = reinterpret_cast<uint32_t *>(peer_wire);
            for (int k = tid; k < pairs; k += nt) {
                const float2 v = active ? src2[k] : make_float2(0.0f, 0.0f);
                union { nv_bfloat16 h[2]; uint32_t u; } pk;
                pk.h[0] = ggml_cuda_cast<nv_bfloat16>(v.x);
                pk.h[1] = ggml_cuda_cast<nv_bfloat16>(v.y);
                dst2[k] = pk.u;
            }
            return;
        }
    }
    for (int i = tid; i < count; i += nt) {
        peer_wire[i] = active ? ggml_cuda_cast<T_wire>(src[i]) : ggml_cuda_cast<T_wire>(0.0f);
    }
}

template <typename T_dst, typename T_wire>
static __global__ void ggml_cuda_ar_p2p_sym_add_kernel(
        T_dst        * __restrict__ dst,
        const T_wire * __restrict__ peer_wire,
        int count,
        bool active) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt  = gridDim.x * blockDim.x;
    if constexpr (std::is_same_v<T_dst, float> && std::is_same_v<T_wire, nv_bfloat16>) {
        if ((count & 1) == 0 &&
            ((reinterpret_cast<uintptr_t>(dst) & 7u) == 0) &&
            ((reinterpret_cast<uintptr_t>(peer_wire) & 3u) == 0)) {
            const int pairs = count >> 1;
            float2 * dst2 = reinterpret_cast<float2 *>(dst);
            const uint32_t * peer2 = reinterpret_cast<const uint32_t *>(peer_wire);
            for (int k = tid; k < pairs; k += nt) {
                const float2 cur = active ? dst2[k] : make_float2(0.0f, 0.0f);
                union { uint32_t u; nv_bfloat16 h[2]; } pk;
                pk.u = peer2[k];
                const nv_bfloat16 l0 = ggml_cuda_cast<nv_bfloat16>(cur.x);
                const nv_bfloat16 l1 = ggml_cuda_cast<nv_bfloat16>(cur.y);
                float2 out;
                out.x = ggml_cuda_cast<float>(l0) + ggml_cuda_cast<float>(pk.h[0]);
                out.y = ggml_cuda_cast<float>(l1) + ggml_cuda_cast<float>(pk.h[1]);
                dst2[k] = out;
            }
            return;
        }
    }
    for (int i = tid; i < count; i += nt) {
        const T_wire d_low = active ? ggml_cuda_cast<T_wire>(dst[i]) : ggml_cuda_cast<T_wire>(0.0f);
        dst[i] = ggml_cuda_cast<T_dst>(
            ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(peer_wire[i]));
    }
}

static constexpr int    GGML_CUDA_AR_FUSED_MAX_BLOCKS = 64;
static constexpr int    GGML_CUDA_AR_FUSED_THREADS    = 256;
static constexpr size_t GGML_CUDA_AR_FUSED_HALF_BYTES = 256 * 1024;

// Uncached GPU-side arrival flags + monotonic step counter per block.
struct alignas(128) ggml_cuda_ar_fused_signal {
    uint32_t start[GGML_CUDA_AR_FUSED_MAX_BLOCKS][2];
    uint32_t end[GGML_CUDA_AR_FUSED_MAX_BLOCKS][2];
    uint32_t flag[GGML_CUDA_AR_FUSED_MAX_BLOCKS];
};

static __device__ __forceinline__ uint32_t ggml_cuda_ar_sys_load_acq(const uint32_t * addr) {
#if defined(GGML_USE_HIP)
    return __hip_atomic_load(addr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    return *reinterpret_cast<const volatile uint32_t *>(addr);
#endif
}

static __device__ __forceinline__ void ggml_cuda_ar_sys_store_rel(uint32_t * addr, uint32_t val) {
#if defined(GGML_USE_HIP)
    __hip_atomic_store(addr, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    __threadfence_system();
    *reinterpret_cast<volatile uint32_t *>(addr) = val;
#endif
}

template <typename T_uncached>
static __device__ __forceinline__ T_uncached ggml_cuda_ar_uncached_load(const T_uncached * addr) {
#if defined(GGML_USE_HIP)
    return __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    return *reinterpret_cast<const volatile T_uncached *>(addr);
#endif
}

template <typename T_dst, typename T_wire>
__launch_bounds__(GGML_CUDA_AR_FUSED_THREADS)
static __global__ void ggml_cuda_ar_fused_kernel(
        T_dst                      * __restrict__ data,
        size_t                                    ne,
        int                                       rank,
        bool                                      active,
        ggml_cuda_ar_fused_signal  * __restrict__ self_sg,
        ggml_cuda_ar_fused_signal  * __restrict__ peer_sg,
        uint8_t                    * __restrict__ self_stage,
        uint8_t                    * __restrict__ peer_stage) {
    const int peer = 1 - rank;
    __shared__ uint32_t s_flag;
    if (threadIdx.x == 0) {
        s_flag = self_sg->flag[blockIdx.x] + 1u;
    }
    __syncthreads();
    const uint32_t flag    = s_flag;
    const uint32_t buf_idx = flag & 1u;

    uint8_t *       peer_slot = peer_stage + (size_t) buf_idx * GGML_CUDA_AR_FUSED_HALF_BYTES;
    const uint8_t * self_slot = self_stage + (size_t) buf_idx * GGML_CUDA_AR_FUSED_HALF_BYTES;

    // All eligible inputs have (nbytes & 0xF) == 0, which guarantees (ne & 3u) == 0.
    const size_t n_q   = ne >> 2;
    const size_t q_beg = (size_t(blockIdx.x) * n_q) / gridDim.x;
    const size_t q_end = (size_t(blockIdx.x + 1) * n_q) / gridDim.x;

    if constexpr (std::is_same_v<T_dst, float> && std::is_same_v<T_wire, nv_bfloat16>) {
        const float4 * __restrict__ src_v4 = reinterpret_cast<const float4 *>(data);
        float4       * __restrict__ dst_v4 = reinterpret_cast<float4 *>(data);
        uint64_t     * __restrict__ peer_w = reinterpret_cast<uint64_t *>(peer_slot);
        const uint64_t * __restrict__ self_w = reinterpret_cast<const uint64_t *>(self_slot);

        for (size_t i = q_beg + threadIdx.x; i < q_end; i += blockDim.x) {
            const float4 f = active ? src_v4[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            union { nv_bfloat16 h[4]; uint64_t u; } pk;
            pk.h[0] = ggml_cuda_cast<nv_bfloat16>(f.x);
            pk.h[1] = ggml_cuda_cast<nv_bfloat16>(f.y);
            pk.h[2] = ggml_cuda_cast<nv_bfloat16>(f.z);
            pk.h[3] = ggml_cuda_cast<nv_bfloat16>(f.w);
            peer_w[i] = pk.u;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            __threadfence_system();
            ggml_cuda_ar_sys_store_rel(&peer_sg->start[blockIdx.x][rank], flag);
            while (int32_t(ggml_cuda_ar_sys_load_acq(&self_sg->start[blockIdx.x][peer]) - flag) < 0) {
#if defined(__HIP_DEVICE_COMPILE__)
                __builtin_amdgcn_s_sleep(1);
#endif
            }
        }
        __syncthreads();

        for (size_t i = q_beg + threadIdx.x; i < q_end; i += blockDim.x) {
            union { uint64_t u; nv_bfloat16 h[4]; } pk;
            pk.u = ggml_cuda_ar_uncached_load(&self_w[i]);
            const float4 f = active ? src_v4[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const nv_bfloat16 l0 = ggml_cuda_cast<nv_bfloat16>(f.x);
            const nv_bfloat16 l1 = ggml_cuda_cast<nv_bfloat16>(f.y);
            const nv_bfloat16 l2 = ggml_cuda_cast<nv_bfloat16>(f.z);
            const nv_bfloat16 l3 = ggml_cuda_cast<nv_bfloat16>(f.w);
            float4 out;
            out.x = ggml_cuda_cast<float>(l0) + ggml_cuda_cast<float>(pk.h[0]);
            out.y = ggml_cuda_cast<float>(l1) + ggml_cuda_cast<float>(pk.h[1]);
            out.z = ggml_cuda_cast<float>(l2) + ggml_cuda_cast<float>(pk.h[2]);
            out.w = ggml_cuda_cast<float>(l3) + ggml_cuda_cast<float>(pk.h[3]);
            dst_v4[i] = out;
        }
    } else {
        using wire_bits_t = std::conditional_t<sizeof(T_wire) == 4, uint32_t, uint16_t>;
        T_wire       * __restrict__ peer_w = reinterpret_cast<T_wire *>(peer_slot);
        const T_wire * __restrict__ self_w = reinterpret_cast<const T_wire *>(self_slot);

        const size_t e_beg = q_beg << 2;
        const size_t e_end = q_end << 2;
        for (size_t i = e_beg + threadIdx.x; i < e_end; i += blockDim.x) {
            peer_w[i] = active ? ggml_cuda_cast<T_wire>(data[i]) : ggml_cuda_cast<T_wire>(0.0f);
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            __threadfence_system();
            ggml_cuda_ar_sys_store_rel(&peer_sg->start[blockIdx.x][rank], flag);
            while (int32_t(ggml_cuda_ar_sys_load_acq(&self_sg->start[blockIdx.x][peer]) - flag) < 0) {
#if defined(__HIP_DEVICE_COMPILE__)
                __builtin_amdgcn_s_sleep(1);
#endif
            }
        }
        __syncthreads();

        for (size_t i = e_beg + threadIdx.x; i < e_end; i += blockDim.x) {
            const wire_bits_t raw = ggml_cuda_ar_uncached_load(reinterpret_cast<const wire_bits_t *>(&self_w[i]));
            T_wire pval;
            std::memcpy(&pval, &raw, sizeof(T_wire));
            const T_wire d_low = active ? ggml_cuda_cast<T_wire>(data[i]) : ggml_cuda_cast<T_wire>(0.0f);
            data[i] = ggml_cuda_cast<T_dst>(ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(pval));
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        self_sg->flag[blockIdx.x] = flag;
        ggml_cuda_ar_sys_store_rel(&peer_sg->end[blockIdx.x][rank], flag);
        while (int32_t(ggml_cuda_ar_sys_load_acq(&self_sg->end[blockIdx.x][peer]) - flag) < 0) {
#if defined(__HIP_DEVICE_COMPILE__)
            __builtin_amdgcn_s_sleep(1);
#endif
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline structure
// ---------------------------------------------------------------------------

// Number of slots in the event / arrival ring.  Two slots is sufficient:
// lockstep guarantees the two GPUs are at most one AR (or chunk) apart, so
// slot[N%2] is always safe to reuse -- peer has already consumed slot[N%2]
// from AR N-2 by the time we get to AR N.  acquire_slot's
// cudaEventSynchronize on ev.ker for both devices makes that consumption
// explicit before we overwrite host_buf[slot] for the new AR.
static constexpr int GGML_CUDA_AR_POOL_SIZE = 2;

// Maximum chunk size (bytes per GPU) handled by one chunked kernel launch.
// Larger tensors are reduced by issuing multiple chunked launches.
static constexpr size_t GGML_CUDA_AR_MAX_BYTES = 1024 * 1024; // 1 MB

// Copy-engine path: largest tensor accepted on this path; sets host_large /
// dev_tmp allocation size.
static constexpr size_t GGML_CUDA_AR_COPY_MAX_BYTES = 32 * 1024 * 1024; // 32 MB

// AR wire size at which the copy-engine path takes over from the chunked-
// kernel path.  Override via GGML_CUDA_AR_COPY_THRESHOLD.
static constexpr size_t GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT = 1024 * 1024; // 1 MB
// Per-call CE chunk-size heuristic: chunk_bytes = clamp(nbytes / 4, MIN, MAX).
// The /4 keeps ~4 chunks in flight at any moment (good D2H/H2D overlap with
// the peer); the clamps cover the cases where nbytes/4 is too small (per-
// memcpy fixed cost dominates) or too large (chunk-level pipelining stalls).
// Env var GGML_CUDA_AR_COPY_CHUNK_BYTES can override with a fixed value.
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MIN = 512 * 1024;       // 512 KB
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MAX = 2 * 1024 * 1024;  // 2 MB
// Absolute floor that an env-var override is allowed to set; this caps the
// per-slot copy-event array.  256 KB -> up to 128 chunks per 32 MB tensor.
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN = 256 * 1024;
static constexpr int GGML_CUDA_AR_COPY_MAX_CHUNKS =
    static_cast<int>((GGML_CUDA_AR_COPY_MAX_BYTES + GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN - 1) /
                    GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN);

struct ggml_cuda_ar_event_slot {
    cudaEvent_t app = nullptr;  // upstream computation complete
    cudaEvent_t cpy[GGML_CUDA_AR_COPY_MAX_CHUNKS] = {};  // copy-engine D2H chunks complete
    cudaEvent_t h2d = nullptr;  // copy-engine H2Ds complete (handoff AR stream -> compute stream)
    cudaEvent_t ker = nullptr;  // AllReduce kernel complete
};

// Mapped pinned host allocation: cudaHostAlloc + cudaHostGetDevicePointer
// in one place, with the host handle preserved for cudaFreeHost.  Used where
// the CPU never touches the buffer -- only the device reads/writes via the
// mapped device pointer.  Required on systems where cudaDevAttrCanUseHost-
// PointerForRegisteredMem is 0 and the host pointer can't be used as a
// device pointer.
struct ggml_cuda_ar_host_mapping {
    uint8_t * host = nullptr;   // cudaFreeHost handle; also the H-side ptr for cudaMemcpyAsync
    uint8_t * dev  = nullptr;   // device-side pointer for kernels / cudaMemset

    cudaError_t alloc(size_t bytes) {
        cudaError_t rc = cudaHostAlloc(reinterpret_cast<void **>(&host), bytes,
                                       cudaHostAllocPortable | cudaHostAllocMapped);
        if (rc != cudaSuccess) {
            host = nullptr;
            return rc;
        }
        rc = cudaHostGetDevicePointer(reinterpret_cast<void **>(&dev), host, 0);
        if (rc != cudaSuccess) {
            CUDA_CHECK(cudaFreeHost(host));
            host = nullptr;
            dev  = nullptr;
        }
        return rc;
    }

    void free() {
        if (host) {
            CUDA_CHECK(cudaFreeHost(host));
            host = nullptr;
            dev  = nullptr;
        }
    }
};

struct ggml_cuda_ar_pipeline {
    int      n_devices;
    int      devices[GGML_CUDA_MAX_DEVICES];
    size_t   buf_bytes;    // bytes per device in host_buf[]
    size_t   copy_bytes;   // bytes per device in host_large[] / dev_tmp[]
    size_t   copy_threshold;
    size_t   copy_chunk_bytes;
    size_t   bf16_threshold; // tensors >= this size (bytes) are reduced via FP32->BF16 round-trip; 0 disables
    uint64_t call_count;

    // Per-device resources.
    ggml_cuda_ar_host_mapping host_buf[GGML_CUDA_MAX_DEVICES];   // pinned staging (chunked kernel)
    ggml_cuda_ar_host_mapping host_large[GGML_CUDA_MAX_DEVICES]; // pinned staging (copy-engine)
    char *                    dev_tmp[GGML_CUDA_MAX_DEVICES];    // device scratch for copy-engine path
    cudaStream_t             streams[GGML_CUDA_MAX_DEVICES];   // non-blocking
    ggml_cuda_ar_event_slot  ev_pool[GGML_CUDA_MAX_DEVICES][GGML_CUDA_AR_POOL_SIZE];

    // Copy-engine: per-device "I finished reading my peer's host_large"
    // event.  Indexed by RECORDER device.  Recorded same-device on streams[i]
    // after stage 2's last H2D from host_large[peer].  Waited cross-device
    // by peer's stage-1 stream before the next AR overwrites host_large[peer].
    cudaEvent_t              host_large_read_done[GGML_CUDA_MAX_DEVICES];
    bool                     host_large_read_done_valid;

    // Copy-engine: per-device "my add_kernel is done with dev_tmp" event.
    // Recorded on the compute stream after each add_kernel; the AR stream
    // waits on it before the next copy_impl's H2D overwrites dev_tmp.  Lets us
    // single-buffer dev_tmp despite add_kernel running on a separate stream.
    cudaEvent_t              dev_tmp_kernel_done[GGML_CUDA_MAX_DEVICES];
    bool                     dev_tmp_kernel_done_valid;

    // Arrival ring: ARRIVAL_STRIDE bytes between adjacent ints.  Mapped pinned
    // memory; CPU never reads/writes -- only the kernel and cudaMemset.
    // Use ggml_cuda_ar_arrival_ptr() to index.
    ggml_cuda_ar_host_mapping arrival;

    // P2P event direct-write staging: 8x 256 KiB small slots + 1x 16 MiB large chunk buffer per GPU.
    static constexpr int    P2P_SMALL_SLOTS = 8;
    static constexpr size_t P2P_SMALL_BYTES = 256 * 1024;
    static constexpr size_t P2P_LARGE_BYTES = 16 * 1024 * 1024;

    bool        enable_p2p;
    bool        p2p_ready;
    uint8_t *   p2p_small_buf[GGML_CUDA_MAX_DEVICES];
    uint8_t *   p2p_large_buf[GGML_CUDA_MAX_DEVICES];
    cudaEvent_t p2p_staged[GGML_CUDA_MAX_DEVICES][P2P_SMALL_SLOTS];
    cudaEvent_t p2p_done[GGML_CUDA_MAX_DEVICES][P2P_SMALL_SLOTS];
    cudaEvent_t p2p_staged_large[GGML_CUDA_MAX_DEVICES];
    cudaEvent_t p2p_done_large[GGML_CUDA_MAX_DEVICES];
    bool        p2p_small_used[P2P_SMALL_SLOTS];
    bool        p2p_large_used;
    uint64_t    p2p_seq;

    // Fused GPU-barrier P2P path: uncached ping-pong staging + signal blocks.
    bool                        enable_fused;
    bool                        fused_ready;
    bool                        fused_used;
    size_t                      fused_max_bytes;
    int                         fused_max_blocks;
    ggml_cuda_ar_fused_signal * fused_sig[GGML_CUDA_MAX_DEVICES];
    uint8_t *                   fused_stage[GGML_CUDA_MAX_DEVICES];

    // Atomic dispatch telemetry (F1d verification).
    enum stat_id {
        STAT_HOST_CHUNKED    = 0,
        STAT_HOST_COPY       = 1,
        STAT_P2P_EVENT       = 2,
        STAT_FUSED_DIRECT    = 3,
        STAT_FUSED_TG_LAUNCH = 4,
        STAT_WIRE_BF16       = 5,
        STAT_WIRE_F32        = 6,
        STAT_WIRE_F16        = 7,
        STAT_WIRE_NAT_BF16   = 8,
        STAT_P2P_INELIGIBLE  = 9,
        STAT_COUNT           = 10,
    };
    mutable std::atomic<uint64_t> stats[STAT_COUNT];
};

// Base pointer for the (slot, rank) per-block token block.  The kernel adds
// blockIdx.x * (ARRIVAL_STRIDE/sizeof(int)) internally to land on its own slot.
static int * ggml_cuda_ar_arrival_ptr(const ggml_cuda_ar_pipeline * p, int slot, int rank) {
    const size_t offset = ((size_t)slot * p->n_devices + rank) *
                          GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    return reinterpret_cast<int *>(p->arrival.dev + offset);
}

static bool ggml_cuda_ar_env_bool(const char * name, bool default_value) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    return strcmp(value, "0") != 0;
}

static uint64_t ggml_cuda_ar_env_u64(const char * name, uint64_t default_value) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    char * end = nullptr;
    const unsigned long long parsed = strtoull(value, &end, 10);
    return end != value ? (uint64_t) parsed : default_value;
}

struct ggml_cuda_ar_slot_info {
    int slot;
    int token;
};

static ggml_cuda_ar_slot_info ggml_cuda_ar_acquire_slot(ggml_cuda_ar_pipeline * p) {
    const int  slot        = static_cast<int>(p->call_count % GGML_CUDA_AR_POOL_SIZE);
    const bool pool_lapped = p->call_count >= GGML_CUDA_AR_POOL_SIZE;
    p->call_count++;

    if (pool_lapped) {
        for (int i = 0; i < p->n_devices; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaEventSynchronize(p->ev_pool[i][slot].ker));
        }
    }

    return { slot, (int) p->call_count };
}

// Per-AR copy-engine chunk size: env-var override if set, else heuristic
// (clamp(nbytes/4, HEURISTIC_MIN, HEURISTIC_MAX)).
static size_t ggml_cuda_ar_chunk_bytes(const ggml_cuda_ar_pipeline * p, size_t nbytes) {
    if (p->copy_chunk_bytes > 0) {
        return p->copy_chunk_bytes;
    }
    return std::min(GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MAX,
                    std::max(GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MIN, nbytes / 4));
}

static void ggml_cuda_ar_wait_for_compute(
        ggml_cuda_ar_pipeline * p, ggml_backend_cuda_context * cuda_ctx, int rank, int slot) {
    ggml_cuda_ar_event_slot & ev = p->ev_pool[rank][slot];
    CUDA_CHECK(cudaEventRecord(ev.app, cuda_ctx->stream()));
    CUDA_CHECK(cudaStreamWaitEvent(p->streams[rank], ev.app));
}

// ---------------------------------------------------------------------------
// Init / free
// ---------------------------------------------------------------------------

ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int * devices, size_t n_devices) {

    if (n_devices != 2) {
        GGML_LOG_DEBUG("%s: internal AllReduce only supports n_devices=2 (got %zu); "
                       "falling back\n", __func__, n_devices);
        return nullptr;
    }

    // The chunked kernel uses __nanosleep (NVIDIA, sm70+) or
    // __builtin_amdgcn_s_sleep (AMD).
    for (size_t i = 0; i < n_devices; ++i) {
        const int cc = ggml_cuda_info().devices[devices[i]].cc;
        if (cc < GGML_CUDA_CC_VOLTA) {
            GGML_LOG_DEBUG("%s: internal AllReduce requires compute capability >= %d "
                           "(device %d has cc=%d); falling back\n",
                           __func__, GGML_CUDA_CC_VOLTA, devices[i], cc);
            return nullptr;
        }
    }

    auto * p = new ggml_cuda_ar_pipeline{};
    p->n_devices        = n_devices;
    p->copy_bytes       = GGML_CUDA_AR_COPY_MAX_BYTES;
    p->copy_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_COPY_THRESHOLD", GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT);
    // 0 = use the per-call heuristic (default).  Non-zero env value forces a
    // fixed chunk size for diagnostics, with a floor at COPY_CHUNK_BYTES_MIN.
    p->copy_chunk_bytes = ggml_cuda_ar_env_u64("GGML_CUDA_AR_COPY_CHUNK_BYTES", 0);
    if (p->copy_chunk_bytes > 0 && p->copy_chunk_bytes < GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN) {
        GGML_LOG_WARN("%s: GGML_CUDA_AR_COPY_CHUNK_BYTES=%zu below minimum %zu; clamping\n",
                      __func__, p->copy_chunk_bytes, GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN);
        p->copy_chunk_bytes = GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN;
    }
    // Default 1: BF16 round-trip is always on for F32 inputs (any non-zero
    // ne).  Set GGML_CUDA_AR_BF16_THRESHOLD=0 to disable, or to a larger
    // byte threshold to opt out for small tensors.
    p->bf16_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_BF16_THRESHOLD", 1);
    for (size_t i = 0; i < n_devices; ++i) {
        p->devices[i] = devices[i];
    }

    // Per-device streams and event pools.
    for (size_t i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);

        cudaStream_t stream = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaStreamCreateWithFlags failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        p->streams[i] = stream;

        for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
            bool ok =
                cudaEventCreateWithFlags(&p->ev_pool[i][s].app, cudaEventDisableTiming) == cudaSuccess &&
                cudaEventCreateWithFlags(&p->ev_pool[i][s].h2d, cudaEventDisableTiming) == cudaSuccess &&
                cudaEventCreateWithFlags(&p->ev_pool[i][s].ker, cudaEventDisableTiming) == cudaSuccess;
            for (int c = 0; ok && c < GGML_CUDA_AR_COPY_MAX_CHUNKS; ++c) {
                ok = cudaEventCreateWithFlags(&p->ev_pool[i][s].cpy[c], cudaEventDisableTiming) == cudaSuccess;
            }
            if (!ok) {
                GGML_LOG_ERROR("%s: cudaEventCreate failed for device %d slot %d\n",
                               __func__, p->devices[i], s);
                ggml_cuda_ar_pipeline_free(p);
                return nullptr;
            }
        }

        if (cudaEventCreateWithFlags(&p->host_large_read_done[i], cudaEventDisableTiming) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaEventCreate for host_large_read_done failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        if (cudaEventCreateWithFlags(&p->dev_tmp_kernel_done[i], cudaEventDisableTiming) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaEventCreate for dev_tmp_kernel_done failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    // Arrival ring: cache-line padded so each GPU's int is on its own line.
    const size_t arrival_bytes =
        (size_t)GGML_CUDA_AR_POOL_SIZE * n_devices *
        GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    if (p->arrival.alloc(arrival_bytes) != cudaSuccess) {
        GGML_LOG_ERROR("%s: alloc for arrival ring failed (%zu bytes)\n",
                       __func__, arrival_bytes);
        ggml_cuda_ar_pipeline_free(p);
        return nullptr;
    }
    ggml_cuda_set_device(p->devices[0]);
    if (cudaMemset(p->arrival.dev, 0, arrival_bytes) != cudaSuccess) {
        GGML_LOG_ERROR("%s: cudaMemset for arrival ring failed (%zu bytes)\n",
                       __func__, arrival_bytes);
        ggml_cuda_ar_pipeline_free(p);
        return nullptr;
    }

    // Per-device pinned staging buffers -- POOL_SIZE-deep ring so the chunked-
    // kernel can write the next slot's data while the peer is still reading
    // the previous slot's. Indexed by (slot * buf_bytes) at the call site.
    p->buf_bytes = GGML_CUDA_AR_MAX_BYTES;
    const size_t host_buf_total = (size_t) GGML_CUDA_AR_POOL_SIZE * p->buf_bytes;
    for (size_t i = 0; i < n_devices; ++i) {
        if (p->host_buf[i].alloc(host_buf_total) != cudaSuccess) {
            GGML_LOG_ERROR("%s: alloc for staging failed (%zu bytes)\n",
                           __func__, host_buf_total);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    // Copy-engine path: pinned host staging + device scratch, sized for the
    // largest tensor we accept on this path (GGML_CUDA_AR_COPY_MAX_BYTES).
    // dev_tmp is single-buffered; cross-AR safety is enforced by an explicit
    // cross-stream wait in copy_impl on the prior AR's add_kernel-done event.
    for (size_t i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        if (p->host_large[i].alloc(p->copy_bytes) != cudaSuccess) {
            GGML_LOG_ERROR("%s: alloc for large staging failed (%zu bytes)\n",
                           __func__, p->copy_bytes);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        if (cudaMalloc(reinterpret_cast<void **>(&p->dev_tmp[i]), p->copy_bytes) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaMalloc for copy scratch failed (%zu bytes) on device %d\n",
                           __func__, p->copy_bytes, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    const bool want_tg_env = ggml_cuda_ar_env_bool("GGML_HIP_AR_TG", false);
    p->enable_p2p   = ggml_cuda_ar_env_bool("GGML_HIP_AR_P2P", want_tg_env);
    p->enable_fused = ggml_cuda_ar_env_bool("GGML_HIP_AR_FUSED", want_tg_env);
    p->p2p_ready    = false;
    p->fused_ready  = false;
    p->fused_used   = false;

    const uint64_t req_fused_bytes = ggml_cuda_ar_env_u64("GGML_HIP_AR_FUSED_MAX_BYTES", GGML_CUDA_AR_FUSED_HALF_BYTES);
    p->fused_max_bytes = (size_t) std::min<uint64_t>(std::max<uint64_t>(req_fused_bytes, 16u), GGML_CUDA_AR_FUSED_HALF_BYTES);
    const uint64_t req_fused_blocks = ggml_cuda_ar_env_u64("GGML_HIP_AR_FUSED_MAX_BLOCKS", 16);
    p->fused_max_blocks = (int) std::min<uint64_t>(std::max<uint64_t>(req_fused_blocks, 1u), GGML_CUDA_AR_FUSED_MAX_BLOCKS);

    bool peer_ok = false;
    if ((p->enable_p2p || p->enable_fused) && n_devices == 2) {
        int can01 = 0, can10 = 0;
        cudaError_t pc0 = cudaDeviceCanAccessPeer(&can01, p->devices[0], p->devices[1]);
        cudaError_t pc1 = cudaDeviceCanAccessPeer(&can10, p->devices[1], p->devices[0]);
        if (pc0 == cudaSuccess && pc1 == cudaSuccess && can01 && can10) {
            peer_ok = true;
            for (int i = 0; i < 2; ++i) {
                ggml_cuda_set_device(p->devices[i]);
                cudaError_t pe = cudaDeviceEnablePeerAccess(p->devices[1 - i], 0);
                if (pe == cudaErrorPeerAccessAlreadyEnabled) {
                    (void) cudaGetLastError();
                } else if (pe != cudaSuccess) {
                    (void) cudaGetLastError();
                    peer_ok = false;
                    break;
                }
            }
        } else {
            (void) cudaGetLastError();
        }
        if (peer_ok) {
            if (p->enable_p2p) {
                const size_t small_total = (size_t) ggml_cuda_ar_pipeline::P2P_SMALL_SLOTS * ggml_cuda_ar_pipeline::P2P_SMALL_BYTES;
                bool alloc_ok = true;
                for (int i = 0; i < 2 && alloc_ok; ++i) {
                    ggml_cuda_set_device(p->devices[i]);
                    if (cudaMalloc(reinterpret_cast<void **>(&p->p2p_small_buf[i]), small_total) != cudaSuccess ||
                        cudaMalloc(reinterpret_cast<void **>(&p->p2p_large_buf[i]), ggml_cuda_ar_pipeline::P2P_LARGE_BYTES) != cudaSuccess) {
                        (void) cudaGetLastError();
                        alloc_ok = false;
                        break;
                    }
                    for (int s = 0; s < ggml_cuda_ar_pipeline::P2P_SMALL_SLOTS && alloc_ok; ++s) {
                        if (cudaEventCreateWithFlags(&p->p2p_staged[i][s], cudaEventDisableTiming) != cudaSuccess ||
                            cudaEventCreateWithFlags(&p->p2p_done[i][s],   cudaEventDisableTiming) != cudaSuccess) {
                            (void) cudaGetLastError();
                            alloc_ok = false;
                        }
                    }
                    if (alloc_ok &&
                        (cudaEventCreateWithFlags(&p->p2p_staged_large[i], cudaEventDisableTiming) != cudaSuccess ||
                         cudaEventCreateWithFlags(&p->p2p_done_large[i],   cudaEventDisableTiming) != cudaSuccess)) {
                        (void) cudaGetLastError();
                        alloc_ok = false;
                    }
                }
                if (alloc_ok) {
                    p->p2p_ready = true;
                } else {
                    GGML_LOG_WARN("%s: P2P allocation failed; disabling P2P direct-write\n", __func__);
                    for (int i = 0; i < 2; ++i) {
                        ggml_cuda_set_device(p->devices[i]);
                        if (p->p2p_small_buf[i]) { (void) cudaFree(p->p2p_small_buf[i]); p->p2p_small_buf[i] = nullptr; }
                        if (p->p2p_large_buf[i]) { (void) cudaFree(p->p2p_large_buf[i]); p->p2p_large_buf[i] = nullptr; }
                        for (int s = 0; s < ggml_cuda_ar_pipeline::P2P_SMALL_SLOTS; ++s) {
                            if (p->p2p_staged[i][s]) { (void) cudaEventDestroy(p->p2p_staged[i][s]); p->p2p_staged[i][s] = nullptr; }
                            if (p->p2p_done[i][s])   { (void) cudaEventDestroy(p->p2p_done[i][s]);   p->p2p_done[i][s]   = nullptr; }
                        }
                        if (p->p2p_staged_large[i]) { (void) cudaEventDestroy(p->p2p_staged_large[i]); p->p2p_staged_large[i] = nullptr; }
                        if (p->p2p_done_large[i])   { (void) cudaEventDestroy(p->p2p_done_large[i]);   p->p2p_done_large[i]   = nullptr; }
                    }
                }
            }
            if (p->enable_fused) {
                bool fused_ok = true;
                for (int i = 0; i < 2 && fused_ok; ++i) {
                    ggml_cuda_set_device(p->devices[i]);
#if defined(GGML_USE_HIP)
                    if (hipExtMallocWithFlags(reinterpret_cast<void **>(&p->fused_sig[i]), sizeof(ggml_cuda_ar_fused_signal), hipDeviceMallocUncached) != hipSuccess ||
                        hipMemset(p->fused_sig[i], 0, sizeof(ggml_cuda_ar_fused_signal)) != hipSuccess ||
                        hipExtMallocWithFlags(reinterpret_cast<void **>(&p->fused_stage[i]), 2 * GGML_CUDA_AR_FUSED_HALF_BYTES, hipDeviceMallocUncached) != hipSuccess ||
                        hipMemset(p->fused_stage[i], 0, 2 * GGML_CUDA_AR_FUSED_HALF_BYTES) != hipSuccess ||
                        hipDeviceSynchronize() != hipSuccess) {
                        (void) hipGetLastError();
                        fused_ok = false;
                        break;
                    }
#else
                    if (cudaMalloc(reinterpret_cast<void **>(&p->fused_sig[i]), sizeof(ggml_cuda_ar_fused_signal)) != cudaSuccess ||
                        cudaMemset(p->fused_sig[i], 0, sizeof(ggml_cuda_ar_fused_signal)) != cudaSuccess ||
                        cudaMalloc(reinterpret_cast<void **>(&p->fused_stage[i]), 2 * GGML_CUDA_AR_FUSED_HALF_BYTES) != cudaSuccess ||
                        cudaMemset(p->fused_stage[i], 0, 2 * GGML_CUDA_AR_FUSED_HALF_BYTES) != cudaSuccess ||
                        cudaDeviceSynchronize() != cudaSuccess) {
                        (void) cudaGetLastError();
                        fused_ok = false;
                        break;
                    }
#endif
                }
                if (fused_ok) {
                    p->fused_ready = true;
                } else {
                    GGML_LOG_WARN("%s: fused uncached allocation failed; disabling fused GPU-barrier AR\n", __func__);
                    for (int i = 0; i < 2; ++i) {
                        ggml_cuda_set_device(p->devices[i]);
                        if (p->fused_stage[i]) { (void) cudaFree(p->fused_stage[i]); p->fused_stage[i] = nullptr; }
                        if (p->fused_sig[i])   { (void) cudaFree(p->fused_sig[i]);   p->fused_sig[i]   = nullptr; }
                    }
                }
            }
        } else {
            GGML_LOG_WARN("%s: P2P peer access unavailable/failed; disabling P2P & fused transports\n", __func__);
        }
    }

    GGML_LOG_INFO("%s: initialized AllReduce pipeline: %zu GPUs, "
                  "%zu KB chunked kernel staging + %zu MB copy-engine staging per GPU, "
                  "p2p_ready=%d, fused_ready=%d (max_bytes=%zu, max_blocks=%d)\n",
                  __func__, n_devices, p->buf_bytes >> 10, p->copy_bytes >> 20,
                  (int) p->p2p_ready, (int) p->fused_ready, p->fused_max_bytes, p->fused_max_blocks);

    return p;
}

void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * p) {
    if (!p) {
        return;
    }

    // Drain all in-flight kernels before tearing down resources.
    for (int i = 0; i < p->n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        if (p->p2p_ready || p->fused_used) {
            CUDA_CHECK(cudaDeviceSynchronize());
        } else if (p->streams[i]) {
            CUDA_CHECK(cudaStreamSynchronize(p->streams[i]));
        }
    }

    for (int i = 0; i < p->n_devices; ++i) {
        p->host_buf[i].free();
        p->host_large[i].free();
        if (p->dev_tmp[i]) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaFree(p->dev_tmp[i]));
        }
        ggml_cuda_set_device(p->devices[i]);
        for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
            if (p->ev_pool[i][s].app) { CUDA_CHECK(cudaEventDestroy(p->ev_pool[i][s].app)); }
            for (int c = 0; c < GGML_CUDA_AR_COPY_MAX_CHUNKS; ++c) {
                if (p->ev_pool[i][s].cpy[c]) { CUDA_CHECK(cudaEventDestroy(p->ev_pool[i][s].cpy[c])); }
            }
            if (p->ev_pool[i][s].h2d) { CUDA_CHECK(cudaEventDestroy(p->ev_pool[i][s].h2d)); }
            if (p->ev_pool[i][s].ker) { CUDA_CHECK(cudaEventDestroy(p->ev_pool[i][s].ker)); }
        }
        if (p->host_large_read_done[i]) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaEventDestroy(p->host_large_read_done[i]));
        }
        if (p->dev_tmp_kernel_done[i]) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaEventDestroy(p->dev_tmp_kernel_done[i]));
        }
        if (p->streams[i]) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaStreamDestroy(p->streams[i]));
        }
    }
    for (int i = 0; i < p->n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        if (p->fused_stage[i]) { CUDA_CHECK(cudaFree(p->fused_stage[i])); p->fused_stage[i] = nullptr; }
        if (p->fused_sig[i])   { CUDA_CHECK(cudaFree(p->fused_sig[i]));   p->fused_sig[i]   = nullptr; }
        if (p->p2p_small_buf[i]) { CUDA_CHECK(cudaFree(p->p2p_small_buf[i])); p->p2p_small_buf[i] = nullptr; }
        if (p->p2p_large_buf[i]) { CUDA_CHECK(cudaFree(p->p2p_large_buf[i])); p->p2p_large_buf[i] = nullptr; }
        for (int s = 0; s < ggml_cuda_ar_pipeline::P2P_SMALL_SLOTS; ++s) {
            if (p->p2p_staged[i][s]) { CUDA_CHECK(cudaEventDestroy(p->p2p_staged[i][s])); p->p2p_staged[i][s] = nullptr; }
            if (p->p2p_done[i][s])   { CUDA_CHECK(cudaEventDestroy(p->p2p_done[i][s]));   p->p2p_done[i][s]   = nullptr; }
        }
        if (p->p2p_staged_large[i]) { CUDA_CHECK(cudaEventDestroy(p->p2p_staged_large[i])); p->p2p_staged_large[i] = nullptr; }
        if (p->p2p_done_large[i])   { CUDA_CHECK(cudaEventDestroy(p->p2p_done_large[i]));   p->p2p_done_large[i]   = nullptr; }
    }
    p->arrival.free();
    delete p;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

// Asymmetric copy_impl: data sent over PCIe in T_src precision (one element of
// nbytes per ne element); accumulated locally into a T_dst buffer.  When
// T_src == T_dst this is the original homogeneous reduction.  When they differ
// (e.g. BF16 wire / F32 accumulator) the add kernel rounds dst through T_src
// for bit-equivalence between GPUs and we skip the otherwise-needed
// post-conversion entirely.
template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_impl(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne,
        size_t                  nbytes) {
    GGML_ASSERT(p->n_devices == 2);
    GGML_ASSERT(nbytes <= p->copy_bytes);
    GGML_ASSERT(ne <= std::numeric_limits<int>::max());

    const size_t chunk_bytes = ggml_cuda_ar_chunk_bytes(p, nbytes);
    GGML_ASSERT(chunk_bytes > 0);

    const int slot = ggml_cuda_ar_acquire_slot(p).slot;
    const size_t copy_chunks = (nbytes + chunk_bytes - 1) / chunk_bytes;
    GGML_ASSERT(copy_chunks <= GGML_CUDA_AR_COPY_MAX_CHUNKS);

    ggml_backend_cuda_context * cuda_ctx[2] = {};

    // Stage 1: both GPUs copy their local contribution to pinned host memory.
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        cuda_ctx[i] = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        GGML_ASSERT(cuda_ctx[i]->device == p->devices[i]);

        ggml_cuda_ar_wait_for_compute(p, cuda_ctx[i], i, slot);

        // Wait for peer's H2D from our host_large[i] (recorded in the
        // previous AR's stage 2) to complete before we overwrite host_large[i].
        // host_large_read_done[peer] = peer finished reading host_large[i].
        // No-op on the first AR -- no prior record exists.
        if (p->host_large_read_done_valid) {
            const int peer = 1 - i;
            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->host_large_read_done[peer]));
        }

        if (!compute[i]) {
            CUDA_CHECK(cudaMemsetAsync(src_buf[i], 0, nbytes, p->streams[i]));
        }

        for (size_t c = 0; c < copy_chunks; ++c) {
            const size_t offset = c * chunk_bytes;
            const size_t this_bytes = (nbytes - offset) < chunk_bytes ?
                (nbytes - offset) : chunk_bytes;

            CUDA_CHECK(cudaMemcpyAsync(
                p->host_large[i].host + offset, reinterpret_cast<char *>(src_buf[i]) + offset, this_bytes,
                cudaMemcpyDeviceToHost, p->streams[i]));
            CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].cpy[c], p->streams[i]));
        }
    }

    // Stage 2: each GPU waits for each peer D2H chunk, pulls that chunk back to
    // local device scratch (dev_tmp), then performs one device-local add over
    // the assembled peer tensor.  The H2Ds run on the AR stream (copy engine)
    // and the add_kernel runs on the caller's compute stream, so the AR stream
    // stays pure-copy and avoids an in-stream copy->compute engine switch every
    // AR.  dev_tmp is single-buffered: the AR stream waits cross-stream on the
    // prior AR's add_kernel-done event before overwriting it.
    for (int i = 0; i < 2; ++i) {
        const int peer = 1 - i;
        ggml_cuda_set_device(p->devices[i]);

        // Wait for the previous AR's add_kernel (on the compute stream) to
        // finish reading dev_tmp before our H2D overwrites it.  No-op on the
        // first copy_impl call.
        if (p->dev_tmp_kernel_done_valid) {
            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->dev_tmp_kernel_done[i]));
        }

        for (size_t c = 0; c < copy_chunks; ++c) {
            const size_t offset = c * chunk_bytes;
            const size_t this_bytes = (nbytes - offset) < chunk_bytes ?
                (nbytes - offset) : chunk_bytes;

            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->ev_pool[peer][slot].cpy[c]));
            CUDA_CHECK(cudaMemcpyAsync(
                p->dev_tmp[i] + offset, p->host_large[peer].host + offset, this_bytes,
                cudaMemcpyHostToDevice, p->streams[i]));
        }

        // Mark our reads of host_large[peer] complete so peer's next AR can
        // safely overwrite it.
        CUDA_CHECK(cudaEventRecord(p->host_large_read_done[i], p->streams[i]));

        // Hand off from AR stream (copy engine) to compute stream: compute
        // stream waits for all H2Ds to finish, then runs the add_kernel.
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].h2d, p->streams[i]));
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx[i]->stream(), p->ev_pool[i][slot].h2d));

        const int block_size = 256;
        int n_blocks = (int) ((ne + block_size - 1) / block_size);
        if (n_blocks > 1024) {
            n_blocks = 1024;
        }
        ggml_cuda_ar_add_kernel<T_dst, T_src><<<n_blocks, block_size, 0, cuda_ctx[i]->stream()>>>(
            dst_buf[i],
            reinterpret_cast<const T_src *>(p->dev_tmp[i]),
            (int) ne);
        CUDA_CHECK(cudaGetLastError());

        // Record dev_tmp-released on the compute stream so the next copy_impl
        // can wait for the kernel to finish before overwriting dev_tmp.  Also
        // record AR-done as ev.ker for acquire_slot's pool-wraparound sync.
        CUDA_CHECK(cudaEventRecord(p->dev_tmp_kernel_done[i], cuda_ctx[i]->stream()));
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, cuda_ctx[i]->stream()));
    }
    p->host_large_read_done_valid = true;
    p->dev_tmp_kernel_done_valid = true;

    return true;
}

// Outer-level chunker: copy_impl handles up to copy_bytes per call (limited by
// the host_large / dev_tmp allocation size).  When the full AR exceeds that,
// slice the tensor into copy_bytes-sized pieces and call copy_impl repeatedly.
// Each slice goes through its own stage 1 -> stage 2 cycle and acquires its own
// slot, so cross-AR fences and pool wraparound work the same way as for any
// other sequence of small ARs.
template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_outer(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne) {
    const int64_t outer_max_elems = (int64_t) (p->copy_bytes / sizeof(T_src));
    GGML_ASSERT(outer_max_elems > 0);

    bool ok = true;
    for (int64_t outer_start = 0; outer_start < ne && ok; outer_start += outer_max_elems) {
        const int64_t outer_ne     = std::min(outer_max_elems, ne - outer_start);
        const size_t  outer_nbytes = (size_t) outer_ne * sizeof(T_src);

        T_src * src[GGML_CUDA_MAX_DEVICES] = {};
        T_dst * dst[GGML_CUDA_MAX_DEVICES] = {};
        for (int i = 0; i < p->n_devices; ++i) {
            src[i] = src_buf[i] + outer_start;
            dst[i] = dst_buf[i] + outer_start;
        }
        ok = ggml_cuda_ar_allreduce_copy_impl<T_src, T_dst>(
            p, backends, src, dst, compute, outer_ne, outer_nbytes);
    }
    return ok;
}

enum class ggml_cuda_ar_path {
    HOST_CHUNKED,
    HOST_COPY,
    P2P_EVENT,
    FUSED,
};

struct ggml_cuda_ar_policy {
    ggml_type         input_type;
    ggml_type         wire_type;
    bool              use_bf16;
    size_t            wire_type_size;
    size_t            wire_nbytes;
    ggml_cuda_ar_path path;
};

static ggml_cuda_ar_policy ggml_cuda_ar_select_policy(
        const ggml_cuda_ar_pipeline * p,
        ggml_type                     input_type,
        int64_t                       ne,
        size_t                        input_nbytes,
        bool                          allow_fused) {
    const bool use_bf16 =
        input_type == GGML_TYPE_F32 &&
        p->bf16_threshold > 0 &&
        input_nbytes >= p->bf16_threshold;
    const ggml_type wire_type = use_bf16 ? GGML_TYPE_BF16 : input_type;
    const size_t wire_type_size = ggml_type_size(wire_type);
    const size_t wire_nbytes = (size_t) ne * wire_type_size;

    ggml_cuda_ar_path path = ggml_cuda_ar_path::HOST_CHUNKED;
    if (allow_fused && p->enable_fused && p->fused_ready &&
        wire_nbytes > 0 && wire_nbytes <= p->fused_max_bytes &&
        ((ne & 3) == 0)) {
        path = ggml_cuda_ar_path::FUSED;
    } else if (p->enable_p2p && p->p2p_ready) {
        path = ggml_cuda_ar_path::P2P_EVENT;
    } else if (p->copy_threshold > 0 && wire_nbytes >= p->copy_threshold) {
        path = ggml_cuda_ar_path::HOST_COPY;
    }
    return {input_type, wire_type, use_bf16, wire_type_size, wire_nbytes, path};
}

static void ggml_cuda_ar_record_wire_stat(ggml_cuda_ar_pipeline * p, const ggml_cuda_ar_policy & pol) {
    if (pol.use_bf16) {
        p->stats[ggml_cuda_ar_pipeline::STAT_WIRE_BF16].fetch_add(1, std::memory_order_relaxed);
    } else if (pol.wire_type == GGML_TYPE_F32) {
        p->stats[ggml_cuda_ar_pipeline::STAT_WIRE_F32].fetch_add(1, std::memory_order_relaxed);
    } else if (pol.wire_type == GGML_TYPE_F16) {
        p->stats[ggml_cuda_ar_pipeline::STAT_WIRE_F16].fetch_add(1, std::memory_order_relaxed);
    } else if (pol.wire_type == GGML_TYPE_BF16) {
        p->stats[ggml_cuda_ar_pipeline::STAT_WIRE_NAT_BF16].fetch_add(1, std::memory_order_relaxed);
    }
}

template <typename T_dst, typename T_wire>
static bool ggml_cuda_ar_p2p_event_exec(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors,
        const bool              compute_flag[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne,
        const ggml_cuda_ar_policy & pol) {
    const size_t input_elem_bytes = sizeof(T_dst);
    const size_t wire_elem_bytes  = sizeof(T_wire);
    const bool   use_small        = pol.wire_nbytes <= ggml_cuda_ar_pipeline::P2P_SMALL_BYTES;
    const size_t max_chunk_bytes  = use_small ? ggml_cuda_ar_pipeline::P2P_SMALL_BYTES : ggml_cuda_ar_pipeline::P2P_LARGE_BYTES;
    const int64_t max_chunk_elems = (int64_t) (max_chunk_bytes / wire_elem_bytes);

    ggml_backend_cuda_context * cuda_ctx[2] = {
        static_cast<ggml_backend_cuda_context *>(backends[0]->context),
        static_cast<ggml_backend_cuda_context *>(backends[1]->context),
    };

    for (int64_t chunk_start = 0; chunk_start < ne; chunk_start += max_chunk_elems) {
        const int64_t chunk_elems = std::min(max_chunk_elems, ne - chunk_start);
        const size_t  chunk_wire_bytes = (size_t) chunk_elems * wire_elem_bytes;

        int slot = 0;
        bool need_wait_done = false;
        if (use_small) {
            slot = (int) (p->p2p_seq++ % ggml_cuda_ar_pipeline::P2P_SMALL_SLOTS);
            need_wait_done = p->p2p_small_used[slot];
        } else {
            need_wait_done = p->p2p_large_used;
        }

        // Phase 1: push local wire representation to peer's P2P recv buffer on compute stream.
        for (int i = 0; i < 2; ++i) {
            const int peer = 1 - i;
            ggml_cuda_set_device(p->devices[i]);
            cudaStream_t stream = cuda_ctx[i]->stream();

            if (need_wait_done) {
                cudaEvent_t ev_done = use_small ? p->p2p_done[peer][slot] : p->p2p_done_large[peer];
                CUDA_CHECK(cudaStreamWaitEvent(stream, ev_done, 0));
            }

            const char * src_ptr = static_cast<const char *>(tensors[i]->data) + chunk_start * (int64_t) input_elem_bytes;
            uint8_t * peer_recv = use_small
                ? p->p2p_small_buf[peer] + (size_t) slot * ggml_cuda_ar_pipeline::P2P_SMALL_BYTES
                : p->p2p_large_buf[peer];

            if (!pol.use_bf16 && !use_small) {
                if (compute_flag[i]) {
                    CUDA_CHECK(cudaMemcpyPeerAsync(
                        peer_recv, p->devices[peer],
                        src_ptr, p->devices[i],
                        chunk_wire_bytes, stream));
                } else {
                    CUDA_CHECK(cudaMemsetAsync(peer_recv, 0, chunk_wire_bytes, stream));
                }
            } else {
                const int block_size = 256;
                int64_t work = (pol.use_bf16 && ((chunk_elems & 1) == 0)) ? (chunk_elems >> 1) : chunk_elems;
                int n_blocks = (int) std::min<int64_t>(512, (work + block_size - 1) / block_size);
                n_blocks = std::max(n_blocks, 1);
                ggml_cuda_ar_p2p_write_kernel<T_dst, T_wire><<<n_blocks, block_size, 0, stream>>>(
                    reinterpret_cast<const T_dst *>(src_ptr),
                    reinterpret_cast<T_wire *>(peer_recv),
                    (int) chunk_elems,
                    compute_flag[i]);
                CUDA_CHECK(cudaGetLastError());
            }

            cudaEvent_t ev_staged = use_small ? p->p2p_staged[i][slot] : p->p2p_staged_large[i];
            CUDA_CHECK(cudaEventRecord(ev_staged, stream));
        }

        // Phase 2: wait for peer's staged write and perform symmetric accumulation into local tensor.
        for (int i = 0; i < 2; ++i) {
            const int peer = 1 - i;
            ggml_cuda_set_device(p->devices[i]);
            cudaStream_t stream = cuda_ctx[i]->stream();

            cudaEvent_t ev_staged_peer = use_small ? p->p2p_staged[peer][slot] : p->p2p_staged_large[peer];
            CUDA_CHECK(cudaStreamWaitEvent(stream, ev_staged_peer, 0));

            char * dst_ptr = static_cast<char *>(tensors[i]->data) + chunk_start * (int64_t) input_elem_bytes;
            const uint8_t * self_recv = use_small
                ? p->p2p_small_buf[i] + (size_t) slot * ggml_cuda_ar_pipeline::P2P_SMALL_BYTES
                : p->p2p_large_buf[i];

            const int block_size = 256;
            int64_t work = (pol.use_bf16 && ((chunk_elems & 1) == 0)) ? (chunk_elems >> 1) : chunk_elems;
            int n_blocks = (int) std::min<int64_t>(512, (work + block_size - 1) / block_size);
            n_blocks = std::max(n_blocks, 1);
            ggml_cuda_ar_p2p_sym_add_kernel<T_dst, T_wire><<<n_blocks, block_size, 0, stream>>>(
                reinterpret_cast<T_dst *>(dst_ptr),
                reinterpret_cast<const T_wire *>(self_recv),
                (int) chunk_elems,
                compute_flag[i]);
            CUDA_CHECK(cudaGetLastError());

            cudaEvent_t ev_done = use_small ? p->p2p_done[i][slot] : p->p2p_done_large[i];
            CUDA_CHECK(cudaEventRecord(ev_done, stream));
        }

        if (use_small) {
            p->p2p_small_used[slot] = true;
        } else {
            p->p2p_large_used = true;
        }
    }
    return true;
}

bool ggml_cuda_ar_fused_prepare(
        const ggml_cuda_ar_pipeline * p,
        ggml_tensor                ** tensors,
        ggml_cuda_ar_fused_plan     * plan) {
    if (!p || !p->enable_fused || !p->fused_ready || p->n_devices != 2 || !tensors || !plan) {
        return false;
    }
    if (!tensors[0] || !tensors[1]) {
        return false;
    }
    const ggml_type input_type = tensors[0]->type;
    if (input_type != GGML_TYPE_F32 && input_type != GGML_TYPE_F16 && input_type != GGML_TYPE_BF16) {
        return false;
    }
    const int64_t ne = ggml_nelements(tensors[0]);
    if (ne == 0) {
        *plan = ggml_cuda_ar_fused_plan{};
        return true;
    }
    if (tensors[1]->type != input_type || ggml_nelements(tensors[1]) != ne ||
        !ggml_is_contiguously_allocated(tensors[0]) || !ggml_is_contiguously_allocated(tensors[1])) {
        return false;
    }
    const size_t input_nbytes = ggml_nbytes(tensors[0]);
    if (ggml_nbytes(tensors[1]) != input_nbytes || (input_nbytes & 0xF) != 0 ||
        ((uintptr_t) tensors[0]->data & 0xF) != 0 || ((uintptr_t) tensors[1]->data & 0xF) != 0) {
        return false;
    }
    const ggml_cuda_ar_policy pol = ggml_cuda_ar_select_policy(p, input_type, ne, input_nbytes, /*allow_fused=*/ true);
    if (pol.path != ggml_cuda_ar_path::FUSED) {
        return false;
    }
    const size_t n_q = (size_t) ne >> 2;
    size_t b = (n_q + GGML_CUDA_AR_FUSED_THREADS - 1) / GGML_CUDA_AR_FUSED_THREADS;
    if (b == 0) { b = 1; }
    if (b > (size_t) p->fused_max_blocks) { b = (size_t) p->fused_max_blocks; }

    plan->ne         = (size_t) ne;
    plan->input_type = pol.input_type;
    plan->wire_type  = pol.wire_type;
    plan->use_bf16   = pol.use_bf16;
    plan->blocks     = (int) b;
    return true;
}

template <typename T_dst, typename T_wire>
static void ggml_cuda_ar_fused_launch_typed(
        ggml_cuda_ar_pipeline         * p,
        int                             rank,
        cudaStream_t                    stream,
        ggml_tensor                   * tensor_r,
        const ggml_cuda_ar_fused_plan * plan) {
    const bool active = (tensor_r->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
    ggml_cuda_ar_fused_kernel<T_dst, T_wire><<<plan->blocks, GGML_CUDA_AR_FUSED_THREADS, 0, stream>>>(
        static_cast<T_dst *>(tensor_r->data),
        plan->ne,
        rank,
        active,
        p->fused_sig[rank],
        p->fused_sig[1 - rank],
        p->fused_stage[rank],
        p->fused_stage[1 - rank]);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_ar_fused_launch_rank(
        ggml_cuda_ar_pipeline         * p,
        ggml_backend_t                * backends,
        int                             rank,
        ggml_tensor                   * tensor_r,
        const ggml_cuda_ar_fused_plan * plan,
        bool                            count_as_tg) {
    GGML_ASSERT(p && p->fused_ready && plan);
    if (plan->ne == 0) {
        return;
    }
    GGML_ASSERT(plan->blocks > 0);
    GGML_ASSERT(rank == 0 || rank == 1);
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backends[rank]->context);
    GGML_ASSERT(ctx->device == p->devices[rank]);
    cudaStream_t s = ctx->stream();
    ggml_cuda_set_device(p->devices[rank]);
    p->fused_used = true;

    if (rank == 0 && count_as_tg) {
        p->stats[ggml_cuda_ar_pipeline::STAT_FUSED_TG_LAUNCH].fetch_add(1, std::memory_order_relaxed);
        const ggml_cuda_ar_policy pol = {
            plan->input_type, plan->wire_type, plan->use_bf16,
            ggml_type_size(plan->wire_type), plan->ne * ggml_type_size(plan->wire_type),
            ggml_cuda_ar_path::FUSED
        };
        ggml_cuda_ar_record_wire_stat(p, pol);
    }

    if (plan->use_bf16) {
        ggml_cuda_ar_fused_launch_typed<float, nv_bfloat16>(p, rank, s, tensor_r, plan);
        return;
    }
    switch (plan->input_type) {
        case GGML_TYPE_F32:  ggml_cuda_ar_fused_launch_typed<float,       float>      (p, rank, s, tensor_r, plan); break;
        case GGML_TYPE_F16:  ggml_cuda_ar_fused_launch_typed<half,        half>       (p, rank, s, tensor_r, plan); break;
        case GGML_TYPE_BF16: ggml_cuda_ar_fused_launch_typed<nv_bfloat16, nv_bfloat16>(p, rank, s, tensor_r, plan); break;
        default: GGML_ASSERT(false);
    }
}

size_t ggml_cuda_ar_get_stats(const ggml_cuda_ar_pipeline * p, uint64_t * out, size_t cap) {
    if (!p || !out) {
        return 0;
    }
    const size_t n = std::min<size_t>(cap, ggml_cuda_ar_pipeline::STAT_COUNT);
    for (size_t i = 0; i < n; ++i) {
        out[i] = p->stats[i].load(std::memory_order_relaxed);
    }
    return n;
}

void ggml_cuda_ar_reset_stats(ggml_cuda_ar_pipeline * p) {
    if (!p) {
        return;
    }
    for (size_t i = 0; i < ggml_cuda_ar_pipeline::STAT_COUNT; ++i) {
        p->stats[i].store(0, std::memory_order_relaxed);
    }
}

bool ggml_cuda_ar_allreduce(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors) {
    GGML_ASSERT(p != nullptr);

    const int n = p->n_devices;
    GGML_ASSERT(n == 2);

    const ggml_type input_type = tensors[0]->type;
    GGML_ASSERT(input_type == GGML_TYPE_F32 || input_type == GGML_TYPE_F16 || input_type == GGML_TYPE_BF16);

    const int64_t ne = ggml_nelements(tensors[0]);
    GGML_ASSERT(ne > 0);

    const size_t   input_nbytes = ggml_nbytes(tensors[0]);

    const ggml_cuda_ar_policy pol = ggml_cuda_ar_select_policy(p, input_type, ne, input_nbytes, /*allow_fused=*/ true);
    const bool      use_bf16    = pol.use_bf16;
    const ggml_type kernel_type = pol.wire_type;
    const size_t    type_size   = pol.wire_type_size;
    GGML_ASSERT(p->buf_bytes >= type_size);
    const size_t    nbytes      = pol.wire_nbytes;

    ggml_cuda_ar_record_wire_stat(p, pol);

    bool compute_flag[GGML_CUDA_MAX_DEVICES] = {};
    for (int i = 0; i < n; ++i) {
        compute_flag[i] = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
    }

    if (pol.path == ggml_cuda_ar_path::FUSED) {
        ggml_cuda_ar_fused_plan plan{};
        if (ggml_cuda_ar_fused_prepare(p, tensors, &plan)) {
            p->stats[ggml_cuda_ar_pipeline::STAT_FUSED_DIRECT].fetch_add(1, std::memory_order_relaxed);
            for (int i = 0; i < 2; ++i) {
                ggml_cuda_ar_fused_launch_rank(p, backends, i, tensors[i], &plan, /*count_as_tg=*/ false);
            }
            return true;
        }
    }

    if (pol.path == ggml_cuda_ar_path::P2P_EVENT) {
        p->stats[ggml_cuda_ar_pipeline::STAT_P2P_EVENT].fetch_add(1, std::memory_order_relaxed);
        if (use_bf16) {
            return ggml_cuda_ar_p2p_event_exec<float, nv_bfloat16>(p, backends, tensors, compute_flag, ne, pol);
        }
        switch (input_type) {
            case GGML_TYPE_F32:  return ggml_cuda_ar_p2p_event_exec<float,       float>      (p, backends, tensors, compute_flag, ne, pol);
            case GGML_TYPE_F16:  return ggml_cuda_ar_p2p_event_exec<half,        half>       (p, backends, tensors, compute_flag, ne, pol);
            case GGML_TYPE_BF16: return ggml_cuda_ar_p2p_event_exec<nv_bfloat16, nv_bfloat16>(p, backends, tensors, compute_flag, ne, pol);
            default: GGML_ASSERT(false);
        }
    }

    const bool use_copy_engine = (pol.path == ggml_cuda_ar_path::HOST_COPY);
    p->stats[use_copy_engine ? ggml_cuda_ar_pipeline::STAT_HOST_COPY : ggml_cuda_ar_pipeline::STAT_HOST_CHUNKED]
        .fetch_add(1, std::memory_order_relaxed);

    // BF16 inactive-shard zeroing: when use_bf16 is on, the combined kernel
    // (chunked kernel path) and the combined add kernel (copy_engine path)
    // both accumulate into the F32 tensor data directly, so an inactive
    // shard's accumulator must start at zero.
    if (use_bf16) {
        for (int i = 0; i < n; ++i) {
            if (!compute_flag[i]) {
                auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
                GGML_ASSERT(cuda_ctx->device == p->devices[i]);
                ggml_cuda_set_device(p->devices[i]);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, (size_t) ne * sizeof(float), cuda_ctx->stream()));
            }
        }
    }

    // Pre-convert F32 -> BF16 into bf16_tmp ONLY for the copy_engine + use_bf16
    // path; the chunked kernel path's combined kernel does the conversion
    // inline as it writes to host_buf.
    ggml_cuda_pool_alloc<nv_bfloat16> bf16_tmp[GGML_CUDA_MAX_DEVICES];
    void * copy_src_ptr[GGML_CUDA_MAX_DEVICES] = {};

    if (use_copy_engine && use_bf16) {
        to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
        for (int i = 0; i < n; ++i) {
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            GGML_ASSERT(cuda_ctx->device == p->devices[i]);
            bf16_tmp[i].pool = &cuda_ctx->pool();
            bf16_tmp[i].alloc(ne);
            ggml_cuda_set_device(p->devices[i]);
            if (compute_flag[i]) {
                to_bf16(tensors[i]->data, bf16_tmp[i].get(), ne, cuda_ctx->stream());
                CUDA_CHECK(cudaGetLastError());
            } else {
                CUDA_CHECK(cudaMemsetAsync(bf16_tmp[i].get(), 0, nbytes, cuda_ctx->stream()));
            }
            copy_src_ptr[i] = bf16_tmp[i].get();
        }
    }

    bool ok = true;
    if (use_copy_engine) {
        // After up-front BF16 conversion, the tmp buffers already hold the
        // (possibly zeroed-for-inactive) data, so the inner path can treat
        // every shard as compute.
        bool inner_compute[GGML_CUDA_MAX_DEVICES];
        for (int i = 0; i < n; ++i) {
            inner_compute[i] = use_bf16 ? true : compute_flag[i];
        }

        // Dispatch into copy_impl with explicit src/dst types.  When use_bf16
        // is on, the wire type is BF16 (src = bf16_tmp) and the accumulator
        // is F32 (dst = tensors[i]->data); the combined add kernel rounds dst
        // through BF16 for bit-equivalence and writes F32 directly, so no
        // post-conversion is needed.  Otherwise src == dst (same native type).
        if (use_bf16) {
            GGML_ASSERT(kernel_type == GGML_TYPE_BF16);
            nv_bfloat16 * src[GGML_CUDA_MAX_DEVICES] = {};
            float       * dst[GGML_CUDA_MAX_DEVICES] = {};
            for (int i = 0; i < n; ++i) {
                src[i] = static_cast<nv_bfloat16 *>(copy_src_ptr[i]);
                dst[i] = static_cast<float *>(tensors[i]->data);
            }
            ok = ggml_cuda_ar_allreduce_copy_outer<nv_bfloat16, float>(
                p, backends, src, dst, inner_compute, ne);
        } else {
            switch (kernel_type) {
                case GGML_TYPE_F32: {
                    float * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<float *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<float, float>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                case GGML_TYPE_BF16: {
                    nv_bfloat16 * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<nv_bfloat16 *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<nv_bfloat16, nv_bfloat16>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                case GGML_TYPE_F16: {
                    half * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<half *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<half, half>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                default:
                    GGML_ASSERT(false);
            }
        }
    } else {
        // host_buf carries T_wire-typed data; max_chunk_elems is the count that
        // fits in one host_buf at the wire size.
        const size_t max_chunk_elems = p->buf_bytes / type_size;
        const size_t input_type_size = ggml_type_size(input_type);

        // Chunked kernel path runs entirely on the caller's compute stream:
        // since AR is a barrier here, same-stream ordering subsumes any
        // cross-stream event handshake that the copy-engine path needs, and
        // skips the cross-stream scheduling overhead that was hurting the
        // small-tensor (tg) latency on the AR-stream variant.  Only ev.ker is
        // still recorded at end-of-AR for acquire_slot's pool-wraparound check.
        for (int64_t chunk_start = 0; chunk_start < ne; chunk_start += (int64_t) max_chunk_elems) {
            const size_t remaining_elems = (size_t) (ne - chunk_start);
            const size_t chunk_elems = remaining_elems < max_chunk_elems ? remaining_elems : max_chunk_elems;
            const size_t chunk_dst_bytes  = chunk_elems * input_type_size;

            const auto [slot, token] = ggml_cuda_ar_acquire_slot(p);
            const bool last_chunk = chunk_start + (int64_t) chunk_elems == ne;

            for (int i = 0; i < n; ++i) {
                const int peer = 1 - i;  // valid for n == 2 only
                ggml_cuda_set_device(p->devices[i]);
                auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
                GGML_ASSERT(cuda_ctx->device == p->devices[i]);
                cudaStream_t stream = cuda_ctx->stream();

                char * data = static_cast<char *>(tensors[i]->data) + chunk_start * (int64_t) input_type_size;

                // Match NCCL/meta-backend semantics: inactive shards contribute
                // zeros.  On the BF16 path the F32 tensor data was already
                // zeroed up-front (above), so per-chunk zeroing isn't needed.
                if (!compute_flag[i] && !use_bf16) {
                    CUDA_CHECK(cudaMemsetAsync(data, 0, chunk_dst_bytes, stream));
                }

#define LAUNCH_AR_KERNEL(T_dst, T_wire) \
                ggml_cuda_ar_kernel<T_dst, T_wire><<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, stream>>>( \
                    reinterpret_cast<const T_dst *>(data), \
                    reinterpret_cast<T_dst *>(data), \
                    reinterpret_cast<T_wire *>(p->host_buf[i].dev + (size_t) slot * p->buf_bytes), \
                    reinterpret_cast<const T_wire *>(p->host_buf[peer].dev + (size_t) slot * p->buf_bytes), \
                    static_cast<int>(chunk_elems), \
                    ggml_cuda_ar_arrival_ptr(p, slot, i), \
                    ggml_cuda_ar_arrival_ptr(p, slot, peer), \
                    token)

                if (use_bf16) {
                    GGML_ASSERT(input_type == GGML_TYPE_F32);
                    LAUNCH_AR_KERNEL(float, nv_bfloat16);
                } else {
                    switch (input_type) {
                        case GGML_TYPE_F32:  LAUNCH_AR_KERNEL(float,       float);       break;
                        case GGML_TYPE_F16:  LAUNCH_AR_KERNEL(half,        half);        break;
                        case GGML_TYPE_BF16: LAUNCH_AR_KERNEL(nv_bfloat16, nv_bfloat16); break;
                        default: GGML_ASSERT(false);
                    }
                }

#undef LAUNCH_AR_KERNEL
                CUDA_CHECK(cudaGetLastError());

                if (last_chunk) {
                    CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, stream));
                }
            }
        }
    }

    return ok;
}

#else // defined(GGML_USE_MUSA)

// MUSA lacks the host-mapped pinned-memory APIs (cudaHostAllocPortable
// / cudaHostAllocMapped / cudaHostGetDevicePointer) and a device-side
// sleep intrinsic that this implementation relies on, so the internal
// AllReduce is unavailable there. The dispatcher in ggml-cuda.cu treats
// a nullptr pipeline as "init failed" and silently falls back to the meta
// backend's generic AllReduce.
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int *, size_t) {
    return nullptr;
}
void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline *) {
}
bool ggml_cuda_ar_allreduce(ggml_cuda_ar_pipeline *, ggml_backend_t *, ggml_tensor **) {
    return false;
}
size_t ggml_cuda_ar_get_stats(const ggml_cuda_ar_pipeline *, uint64_t *, size_t) {
    return 0;
}
void ggml_cuda_ar_reset_stats(ggml_cuda_ar_pipeline *) {
}
bool ggml_cuda_ar_fused_prepare(const ggml_cuda_ar_pipeline *, ggml_tensor **, ggml_cuda_ar_fused_plan *) {
    return false;
}
void ggml_cuda_ar_fused_launch_rank(ggml_cuda_ar_pipeline *, ggml_backend_t *, int, ggml_tensor *, const ggml_cuda_ar_fused_plan *, bool) {
}

#endif // !defined(GGML_USE_MUSA)
