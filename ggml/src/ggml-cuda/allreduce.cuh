#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>
#include <cstdint>

// Opaque pipeline context -- owns all pinned buffers, streams, and events.
struct ggml_cuda_ar_pipeline;

// Allocate a pipeline for n_devices GPUs.
// devices[] holds the GPU device IDs in rank order.
// Returns nullptr on allocation failure.
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(
    const int * devices, size_t n_devices);

// Release all resources owned by the pipeline.
void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * pipeline);

// Execute an in-place AllReduce (sum) across tensors[0..n_devices-1].
// tensors[i] must live on the device managed by backends[i] and be
// contiguous F32, F16, or BF16.
// Preconditions are checked by the CUDA comm dispatcher before calling this.
// Returns true once the reduction work has been enqueued successfully.
bool ggml_cuda_ar_allreduce(
    ggml_cuda_ar_pipeline * pipeline,
    ggml_backend_t        * backends,
    ggml_tensor           ** tensors);

size_t ggml_cuda_ar_get_stats(const ggml_cuda_ar_pipeline * pipeline, uint64_t * out, size_t cap);
void ggml_cuda_ar_reset_stats(ggml_cuda_ar_pipeline * pipeline);

// Launch plan for single-kernel fused GPU-barrier P2P AllReduce (graph-capturable per rank).
struct ggml_cuda_ar_fused_plan {
    size_t    ne         = 0;
    ggml_type input_type = GGML_TYPE_F32;
    ggml_type wire_type  = GGML_TYPE_BF16;
    bool      use_bf16   = false;
    int       blocks     = 0;
};

// Check whether the fused GPU-barrier path is active and eligible for tensors[0..1],
// populating `plan`. Never allocates or synchronizes the GPU.
bool ggml_cuda_ar_fused_prepare(
    const ggml_cuda_ar_pipeline * pipeline,
    ggml_tensor                ** tensors,
    ggml_cuda_ar_fused_plan     * plan);

// Enqueue one rank's fused GPU-barrier kernel onto backends[rank]->stream().
// Capturable inside a HIP/CUDA graph; caller must invoke for both ranks 0 and 1.
void ggml_cuda_ar_fused_launch_rank(
    ggml_cuda_ar_pipeline         * pipeline,
    ggml_backend_t                * backends,
    int                             rank,
    ggml_tensor                   * tensor_r,
    const ggml_cuda_ar_fused_plan * plan,
    bool                            count_as_tg = false);

