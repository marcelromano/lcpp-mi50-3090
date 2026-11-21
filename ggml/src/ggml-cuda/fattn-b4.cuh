#pragma once
#include "common.cuh"

// Opt-in specializations; false preserves the upstream selector and its allocation contract.
bool ggml_cuda_fattn_b4_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
