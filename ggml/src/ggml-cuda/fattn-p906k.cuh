#pragma once
#include "common.cuh"

// gfx906 prompt attention (opt-in GGML_HIP_FA_P906=100|101). Returns false when not applicable.
bool ggml_cuda_flash_attn_ext_p906k_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
