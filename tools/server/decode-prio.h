#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

// Invalid, zero or unset means the production scheduler. Strict decimal only.
inline int32_t server_decode_prio_prompt_tokens(const char * value) {
    if (!value || !*value) {
        return 0;
    }
    int32_t result = 0;
    for (const char * p = value; *p; ++p) {
        if (*p < '0' || *p > '9' ||
            result > (std::numeric_limits<int32_t>::max() - (*p - '0')) / 10) {
            return 0;
        }
        result = result * 10 + (*p - '0');
    }
    return result;
}

// Global prompt allowance, not per slot. Preserve every sampled/drafted token.
// With no decode work, retain full production prefill chunks.
inline int32_t server_decode_prio_batch_limit(int32_t n_batch, int32_t n_decode, int32_t cap) {
    if (cap <= 0 || n_decode <= 0 || n_decode >= n_batch) {
        return n_batch;
    }
    return n_decode + std::min(cap, n_batch - n_decode);
}
