// Dedicated test for multi-rank AllReduce communicator and meta-backend
// whole-token HIP/CUDA graphs (Bloco 7: v1.0.0 + the previous revision P2P + the previous revision fused/TG).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-impl.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

static int g_failures = 0;

#define CHECK(cond, ...)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "FAIL [%s:%d]: ", __func__, __LINE__);      \
            std::fprintf(stderr, __VA_ARGS__);                               \
            std::fprintf(stderr, "\n");                                      \
            ++g_failures;                                                    \
            return false;                                                    \
        }                                                                    \
    } while (0)

#define CHECK_NO_RET(cond, ...)                                              \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "FAIL [%s:%d]: ", __func__, __LINE__);      \
            std::fprintf(stderr, __VA_ARGS__);                               \
            std::fprintf(stderr, "\n");                                      \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

static void set_env_val(const char * name, const char * val) {
#ifdef _WIN32
    _putenv_s(name, val ? val : "");
#else
    if (val == nullptr) {
        unsetenv(name);
    } else {
        setenv(name, val, 1);
    }
#endif
}

struct cuda_proc_api {
    using comm_init_t             = void * (*)(ggml_backend_t *, size_t);
    using comm_free_t             = void (*)(void *);
    using comm_allreduce_tensor_t = bool (*)(void *, ggml_tensor **);
    using ar_get_stats_t          = size_t (*)(void *, uint64_t *, size_t);
    using ar_reset_stats_t        = void (*)(void *);
    using graph_owner_depth_t     = uint32_t (*)(ggml_backend_t);
    using graph_epoch_t           = uint64_t (*)(ggml_backend_t);
    using graph_invalidate_t      = void (*)(ggml_backend_t);
    using test_fail_cap_t         = void (*)(ggml_backend_t, bool);

    comm_init_t             comm_init             = nullptr;
    comm_free_t             comm_free             = nullptr;
    comm_allreduce_tensor_t comm_allreduce_tensor = nullptr;
    ar_get_stats_t          ar_get_stats          = nullptr;
    ar_reset_stats_t        ar_reset_stats        = nullptr;
    graph_owner_depth_t     graph_owner_depth     = nullptr;
    graph_epoch_t           graph_epoch           = nullptr;
    graph_invalidate_t      graph_invalidate      = nullptr;
    test_fail_cap_t         test_fail_cap         = nullptr;

    static cuda_proc_api resolve(ggml_backend_reg_t reg) {
        cuda_proc_api api;
        if (!reg) {
            return api;
        }
        api.comm_init             = (comm_init_t)             ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init");
        api.comm_free             = (comm_free_t)             ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free");
        api.comm_allreduce_tensor = (comm_allreduce_tensor_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor");
        api.ar_get_stats          = (ar_get_stats_t)          ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_ar_get_stats");
        api.ar_reset_stats        = (ar_reset_stats_t)        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_ar_reset_stats");
        api.graph_owner_depth     = (graph_owner_depth_t)     ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_graph_owner_depth");
        api.graph_epoch           = (graph_epoch_t)           ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_graph_epoch");
        api.graph_invalidate      = (graph_invalidate_t)      ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_graph_invalidate");
        api.test_fail_cap         = (test_fail_cap_t)         ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_test_fail_next_capture");
        return api;
    }
};

enum ar_stat_idx {
    AR_STAT_HOST_CHUNKED    = 0,
    AR_STAT_HOST_COPY       = 1,
    AR_STAT_P2P_EVENT       = 2,
    AR_STAT_FUSED_DIRECT    = 3,
    AR_STAT_FUSED_TG_LAUNCH = 4,
    AR_STAT_WIRE_BF16       = 5,
    AR_STAT_WIRE_F32        = 6,
    AR_STAT_WIRE_F16        = 7,
    AR_STAT_WIRE_NAT_BF16   = 8,
    AR_STAT_P2P_INELIGIBLE  = 9,
    AR_STAT_COUNT           = 10,
};

static float quant_bf16_roundtrip(float x) {
    return ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));
}

static float quant_f16_roundtrip(float x) {
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
}

static void encode_elem(ggml_type t, float v, void * dst, size_t idx) {
    switch (t) {
        case GGML_TYPE_F32:  reinterpret_cast<float *>(dst)[idx]       = v; break;
        case GGML_TYPE_F16:  reinterpret_cast<ggml_fp16_t *>(dst)[idx] = ggml_fp32_to_fp16(v); break;
        case GGML_TYPE_BF16: reinterpret_cast<ggml_bf16_t *>(dst)[idx] = ggml_fp32_to_bf16(v); break;
        default: std::abort();
    }
}

static float decode_elem(ggml_type t, const void * src, size_t idx) {
    switch (t) {
        case GGML_TYPE_F32:  return reinterpret_cast<const float *>(src)[idx];
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(reinterpret_cast<const ggml_fp16_t *>(src)[idx]);
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(reinterpret_cast<const ggml_bf16_t *>(src)[idx]);
        default: std::abort();
    }
}

static float expected_wire_sym(ggml_type t, bool use_bf16_wire, float a, float b) {
    if (t == GGML_TYPE_F32 && use_bf16_wire) {
        return quant_bf16_roundtrip(a) + quant_bf16_roundtrip(b);
    }
    if (t == GGML_TYPE_F32) {
        return a + b;
    }
    if (t == GGML_TYPE_F16) {
        return quant_f16_roundtrip(quant_f16_roundtrip(a) + quant_f16_roundtrip(b));
    }
    return quant_bf16_roundtrip(quant_bf16_roundtrip(a) + quant_bf16_roundtrip(b));
}

static double compute_nmse(const std::vector<float> & ref, const std::vector<float> & got) {
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double) got[i] - (double) ref[i];
        num += d * d;
        den += (double) ref[i] * (double) ref[i];
    }
    return num / std::max(den, 1e-12);
}

static std::vector<ggml_backend_dev_t> find_gpu_devices() {
    std::vector<ggml_backend_dev_t> gpus;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpus.push_back(dev);
        }
    }
    return gpus;
}

static ggml_backend_dev_t find_cpu_device() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            return dev;
        }
    }
    return nullptr;
}

struct meta_test_ud {
    int64_t k0 = 0;
    int64_t k1 = 0;
    int64_t m  = 0;
    int64_t n  = 0;
};

static ggml_backend_meta_split_state test_split_state_cb(const ggml_tensor * tensor, void * ud_v) {
    const auto * ud = static_cast<const meta_test_ud *>(ud_v);
    ggml_backend_meta_split_state ss = {};
    if (std::strncmp(tensor->name, "W_", 2) == 0 || std::strncmp(tensor->name, "X_", 2) == 0) {
        ss.axis       = GGML_BACKEND_SPLIT_AXIS_0;
        ss.n_segments = 1;
        ss.nr[0]      = 1;
        ss.ne[0]      = ud->k0;
        ss.ne[1]      = ud->k1;
        if (std::strstr(tensor->name,"repeat")) { ss.nr[0]=4;ss.ne[0]/=4;ss.ne[1]/=4; }
        return ss;
    }
    ss.axis       = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    ss.n_segments = 1;
    ss.nr[0]      = 1;
    return ss;
}

// Build a meta graph with n_layers sequential PARTIAL matmul -> AR -> scale -> add stages.
// Returns output tensor `cur` plus `scale_nodes` (to allow op_params mutation on same uid).
struct built_meta_graph {
    ggml_context_ptr ctx;
    ggml_cgraph *    gf = nullptr;
    std::vector<ggml_tensor *> weights;
    ggml_tensor *    x_in  = nullptr;
    ggml_tensor *    scale0 = nullptr;
    ggml_tensor *    out   = nullptr;
};

static built_meta_graph build_layered_partial_graph(
        int64_t K, int64_t M, int64_t N, int n_layers, uint64_t uid, float scale_val) {
    built_meta_graph bg;
    const size_t n_tensors = (size_t) n_layers * 5 + 16;
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(n_tensors * 2, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    bg.ctx.reset(ggml_init(params));
    ggml_context * ctx = bg.ctx.get();

    bg.x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_set_name(bg.x_in, "X_in");
    ggml_set_input(bg.x_in);

    ggml_tensor * bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, N);
    ggml_set_name(bias, "B_mirrored");
    ggml_set_input(bias);

    ggml_tensor * acc = bias;
    for (int l = 0; l < n_layers; ++l) {
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        char wname[32];
        std::snprintf(wname, sizeof(wname), "W_%d", l);
        ggml_set_name(w, wname);
        ggml_set_input(w);
        bg.weights.push_back(w);

        ggml_tensor * mm = ggml_mul_mat(ctx, w, bg.x_in); // AXIS_0 x AXIS_0 -> PARTIAL!
        // Consuming PARTIAL with MIRRORED forces meta backend to emit comm_allreduce + subgraph split:
        ggml_tensor * scaled = ggml_scale(ctx, mm, scale_val);
        if (l == 0) {
            bg.scale0 = scaled;
        }
        acc = ggml_add(ctx, acc, scaled);
    }
    bg.out = acc;
    ggml_set_output(bg.out);

    bg.gf = ggml_new_graph_custom(ctx, n_tensors * 2, false);
    ggml_build_forward_expand(bg.gf, bg.out);
    bg.gf->uid = uid;
    return bg;
}

static bool test_meta_graphs_for_devs(
        const char * label,
        ggml_backend_dev_t devs[2],
        bool has_gpu,
        const cuda_proc_api * api) {
    std::printf("[meta-graph:%s] starting (K=128, M=2048, N=1)\n", label);

    const int64_t K = 128;
    const int64_t M = 2048;
    const int64_t N = 1;

    // Helper lambda to run a whole-token graph check for given split & n_layers
    auto init_meta = [&](int64_t k0, int64_t k1, meta_test_ud & ud, ggml_backend_dev_t * out_mdev) -> ggml_backend_t {
        ud = {k0, k1, M, N};
        ggml_backend_dev_t mdev = ggml_backend_meta_device(devs, 2, test_split_state_cb, &ud);
        if (out_mdev) {
            *out_mdev = mdev;
        }
        return ggml_backend_dev_init(mdev, nullptr);
    };

    // 1) Active/active multi-uid + TG capture + pointer/op_params/epoch invalidation + small-subgraph skip
    {
        if (has_gpu) {
            set_env_val("GGML_HIP_AR_TG", "1");
            set_env_val("GGML_META_TOKEN_GRAPH", "1");
            set_env_val("GGML_META_TG_WARM", "2");
            set_env_val("GGML_META_TG_MIN_SUBGRAPHS", "8");
        }

        meta_test_ud ud{};
        ggml_backend_dev_t mdev = nullptr;
        ggml_backend_t mback = init_meta(K / 2, K / 2, ud, &mdev);
        CHECK(mback != nullptr, "meta backend init failed");
        ggml_backend_buffer_type_t mbuft = ggml_backend_dev_buffer_type(mdev);

        ggml_backend_t sb0 = ggml_backend_meta_simple_backend(mback, 0);
        ggml_backend_t sb1 = ggml_backend_meta_simple_backend(mback, 1);

        // Large graph: 10 layers -> 10 AR boundaries (>= 10 subgraphs >= 8 -> TG eligible)
        const int n_layers = 10;
        built_meta_graph gA = build_layered_partial_graph(K, M, N, n_layers, /*uid=*/ 0x1001ULL, 1.0f);
        built_meta_graph gB = build_layered_partial_graph(K, M, N, n_layers, /*uid=*/ 0x1002ULL, 1.0f);
        built_meta_graph gS = build_layered_partial_graph(K, M, N, /*n_layers=*/ 2, /*uid=*/ 0x2001ULL, 1.0f); // MTP-small (<8 subgraphs)

        ggml_gallocr_t gallocr = ggml_gallocr_new(mbuft);

        auto alloc_compute_graph = [&](ggml_gallocr_t ga, ggml_cgraph * gf) -> bool {
            for (int i = 0; i < gf->n_nodes; ++i) {
                gf->nodes[i]->data   = nullptr;
                gf->nodes[i]->buffer = nullptr;
            }
            return ggml_gallocr_alloc_graph(ga, gf);
        };

        auto alloc_and_fill = [&](built_meta_graph & g, float w_base, float x_base, float b_val,
                                  ggml_context_ptr & leaf_ctx, ggml_backend_buffer_ptr & leaf_buf) {
            ggml_init_params lp = {ggml_tensor_overhead() * (g.weights.size() + 4), nullptr, true};
            leaf_ctx.reset(ggml_init(lp));
            // Reserve buffer on graph tensors (all inputs live inside g.ctx and need buffer allocation)
            std::vector<ggml_tensor *> inputs;
            inputs.push_back(g.x_in);
            // Collect bias & weights via graph leafs
            for (int i = 0; i < g.gf->n_leafs; ++i) {
                if (g.gf->leafs[i] != g.x_in) {
                    inputs.push_back(g.gf->leafs[i]);
                }
            }
            // Allocate g's input tensors in one meta buffer, then gallocr compute tensors
            ggml_backend_t tmp_back = mback;
            GGML_UNUSED(tmp_back);
            size_t total_sz = 0;
            for (auto * t : inputs) {
                total_sz += ggml_backend_buft_get_alloc_size(mbuft, t) + 64;
            }
            leaf_buf.reset(ggml_backend_buft_alloc_buffer(mbuft, total_sz));
            ggml_backend_buffer_set_usage(leaf_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            char * base = (char *) ggml_backend_buffer_get_base(leaf_buf.get());
            ggml_tallocr ta = ggml_tallocr_new(leaf_buf.get());
            for (auto * t : inputs) {
                ggml_tallocr_alloc(&ta, t);
            }
            GGML_UNUSED(base);

            std::vector<float> x_data(K * N);
            for (int64_t i = 0; i < K * N; ++i) {
                x_data[i] = x_base + 0.01f * (float) ((i % 13) - 6);
            }
            ggml_backend_tensor_set(g.x_in, x_data.data(), 0, x_data.size() * sizeof(float));

            std::vector<float> b_data(M * N, b_val);
            for (auto * t : inputs) {
                if (std::strcmp(t->name, "B_mirrored") == 0) {
                    ggml_backend_tensor_set(t, b_data.data(), 0, b_data.size() * sizeof(float));
                } else if (std::strncmp(t->name, "W_", 2) == 0) {
                    int lidx = std::atoi(t->name + 2);
                    std::vector<float> w_data(K * M);
                    for (int64_t m = 0; m < M; ++m) {
                        for (int64_t k = 0; k < K; ++k) {
                            w_data[m * K + k] = w_base * (float) (((m + k + lidx) % 7) - 3) * 0.02f;
                        }
                    }
                    ggml_backend_tensor_set(t, w_data.data(), 0, w_data.size() * sizeof(float));
                }
            }
        };

        auto cpu_oracle = [&](float w_base, float x_base, float b_val, int nl, float scale_val) -> std::vector<float> {
            std::vector<float> x_data(K * N);
            for (int64_t i = 0; i < K * N; ++i) {
                x_data[i] = x_base + 0.01f * (float) ((i % 13) - 6);
            }
            std::vector<float> out(M * N, b_val);
            for (int l = 0; l < nl; ++l) {
                for (int64_t m = 0; m < M; ++m) {
                    double s0 = 0.0, s1 = 0.0;
                    for (int64_t k = 0; k < K / 2; ++k) {
                        double wv = w_base * (double) (((m + k + l) % 7) - 3) * 0.02;
                        s0 += wv * (double) x_data[k];
                    }
                    for (int64_t k = K / 2; k < K; ++k) {
                        double wv = w_base * (double) (((m + k + l) % 7) - 3) * 0.02;
                        s1 += wv * (double) x_data[k];
                    }
                    out[m] += (float) ((s0 + s1) * (double) scale_val);
                }
            }
            return out;
        };

        ggml_context_ptr lcA, lcB, lcS;
        ggml_backend_buffer_ptr lbA, lbB, lbS;
        alloc_and_fill(gA, 1.0f, 0.5f, 0.25f, lcA, lbA);
        alloc_and_fill(gB, 0.75f, -0.3f, 0.1f, lcB, lbB);
        alloc_and_fill(gS, 1.0f, 0.5f, 0.25f, lcS, lbS);

        CHECK(ggml_gallocr_reserve(gallocr, gA.gf), "gallocr_reserve gA failed");

        std::vector<float> refA = cpu_oracle(1.0f, 0.5f, 0.25f, n_layers, 1.0f);
        std::vector<float> refS = cpu_oracle(1.0f, 0.5f, 0.25f, 2, 1.0f);
        std::vector<float> got(M * N);

        // Warmup + capture + replay of gA, interleaved with gS (small <8 subgraphs)
        for (int iter = 0; iter < 6; ++iter) {
            CHECK(alloc_compute_graph(gallocr, gA.gf), "alloc gA iter");
            CHECK(ggml_backend_graph_compute(mback, gA.gf) == GGML_STATUS_SUCCESS, "compute gA iter");
            ggml_backend_tensor_get(gA.out, got.data(), 0, got.size() * sizeof(float));
            double nmse = compute_nmse(refA, got);
            CHECK(nmse < 1e-4, "gA iter=%d nmse=%.6e too high", iter, nmse);

            CHECK(alloc_compute_graph(gallocr, gS.gf), "alloc gS iter");
            CHECK(ggml_backend_graph_compute(mback, gS.gf) == GGML_STATUS_SUCCESS, "compute gS iter");
            ggml_backend_tensor_get(gS.out, got.data(), 0, got.size() * sizeof(float));
            double nmseS = compute_nmse(refS, got);
            CHECK(nmseS < 1e-4, "gS iter=%d nmse=%.6e too high", iter, nmseS);
        }

        // Compute gB on alternating UID
        std::vector<float> refB = cpu_oracle(0.75f, -0.3f, 0.1f, n_layers, 1.0f);
        for (int iter = 0; iter < 4; ++iter) {
            CHECK(alloc_compute_graph(gallocr, gB.gf), "alloc gB");
            CHECK(ggml_backend_graph_compute(mback, gB.gf) == GGML_STATUS_SUCCESS, "compute gB");
            ggml_backend_tensor_get(gB.out, got.data(), 0, got.size() * sizeof(float));
            double nmseB = compute_nmse(refB, got);
            CHECK(nmseB < 1e-4, "gB iter=%d nmse=%.6e too high", iter, nmseB);
        }

        // Same-UID op_params mutation test: change scale0 from 1.0f -> 2.5f WITHOUT changing uid!
        {
            float new_scale = 2.5f;
            std::memcpy(gA.scale0->op_params, &new_scale, sizeof(new_scale));
            CHECK(alloc_compute_graph(gallocr, gA.gf), "alloc gA before op_params mut");
            CHECK(ggml_backend_graph_compute(mback, gA.gf) == GGML_STATUS_SUCCESS, "compute gA after op_params mut");
            ggml_backend_tensor_get(gA.out, got.data(), 0, got.size() * sizeof(float));
            // Only layer 0 was scaled by 2.5 (layers 1..9 stayed 1.0): compute exact oracle
            std::vector<float> refA_mut = refA;
            std::vector<float> refL0 = cpu_oracle(1.0f, 0.5f, 0.0f, 1, 1.5f); // +1.5x layer 0
            for (size_t i = 0; i < refA_mut.size(); ++i) {
                refA_mut[i] += refL0[i];
            }
            double nmse_mut = compute_nmse(refA_mut, got);
            CHECK(nmse_mut < 1e-4, "op_params mutation on same uid stale graph! nmse=%.6e", nmse_mut);
            // Restore scale0 = 1.0f, switch via gB (so meta rebuilds bcj while TG keeps 0x1001 cached!), then re-warm gA
            float orig_scale = 1.0f;
            std::memcpy(gA.scale0->op_params, &orig_scale, sizeof(orig_scale));
            CHECK(alloc_compute_graph(gallocr, gB.gf), "alloc gB switch");
            CHECK(ggml_backend_graph_compute(mback, gB.gf) == GGML_STATUS_SUCCESS, "compute gB switch");
            for (int r = 0; r < 3; ++r) {
                CHECK(alloc_compute_graph(gallocr, gA.gf), "alloc gA restore");
                CHECK(ggml_backend_graph_compute(mback, gA.gf) == GGML_STATUS_SUCCESS, "compute gA restore");
            }
            CHECK(alloc_compute_graph(gallocr, gB.gf), "alloc gB switch 2");
            CHECK(ggml_backend_graph_compute(mback, gB.gf) == GGML_STATUS_SUCCESS, "compute gB switch 2");
        }

        // Same-UID pointer swap mutation test: allocate a fresh gA_clone with SAME uid=0x1001ULL & different input data!
        {
            built_meta_graph gA2 = build_layered_partial_graph(K, M, N, n_layers, /*uid=*/ 0x1001ULL, 1.0f);
            ggml_context_ptr lcA2;
            ggml_backend_buffer_ptr lbA2;
            alloc_and_fill(gA2, 0.5f, -0.8f, -0.2f, lcA2, lbA2);
            CHECK(alloc_compute_graph(gallocr, gA2.gf), "alloc gA2 pointer swap");
            CHECK(ggml_backend_graph_compute(mback, gA2.gf) == GGML_STATUS_SUCCESS, "compute gA2 pointer swap");
            ggml_backend_tensor_get(gA2.out, got.data(), 0, got.size() * sizeof(float));
            std::vector<float> refA2 = cpu_oracle(0.5f, -0.8f, -0.2f, n_layers, 1.0f);
            double nmse_swap = compute_nmse(refA2, got);
            CHECK(nmse_swap < 1e-4, "pointer/graph swap on same uid failed! nmse=%.6e", nmse_swap);
            // Switch via gB again so gA.gf triggers rebuild after gA2
            CHECK(alloc_compute_graph(gallocr, gB.gf), "alloc gB post-swap");
            CHECK(ggml_backend_graph_compute(mback, gB.gf) == GGML_STATUS_SUCCESS, "compute gB post-swap");
        }

        // GPU-only epoch invalidation & capture-failure rollback tests
        if (has_gpu && api && api->graph_epoch && api->graph_invalidate) {
            const uint64_t e0 = api->graph_epoch(sb0);
            api->graph_invalidate(sb0);
            const uint64_t e1 = api->graph_epoch(sb0);
            CHECK(e1 > e0, "graph_invalidate did not bump epoch (%llu -> %llu)", (unsigned long long) e0, (unsigned long long) e1);

            for (int r = 0; r < 4; ++r) {
                CHECK(alloc_compute_graph(gallocr, gA.gf), "alloc gA post-epoch");
                CHECK(ggml_backend_graph_compute(mback, gA.gf) == GGML_STATUS_SUCCESS, "compute gA post-epoch");
            }
            ggml_backend_tensor_get(gA.out, got.data(), 0, got.size() * sizeof(float));
            CHECK(compute_nmse(refA, got) < 1e-4, "post-epoch output mismatch");
        }

        if (has_gpu && api && api->test_fail_cap) {
            built_meta_graph gF = build_layered_partial_graph(K, M, N, n_layers, /*uid=*/ 0x9999ULL, 1.0f);
            ggml_context_ptr lcF;
            ggml_backend_buffer_ptr lbF;
            alloc_and_fill(gF, 1.2f, 0.4f, 0.3f, lcF, lbF);
            std::vector<float> refF = cpu_oracle(1.2f, 0.4f, 0.3f, n_layers, 1.0f);

            // Call 1: warmup 1
            CHECK(alloc_compute_graph(gallocr, gF.gf), "alloc gF call 1");
            CHECK(ggml_backend_graph_compute(mback, gF.gf) == GGML_STATUS_SUCCESS, "compute gF call 1");
            // Arm capture failure for call 2 (capture attempt!)
            api->test_fail_cap(sb1, true);
            CHECK(alloc_compute_graph(gallocr, gF.gf), "alloc gF call 2");
            CHECK(ggml_backend_graph_compute(mback, gF.gf) == GGML_STATUS_SUCCESS, "compute gF fallback after capture fail");
            api->test_fail_cap(sb1, false);
            ggml_backend_tensor_get(gF.out, got.data(), 0, got.size() * sizeof(float));
            double nmseF = compute_nmse(refF, got);
            CHECK(nmseF < 1e-4, "capture-failure fallback produced wrong/double-AR result! nmse=%.6e", nmseF);
        }

#ifdef GGML_META_HAS_TG_STATS
        if (has_gpu) {
            uint64_t tg_stats[8] = {};
            size_t n_st = ggml_backend_meta_tg_get_stats(mback, tg_stats, 8);
            CHECK(n_st >= 7, "meta tg stats too short");
            // [0]=captures, [1]=launches, [2]=evicts, [3]=sig_inval, [4]=epoch_inval, [5]=cap_fails, [6]=skipped_small
            std::printf("  [TG stats] captures=%llu launches=%llu evicts=%llu sig_inv=%llu epoch_inv=%llu cap_fail=%llu skip_small=%llu\n",
                (unsigned long long) tg_stats[0], (unsigned long long) tg_stats[1],
                (unsigned long long) tg_stats[2], (unsigned long long) tg_stats[3],
                (unsigned long long) tg_stats[4], (unsigned long long) tg_stats[5],
                (unsigned long long) tg_stats[6]);
            CHECK(tg_stats[0] >= 2, "expected >=2 TG captures, got %llu", (unsigned long long) tg_stats[0]);
            CHECK(tg_stats[1] >= 4, "expected >=4 TG launches, got %llu", (unsigned long long) tg_stats[1]);
            CHECK(tg_stats[3] >= 1, "expected >=1 signature invalidation, got %llu", (unsigned long long) tg_stats[3]);
            CHECK(tg_stats[4] >= 1, "expected >=1 epoch invalidation, got %llu", (unsigned long long) tg_stats[4]);
            CHECK(tg_stats[5] >= 1, "expected >=1 capture failure recorded, got %llu", (unsigned long long) tg_stats[5]);
            CHECK(tg_stats[6] >= 1, "expected >=1 small-subgraph skip, got %llu", (unsigned long long) tg_stats[6]);
        }
#endif

        ggml_gallocr_free(gallocr);
        if (has_gpu && api && api->graph_owner_depth) {
            // mback owns and frees sb0/sb1; inspect the live backend, not freed storage.
            CHECK(api->graph_owner_depth(sb0) == 0, "outer_graph_depth leaked on sb0");
        }
        ggml_backend_free(mback);
    }

    // 2) Inactive shard (`k1 == 0`) with NaN poison in rank 1 weight slice must contribute exact 0!
    {
        meta_test_ud ud{};
        ggml_backend_dev_t mdev = nullptr;
        ggml_backend_t mback = init_meta(K, 0, ud, &mdev);
        CHECK(mback != nullptr, "meta inactive backend init failed");
        ggml_backend_buffer_type_t mbuft = ggml_backend_dev_buffer_type(mdev);
        built_meta_graph g = build_layered_partial_graph(K, M, N, 2, /*uid=*/ 0x3001ULL, 1.0f);

        std::vector<ggml_tensor *> inputs = {g.x_in};
        for (int i = 0; i < g.gf->n_leafs; ++i) {
            if (g.gf->leafs[i] != g.x_in) {
                inputs.push_back(g.gf->leafs[i]);
            }
        }
        size_t total_sz = 0;
        for (auto * t : inputs) {
            total_sz += ggml_backend_buft_get_alloc_size(mbuft, t) + 64;
        }
        ggml_backend_buffer_ptr lb(ggml_backend_buft_alloc_buffer(mbuft, total_sz));
        ggml_backend_buffer_set_usage(lb.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_tallocr ta = ggml_tallocr_new(lb.get());
        for (auto * t : inputs) {
            ggml_tallocr_alloc(&ta, t);
        }
        std::vector<float> x_data(K * N, 0.5f);
        ggml_backend_tensor_set(g.x_in, x_data.data(), 0, x_data.size() * sizeof(float));
        std::vector<float> b_data(M * N, 1.0f);
        std::vector<float> w_data(K * M, 0.125f);
        for (auto * t : inputs) {
            if (std::strcmp(t->name, "B_mirrored") == 0) {
                ggml_backend_tensor_set(t, b_data.data(), 0, b_data.size() * sizeof(float));
            } else if (std::strncmp(t->name, "W_", 2) == 0) {
                ggml_backend_tensor_set(t, w_data.data(), 0, w_data.size() * sizeof(float));
            }
        }
        ggml_gallocr_t gallocr = ggml_gallocr_new(mbuft);

        auto alloc_compute_graph = [&](ggml_gallocr_t ga, ggml_cgraph * gf) -> bool {
            for (int i = 0; i < gf->n_nodes; ++i) {
                gf->nodes[i]->data   = nullptr;
                gf->nodes[i]->buffer = nullptr;
            }
            return ggml_gallocr_alloc_graph(ga, gf);
        };
        CHECK(alloc_compute_graph(gallocr, g.gf), "alloc inactive");
        CHECK(ggml_backend_graph_compute(mback, g.gf) == GGML_STATUS_SUCCESS, "compute inactive");

        std::vector<float> got(M * N);
        ggml_backend_tensor_get(g.out, got.data(), 0, got.size() * sizeof(float));
        // Each layer adds K * 0.5 * 0.125 = 128 * 0.0625 = 8.0; 2 layers + bias 1.0 = 17.0
        for (size_t i = 0; i < got.size(); ++i) {
            CHECK(std::fabs(got[i] - 17.0f) < 1e-2f, "inactive shard wrong at %zu: got=%f exp=17.0", i, got[i]);
        }
        ggml_gallocr_free(gallocr);
        ggml_backend_free(mback);
    }

    std::printf("[meta-graph:%s] PASS\n", label);
    return true;
}

static bool test_comm_direct_gpu(ggml_backend_dev_t gpus[2], const cuda_proc_api & api, bool do_bench) {
    CHECK(api.comm_init && api.comm_free && api.comm_allreduce_tensor, "missing CUDA comm proc addresses");

    auto run_with_env = [&](const char * mode_name, const char * p2p, const char * fused, const char * tg, const char * bf16_thresh,
                            auto && body) -> bool {
        set_env_val("GGML_HIP_AR_P2P", p2p);
        set_env_val("GGML_HIP_AR_FUSED", fused);
        set_env_val("GGML_HIP_AR_TG", tg);
        set_env_val("GGML_CUDA_AR_BF16_THRESHOLD", bf16_thresh);

        ggml_backend_t b[2] = {
            ggml_backend_dev_init(gpus[0], nullptr),
            ggml_backend_dev_init(gpus[1], nullptr),
        };
        CHECK(b[0] && b[1], "gpu backend init failed");
        void * comm = api.comm_init(b, 2);
        CHECK(comm != nullptr, "comm_init failed");

        bool ok = body(mode_name, b, comm);

        ggml_backend_synchronize(b[0]);
        ggml_backend_synchronize(b[1]);
        api.comm_free(comm);
        ggml_backend_free(b[0]);
        ggml_backend_free(b[1]);
        return ok;
    };

    // Helper: create tensor on each device inside allocated buffer, run allreduce, check bitwise + oracle
    auto check_single_ar = [&](void * comm, ggml_backend_t b[2], ggml_type dtype, int64_t ne,
                               bool active0, bool active1, bool expect_internal, bool use_bf16_oracle,
                               const std::vector<float> * custom0 = nullptr,
                               const std::vector<float> * custom1 = nullptr,
                               size_t ptr_offset_bytes = 0) -> bool {
        ggml_init_params gp = {ggml_tensor_overhead() * 4, nullptr, true};
        ggml_context_ptr ctx(ggml_init(gp));
        ggml_tensor * t[2] = {
            ggml_new_tensor_1d(ctx.get(), dtype, std::max<int64_t>(ne, 1)),
            ggml_new_tensor_1d(ctx.get(), dtype, std::max<int64_t>(ne, 1)),
        };
        if (ne == 0) {
            t[0]->ne[0] = 0;
            t[1]->ne[0] = 0;
        }
        if (active0) { t[0]->flags |= GGML_TENSOR_FLAG_COMPUTE; }
        if (active1) { t[1]->flags |= GGML_TENSOR_FLAG_COMPUTE; }

        const size_t nbytes = (size_t) std::max<int64_t>(ne, 1) * ggml_type_size(dtype) + ptr_offset_bytes + 64;
        ggml_backend_buffer_ptr buf0(ggml_backend_alloc_buffer(b[0], nbytes));
        ggml_backend_buffer_ptr buf1(ggml_backend_alloc_buffer(b[1], nbytes));
        t[0]->buffer = buf0.get();
        t[1]->buffer = buf1.get();
        t[0]->data   = (char *) ggml_backend_buffer_get_base(buf0.get()) + ptr_offset_bytes;
        t[1]->data   = (char *) ggml_backend_buffer_get_base(buf1.get()) + ptr_offset_bytes;

        const size_t payload_bytes = (size_t) ne * ggml_type_size(dtype);
        std::vector<uint8_t> raw0(std::max<size_t>(payload_bytes, 1), 0), raw1(std::max<size_t>(payload_bytes, 1), 0);
        std::vector<float> f0(ne), f1(ne), ref_wire(ne), ref_f32(ne);
        const float nan_val = std::numeric_limits<float>::quiet_NaN();

        for (int64_t i = 0; i < ne; ++i) {
            float v0 = custom0 ? (*custom0)[i] : (active0 ? 0.125f * std::sin((float) (i % 97) * 0.1f) + 0.05f : nan_val);
            float v1 = custom1 ? (*custom1)[i] : (active1 ? 0.125f * std::cos((float) (i % 89) * 0.1f) - 0.03f : nan_val);
            encode_elem(dtype, v0, raw0.data(), (size_t) i);
            encode_elem(dtype, v1, raw1.data(), (size_t) i);
            const float eff0 = active0 ? v0 : 0.0f;
            const float eff1 = active1 ? v1 : 0.0f;
            f0[i] = eff0;
            f1[i] = eff1;
            ref_wire[i] = expected_wire_sym(dtype, use_bf16_oracle, eff0, eff1);
            ref_f32[i]  = eff0 + eff1;
        }
        if (payload_bytes > 0) {
            ggml_backend_tensor_set(t[0], raw0.data(), 0, payload_bytes);
            ggml_backend_tensor_set(t[1], raw1.data(), 0, payload_bytes);
        }

        const bool ok = api.comm_allreduce_tensor(comm, t);
        CHECK(ok == expect_internal, "comm_allreduce_tensor returned %d expected %d (ne=%lld dtype=%d off=%zu)",
              (int) ok, (int) expect_internal, (long long) ne, (int) dtype, ptr_offset_bytes);
        if (!ok || ne == 0) {
            return true;
        }

        ggml_backend_synchronize(b[0]);
        ggml_backend_synchronize(b[1]);

        std::vector<uint8_t> out0(std::max<size_t>(payload_bytes, 1)), out1(std::max<size_t>(payload_bytes, 1));
        if (payload_bytes > 0) {
            ggml_backend_tensor_get(t[0], out0.data(), 0, payload_bytes);
            ggml_backend_tensor_get(t[1], out1.data(), 0, payload_bytes);
        }

        if (std::getenv("T2012_AR") && dtype == GGML_TYPE_F32 && use_bf16_oracle) {
            // Fork contract: local F32 + rounded peer. Ranks are NOT identical.
            for (int64_t i = 0; i < ne; ++i) {
                const float got[2] = {decode_elem(dtype,out0.data(),i), decode_elem(dtype,out1.data(),i)};
                const float expected[2] = {f0[i] + quant_bf16_roundtrip(f1[i]), f1[i] + quant_bf16_roundtrip(f0[i])};
                for (int rank = 0; rank < 2; ++rank) {
                    CHECK(std::memcmp(&got[rank], &expected[rank], sizeof(float)) == 0,
                        "fork oracle mismatch rank=%d ne=%lld i=%lld got=%g exp=%g", rank,
                        (long long)ne, (long long)i, got[rank], expected[rank]);
                }
            }
        } else if (custom0 == nullptr && custom1 == nullptr) {
            // Bitwise symmetry between rank 0 and rank 1
            for (int64_t i = 0; i < ne; ++i) {
                float g0 = decode_elem(dtype, out0.data(), (size_t) i);
                float g1 = decode_elem(dtype, out1.data(), (size_t) i);
                uint32_t u0 = 0, u1 = 0;
                std::memcpy(&u0, &g0, 4);
                std::memcpy(&u1, &g1, 4);
                CHECK(u0 == u1, "rank0!=rank1 at i=%lld (ne=%lld dtype=%d): %g (0x%08x) vs %g (0x%08x)",
                      (long long) i, (long long) ne, (int) dtype, g0, u0, g1, u1);
                CHECK(std::fabs(g0 - ref_wire[i]) <= 1e-5f * (1.0f + std::fabs(ref_wire[i])),
                      "wire mismatch at i=%lld (ne=%lld dtype=%d bf16=%d): got=%g exp_wire=%g (a=%g b=%g)",
                      (long long) i, (long long) ne, (int) dtype, (int) use_bf16_oracle, g0, ref_wire[i], f0[i], f1[i]);
            }
        } else {
            // Edge-value verification (NaN / Inf / -0.0 / cancellation)
            for (int64_t i = 0; i < ne; ++i) {
                float g0 = decode_elem(dtype, out0.data(), (size_t) i);
                float g1 = decode_elem(dtype, out1.data(), (size_t) i);
                if (std::isnan(ref_wire[i])) {
                    CHECK(std::isnan(g0) && std::isnan(g1), "NaN lost at i=%lld: g0=%g g1=%g", (long long) i, g0, g1);
                } else if (std::isinf(ref_wire[i])) {
                    CHECK(std::isinf(g0) && std::signbit(g0) == std::signbit(ref_wire[i]) &&
                          std::isinf(g1) && std::signbit(g1) == std::signbit(ref_wire[i]),
                          "Inf mismatch at i=%lld", (long long) i);
                } else {
                    CHECK(std::fabs(g0 - ref_wire[i]) <= 1e-4f * (1.0f + std::fabs(ref_wire[i])),
                          "edge val mismatch at i=%lld: g0=%g exp=%g", (long long) i, g0, ref_wire[i]);
                }
            }
        }
        return true;
    };

    const int64_t sizes[] = {
        0, 4, 512, 2048, 5120, 10240, 20479, 20480, 20481,
        65536, 65537, 131072, 131073, 300000, 2621440, 9437184
    };
    const int64_t fused_sizes[] = {0, 4, 512, 2048, 5120, 10240, 65536, 131072, 300000};

    // Matrix across transports U (upstream host staged), P (P2P event), F (fused), and F32 wire control
    struct mode_cfg {
        const char * name;
        const char * p2p;
        const char * fused;
        const char * thresh;
        bool use_bf16;
        bool small_only;
    } modes[] = {
        {"U_host_bf16",  "0", "0", "1", true,  true},
        {"P_event_bf16", "1", "0", "1", true,  false},
        {"F_fused_bf16", "0", "1", "1", true,  true},
        {"P_event_f32",  "1", "0", "0", false, true},
        {"F_fused_f32",  "0", "1", "0", false, true},
    };

    if (std::getenv("T2012_AR")) {
        for (const char * fused : {"0", "1"}) {
            CHECK(run_with_env("t2012_fork", "1", fused, "0", "1", [&](const char *, ggml_backend_t b[2], void * comm) {
                for (int repeat = 0; repeat < 8; ++repeat) {
                    for (int64_t ne : {4, 5120, 10240, 20480, 131072, 300000}) {
                        for (auto active : {std::pair<bool,bool>{true,true}, {true,false}, {false,true}, {false,false}}) {
                            CHECK(check_single_ar(comm,b,GGML_TYPE_F32,ne,active.first,active.second,true,true), "t2012 fork numeric");
                        }
                    }
                }
                uint64_t st[AR_STAT_COUNT] = {};
                api.ar_get_stats(comm,st,AR_STAT_COUNT);
                std::printf("t2012 fork fused=%llu p2p=%llu bf16=%llu\n", (unsigned long long)st[AR_STAT_FUSED_DIRECT],
                    (unsigned long long)st[AR_STAT_P2P_EVENT], (unsigned long long)st[AR_STAT_WIRE_BF16]);
                CHECK(st[AR_STAT_P2P_EVENT] > 0, "P2P not exercised");
                if (fused[0] == '1') CHECK(st[AR_STAT_FUSED_DIRECT] > 0, "fused not exercised");
                return true;
            }), "t2012 fork mode");
        }
        return true;
    }

    // only the touched F32/BF16 fused path, same strict oracle as the full matrix.
    if (std::getenv("T2011_AR")) {
        return run_with_env("t2011_fused", "1", "1", "0", "1", [&](const char *, ggml_backend_t b[2], void * comm) {
            for (int repeat = 0; repeat < 8; ++repeat) {
                for (int64_t ne : {5120, 10240, 20480}) {
                    for (auto active : {std::pair<bool,bool>{true,true}, {true,false}, {false,true}, {false,false}}) {
                        CHECK(check_single_ar(comm,b,GGML_TYPE_F32,ne,active.first,active.second,true,true), "t2011 fused numeric");
                    }
                }
            }
            CHECK(api.ar_get_stats != nullptr, "missing AR telemetry");
            uint64_t st[AR_STAT_COUNT] = {};
            api.ar_get_stats(comm,st,AR_STAT_COUNT);
            std::printf("t2011 fused calls=%llu bf16=%llu\n", (unsigned long long)st[AR_STAT_FUSED_DIRECT], (unsigned long long)st[AR_STAT_WIRE_BF16]);
            CHECK(st[AR_STAT_FUSED_DIRECT] >= 96, "fused path was not exercised");
            return true;
        });
    }
    for (const auto & mc : modes) {
        bool ok = run_with_env(mc.name, mc.p2p, mc.fused, "0", mc.thresh, [&](const char * mname, ggml_backend_t b[2], void * comm) {
            std::printf("[comm:%s] matrix check...\n", mname);
            const int64_t * sz_list = mc.small_only ? fused_sizes : sizes;
            const size_t n_sz = mc.small_only ? (sizeof(fused_sizes) / sizeof(fused_sizes[0])) : (sizeof(sizes) / sizeof(sizes[0]));
            const ggml_type dtypes[] = {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16};

            for (ggml_type dt : dtypes) {
                if (!mc.use_bf16 && dt != GGML_TYPE_F32) {
                    continue;
                }
                for (size_t si = 0; si < n_sz; ++si) {
                    int64_t ne = sz_list[si];
                    // Kernel internal requires (nbytes & 0xF) == 0 for >0 bytes; skip odd element counts on F16/BF16 if not 16B aligned here, test those separately!
                    if (ne > 0 && ((ne * (int64_t) ggml_type_size(dt)) & 0xF) != 0) {
                        CHECK(check_single_ar(comm, b, dt, ne, true, true, /*expect_internal=*/ false, mc.use_bf16),
                              "non-16B failed");
                        continue;
                    }
                    // Test {T,T}, plus {T,F}, {F,T}, {F,F} on representative sizes
                    CHECK(check_single_ar(comm, b, dt, ne, true, true, true, mc.use_bf16), "TT");
                    if (ne == 0 || ne == 512 || ne == 5120 || ne == 65536 || ne == 300000) {
                        CHECK(check_single_ar(comm, b, dt, ne, true,  false, true, mc.use_bf16), "TF");
                        CHECK(check_single_ar(comm, b, dt, ne, false, true,  true, mc.use_bf16), "FT");
                        CHECK(check_single_ar(comm, b, dt, ne, false, false, true, mc.use_bf16), "FF");
                    }
                }
            }

            // Edge values (F32): NaN, +Inf, -Inf, -0.0, cancellation
            {
                std::vector<float> c0(16, 0.0f), c1(16, 0.0f);
                c0[0] = std::numeric_limits<float>::quiet_NaN(); c1[0] = 1.0f;
                c0[1] = std::numeric_limits<float>::infinity();  c1[1] = 1.0f;
                c0[2] = -std::numeric_limits<float>::infinity(); c1[2] = 1.0f;
                c0[3] = -0.0f;                                   c1[3] = -0.0f;
                c0[4] = 65504.0f;                                c1[4] = -65504.0f;
                c0[5] = 1e-38f;                                  c1[5] = 1e-38f;
                c0[6] = 0.25f;                                   c1[6] = -0.125f;
                CHECK(check_single_ar(comm, b, GGML_TYPE_F32, 16, true, true, true, mc.use_bf16, &c0, &c1), "edge values");
            }

            // Unaligned pointer fallback check (+4 bytes offset)
            CHECK(check_single_ar(comm, b, GGML_TYPE_F32, 64, true, true, /*expect_internal=*/ false, mc.use_bf16, nullptr, nullptr, 4), "unaligned");
            // Non-16B size check (ne=3 F32 = 12 bytes) -> graceful false
            CHECK(check_single_ar(comm, b, GGML_TYPE_F32, 3, true, true, /*expect_internal=*/ false, mc.use_bf16), "ne=3 graceful fallback");

            // Dispatch telemetry verification (F1d)
            if (api.ar_get_stats && api.ar_reset_stats) {
                api.ar_reset_stats(comm);
                CHECK(check_single_ar(comm, b, GGML_TYPE_F32, 5120, true, true, true, mc.use_bf16), "dispatch 5120");
                CHECK(check_single_ar(comm, b, GGML_TYPE_F32, 300000, true, true, true, mc.use_bf16), "dispatch 300k");
                uint64_t st[AR_STAT_COUNT] = {};
                api.ar_get_stats(comm, st, AR_STAT_COUNT);
                std::printf("  [F1d %s] chunked=%llu copy=%llu p2p_ev=%llu fused=%llu bf16_w=%llu f32_w=%llu\n",
                    mname,
                    (unsigned long long) st[AR_STAT_HOST_CHUNKED],
                    (unsigned long long) st[AR_STAT_HOST_COPY],
                    (unsigned long long) st[AR_STAT_P2P_EVENT],
                    (unsigned long long) st[AR_STAT_FUSED_DIRECT],
                    (unsigned long long) st[AR_STAT_WIRE_BF16],
                    (unsigned long long) st[AR_STAT_WIRE_F32]);
                if (std::strcmp(mc.p2p, "1") == 0) {
                    CHECK(st[AR_STAT_P2P_EVENT] >= 2, "P2P mode did not dispatch P2P event path!");
                } else if (std::strcmp(mc.fused, "1") == 0) {
                    CHECK(st[AR_STAT_FUSED_DIRECT] >= 1, "Fused mode did not dispatch fused path on 5120!");
                    CHECK(st[AR_STAT_HOST_CHUNKED] + st[AR_STAT_HOST_COPY] >= 1, "300k did not fallback to host path when P2P=0!");
                } else {
                    CHECK(st[AR_STAT_HOST_CHUNKED] + st[AR_STAT_HOST_COPY] >= 2, "U mode did not dispatch host path!");
                }
                if (mc.use_bf16) {
                    CHECK(st[AR_STAT_WIRE_BF16] >= 2 && st[AR_STAT_WIRE_F32] == 0, "BF16 wire policy mismatch");
                } else {
                    CHECK(st[AR_STAT_WIRE_F32] >= 2 && st[AR_STAT_WIRE_BF16] == 0, "F32 wire policy mismatch");
                }
            }

            // Rank skew / liveness loop (200 alternating ARs with extra asymmetric work on rank 0)
            {
                const int64_t ne_skew = 5120;
                for (int iter = 0; iter < 50; ++iter) {
                    CHECK(check_single_ar(comm, b, GGML_TYPE_F32, ne_skew, true, true, true, mc.use_bf16), "skew 5120");
                }
            }
            return true;
        });
        if (!ok) {
            return false;
        }
    }

    if (do_bench) {
        const char * bench_modes[][4] = {
            {"U", "0", "0", "1"},
            {"P", "1", "0", "1"},
            {"F", "0", "1", "1"},
        };
        for (const auto & bm : bench_modes) {
            run_with_env(bm[0], bm[1], bm[2], "0", bm[3], [&](const char * mname, ggml_backend_t b[2], void * comm) {
                ggml_init_params gp = {ggml_tensor_overhead() * 4, nullptr, true};
                ggml_context_ptr ctx(ggml_init(gp));
                const int64_t ne = 5120;
                ggml_tensor * t[2] = {ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, ne), ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, ne)};
                t[0]->flags |= GGML_TENSOR_FLAG_COMPUTE;
                t[1]->flags |= GGML_TENSOR_FLAG_COMPUTE;
                ggml_backend_buffer_ptr b0(ggml_backend_alloc_buffer(b[0], ne * 4 + 64));
                ggml_backend_buffer_ptr b1(ggml_backend_alloc_buffer(b[1], ne * 4 + 64));
                t[0]->buffer = b0.get(); t[0]->data = ggml_backend_buffer_get_base(b0.get());
                t[1]->buffer = b1.get(); t[1]->data = ggml_backend_buffer_get_base(b1.get());
                for (int w = 0; w < 10; ++w) { api.comm_allreduce_tensor(comm, t); }
                ggml_backend_synchronize(b[0]); ggml_backend_synchronize(b[1]);
                auto t0 = std::chrono::steady_clock::now();
                for (int it = 0; it < 129; ++it) {
                    api.comm_allreduce_tensor(comm, t);
                }
                ggml_backend_synchronize(b[0]); ggml_backend_synchronize(b[1]);
                auto t1 = std::chrono::steady_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                std::printf("BENCH [%s] 129x AR(5120 F32): %.3f ms total (%.2f us/AR)\n", mname, ms, (ms * 1000.0) / 129.0);
                return true;
            });
        }
    }

    return true;
}

static bool test_t2012_copy(ggml_backend_dev_t devs[2]) {
    for (int rows : {3,1024}) for (int mode : {0,1,4}) {
        if(rows>3 && mode!=0) continue;
        const bool split=mode != 0;
        meta_test_ud ud{2048,3072,0,0};
        auto * dev = ggml_backend_meta_device(devs,2,test_split_state_cb,&ud);
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        ggml_backend_ptr consumer;
        ggml_backend_event_ptr ready,done;
        if (std::getenv("T2012_EVENTS")) {
            consumer.reset(ggml_backend_dev_init(dev,nullptr));
            ready.reset(ggml_backend_event_new(dev));done.reset(ggml_backend_event_new(dev));
            CHECK(ready && done,"meta events must be available");
        }
        ggml_init_params params={8*ggml_tensor_overhead(),nullptr,true};
        ggml_context_ptr ctx(ggml_init(params));
        auto * src=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,5120,rows);
        auto * dst=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,5120,rows);
        ggml_set_name(src,mode==4 ? "W_repeat" : (split ? "W_src" : "mirror_src"));
        ggml_set_name(dst,mode==4 ? "X_repeat" : (split ? "X_dst" : "mirror_dst"));
        auto * snapshot=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,5120,rows);
        ggml_set_name(snapshot,"snapshot_mirrored");
        ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
        CHECK(buf != nullptr,"copy test allocation");
        std::vector<float> data(5120*rows),got(data.size());
        for (int rep=0;rep<8;++rep) {
            for(size_t i=0;i<data.size();++i) data[i]=float(i)*0.125f+rep;
            ggml_backend_tensor_set(src,data.data(),0,ggml_nbytes(src));
            ggml_backend_tensor_copy_async(backend.get(),backend.get(),src,dst);
            ggml_backend_synchronize(backend.get());
            ggml_backend_tensor_get(dst,got.data(),0,ggml_nbytes(dst));
            CHECK(data==got,"meta D2D copy oracle split=%d rep=%d",int(split),rep);
            ggml_backend_tensor_copy_async(backend.get(),backend.get(),dst,src);
            ggml_backend_synchronize(backend.get());
            ggml_backend_tensor_get(src,got.data(),0,ggml_nbytes(src));
            CHECK(data==got,"meta D2D reverse oracle");
            ggml_backend_tensor_copy_async(backend.get(),backend.get(),src,snapshot);
            ggml_backend_synchronize(backend.get());
            std::vector<float> poison(data.size(),-777.0f);
            ggml_backend_tensor_set(dst,poison.data(),0,ggml_nbytes(dst));
            ggml_backend_tensor_copy_async(backend.get(),backend.get(),snapshot,dst);
            ggml_backend_synchronize(backend.get());
            ggml_backend_tensor_get(dst,got.data(),0,ggml_nbytes(dst));
            CHECK(data==got,"snapshot gather/scatter oracle");
            const ggml_tensor * tt[]={snapshot};size_t off[]={0},sz[]={ggml_nbytes(snapshot)},data_off[]={0};
            ggml_backend_tensor_get_n(tt,got.data(),off,sz,data_off,1);
            CHECK(data==got,"snapshot batched export oracle");
            off[0]=sizeof(float);sz[0]-=2*sizeof(float);data_off[0]=sizeof(float);
            std::fill(got.begin(),got.end(),-1234.0f);
            ggml_backend_tensor_get_n(tt,got.data(),off,sz,data_off,1);
            CHECK(got.front()==-1234.0f && got.back()==-1234.0f,"export canaries");
            CHECK(std::equal(got.begin()+1,got.end()-1,data.begin()+1),"export nonzero offsets");
            if (consumer) {
                ggml_backend_tensor_copy_async(backend.get(),backend.get(),src,snapshot);
                ggml_backend_event_record(ready.get(),backend.get());
                ggml_backend_event_wait(consumer.get(),ready.get());
                ggml_backend_tensor_copy_async(consumer.get(),consumer.get(),snapshot,dst);
                ggml_backend_event_record(done.get(),consumer.get());
                ggml_backend_event_wait(backend.get(),done.get());
                ggml_backend_tensor_copy_async(backend.get(),backend.get(),dst,src);
                ggml_backend_event_record(ready.get(),backend.get());
                ggml_backend_event_synchronize(ready.get());
                ggml_backend_tensor_get(src,got.data(),0,ggml_nbytes(src));
                CHECK(data==got,"composed event cross-context chain");
            }
        }
    }
    std::printf("t2012-copy PASS: 128 exact copies + 64 full/offset/canary exports (20 MiB chunk-boundary case)\n");
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    bool cpu_only    = false;
    bool require_gpu = false;
    bool do_bench    = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--cpu-only") == 0) {
            cpu_only = true;
        } else if (std::strcmp(argv[i], "--require-gpu") == 0) {
            require_gpu = true;
        } else if (std::strcmp(argv[i], "--bench") == 0) {
            do_bench = true;
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::printf("Usage: %s [--cpu-only] [--require-gpu] [--bench]\n", argv[0]);
            return 0;
        }
    }

    ggml_backend_load_all();
    if (std::getenv("T2012_GRAPH")) {
        auto gpus=find_gpu_devices();
        if (gpus.size()<2) return 1;
        ggml_backend_dev_t devs[2]={gpus[0],gpus[1]};
        // Do not inherit the failed transport matrix's U_host env (FUSED=0).
        set_env_val("GGML_HIP_AR_P2P","1");set_env_val("GGML_HIP_AR_FUSED","1");
        set_env_val("GGML_CUDA_AR_BF16_THRESHOLD","1");
        auto api=cuda_proc_api::resolve(ggml_backend_dev_backend_reg(gpus[0]));
        return test_meta_graphs_for_devs("gpu2-directed",devs,true,&api) && g_failures==0 ? 0 : 1;
    }
    if (std::getenv("T2012_COPY")) {
        auto gpus=find_gpu_devices();
        if (gpus.size()<2) return 1;
        ggml_backend_dev_t devs[2]={gpus[0],gpus[1]};
        return test_t2012_copy(devs) && g_failures==0 ? 0 : 1;
    }
    if (std::getenv("T1993_HANDSHAKE")) {
        std::printf("T1993_READY\n");
        std::fflush(stdout);
        if (std::getchar() == EOF) { return 1; }
    }

    // Always run CPU 2-rank meta graph test
    ggml_backend_dev_t cpu_dev = find_cpu_device();
    if (cpu_dev != nullptr && !(std::getenv("T2011_AR") || std::getenv("T2012_AR"))) {
        ggml_backend_dev_t cpu_pair[2] = {cpu_dev, cpu_dev};
        test_meta_graphs_for_devs("cpu2", cpu_pair, false, nullptr);
    } else {
        std::printf("SKIP cpu2 (no CPU device found)\n");
    }

    if (!cpu_only) {
        std::vector<ggml_backend_dev_t> gpus = find_gpu_devices();
        if (gpus.size() < 2) {
            if (require_gpu) {
                std::fprintf(stderr, "FAIL: --require-gpu set, found %zu GPUs (<2)\n", gpus.size());
                return 1;
            }
            std::printf("SKIP gpu2 tests (found %zu GPUs < 2)\n", gpus.size());
        } else {
            ggml_backend_dev_t gpu_pair[2] = {gpus[0], gpus[1]};
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(gpus[0]);
            cuda_proc_api api = cuda_proc_api::resolve(reg);
            test_comm_direct_gpu(gpu_pair, api, do_bench);
            if (!(std::getenv("T2011_AR") || std::getenv("T2012_AR"))) test_meta_graphs_for_devs("gpu2", gpu_pair, true, &api);
        }
    }

    if (g_failures > 0) {
        std::fprintf(stderr, "TOTAL FAILURES: %d\n", g_failures);
        return 1;
    }
    std::printf("ALL PASSED\n");
    return 0;
}
