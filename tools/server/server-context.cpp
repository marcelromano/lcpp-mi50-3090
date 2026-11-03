#include "server-context.h"
#ifdef LLAMA_KV_ADMISSION_CPU_TEST_DRIVER
#include "../../src/llama-memory-hybrid.h"
#endif
#include <cstdlib>

// CPU integration instrumentation is absent unless explicitly compiled in.
static const char * kv_admission_cpu_test_env(const char * name) {
#ifdef LLAMA_KV_ADMISSION_CPU_TEST_DRIVER
    return getenv(name);
#else
    (void) name;
    return nullptr;
#endif
}
#include "server-chat.h"
#include "server-common.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-slot-operation.h"
#include "kv_planner.hpp"
#include "kv_resource_view.hpp"
#include "kv_restore_map.hpp"
#include "server-schema.h"
#include "server-stream.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "../../src/llama-ext.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <optional>
#include <cstddef>
#include <cinttypes>
#include <exception>
#include <memory>
#include <new>
#include <filesystem>
#include <utility>
#include <fstream>
#include <cctype>
#include <mutex>
#include <random>
#include <unordered_map>

// side file <slot file>.ckpt: the slot's context checkpoints (hybrid/recurrent models lose them otherwise)
// layout: magic u32 | version u32 | n_tokens_total u64 | tokens_hash u64 | n u32 |
//         n x { n_tokens i64 | pos_min i32 | pos_max i32 | size_tgt u64 | size_dft u64 | size_spec u64 | tgt | dft | spec }
static constexpr uint32_t SLOT_CKPT_MAGIC   = 0x504b434c; // "LCKP"
static constexpr uint32_t SLOT_CKPT_VERSION = 1;

static uint64_t slot_ckpt_hash_tokens(const llama_tokens & tokens) {
    uint64_t h = 1469598103934665603ULL;
    for (const llama_token t : tokens) {
        h ^= (uint64_t) (uint32_t) t;
        h *= 1099511628211ULL;
    }
    return h;
}

template <typename T>
static void slot_ckpt_put(std::ofstream & f, const T & v) {
    f.write(reinterpret_cast<const char *>(&v), sizeof(T));
}

template <typename T>
static bool slot_ckpt_get(std::ifstream & f, T & v) {
    f.read(reinterpret_cast<char *>(&v), sizeof(T));
    return (bool) f;
}

// returns the number of checkpoints written (0 = side file removed/not written)
static size_t slot_ckpt_save(const std::string & path, const llama_tokens & tokens, const std::list<common_prompt_checkpoint> & cks) {
    std::error_code ec;
    if (cks.empty()) {
        std::filesystem::remove(path, ec); // do not leave a stale side file next to a fresh slot file
        return 0;
    }
    const std::string tmp = path + ".tmp";
    size_t expected_size = 0;
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            return 0;
        }
        slot_ckpt_put<uint32_t>(f, SLOT_CKPT_MAGIC);
        slot_ckpt_put<uint32_t>(f, SLOT_CKPT_VERSION);
        slot_ckpt_put<uint64_t>(f, tokens.size());
        slot_ckpt_put<uint64_t>(f, slot_ckpt_hash_tokens(tokens));
        slot_ckpt_put<uint32_t>(f, (uint32_t) cks.size());
        for (const auto & c : cks) {
            slot_ckpt_put<int64_t>(f, c.n_tokens);
            slot_ckpt_put<int32_t>(f, c.pos_min);
            slot_ckpt_put<int32_t>(f, c.pos_max);
            slot_ckpt_put<uint64_t>(f, c.data_tgt.size());
            slot_ckpt_put<uint64_t>(f, c.data_dft.size());
            slot_ckpt_put<uint64_t>(f, c.data_spec.size());
            f.write(reinterpret_cast<const char *>(c.data_tgt.data()),  c.data_tgt.size());
            f.write(reinterpret_cast<const char *>(c.data_dft.data()),  c.data_dft.size());
            f.write(reinterpret_cast<const char *>(c.data_spec.data()), c.data_spec.size());
        }
        const auto end = f.tellp();
        if (end < 0) {
            f.close();
            std::filesystem::remove(tmp, ec);
            return 0;
        }
        expected_size = (size_t) end;
        f.flush();
        if (!f) {
            f.close();
            std::filesystem::remove(tmp, ec);
            return 0;
        }
        f.close();
        if (f.fail()) {
            std::filesystem::remove(tmp, ec);
            return 0;
        }
    }
    const auto tmp_size = std::filesystem::file_size(tmp, ec);
    if (ec || tmp_size != expected_size) {
        std::filesystem::remove(tmp, ec);
        return 0;
    }
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return 0;
    }
    const auto final_size = std::filesystem::file_size(path, ec);
    if (ec || final_size != expected_size) {
        return 0;
    }
    return cks.size();
}

// loads into `out` only if the file is consistent with the restored tokens; returns the count loaded
static size_t slot_ckpt_load(const std::string & path, const llama_tokens & tokens, size_t max_keep, std::list<common_prompt_checkpoint> & out, bool bounded) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return 0;
    }
    std::ifstream f(path, std::ios::binary);
    uint32_t magic = 0, version = 0, n = 0;
    uint64_t n_total = 0, hash = 0;
    if (!f || !slot_ckpt_get(f, magic) || !slot_ckpt_get(f, version) || !slot_ckpt_get(f, n_total) ||
        !slot_ckpt_get(f, hash) || !slot_ckpt_get(f, n)) {
        SRV_WRN("slot checkpoint file '%s' unreadable, ignoring\n", path.c_str());
        return 0;
    }
    if (magic != SLOT_CKPT_MAGIC || version != SLOT_CKPT_VERSION) {
        SRV_WRN("slot checkpoint file '%s' has unknown magic/version (%u), ignoring\n", path.c_str(), version);
        return 0;
    }
    if (n_total != tokens.size() || hash != slot_ckpt_hash_tokens(tokens)) {
        SRV_WRN("slot checkpoint file '%s' does not match the restored tokens (stale?), ignoring\n", path.c_str());
        return 0;
    }
    std::vector<common_prompt_checkpoint> tmp;
    for (uint32_t i = 0; i < n; i++) {
        common_prompt_checkpoint c;
        int64_t  nt = 0;
        int32_t  p0 = 0, p1 = 0;
        uint64_t s_tgt = 0, s_dft = 0, s_spec = 0;
        if (!slot_ckpt_get(f, nt) || !slot_ckpt_get(f, p0) || !slot_ckpt_get(f, p1) ||
            !slot_ckpt_get(f, s_tgt) || !slot_ckpt_get(f, s_dft) || !slot_ckpt_get(f, s_spec) ||
            nt < 0 || (uint64_t) nt > n_total) {
            SRV_WRN("slot checkpoint file '%s' is corrupt, ignoring\n", path.c_str());
            return 0;
        }
        const uint64_t ckpt_cap = (uint64_t) nt * 1024ULL * 1024ULL + (1ULL << 20);
        if (bounded && (s_tgt > ckpt_cap || s_dft > ckpt_cap || s_spec > (1ULL << 30))) {
            SRV_WRN("slot checkpoint file '%s' is corrupt, ignoring\n", path.c_str());
            return 0;
        }
        c.n_tokens = nt;
        c.pos_min  = p0;
        c.pos_max  = p1;
        c.data_tgt.resize(s_tgt);
        c.data_dft.resize(s_dft);
        c.data_spec.resize(s_spec);
        f.read(reinterpret_cast<char *>(c.data_tgt.data()),  s_tgt);
        f.read(reinterpret_cast<char *>(c.data_dft.data()),  s_dft);
        f.read(reinterpret_cast<char *>(c.data_spec.data()), s_spec);
        if (!f) {
            SRV_WRN("slot checkpoint file '%s' is truncated, ignoring\n", path.c_str());
            return 0;
        }
        tmp.push_back(std::move(c));
    }
    while (tmp.size() > max_keep) {
        tmp.erase(tmp.begin());
    }
    const size_t n_loaded = tmp.size();
    for (auto & c : tmp) {
        out.push_back(std::move(c));
    }
    return n_loaded;
}

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

using json = nlohmann::ordered_json;


#ifdef LLAMA_EXP_SLOT_DFT
// experimental: draft (MTP) sidecar file `<slot file>.dft`
//   v1  : plain llama_state_seq_save_file() of ctx_dft (magic 'ggsq') - KV only
//   v2 : magic 'LDF2', u32 version = 2, u32 n_tokens, u64 hash(tokens), u64 n_spec, u64 n_kv,
//                 spec blob (MTP pending_h, may be empty), KV blob (llama_state_seq_get_data_ext, flags NONE)
// the loader accepts both; the saver writes v2.
static constexpr uint32_t SLOT_DFT_MAGIC_V2 = 0x3246444cu; // 'LDF2' (little-endian bytes)
static constexpr uint32_t SLOT_DFT_VERSION  = 2;

static uint64_t slot_dft_hash_tokens(const llama_token * tokens, size_t n) {
    uint64_t h = 1469598103934665603ull; // FNV-1a
    for (size_t i = 0; i < n; ++i) {
        const uint32_t t = (uint32_t) tokens[i];
        for (int b = 0; b < 4; ++b) {
            h ^= (uint8_t) (t >> (8 * b));
            h *= 1099511628211ull;
        }
    }
    return h;
}

// returns bytes written (0 on failure); n_spec_out = size of the speculative blob stored
static size_t slot_dft_save(llama_context * ctx_dft, common_speculative * spec, llama_seq_id seq_id,
        const std::string & filepath, const llama_token * tokens, size_t n_tokens, size_t & n_spec_out) {
    n_spec_out = 0;

    std::vector<uint8_t> blob_spec;
    if (spec && common_speculative_get_state(spec, seq_id, blob_spec)) {
        n_spec_out = blob_spec.size();
    } else {
        blob_spec.clear();
    }

    const size_t n_kv = llama_state_seq_get_size_ext(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (n_kv == 0) {
        return 0;
    }

    std::vector<uint8_t> blob_kv(n_kv);
    if (llama_state_seq_get_data_ext(ctx_dft, blob_kv.data(), blob_kv.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) != n_kv) {
        return 0;
    }

    const std::string tmp_path = filepath + ".tmp";
    std::ofstream f(tmp_path, std::ios::binary | std::ios::trunc);
    if (!f) {
        return 0;
    }

    const uint32_t magic   = SLOT_DFT_MAGIC_V2;
    const uint32_t version = SLOT_DFT_VERSION;
    const uint32_t n_tok   = (uint32_t) n_tokens;
    const uint64_t hash    = slot_dft_hash_tokens(tokens, n_tokens);
    const uint64_t n_sp    = blob_spec.size();
    const uint64_t n_k     = blob_kv.size();

    f.write((const char *) &magic,   sizeof(magic));
    f.write((const char *) &version, sizeof(version));
    f.write((const char *) &n_tok,   sizeof(n_tok));
    f.write((const char *) &hash,    sizeof(hash));
    f.write((const char *) &n_sp,    sizeof(n_sp));
    f.write((const char *) &n_k,     sizeof(n_k));
    f.write((const char *) blob_spec.data(), (std::streamsize) blob_spec.size());
    f.write((const char *) blob_kv.data(),   (std::streamsize) blob_kv.size());
    f.flush();

    if (!f) {
        f.close();
        std::error_code ec;
        std::filesystem::remove(tmp_path, ec);
        return 0;
    }

    const size_t expected_size = sizeof(magic) + sizeof(version) + sizeof(n_tok) + sizeof(hash) + sizeof(n_sp) + sizeof(n_k) + blob_spec.size() + blob_kv.size();
    f.close();
    if (f.fail()) {
        std::error_code ec;
        std::filesystem::remove(tmp_path, ec);
        return 0;
    }
    std::error_code ec;
    const auto tmp_size = std::filesystem::file_size(tmp_path, ec);
    if (ec || tmp_size != expected_size) {
        std::filesystem::remove(tmp_path, ec);
        return 0;
    }
    std::filesystem::rename(tmp_path, filepath, ec);
    if (ec) {
        std::filesystem::remove(tmp_path, ec);
        return 0;
    }
    const auto final_size = std::filesystem::file_size(filepath, ec);
    return !ec && final_size == expected_size ? expected_size : 0;
}

// Draft v1 is a native sequence blob and must share the restore token budget.
struct DraftRestoreBridge {
    kvplan::Planner * planner;
    kvplan::RestoreToken * token;
    kvplan::RestoreKeyMap map;
    std::atomic<uint64_t> * clears;
    uint64_t unmapped = 0;
    static bool consume(void * user, uint32_t kind, uint32_t stream, uint64_t count) {
        auto * bridge = static_cast<DraftRestoreBridge *>(user);
        return kvplan::restore_consume(*bridge->planner, *bridge->token, bridge->map, kind, stream, count, bridge->unmapped);
    }
    static void before_replace(void * user) {
        auto * bridge = static_cast<DraftRestoreBridge *>(user);
        if (bridge->clears) ++*bridge->clears;
    }
};

// returns bytes read (0 on any failure; the caller then clears the draft sequence).
// version_out: 1 or 2; spec_restored_out: true when the speculative blob was applied (v2 only)
static size_t slot_dft_load(llama_context * ctx_dft, common_speculative * spec, llama_seq_id seq_id,
        const std::string & filepath, const llama_tokens & tokens, int & version_out, bool & spec_restored_out, bool bounded,
        kvplan::Planner * planner, kvplan::RestoreToken * restore_token, const kvplan::RestoreKeyMap * resource_key,
        std::atomic<uint64_t> * clears) {
    version_out       = 0;
    spec_restored_out = false;

    uint32_t magic = 0;
    {
        std::ifstream f(filepath, std::ios::binary);
        if (!f || !f.read((char *) &magic, sizeof(magic))) {
            return 0;
        }
    }

    if (magic == LLAMA_FILE_MAGIC_GGSQ) {
        // v1: KV only
        version_out = 1;

        llama_tokens tokens_dft(tokens.size() + 1);
        size_t n_dft = 0;

        size_t n = 0;
        if (bounded && planner && restore_token && resource_key) {
            DraftRestoreBridge bridge{planner, restore_token, *resource_key, clears};
            llama_state_seq_limits limits{};
            limits.user = &bridge;
            limits.consume = DraftRestoreBridge::consume;
            limits.before_replace = DraftRestoreBridge::before_replace;
            limits.max_tensor_bytes = llama_state_seq_restore_tensor_capacity(ctx_dft);
            limits.max_snapshot_bytes = llama_state_seq_restore_file_capacity(ctx_dft, seq_id);
            n = llama_state_seq_load_file_bounded(ctx_dft, filepath.c_str(), seq_id,
                    tokens_dft.data(), tokens_dft.size(), &n_dft, &limits);
            SRV_INF("B3_DFT_V1 bytes=%zu limit_refused=%d cleared=%" PRIu64 "\\n",
                    n, limits.limit_refused, limits.clear_count);
        } else {
            n = llama_state_seq_load_file(ctx_dft, filepath.c_str(), seq_id,
                    tokens_dft.data(), tokens_dft.size(), &n_dft);
        }
        if (n == 0 || n_dft != tokens.size() || !std::equal(tokens.begin(), tokens.end(), tokens_dft.begin())) {
            SRV_WRN("B3_DFT_REJECT native_bytes=%zu token_mismatch=%d\n", n,
                    n > 0 && (n_dft != tokens.size() || !std::equal(tokens.begin(), tokens.end(), tokens_dft.begin())));
            return 0;
        }
        return n;
    }

    if (magic != SLOT_DFT_MAGIC_V2) {
        return 0;
    }

    std::ifstream f(filepath, std::ios::binary);

    uint32_t version = 0;
    uint32_t n_tok   = 0;
    uint64_t hash    = 0;
    uint64_t n_sp    = 0;
    uint64_t n_k     = 0;

    f.read((char *) &magic,   sizeof(magic));
    f.read((char *) &version, sizeof(version));
    f.read((char *) &n_tok,   sizeof(n_tok));
    f.read((char *) &hash,    sizeof(hash));
    f.read((char *) &n_sp,    sizeof(n_sp));
    f.read((char *) &n_k,     sizeof(n_k));

    const uint64_t dft_cap = llama_state_seq_restore_file_capacity(ctx_dft, seq_id);
    if (!f || version != SLOT_DFT_VERSION || n_tok != tokens.size() ||
            hash != slot_dft_hash_tokens(tokens.data(), tokens.size()) || n_sp > (1u << 20) || n_k == 0 ||
            (bounded && dft_cap > 0 && n_k > dft_cap)) {
        return 0;
    }

    version_out = 2;

    std::vector<uint8_t> blob_spec(n_sp);
    std::vector<uint8_t> blob_kv(n_k);

    f.read((char *) blob_spec.data(), (std::streamsize) n_sp);
    f.read((char *) blob_kv.data(),   (std::streamsize) n_k);
    if (!f) {
        return 0;
    }

    if (llama_state_seq_set_data_ext(ctx_dft, blob_kv.data(), blob_kv.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == 0) {
        return 0;
    }

    if (spec && !blob_spec.empty()) {
        common_speculative_set_state(spec, seq_id, blob_spec);
        spec_restored_out = true;
    }

    return sizeof(magic) + sizeof(version) + sizeof(n_tok) + sizeof(hash) + sizeof(n_sp) + sizeof(n_k) + n_sp + n_k;
}
#endif

#ifdef LLAMA_EXP_SLOT_DIAG
// experimental diagnostics: LLAMA_SLOT_DIAG=1 prints, to stderr, one parseable line per dump:
//   DIAG <tag> slot=<id> <what> pos=[min,max] bytes=<n> fnv=<hex64> norm=<l2> nonfinite=<n> chunks=<h32>:<norm>,...
// tags: save | restore | post-prefill ; plus per-round "DIAGR" lines (accepted/drafted, ids, running hash of emitted ids)
// compare the same <what> across runs (native prefill vs restore + tail): equal fnv => bit-identical state
#include <cmath>
#include <cstdio>
#include <cstdlib>

static bool slot_diag_on() {
    static const bool on = [] { const char * e = getenv("LLAMA_SLOT_DIAG"); return e && atoi(e) != 0; }();
    return on;
}

static int slot_diag_rounds() {
    static const int n = [] { const char * e = getenv("LLAMA_SLOT_DIAG_ROUNDS"); return e ? atoi(e) : 48; }();
    return n;
}

static uint64_t slot_diag_fnv(const uint8_t * p, size_t n, uint64_t h = 1469598103934665603ull) {
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

// elem: 0 = hash only, 2 = interpret as f16, 4 = interpret as f32 (norm over the chunk, header bytes included)
static void slot_diag_blob(const char * tag, int slot_id, const char * what, const std::vector<uint8_t> & b,
        int elem, llama_pos pmin, llama_pos pmax) {
    constexpr int N_CHUNKS = 32;

    double   norm_all = 0.0;
    uint64_t nonfin   = 0;

    std::string chunks;
    const size_t csz = std::max<size_t>(4, ((b.size() + N_CHUNKS - 1) / N_CHUNKS + 3) & ~(size_t) 3);

    for (size_t off = 0; off < b.size(); off += csz) {
        const size_t n = std::min(csz, b.size() - off);
        const uint64_t h = slot_diag_fnv(b.data() + off, n);

        double nrm = 0.0;
        if (elem == 4) {
            for (size_t i = 0; i + 4 <= n; i += 4) {
                float v; std::memcpy(&v, b.data() + off + i, 4);
                if (!std::isfinite(v)) { nonfin++; continue; }
                nrm += (double) v * v;
            }
        } else if (elem == 2) {
            for (size_t i = 0; i + 2 <= n; i += 2) {
                ggml_fp16_t hv; std::memcpy(&hv, b.data() + off + i, 2);
                const float v = ggml_fp16_to_fp32(hv);
                if (!std::isfinite(v)) { nonfin++; continue; }
                nrm += (double) v * v;
            }
        }
        norm_all += nrm;

        char buf[64];
        snprintf(buf, sizeof(buf), "%s%08x:%.5g", chunks.empty() ? "" : ",", (unsigned) (h & 0xffffffffu), std::sqrt(nrm));
        chunks += buf;
    }

    fprintf(stderr, "DIAG %s slot=%d %s pos=[%d,%d] bytes=%zu fnv=%016llx norm=%.6g nonfinite=%llu chunks=%s\n",
            tag, slot_id, what, (int) pmin, (int) pmax, b.size(), (unsigned long long) slot_diag_fnv(b.data(), b.size()),
            std::sqrt(norm_all), (unsigned long long) nonfin, chunks.c_str());
    fflush(stderr);
}

static void slot_diag_state(const char * tag, llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec,
        int slot_id, const llama_tokens & tokens) {
    if (!slot_diag_on()) {
        return;
    }

    fprintf(stderr, "DIAG %s slot=%d tokens n=%zu fnv=%016llx\n", tag, slot_id, tokens.size(),
            (unsigned long long) slot_diag_fnv((const uint8_t *) tokens.data(), tokens.size() * sizeof(llama_token)));

    {
        auto * mem = llama_get_memory(ctx_tgt);
        const llama_pos pmin = llama_memory_seq_pos_min(mem, slot_id);
        const llama_pos pmax = llama_memory_seq_pos_max(mem, slot_id);

        // PARTIAL_ONLY on a hybrid model = recurrent (GDN) state only, f32 by default
        std::vector<uint8_t> b(llama_state_seq_get_size_ext(ctx_tgt, slot_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
        if (!b.empty() && llama_state_seq_get_data_ext(ctx_tgt, b.data(), b.size(), slot_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == b.size()) {
            slot_diag_blob(tag, slot_id, "tgt-recurrent(f32)", b, 4, pmin, pmax);
        } else {
            fprintf(stderr, "DIAG %s slot=%d tgt-recurrent unavailable\n", tag, slot_id);
        }

        std::vector<uint8_t> a(llama_state_seq_get_size_ext(ctx_tgt, slot_id, LLAMA_STATE_SEQ_FLAGS_NONE));
        if (!a.empty() && llama_state_seq_get_data_ext(ctx_tgt, a.data(), a.size(), slot_id, LLAMA_STATE_SEQ_FLAGS_NONE) == a.size()) {
            slot_diag_blob(tag, slot_id, "tgt-full(hash)", a, 0, pmin, pmax);
        }
    }

    if (ctx_dft) {
        auto * mem = llama_get_memory(ctx_dft);
        const llama_pos pmin = llama_memory_seq_pos_min(mem, slot_id);
        const llama_pos pmax = llama_memory_seq_pos_max(mem, slot_id);

        // assumes the default draft KV type (F16, -ctkd/-ctvd not set); the norm is meaningless otherwise, the hash is not
        std::vector<uint8_t> d(llama_state_seq_get_size_ext(ctx_dft, slot_id, LLAMA_STATE_SEQ_FLAGS_NONE));
        if (!d.empty() && llama_state_seq_get_data_ext(ctx_dft, d.data(), d.size(), slot_id, LLAMA_STATE_SEQ_FLAGS_NONE) == d.size()) {
            slot_diag_blob(tag, slot_id, "dft-kv(f16)", d, 2, pmin, pmax);
        } else {
            fprintf(stderr, "DIAG %s slot=%d dft-kv empty/unavailable pos=[%d,%d]\n", tag, slot_id, (int) pmin, (int) pmax);
        }
    }

    {
        std::vector<uint8_t> h;
        if (spec && common_speculative_get_state(spec, slot_id, h) && h.size() > sizeof(int32_t)) {
            const size_t nf = (h.size() - sizeof(int32_t)) / sizeof(float);
            std::vector<float> f(nf);
            std::memcpy(f.data(), h.data() + sizeof(int32_t), nf * sizeof(float));
            double nrm = 0.0;
            for (float v : f) { nrm += (double) v * v; }
            fprintf(stderr, "DIAG %s slot=%d pending_h n=%zu fnv=%016llx norm=%.6g first=%.5g,%.5g,%.5g,%.5g\n", tag, slot_id, nf,
                    (unsigned long long) slot_diag_fnv(h.data(), h.size()), std::sqrt(nrm),
                    nf > 0 ? f[0] : 0.f, nf > 1 ? f[1] : 0.f, nf > 2 ? f[2] : 0.f, nf > 3 ? f[3] : 0.f);
        } else {
            fprintf(stderr, "DIAG %s slot=%d pending_h absent (not valid yet, or  not built)\n", tag, slot_id);
        }
    }
    fflush(stderr);
}

// per-slot round bookkeeping (slot ids are small)
static uint64_t g_diag_gen_hash[256];
static int      g_diag_round[256];

static void slot_diag_round(const char * kind, int slot_id, int n_past, size_t n_draft, size_t n_acc, bool replay,
        const llama_tokens & ids) {
    if (!slot_diag_on()) {
        return;
    }
    const int s = slot_id & 255;
    const int r = g_diag_round[s]++;
    for (llama_token t : ids) {
        g_diag_gen_hash[s] = slot_diag_fnv((const uint8_t *) &t, sizeof(t), g_diag_gen_hash[s]);
    }
    if (r < slot_diag_rounds() || (r % 16) == 0) {
        std::string l;
        for (size_t i = 0; i < ids.size() && i < 8; ++i) {
            l += (i ? "," : "") + std::to_string(ids[i]);
        }
        fprintf(stderr, "DIAGR %s slot=%d r=%d n_past=%d drafted=%zu accepted=%zu replay=%d ids=%s gen_fnv=%016llx\n",
                kind, slot_id, r, n_past, n_draft, n_acc, (int) replay, l.c_str(), (unsigned long long) g_diag_gen_hash[s]);
        fflush(stderr);
    }
}
#endif

constexpr int HTTP_POLLING_SECONDS = 1;

static uint32_t server_n_outputs_max(const common_params & params) {
    const uint32_t n_batch  = params.n_batch;

    if (params.embedding ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        return n_batch;
    }

    const uint32_t n_outputs_per_seq = 1 + common_speculative_n_max(&params.speculative);

    const uint64_t n_outputs = (uint64_t) params.n_parallel * n_outputs_per_seq;

    return std::max<uint32_t>(1, std::min<uint64_t>(n_batch, n_outputs));
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

struct server_slot; // forward declaration

struct server_batch {
    llama_batch batch;
    bool batch_rendered = false;

    struct token {
        int32_t id_slot;
        llama_token token;
        llama_pos pos;
        bool output;
    };
    std::vector<token> tokens;
    int32_t n_tokens_alloc = 0;
    int32_t n_embd = 0;

    // track if given slot can be batched with slots already in the batch
    server_slot * slot_batched = nullptr;

    // in embd mode, we temporarily swap out the tokens arr and restore it on clear()
    bool has_embd = false;
    llama_token * tokens_ptr = nullptr;
    std::vector<float> embd;

    float  alora_scale       = -1.0f;
    size_t alora_disabled_id = 0;

    server_batch() {
        batch.pos = nullptr; // sentinel: uninitialized batch
    }

    ~server_batch() {
        if (batch.pos != nullptr) {
            clear();
            llama_batch_free(batch);
        }
    }

    void init(int32_t n_tokens_alloc, int32_t n_embd) {
        this->n_tokens_alloc = n_tokens_alloc;
        this->n_embd = n_embd;
        batch = llama_batch_init(n_tokens_alloc, 0, 1);
        tokens_ptr = batch.token;
        tokens.reserve(n_tokens_alloc);
    }

    bool add(int32_t id_slot, llama_token token, llama_pos pos, bool output) {
        GGML_ASSERT(!has_embd); // cannot mix tokens + embd in same batch
        GGML_ASSERT(batch.pos != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, token, pos, output });
        return true;
    }

    bool add(int32_t id_slot, const std::vector<float> & embd_in, llama_pos pos, bool output) {
        GGML_ASSERT(batch.pos != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, LLAMA_TOKEN_NULL, pos, output });
        has_embd = true;
        embd.insert(embd.end(), embd_in.begin(), embd_in.end());
        return true;
    }

    void clear() {
        tokens.clear();
        embd.clear();
        common_batch_clear(batch);
        slot_batched      = nullptr;
        alora_scale       = -1.0f;
        alora_disabled_id = 0;
        batch_rendered    = false;
        has_embd          = false;
        if (batch.token == nullptr) {
            batch.token = tokens_ptr;
            batch.embd  = nullptr;
        }
    }

    int32_t size() const {
        return (int32_t)tokens.size();
    }

    void set_output(int32_t idx, bool output) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].output = output;
    }

    void render() {
        GGML_ASSERT(!batch_rendered);
        GGML_ASSERT(batch.pos != nullptr);
        common_batch_clear(batch);
        for (int32_t i = 0; i < size(); i++) {
            const auto & t = tokens[i];
            common_batch_add(batch, t.token, t.pos, { t.id_slot }, t.output);
        }
        if (has_embd) {
            batch.token = nullptr; // will be restored on clear()
            batch.embd  = embd.data();
        }
        batch_rendered = true;
    }

    llama_batch get_view(int32_t off, int32_t n_tokens) const {
        GGML_ASSERT(batch.pos != nullptr);
        GGML_ASSERT(batch_rendered);
        GGML_ASSERT(off >= 0 && off < size());
        GGML_ASSERT(n_tokens > 0 && off + n_tokens <= size());

        auto * token = batch.token ? batch.token + off          : nullptr;
        auto * embd  = batch.embd  ? batch.embd  + off * n_embd : nullptr;

        llama_batch view = {
            n_tokens,
            token,
            embd,
            batch.pos      + off,
            batch.n_seq_id + off,
            batch.seq_id   + off,
            batch.logits   + off,
        };

        return view;
    }
};

struct server_slot {
    int id;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    common_memory mem;

    // multimodal
    mtmd_context * mctx = nullptr;
    mtmd::batch_ptr mbatch = nullptr;

    // speculative decoding
    common_speculative * spec;

    llama_tokens spec_draft;
    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    common_prompt_checkpoint spec_ckpt;
    bool spec_is_replay = false;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx       = 0;  // context size per slot
    int32_t n_keep      = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;

    int32_t n_prompt_tokens_cache     = 0;
    int32_t n_prompt_tokens_processed = 0;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    server_prompt prompt;

    bool prompt_save(server_prompt_cache & prompt_cache) const {
        if (prompt.tokens.size() == 0) {
            return false;
        }

        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        const size_t cur_size = cur_size_tgt + cur_size_dft;

        SRV_TRC(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft);
        if (cur == nullptr) {
            return false;
        }

        llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (ctx_dft) {
            llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }

        return true;
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear() {
        SLT_TRC(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        mem.seq_rm(id, -1, -1);

        prompt.clear();
    }

    void clear_after_restore_failure() {
        prompt_clear(); // also drops every old target/draft checkpoint
        if (ctx_dft && !llama_memory_seq_rm(llama_get_memory(ctx_dft), id, -1, -1)) {
            SLT_WRN(*this, "failed to clear draft sequence %d after restore failure\n", id);
        }
        common_speculative_clear_state(spec, id);
        spec_draft.clear();
        spec_prompt.clear();
        spec_i_batch.clear();
        spec_ckpt.clear();
        spec_is_replay = false;
        json_schema = json::object();
        smpl.reset();
        n_keep = 0;
        n_prompt_tokens_cache = 0;
        n_prompt_tokens_processed = 0;
        n_decoded = 0;
        n_decoded_last = 0;
        n_remaining = -1;
        i_batch = -1;
        last_nl_pos = 0;
        generated_text.clear();
        debug_generated_text.clear();
        generated_tokens.clear();
        generated_token_probs.clear();
        inp_embd.clear();
        n_sent_text = 0;
        sampled = LLAMA_TOKEN_NULL;
        has_next_token = true;
        has_new_line = false;
        truncated = false;
        stop = STOP_TYPE_NONE;
        stopping_word.clear();
        n_draft_total = n_draft_accepted = n_draft_verif_steps = 0;
        n_accepted_per_pos.clear();
        alora_invocation_start = -1;
        mbatch.reset();
        if (ctx_tgt) llama_set_sampler(ctx_tgt, id, nullptr);
    }

    void clear_optional_draft_restore() {
        if (ctx_dft && !llama_memory_seq_rm(llama_get_memory(ctx_dft), id, -1, -1)) {
            SLT_WRN(*this, "failed to clear draft sequence %d after optional restore failure\n", id);
        }
        common_speculative_clear_state(spec, id);
        spec_draft.clear();
        spec_prompt.clear();
        spec_i_batch.clear();
        spec_ckpt.clear_dft();
        spec_is_replay = false;
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // for TTS models, this is the embd generated from prev step, decode this to generate next hidden state
    // corresponding to one token position (size = n_embd)
    std::vector<float> inp_embd;

    // stats
    size_t n_sent_text = 0; // number of sent text character

    // TODO @ngxson : move all metrics to a sub-struct for clarity
    int64_t t_start_process_prompt;
    int64_t t_start_generation;
    int64_t t_print_last = 0;
    int32_t n_decoded_last = 0;

    double t_prompt_processing = 0.0; // ms
    double t_token_generation = 0.0;  // ms

    std::function<void(int /* id_slot */)> callback_on_release;

    // Speculative decoding stats
    int32_t n_draft_total = 0;      // Total draft tokens generated
    int32_t n_draft_accepted = 0;   // Draft tokens actually accepted
    int32_t n_draft_verif_steps = 0; // Total draft token verification steps by the target model
    std::vector<int32_t> n_accepted_per_pos; // Accepted tokens per draft position

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        spec_is_replay = false;

        n_prompt_tokens_cache = 0;

        last_nl_pos    = 0;
        generated_text = "";
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        if (can_speculate()) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        // clear speculative decoding stats
        n_draft_total = 0;
        n_draft_accepted = 0;
        n_draft_verif_steps = 0;
        n_accepted_per_pos.clear();

        task_prev = std::move(task);
        task.reset();

        llama_set_sampler(ctx_tgt, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;

        // clear multimodal state
        mbatch.reset();
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd() || (spec && common_speculative_need_embd(spec));
    }

    bool need_embd_nextn() const {
        GGML_ASSERT(task);
        return spec && common_speculative_need_embd_nextn(spec);
    }

    // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
    // also we cannot split if the pooling would require any past tokens
    // (MTP supports splitting — uses task->need_embd() not need_embd())
    bool can_split() const {
        GGML_ASSERT(task);

        return
            !task->need_embd() ||
            (llama_get_memory(ctx_tgt) && llama_pooling_type(ctx_tgt) == LLAMA_POOLING_TYPE_LAST);
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        return task->type == other_slot.task->type
            && inp_embd.size() == other_slot.inp_embd.size()
            && are_lora_equal(lora, other_slot.lora);
    }

    bool has_budget(const common_params & global_params) {
        GGML_ASSERT(task);

        if (task->params.n_predict == -1 && global_params.n_predict == -1) {
            return true; // limitless
        }

        n_remaining = -1;

        if (task->params.n_predict != -1) {
            n_remaining = task->params.n_predict - n_decoded;
        } else if (global_params.n_predict != -1) {
            n_remaining = global_params.n_predict - n_decoded;
        }

        return n_remaining > 0; // no budget
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return !!spec;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    // add sampled token of this slot to the batch, optionally add the speculative draft tokens if any
    void handle_last_sampled_token(server_batch & batch) {
        bool add_ok = true;
        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.size();

            if (!inp_embd.empty()) {
                add_ok &= batch.add(id, inp_embd, prompt.tokens.pos_next(), true);
            } else {
                add_ok &= batch.add(id, sampled, prompt.tokens.pos_next(), true);
            }

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.size());
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.size() + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            add_ok &= batch.add(id, sampled, pos0++, true);
            for (auto token : spec_draft) {
                add_ok &= batch.add(this->id, token, pos0++, true);
            }
        }

        GGML_ASSERT(add_ok && "batch must be large enough to hold the sampled and draft tokens");

        prompt.tokens.push_back(sampled);
        prompt.tokens.insert(spec_draft);
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used        =  ggml_time_us();
            t_token_generation = (ggml_time_us() - t_start_generation) / 1e3;

            state = SLOT_STATE_IDLE;

            // do not keep context of the child slots - the parent's context is enough
            if (task->is_child()) {
                prompt_clear();
            }

            reset();

            callback_on_release(id);
        }
    }

    result_timings get_timings() const {
        result_timings timings;
        timings.cache_n = n_prompt_tokens_cache;

        timings.prompt_n            = n_prompt_tokens_processed;
        timings.prompt_ms           = t_prompt_processing;
        timings.prompt_per_token_ms = t_prompt_processing / n_prompt_tokens_processed;
        timings.prompt_per_second   = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        timings.predicted_n            = n_decoded;
        timings.predicted_ms           = t_token_generation;
        timings.predicted_per_token_ms = t_token_generation / n_decoded;
        timings.predicted_per_second   = 1e3 / t_token_generation * n_decoded;

        // Add speculative metrics
        if (n_draft_total > 0) {
            timings.draft_n          = n_draft_total;
            timings.draft_n_accepted = n_draft_accepted;
        }

        return timings;
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (n_decoded < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        const double n_gen_second     = 1e3 / (t_token_generation)   * (n_decoded);
        const double n_gen_second_win = 1e6 / (t_now - t_print_last) * (n_decoded - n_decoded_last);

        t_print_last = t_now;
        n_decoded_last = n_decoded;

        SLT_INF(*this, "n_decoded = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", n_decoded, n_gen_second, n_gen_second_win);
    }

    void print_timings_pp() const {
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;
        const double f_progress = (float) prompt.n_tokens() / task->n_tokens();

        if (t_prompt_processing < 3000.0) {
            return;
        }

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                n_prompt_tokens_processed, f_progress, t_prompt_processing / 1e3, n_prompt_second);
    }

    void print_timings() const {
        const double t_prompt        =       t_prompt_processing / n_prompt_tokens_processed;
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        const double t_gen        =       t_token_generation / n_decoded;
        const double n_gen_second = 1e3 / t_token_generation * n_decoded;

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_processing, n_prompt_tokens_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_token_generation, n_decoded, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_processing + t_token_generation, n_prompt_tokens_processed + n_decoded);

        SLT_INF(*this,
                "   graphs reused = %10d\n",
                llama_perf_context(ctx_tgt).n_reused);

        if (n_draft_total > 0) {
            const float  draft_ratio  = (float) n_draft_accepted / n_draft_total;
            const double mean_acc_len = n_draft_verif_steps > 0 ? 1.0 + (double) n_draft_accepted / (double) n_draft_verif_steps : 1.0;

            std::string acceptance_rates_per_pos;
            if (n_draft_verif_steps > 0) {
                for (size_t i = 0; i < n_accepted_per_pos.size(); ++i) {
                    if (i > 0) {
                        acceptance_rates_per_pos += ", ";
                    }
                    acceptance_rates_per_pos += string_format("%.3f", (double) n_accepted_per_pos[i] / (double) n_draft_verif_steps);
                }
            }

            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n",
                    draft_ratio, n_draft_accepted, n_draft_total, mean_acc_len);
            SLT_TRC(*this,
                    "     acc per pos = (%s)\n", acceptance_rates_per_pos.c_str());
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        // to_json runs in the serialized metrics task, never in the HTTP thread.
        llama_memory_seq_capacity capacity = {};
        const bool known = llama_memory_get_seq_capacity(llama_get_memory(ctx_tgt), id, &capacity);
        res["kv_capacity"] = {{"known", known}, {"scope", "attention_kv_stream"}};
        if (known) {
            res["kv_capacity"].update({
                {"unit", "cells"}, {"stream_id", capacity.stream_id},
                {"stream_count", capacity.stream_count}, {"capacity_cells", capacity.capacity_cells},
                {"used_cells", capacity.used_cells}, {"seq_cells", capacity.seq_cells},
                {"seq_exclusive_cells", capacity.seq_exclusive_cells},
            });
        } else {
            res["kv_capacity"]["reason"] = "unsupported_or_pending_memory_layout";
        }

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["n_prompt_tokens"]           = (int32_t) prompt.tokens.size();
            res["n_prompt_tokens_processed"] = n_prompt_tokens_processed;
            res["n_prompt_tokens_cache"]     = n_prompt_tokens_cache;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = {
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining},
                    {"n_decoded",      n_decoded},
                }
            };

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated_text.empty() ? debug_generated_text : generated_text;
            }
        }

        return res;
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        mem.seq_rm(other.id,     -1, -1);
        mem.seq_cp(id, other.id, -1, -1);

        other.n_decoded   = n_decoded;
        other.n_remaining = n_remaining;
        other.i_batch     = i_batch;

        other.t_start_process_prompt    = t_start_process_prompt;
        other.t_prompt_processing       = t_prompt_processing;
        other.n_prompt_tokens_cache     = n_prompt_tokens_cache;
        other.n_prompt_tokens_processed = n_prompt_tokens_processed;

        other.prompt = prompt.clone();
        other.init_sampler();
    }

    // returns 0 on success
    // caller need to update prompt.tokens after a successful call to keep track of the processing progress
    int process_mtmd_chunk(size_t idx, size_t & n_tokens_out) {
        GGML_ASSERT(mctx);
        const auto & input_tokens = task->tokens;
        const auto & chunk = input_tokens.find_chunk(idx);
        int32_t res = 0;

        auto try_decode = [&]() -> int32_t {
            if (mbatch) {
                float * embd = mtmd_batch_get_output_embd(mbatch.get(), chunk.get());
                if (embd) {
                    void * cb_data = spec;
                    static auto cb = [](llama_batch batch, void * user_data) {
                        common_speculative * spec = static_cast<common_speculative *>(user_data);
                        if (!common_speculative_process(spec, batch)) {
                            return 1;
                        }
                        return 0;
                    };

                    llama_pos new_n_past; // unused for now
                    res = mtmd_helper_decode_image_chunk(
                        mctx,
                        ctx_tgt,
                        chunk.get(),
                        embd,
                        prompt.tokens.pos_next(),
                        id,
                        llama_n_batch(ctx_tgt),
                        &new_n_past,
                        cb,
                        cb_data
                    );
                    if (res != 0) {
                        SLT_ERR(*this, "failed to decode mtmd chunk, idx = %zu, res = %d\n", idx, res);
                        return -1;
                    }
                    n_tokens_out = mtmd_input_chunk_get_n_tokens(chunk.get());
                    return 0; // success
                }
            }
            return 1; // (non-error) need to create & encode batch
        };

        // if the batch is already exist, try searching & encode
        res = try_decode();
        if (res == 0) {
            return 0;
        }
        if (res < 0) {
            // fatal error
            return res;
        }

        // otherwise, the batch is either uninitialized or is used up
        // we need to create & encode a new batch
        mbatch.reset(mtmd_batch_init(mctx));
        res = mtmd_batch_add_chunk(mbatch.get(), chunk.get());
        GGML_ASSERT(res == 0); // we should never have an empty batch

        // try batching as much as possible
        int n_added = 1;
        size_t idx_cur = idx;
        while (res == 0) {
            auto [next_chunk, next_idx] = input_tokens.find_next_media_chunk(idx_cur);
            if (next_chunk == nullptr) {
                break;
            }
            res = mtmd_batch_add_chunk(mbatch.get(), next_chunk->get());
            n_added += (res == 0 ? 1 : 0);
            idx_cur = next_idx;
            SLT_DBG(*this, "try adding media chunk idx = %zu to batch, res = %d\n", next_idx, res);
            // if res != 0, batch is full or chunk is not compatible -> this loop breaks
        }

        // TODO @ngxson : move this log line to debug when it become more stable
        SLT_TRC(*this, "encoding mtmd batch from idx = %zu, n_chunks = %d\n", idx, n_added);

        res = mtmd_batch_encode(mbatch.get());
        if (res != 0) {
            SLT_ERR(*this, "failed to encode mtmd batch for chunk idx = %zu, res = %d\n", idx, res);
            return -1;
        }

        return try_decode();
    }
};



//
// server_metrics
//

struct server_metrics {
    int64_t t_start = 0;

    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_tokens_max = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_slots_total = 0;

    void init() {
        t_start = ggml_time_us();
    }

    void on_prompt_eval(const server_slot & slot) {
        n_prompt_tokens_processed_total += slot.n_prompt_tokens_processed;
        n_prompt_tokens_processed       += slot.n_prompt_tokens_processed;
        t_prompt_processing             += slot.t_prompt_processing;
        t_prompt_processing_total       += slot.t_prompt_processing;

        n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
    }

    void on_prediction(const server_slot & slot) {
        n_tokens_predicted_total   += slot.n_decoded;
        n_tokens_predicted         += slot.n_decoded;
        t_tokens_generation        += slot.t_token_generation;
        t_tokens_generation_total  += slot.t_token_generation;
    }

    void on_decoded(const std::vector<server_slot> & slots) {
        n_decode_total++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                n_busy_slots_total++;
            }
            n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }
    }

    void reset_bucket() {
        n_prompt_tokens_processed = 0;
        t_prompt_processing       = 0;
        n_tokens_predicted        = 0;
        t_tokens_generation       = 0;
    }
};


//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    mtmd_context * mctx = nullptr;
    const llama_vocab * vocab = nullptr;

    server_queue    queue_tasks;
    server_response queue_results;
    mutable slot_operation_registry slot_operations;
    kvplan::Planner kv_admission_planner;
    // CPU observability: atomic reads are safe from the HTTP props thread.
    std::atomic<uint64_t> b3_limit{0}, b3_limit_rec{0}, b3_reserved{0}, b3_fenced{0}, b3_released{0}, b3_clears{0}, b3_refused{0};

    uint64_t kv_admission_epoch = 0;
    std::vector<std::optional<kvplan::Owner>> kv_admission_slot_owners;
    bool kv_admission_release_failed = false;
    bool kv_admission_layout_supported = false;
    std::string kv_admission_layout_reason = "layout_not_checked";
    uint64_t kv_admission_test_launch_count = 0;
    uint64_t kv_admission_test_commit_count = 0;

    bool kv_admission_simple_draft() const {
        if (!ctx_dft || !spec) return false;
        size_t active_types = 0;
        for (const auto type : params_base.speculative.types) {
            if (type == COMMON_SPECULATIVE_TYPE_NONE) continue;
            if (type != COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE) return false;
            ++active_types;
        }
        return active_types == 1;
    }

    // Bounded MTP support is based on the actual contexts. Qwen35 MTP reuses
    // the target model object, but owns an attention-only context/memory; the
    // target hybrid context owns both attention and recurrent state.
    bool kv_admission_mtp_hybrid() const {
        if (!ctx_tgt || !ctx_dft || !spec || !model_tgt || !llama_model_is_hybrid(model_tgt)) return false;
        size_t active_types = 0;
        for (const auto type : params_base.speculative.types) {
            if (type == COMMON_SPECULATIVE_TYPE_NONE) continue;
            if (type != COMMON_SPECULATIVE_TYPE_DRAFT_MTP) return false;
            ++active_types;
        }
        if (active_types != 1 || llama_get_model(ctx_dft) != model_tgt ||
                llama_get_ctx_other(ctx_dft) != nullptr ||
                llama_get_memory(ctx_dft) == llama_get_memory(ctx_tgt) ||
                llama_model_n_layer_nextn(model_tgt) <= 0 || llama_n_rs_seq(ctx_dft) != 0) return false;
        for (const auto & slot : slots) {
            llama_memory_resource_capacity target = {}, draft = {};
            if (!llama_memory_get_resource_capacity(llama_get_memory(ctx_tgt), slot.id, &target, sizeof(target)) ||
                    !llama_memory_get_resource_capacity(llama_get_memory(ctx_dft), slot.id, &draft, sizeof(draft)) ||
                    !target.attention_known || !target.recurrent_known ||
                    !draft.attention_known || draft.recurrent_known ||
                    target.recurrent_rollback_groups != llama_n_rs_seq(ctx_tgt) + 1 ||
                    !target.recurrent_rollback_groups) return false;
        }
        return true;
    }

    bool kv_admission_supported_draft() const {
        return kv_admission_simple_draft() || kv_admission_mtp_hybrid();
    }

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_state_callback_t callback_state = [](server_state, json) -> void {};

    server_context_impl() {
        static std::atomic<uint64_t> next_epoch{1};
        kv_admission_epoch = next_epoch.fetch_add(1, std::memory_order_relaxed);
        if (kv_admission_epoch == 0) {
            kv_admission_epoch = next_epoch.fetch_add(1, std::memory_order_relaxed);
        }
        mtmd_helper_log_set(common_log_default_callback, nullptr);
    }

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;

    server_batch batch;

    llama_model   * model_dft = nullptr;
    llama_context * ctx_dft   = nullptr;

    common_speculative_init_result_ptr spec_init;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    common_speculative_ptr spec;

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // slots / clients
    std::vector<server_slot> slots;

    int trace = 0;
    int slots_debug = 0;
    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    server_metrics metrics;

    json json_ui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    int64_t t_last_load_progress_ms = 0;

    void destroy() {
        spec.reset();
        spec_init.reset();

        ctx_dft   = nullptr;
        model_dft = nullptr;

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    struct load_progress_data {
        server_context_impl * ctx;
        std::string stage;
        std::vector<std::string> stages;
        int64_t t_last_load_progress_ms = 0;
        load_progress_data(server_context_impl * ctx, const std::string & stage) : ctx(ctx), stage(stage) {}
    };
    static bool load_progress_callback(float progress, void * user_data) {
        auto * d = static_cast<load_progress_data *>(user_data);
        GGML_ASSERT(d);
        // always emit the first and final sample; throttle the rest to one per 200ms
        {
            auto & t_last = d->t_last_load_progress_ms;
            const int64_t t_now = ggml_time_ms();
            const bool first = t_last == 0;
            const bool done  = progress >= 1.0f;
            const bool throttled = !first && !done && (t_now - t_last) < 200;
            if (throttled) {
                return true;
            }
            t_last = t_now;
        }
        if (d->ctx->callback_state) {
            d->ctx->callback_state(SERVER_STATE_LOADING, {
                {"stages", d->stages},
                {"current", d->stage},
                {"value", progress},
            });
        }
        return true;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        load_progress_data load_progress_text  (this, "text_model");
        load_progress_data load_progress_mmproj(this, "mmproj_model");
        load_progress_data load_progress_spec  (this, "spec_model");

        const bool is_resume = sleeping;

        params_base = params;
        SRV_INF("KV admission enabled=%d scope=%s\n", params_base.kv_admission,
                common_kv_admission_scope_name(params_base.kv_admission_scope));
        if (params_base.kv_admission) {
            params_base.cache_ram_mib = 0;
            params_base.cache_idle_slots = false;
            params_base.n_ctx_checkpoints = 0;
        }
        slot_operations.configure_owner(params_base.slot_operation_owner);
        slot_operations.configure_engine_id(params_base.slot_operation_engine_id);
        params_base.n_outputs_max = server_n_outputs_max(params_base);

        const bool has_mmproj = !params.mmproj.path.empty();
        const bool has_draft = params.speculative.has_dft();
        const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                        params_base.speculative.types.end(),
                                        COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();
        const bool has_spec = has_draft || spec_mtp;

        if (callback_state) {
            std::vector<std::string> stages = {"text_model"};
            if (has_spec) {
                stages.push_back("spec_model");
            }
            if (has_mmproj) {
                stages.push_back("mmproj_model");
            }
            load_progress_text.stages   = stages;
            load_progress_mmproj.stages = stages;
            load_progress_spec.stages   = stages;

            // trigger 0% progress
            load_progress_callback(0.0f, &load_progress_text);
        }


        SRV_INF("loading model '%s'\n", params.model.get_name().c_str());
        SRV_TRC("local path '%s'\n", params.model.path.c_str());

        std::string & mmproj_path = params_base.mmproj.path;
        mtmd_context_params mparams = mtmd_context_params_default();
        if (has_mmproj) {
            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.print_timings    = false;
            mparams.n_threads        = params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;
            mparams.batch_max_tokens = params_base.mtmd_batch_max_tokens;
            mparams.media_marker     = get_media_marker();
            // progress callback
            mparams.progress_callback           = load_progress_callback;
            mparams.progress_callback_user_data = &load_progress_mmproj;
        }

        // optionally get the memory usage of mmproj
        if (has_mmproj && params_base.fit_params) {
            int64_t t_start = ggml_time_us();
            auto mmproj_mem = mtmd_get_memory_usage(mmproj_path.c_str(), mparams);
            int64_t t_elapsed = ggml_time_us() - t_start;
            if (!mmproj_mem.empty()) {
                size_t total = 0;
                for (auto & [dev, size] : mmproj_mem) {
                    total += size;
                }
                SRV_TRC("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB (took %.2f ms)\n", total / (1024.0 * 1024.0), t_elapsed / 1000.0);
                GGML_ASSERT(!params_base.fit_params_target.empty());
                for (auto & [dev, size] : mmproj_mem) {
                    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                        if (ggml_backend_dev_get(i) == dev) {
                            if (i < params_base.fit_params_target.size()) {
                                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", size / (1024.0 * 1024.0), ggml_backend_dev_name(dev));
                                params_base.fit_params_target[i] += size;
                            }
                            break;
                        }
                    }
                }
            } else {
                SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
            }
        }

        // optionally reserve VRAM for the draft / MTP context before fitting the target model
        if (params_base.fit_params) {
            if (has_spec) {
                // MTP draft context lives on the target model, only context+compute are new
                bool measure_model_bytes = has_draft;

                common_params params_dft = common_base_params_to_speculative(params_base);

                auto mparams_dft = common_model_params_to_llama(params_dft);
                auto cparams_dft = common_context_params_to_llama(params_dft);
                if (spec_mtp) {
                    cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
                }
                cparams_dft.n_rs_seq = 0;

                std::vector<ggml_backend_dev_t> devs;
                uint32_t hp_ngl = 0;
                uint32_t hp_nct = 0;
                uint32_t hp_nex = 0;
                try {
                    auto dmd = common_get_device_memory_data(
                        params_dft.model.path.c_str(), &mparams_dft, &cparams_dft,
                        devs, hp_ngl, hp_nct, hp_nex, GGML_LOG_LEVEL_ERROR);

                    GGML_ASSERT(!params_base.fit_params_target.empty());
                    size_t total = 0;

                    std::vector<ggml_backend_dev_t> tgt_devices = params.devices;

                    if (tgt_devices.empty()) {
                        for(size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                           tgt_devices.push_back(ggml_backend_dev_get(i));
                        }
                    }

                    for (size_t j = 0; j < devs.size(); ++j) {
                        const size_t bytes = (measure_model_bytes ? dmd[j].model : 0) + dmd[j].context + dmd[j].compute;
                        total += bytes;
                        for (size_t i = 0; i < tgt_devices.size(); i++) {
                            if (tgt_devices[i] == devs[j]) {
                                SRV_DBG("[spec] adding %.2f MiB to fit_params_target for device %s\n",
                                        bytes / (1024.0 * 1024.0), ggml_backend_dev_name(devs[j]));
                                params_base.fit_params_target[i] += bytes;
                                break;
                            }
                        }
                    }
                    SRV_TRC("[spec] estimated memory usage of %s is %.2f MiB\n",
                            has_draft ? "draft model" : "MTP context",
                            total / (1024.0 * 1024.0));
                } catch (const std::exception & e) {
                    SRV_WRN("[spec] failed to measure %s memory: %s\n",
                            has_draft ? "draft model" : "MTP context", e.what());
                }
            }
        }

        // attach a progress callback
        {
            params_base.load_progress_callback = load_progress_callback;
            params_base.load_progress_callback_user_data = &load_progress_text;
        }

        llama_init = common_init_from_params(params_base);

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        if (ctx_tgt == nullptr) {
            SRV_ERR("failed to create_context with model '%s'\n", params_base.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model_tgt);

        n_ctx = llama_n_ctx(ctx_tgt);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (has_spec) {
            // spec_mtp doesn't use load a model internally, so we report 0.0 and 1.0 manually
            load_progress_callback(0.0f, &load_progress_spec);
            load_progress_spec.t_last_load_progress_ms = 0;  // reset so internal cbs aren't delayed

            {
                common_params params_dft = common_base_params_to_speculative(params_base);

                // progress callback
                params_dft.load_progress_callback           = load_progress_callback;
                params_dft.load_progress_callback_user_data = &load_progress_spec;

                spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
                model_dft = spec_init->model();
                ctx_dft   = spec_init->context();

                if (has_draft && model_dft == nullptr) {
                    SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                    return false;
                }

                if (ctx_dft == nullptr) {
                    SRV_ERR("%s", "failed to create MTP context\n");
                    return false;
                }

                params_base.speculative.draft.ctx_tgt = ctx_tgt;
                params_base.speculative.draft.ctx_dft = ctx_dft;
            }

            load_progress_callback(1.0f, &load_progress_spec);
        }

        if (has_mmproj) {
            if (callback_state) {
                callback_state(SERVER_STATE_LOADING, {{"stage", "mmproj_model"}});
            }

            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }
        }

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        const int n_ctx_train = llama_model_n_ctx_train(model_tgt);

        int n_ctx_slot = llama_n_ctx_seq(ctx_tgt);
        if (n_ctx_slot > n_ctx_train) {
            SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n", n_ctx_slot, n_ctx_train);
            n_ctx_slot = n_ctx_train;
        }

        slots.clear();

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_TRC("%s", "speculative decoding will use checkpoints\n");
        }

        // setup slots
        SRV_INF("initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'\n",
                params_base.n_parallel, n_ctx_slot, params_base.kv_unified ? "true" : "false");

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        kv_admission_slot_owners.clear();
        kv_admission_slot_owners.resize(slots.size());
        kv_admission_release_failed = false;

        // try speculative decoding
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
            }
        }

        if (ctx_dft) {
            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft);
        }

        if (spec) {
            SRV_TRC("%s", "speculative decoding context initialized\n");
        } else {
            spec_init.reset();
            ctx_dft   = nullptr;
            model_dft = nullptr;
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft;
            slot.mem.init(ctx_tgt, ctx_dft);
            slot.spec    = spec.get();
            slot.n_ctx   = n_ctx_slot;

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            SLT_TRC(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                release_kv_admission_slot(id_slot);
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.reset();
        }

        kv_admission_layout_supported = true;
        kv_admission_layout_reason.clear();
        const std::array<llama_context *, 2> admission_contexts = {ctx_tgt, ctx_dft};
        for (size_t ci = 0; ci < admission_contexts.size() && kv_admission_layout_supported; ++ci) {
            llama_context * admission_ctx = admission_contexts[ci];
            if (!admission_ctx) continue;
            const auto role = ci == 0 ? kvplan::Context::target : kvplan::Context::draft;
            for (const auto & slot : slots) {
                llama_memory_resource_capacity raw = {};
                std::vector<kvplan::PhysicalView> physical;
                const bool expect_recurrent = ci == 0 && (llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt));
                const bool expect_attention = !expect_recurrent || (ci == 0 && llama_model_is_hybrid(model_tgt));
                if (!llama_memory_get_resource_capacity(llama_get_memory(admission_ctx), slot.id, &raw, sizeof(raw)) ||
                        !kvplan::normalize_resources(role, kv_admission_epoch, raw, physical) ||
                        !kvplan::layout_matches(physical, expect_attention, expect_recurrent)) {
                    kv_admission_layout_supported = false;
                    kv_admission_layout_reason = ci == 0 ? "unknown_or_invalid_target_memory_layout" : "unknown_or_invalid_draft_memory_layout";
                    break;
                }
            }
        }

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx_tgt);
            const int32_t n_embd  = llama_model_n_embd_inp(model_tgt);
            batch.init(std::max(n_batch, params_base.n_parallel), n_embd);
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_TRC("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_TRC("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_TRC("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
        } else {
            SRV_TRC("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_TRC("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_TRC("context checkpoints enabled, max = %d, min spacing = %d\n",
                    params_base.n_ctx_checkpoints, params_base.checkpoint_min_step);
        } else {
            SRV_TRC("%s", "context checkpoints disabled\n");
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.get_name().empty()) {
            model_name = params_base.model.get_name();
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        if (callback_state) {
            callback_state(SERVER_STATE_READY, {});
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx_tgt   != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task) {
            process_single_task(std::move(task));
        });
        queue_tasks.on_task_cancelled([this](const server_task & task) {
            if (!task.slot_action.operation_id.empty()) {
                slot_operations.cancel_before_start(task.slot_action.operation_id);
            }
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        if (params_base.cache_idle_slots) {
            if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                if (params_base.kv_unified) {
                    SRV_TRC("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                } else {
                    // without a unified KV cache, clearing a slot frees no reusable room, so we only
                    // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
                    SRV_TRC("%s", "idle slots will be saved to prompt cache upon starting a new task\n");
                }
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        {
            const std::string & cfg = params_base.ui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings = json_settings;
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;
            bool enable_thinking = false;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                SRV_TRC("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

                // thinking is enabled if:
                // 1. It's not explicitly disabled via --reasoning off
                // 2. The chat template supports it
                const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
                enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
                SRV_TRC("%s: chat template, thinking = %d\n", __func__, enable_thinking);
            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // IMPORTANT: chat_params is reused across sleeping / resuming states,
            //            never store llama_context/llama_model pointers in chat_params,
            //            as they may be invalidated after sleeping
            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* allow_video           */ mctx ? mtmd_helper_support_video(mctx) : false,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };

            {
                auto caps = common_chat_templates_get_caps(chat_params.tmpls.get());
                auto it = params_base.default_template_kwargs.find("preserve_reasoning");
                bool supported = caps.at("supports_preserve_reasoning");
                bool enabled = it != params_base.default_template_kwargs.end();
                if (supported && !enabled) {
                    SRV_INF("%s", "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve\n");
                }
                if (!supported && enabled) {
                    SRV_WRN("%s", "chat template does NOT support preserving reasoning, --reasoning-preserve has no effect\n");
                }
            }
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_slot_by_cmpl_id(const std::string & cmpl_id) {
        if (cmpl_id.empty()) {
            return nullptr;
        }

        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.task && slot.task->params.oaicompat_cmpl_id == cmpl_id) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        // if a specific slot is requested, use it (still goes through cache update logic below)
        if (task.id_slot != -1) {
            ret = get_slot_by_id(task.id_slot);
            if (ret) {
                SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
            }
        }

        // find the slot that has at least n% prompt similarity
        if (slot_prompt_similarity != 0.0f) {
            float f_sim_best = 0;

            for (server_slot & slot : slots) {
                if (task.id_slot != -1 && slot.id != task.id_slot) {
                    continue;
                }

                // skip the slot if it is not available
                if (slot.is_processing()) {
                    SLT_TRC(slot, " - skipping, is_processing = %d\n", slot.is_processing());
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    SLT_TRC(slot, "%s", " - skipping, slot is empty\n");
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const size_t lcp_len = tokens.get_common_prefix(task.tokens);
                const float f_sim_cur = float(lcp_len) / task.tokens.size();

                SLT_TRC(slot, " - checking sim = %.3f (%zu/%zu) > %.3f\n", f_sim_cur, lcp_len, task.tokens.size(), slot_prompt_similarity);

                // select the current slot if the criteria match
                if (f_sim_cur > f_sim_best && f_sim_cur > slot_prompt_similarity) {
                    f_sim_best = f_sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (f_sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                if (task.id_slot == -1) {
                    SLT_INF(*ret, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                            f_sim_best, slot_prompt_similarity, f_keep);
                }

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = -1;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // select the current slot if the criteria match
                if (!ret || slot.t_last_used <= t_last) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_INF(*ret, "selected slot by LRU, t_last = %" PRId64 "\n", t_last);

                update_cache = true;
            }
        }

        if (ret) {
            update_cache = update_cache && prompt_cache && !params_base.kv_admission;

            // cache prompts only for completion tasks
            update_cache = update_cache && task.type == SERVER_TASK_TYPE_COMPLETION;

            if (update_cache) {
                SRV_TRC("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                ret->prompt_save(*prompt_cache);

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                    ret->prompt_clear();
                }

                prompt_cache->update();

                SRV_TRC("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots() {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        for (auto & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear();

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.clear();
                } else {
                    SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx_tgt)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_pre_sample_logits = task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs;

            bool backend_sampling = true;

            backend_sampling &= task.params.sampling.backend_sampling;

            // TODO: speculative decoding requires multiple samples per batch - not supported yet
            backend_sampling &= !(slot.can_speculate());

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            backend_sampling &= !need_pre_sample_logits;

            // TODO: tmp until backend sampling is fully implemented
            if (backend_sampling) {
                llama_set_sampler(ctx_tgt, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx_tgt, slot.id, nullptr);
            }

            SLT_TRC(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());
        } else {
            slot.smpl.reset();
        }

        slot.task = std::make_unique<const server_task>(std::move(task));

        if (kv_admission_cpu_test_env("LLAMA_KV_ADMISSION_TEST_TRACE")) {
            ++kv_admission_test_launch_count;
            SRV_INF("KV_ADMISSION_TEST launch count=%" PRIu64 " task=%d slot=%d role=%s\n",
                    kv_admission_test_launch_count, slot.task->id, slot.id,
                    slot.ctx_dft ? "target+draft" : "target");
        }

        slot.state = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated_text += token_str;
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_decoded = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), slot.n_decoded, slot.n_ctx);
        }

        // check the limits
        if (slot.n_decoded > 0 && slot.has_next_token && !slot.has_budget(params_base)) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_decoded = %d, n_predict = %d\n", slot.n_decoded, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_decoded = %d, n_indent = %d\n", slot.n_decoded, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && (ggml_time_us() - slot.t_start_generation > 1000.0f*slot.task->params.t_max_predict_ms)) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_decoded = %d, t_max_predict_ms = %d ms\n", slot.n_decoded, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_decoded = %d, n_remaining = %d, next token: %5d '%s'\n", slot.n_decoded, slot.n_remaining, result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p->data[i].p == 0.0) {
                    break;
                }

                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx_tgt, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            std::vector<llama_token_data> cur = get_token_probabilities(ctx_tgt, idx, n_probs_request);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx_tgt, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    // Gate slot save/restore/erase on slot content (does it hold media),
    // not model capability: a multimodal model may hold a pure-text slot.
    bool check_slot_no_media(const server_slot & slot, const int id_task) {
        if (slot.prompt.tokens.has_media()) {
            send_error(id_task,
                "This operation is not supported while the slot holds image/audio tokens (a pure-text prefix is supported)",
                ERROR_TYPE_NOT_SUPPORTED);
            return false;
        }
        return true;
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.n_prompt_tokens_cache;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = (ggml_time_us() - slot.t_start_process_prompt) / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->timings = slot.get_timings();
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated_text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = "";
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated_text);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->timings         = slot.get_timings();
        res->prompt          = slot.task->tokens.detokenize(ctx_tgt, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx_tgt, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.seq_id[i][0]);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx_tgt, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx_tgt, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
            if (free_slots.size() >= n_slots_needed) {
                break;
            }
        }
        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_TRC("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

#ifdef GGML_EXP_CKPT_FAST
    // Recycle the data buffers of erased checkpoints for the next one: a fresh allocation of
    // ~190 MiB pays first-touch page faults (+ munmap on free) on every checkpoint.
    // Assumes the single-threaded main loop; the pool is small and bounded.
    static void ckpt_recycle(common_prompt_checkpoint & ck) {
        static constexpr size_t POOL_MAX = 2;
        auto & pt = ckpt_pool_tgt();
        auto & pd = ckpt_pool_dft();
        if (ck.data_tgt.capacity() > 0 && pt.size() < POOL_MAX) {
            pt.push_back(std::move(ck.data_tgt));
        }
        if (ck.data_dft.capacity() > 0 && pd.size() < POOL_MAX) {
            pd.push_back(std::move(ck.data_dft));
        }
    }

    static void ckpt_take(common_prompt_checkpoint & ck) {
        auto & pt = ckpt_pool_tgt();
        auto & pd = ckpt_pool_dft();
        if (!pt.empty()) {
            ck.data_tgt = std::move(pt.back());
            pt.pop_back();
        }
        if (!pd.empty()) {
            ck.data_dft = std::move(pd.back());
            pd.pop_back();
        }
    }

    static std::vector<common_ckpt_buffer> & ckpt_pool_tgt() { static std::vector<common_ckpt_buffer> p; return p; }
    static std::vector<common_ckpt_buffer> & ckpt_pool_dft() { static std::vector<common_ckpt_buffer> p; return p; }
#endif

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max) {
        const int id_task = slot.task->id;

        // evict checkpoints within min-step of a previous checkpoint, unless they were
        // created by the current task
        int64_t last = -1;
        for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end(); ) {
            if (it->id_task != id_task && last >= 0 && it->n_tokens <= last + params_base.checkpoint_min_step) {
                SLT_TRC(slot, "erasing context checkpoint too close to an earlier one (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                        it->pos_min, it->pos_max, it->n_tokens, (float) it->size() / 1024 / 1024);

#ifdef GGML_EXP_CKPT_FAST
                ckpt_recycle(*it);
#endif
                it = slot.prompt.checkpoints.erase(it);
                continue;
            }

            last = it->n_tokens;
            ++it;
        }

        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed
            const auto & cur = slot.prompt.checkpoints.front();

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);

#ifdef GGML_EXP_CKPT_FAST
            ckpt_recycle(slot.prompt.checkpoints.front());
#endif
            slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
        }

        auto & cur = slot.prompt.checkpoints.emplace_back();
#ifdef GGML_EXP_CKPT_FAST
        ckpt_take(cur);
#endif

        cur.id_task = id_task;

        // [TAG_CHECKPOINTS_FIX_POS_MIN]
        // TODO: here we incorrectly deterimne that the saved checkpoint data covers the [pos_min, pos_max] range
        //       this is not true for SWA models: https://github.com/ggml-org/llama.cpp/pull/24411#issuecomment-4677983225
        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);

        cur.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        cur.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        // stash the draft's speculative state with the checkpoint
        common_speculative_get_state(spec.get(), slot.id, cur.data_spec);

        SLT_TRC(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);
    }

    std::string kv_admission_static_reason() const {
        if (!model_tgt || !ctx_tgt) return "model_context_unavailable";
        const bool any_draft_active = ctx_dft || spec || params_base.speculative.has_dft() ||
            std::any_of(params_base.speculative.types.begin(), params_base.speculative.types.end(),
                [](common_speculative_type type) { return type != COMMON_SPECULATIVE_TYPE_NONE; });
        if (llama_model_is_hybrid(model_tgt) && any_draft_active && !kv_admission_mtp_hybrid()) return "hybrid_target_with_draft_unsupported";
        if (llama_model_is_recurrent(model_tgt) && !llama_model_is_hybrid(model_tgt)) return "target_recurrent_memory_unsupported";
        if (llama_model_n_swa(model_tgt) != 0) return "target_swa_memory_unsupported";
        if (ctx_dft || spec || params_base.speculative.has_dft()) {
            if (kv_admission_mtp_hybrid()) {
                // The per-context raw layout above is authoritative; shared model identity is not.
            } else if (!kv_admission_simple_draft() || !model_dft) return "unsupported_speculative_context";
            if (!kv_admission_mtp_hybrid() && (llama_model_is_recurrent(model_dft) || llama_model_is_hybrid(model_dft))) return "draft_recurrent_or_hybrid_memory_unsupported";
            if (!kv_admission_mtp_hybrid() && llama_model_n_swa(model_dft) != 0) return "draft_swa_memory_unsupported";
            if (ctx_dft && llama_n_ctx_seq(ctx_dft) < slots.size()) return "draft_context_has_fewer_sequences_than_slots";
            if (!kv_admission_mtp_hybrid() && !model_dft) return "unsupported_speculative_context";
        } else if (std::any_of(params_base.speculative.types.begin(), params_base.speculative.types.end(),
                    [](common_speculative_type type) { return type != COMMON_SPECULATIVE_TYPE_NONE; })) {
            return "unsupported_speculative_context";
        }
        return {};
    }

    bool kv_admission_plan_restore_enabled() const {
        return params_base.kv_admission && kv_admission_layout_supported && params_base.kv_admission_restore;
    }

public:
    json b3_restore_metrics() const {
        return json {
                {"last_target_limit_I", b3_limit.load()},
                {"last_target_limit_by_key", json {{"attention", b3_limit.load()}, {"recurrent", b3_limit_rec.load()}}},
                {"reservations", b3_reserved.load()},
                {"fences", b3_fenced.load()},
                {"releases", b3_released.load()},
                {"active_restore_tokens", b3_reserved.load() - b3_released.load()},
                {"executor_clears", b3_clears.load()},
                {"plan_refusals", b3_refused.load()},
        };
    }

private:
    json kv_admission_descriptor() const {
        std::string reason = kv_admission_static_reason();
        if (reason.empty() && !kv_admission_layout_supported) reason = kv_admission_layout_reason;
        const bool restore_enabled = kv_admission_plan_restore_enabled();
        return json {
            {"version", 1},
            {"enabled", params_base.kv_admission},
            {"scope", common_kv_admission_scope_name(params_base.kv_admission_scope)},
            {"supported", reason.empty()},
            {"scope", kv_admission_mtp_hybrid() ? "mtp_hybrid_target_attention_recurrent_plus_mtp_attention" :
                (kv_admission_simple_draft() ? "simple_draft_two_attention_contexts" :
                (model_tgt && llama_model_is_hybrid(model_tgt) ? "hybrid_attention_recurrent_no_draft" :
                 (model_tgt && llama_model_is_recurrent(model_tgt) ? "recurrent_no_draft" : "transformer_attention_no_draft")))},
            {"effective_slot_context", slots.empty() ? 0 : slots.front().n_ctx},
            {"parallel_slots", params_base.n_parallel},
            {"kv_unified", params_base.kv_unified},
            {"engine_id", slot_operations.configured_engine_id},
            {"engine_epoch", slot_operations.engine_epoch},
            {"reason", reason},
            {"plan_restore", json {
                {"enabled", restore_enabled},
                {"cpu_fixture_only", false},
                {"declared_keys", json {
                    {"attention", model_tgt && (!(llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt)) || llama_model_is_hybrid(model_tgt))},
                    {"recurrent", model_tgt && (llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt))},
                }},
                {"limits_by_key", json {{"attention", b3_limit.load()}, {"recurrent", b3_limit_rec.load()}}},
                {"reader_fences", true},
                {"output_fences", true},

            }},
        };
    }

    void release_kv_admission_slot(int id_slot) noexcept {
        if (id_slot < 0 || (size_t) id_slot >= kv_admission_slot_owners.size()) return;
        auto & owner = kv_admission_slot_owners[(size_t) id_slot];
        if (!owner) return;
        try {
            const auto result = kv_admission_planner.native_release(*owner);
            if (result.status == kvplan::Status::accepted || result.status == kvplan::Status::already_released) {
                if (kv_admission_cpu_test_env("LLAMA_KV_ADMISSION_TEST_TRACE")) {
                    SRV_INF("KV_ADMISSION_TEST release task=%" PRIu64 " slot=%d status=%s ledger_before=%zu\n",
                            owner->task, id_slot,
                            result.status == kvplan::Status::accepted ? "released" : "already_released",
                            kv_admission_planner.size());
                }
                owner.reset();
                return;
            }
            SRV_ERR("KV admission native release rejected for task=%" PRIu64 " slot=%d: %s\n",
                    owner->task, id_slot, result.reason.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("KV admission native release failed for slot=%d: %s\n", id_slot, e.what());
        } catch (...) {
            SRV_ERR("KV admission native release failed for slot=%d\n", id_slot);
        }
        kv_admission_release_failed = true;
    }

    using kv_admission_slot_resources = std::map<std::pair<kvplan::Context, int>, std::vector<kvplan::ResourceKey>>;

    bool kv_admission_snapshots(std::vector<kvplan::Snapshot> & snapshots,
                                kv_admission_slot_resources & resource_by_slot,
                                std::string & reason) const {
        std::map<kvplan::ResourceKey, kvplan::Snapshot> by_resource;
        std::map<kvplan::ResourceKey, uint32_t> stream_counts;
        const std::array<std::pair<kvplan::Context, llama_context *>, 2> contexts = {{
            {kvplan::Context::target, ctx_tgt}, {kvplan::Context::draft, ctx_dft}
        }};
        for (const auto & context : contexts) {
            if (!context.second) continue;
            for (const auto & slot : slots) {
                if (slot.id < 0 || (size_t) slot.id >= kv_admission_slot_owners.size()) {
                    reason = "invalid_native_slot_id";
                    return false;
                }
                llama_memory_resource_capacity raw = {};
                const char * fault = kv_admission_cpu_test_env("LLAMA_KV_ADMISSION_TEST_FAULT");
                const llama_seq_id query_seq = fault && std::strcmp(fault, "unknown") == 0 &&
                    context.first == kvplan::Context::target && slot.id == 0 ? INT32_MAX : slot.id;
                if (!llama_memory_get_resource_capacity(llama_get_memory(context.second), query_seq, &raw, sizeof(raw))) {
                    reason = "unknown_memory_resource_snapshot";
                    return false;
                }
                std::vector<kvplan::PhysicalView> physical;
                const bool expect_recurrent = context.first == kvplan::Context::target &&
                    (llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt));
                const bool expect_hybrid = context.first == kvplan::Context::target && llama_model_is_hybrid(model_tgt);
                const bool expect_attention = !expect_recurrent || expect_hybrid;
                const bool inject_epoch = fault && std::strcmp(fault, "epoch") == 0 && context.first == kvplan::Context::target;
                const uint64_t snapshot_epoch = inject_epoch && kv_admission_epoch != UINT64_MAX ? kv_admission_epoch + 1 : kv_admission_epoch;
                if (!kvplan::normalize_resources(context.first, snapshot_epoch, raw, physical) ||
                        !kvplan::layout_matches(physical, expect_attention, expect_recurrent)) {
                    reason = "unknown_or_invalid_memory_resource_layout";
                    return false;
                }
                auto & slot_resources = resource_by_slot[std::make_pair(context.first, slot.id)];
                const bool active = slot.is_processing();
                const auto & owner = kv_admission_slot_owners[(size_t) slot.id];
                if (!common_kv_admission_snapshot_owner_valid(params_base.kv_admission_scope, active, owner.has_value())) {
                    reason = "native_owner_snapshot_mismatch";
                    return false;
                }
                for (const auto & capacity : physical) {
                    const auto key = capacity.key;
                    slot_resources.push_back(key);
                    if (capacity.stream_count) {
                        const auto count = stream_counts.emplace(key, capacity.stream_count);
                        if (!count.second && count.first->second != capacity.stream_count) {
                            reason = "inconsistent_stream_count";
                            return false;
                        }
                    }
                    auto it = by_resource.find(key);
                    if (it == by_resource.end()) {
                        kvplan::Snapshot snapshot;
                        snapshot.key = key;
                        snapshot.known = true;
                        snapshot.capacity = capacity.capacity;
                        snapshot.used_real = capacity.used;
                        it = by_resource.emplace(key, std::move(snapshot)).first;
                    } else if (it->second.capacity != capacity.capacity || it->second.used_real != capacity.used) {
                        reason = "inconsistent_stream_snapshot";
                        return false;
                    }
                    if (active && owner.has_value()) {
                        uint64_t total = it->second.seq_exclusive_real;
                        const uint64_t exclusive = capacity.exclusive;
                        if (exclusive > std::numeric_limits<uint64_t>::max() - total) {
                            reason = "exclusive_snapshot_overflow";
                            return false;
                        }
                        it->second.seq_exclusive_real = total + exclusive;
                        kvplan::Owner snapshot_owner = *owner;
                        const char * fault = kv_admission_cpu_test_env("LLAMA_KV_ADMISSION_TEST_FAULT");
                        if (fault && std::strcmp(fault, "binding") == 0 && context.first == kvplan::Context::target && slot.id == 0) {
                            ++snapshot_owner.task;
                        }
                        if (!it->second.exclusive_by_owner.emplace(snapshot_owner, exclusive).second) {
                            reason = "duplicate_native_owner";
                            return false;
                        }
                    }
                }
            }
        }
        snapshots.reserve(by_resource.size());
        for (auto & item : by_resource) snapshots.push_back(std::move(item.second));
        return true;
    }

    // Credit for the resident of the already selected, idle target slot. Proof (proposal.md): the peak counts the
    // whole sequence (prompt+output) and update_slots runs slot.mem.seq_rm(slot, p0 <= n_past, end) on target and draft
    // before any decode that grows the sequence, so the resident exclusive cells/rows are retired or reused first.
    // AND semantics: every resource of the slot must be known and bound, otherwise no credit at all.
    // Everything is staged before the planner call; nothing here mutates KV or the ledger.
    void kv_admission_resident_credit(const server_task & task, const server_slot & slot, const kvplan::Owner & owner,
                                      const kv_admission_slot_resources & resource_by_slot,
                                      const kvplan::Reservation & reservation,
                                      std::vector<kvplan::Snapshot> & snapshots) const {
        if (task.type != SERVER_TASK_TYPE_COMPLETION && task.type != SERVER_TASK_TYPE_INFILL) return;
        if (slot.is_processing() || slot.id < 0 || (size_t) slot.id >= kv_admission_slot_owners.size() ||
                kv_admission_slot_owners[(size_t) slot.id]) return;
        if (task.tokens.has_media() || slot.prompt.tokens.has_media()) return;
        if (owner.slot != (uint64_t) slot.id || owner.task != (uint64_t) task.id || owner.epoch != kv_admission_epoch) return;
        std::vector<std::pair<size_t, uint64_t>> staged; // snapshot index, credit
        const std::array<std::pair<kvplan::Context, bool>, 2> roles = {{{kvplan::Context::target, true}, {kvplan::Context::draft, ctx_dft != nullptr}}};
        for (const auto & role : roles) {
            if (!role.second) continue;
            const auto resources = resource_by_slot.find({role.first, slot.id});
            if (resources == resource_by_slot.end() || resources->second.empty()) return;
            // Re-read the slot's own physical view at the same task boundary (both contexts already synchronized by
            // the preflight; nothing mutates KV in between) and normalize it exactly like the snapshots.
            llama_memory_resource_capacity raw = {};
            std::vector<kvplan::PhysicalView> physical;
            if (!llama_memory_get_resource_capacity(llama_get_memory(role.first == kvplan::Context::target ? ctx_tgt : ctx_dft),
                        slot.id, &raw, sizeof(raw)) || !kvplan::normalize_resources(role.first, owner.epoch, raw, physical) ||
                    physical.size() != resources->second.size()) return;
            for (const auto & key : resources->second) {
                const auto peak = reservation.peak_by_resource.find(key);
                const auto exclusive = std::find_if(physical.begin(), physical.end(), [&key](const kvplan::PhysicalView & v) { return v.key == key; });
                const auto snapshot = std::find_if(snapshots.begin(), snapshots.end(), [&key](const kvplan::Snapshot & v) { return v.key == key; });
                if (peak == reservation.peak_by_resource.end() || exclusive == physical.end() ||
                        snapshot == snapshots.end() || key.epoch != owner.epoch || exclusive->used != snapshot->used_real ||
                        exclusive->exclusive > snapshot->used_real) return;
                staged.emplace_back((size_t) (snapshot - snapshots.begin()), std::min<uint64_t>(exclusive->exclusive, peak->second));
            }
        }
        for (const auto & item : staged) {
            if (!item.second) continue;
            kvplan::Owner credit_owner = owner;
#ifdef LLAMA_KV_ADMISSION_CPU_TEST_DRIVER
            const char * fault = kv_admission_cpu_test_env("LLAMA_KV_ADMISSION_TEST_FAULT");
            if (fault && std::strcmp(fault, "credit_binding") == 0) ++credit_owner.task;
            if (kv_admission_cpu_test_env("LLAMA_KV_ADMISSION_TEST_TRACE")) {
                SRV_INF("KV_ADMISSION_TEST resident_credit task=%d slot=%d role=%s stream=%u kind=%u credit=%" PRIu64 "\n",
                        task.id, slot.id, snapshots[item.first].key.context == kvplan::Context::target ? "target" : "draft",
                        snapshots[item.first].key.stream, (unsigned) snapshots[item.first].key.kind, item.second);
            }
#endif
            snapshots[item.first].resident_credit_by_incoming_owner.emplace(credit_owner, item.second);
        }
    }

    bool kv_admission_peak(const server_task & task, const server_tokens & input, const server_slot & slot,
                           const kvplan::Owner & owner, const std::optional<kvplan::Owner> & parent,
                           llama_context * context, const std::vector<kvplan::ResourceKey> & resources, kvplan::Reservation & out,
                           std::string & reason, error_type & error) const {
        if (task.id < 0 || slot.id < 0 || owner.epoch == 0 || owner.task != (uint64_t) task.id || owner.slot != (uint64_t) slot.id) {
            reason = "invalid_native_task_or_slot_id";
            error = ERROR_TYPE_KV_UNSUPPORTED;
            return false;
        }
        if (input.has_media()) {
            reason = "media_cell_layout_unsupported";
            error = ERROR_TYPE_KV_UNSUPPORTED;
            return false;
        }
        if (slot.n_ctx <= 0 || !context) {
            reason = "invalid_effective_context";
            error = ERROR_TYPE_KV_UNSUPPORTED;
            return false;
        }
        const size_t input_size = input.size();
        const uint64_t context_capacity = llama_n_ctx_seq(context);
        if (input_size > (size_t) std::min<uint64_t>(slot.n_ctx, context_capacity)) {
            reason = "prompt_exceeds_effective_context";
            error = ERROR_TYPE_KV_CAPACITY;
            return false;
        }
        uint64_t output = 0;
        if (task.type == SERVER_TASK_TYPE_COMPLETION || task.type == SERVER_TASK_TYPE_INFILL) {
            int32_t n_predict = task.params.n_predict;
            if (n_predict == -1) n_predict = params_base.n_predict;
            if (n_predict < -1) {
                reason = "invalid_output_limit";
                error = ERROR_TYPE_KV_UNSUPPORTED;
                return false;
            }
            const uint64_t remaining = std::min<uint64_t>(slot.n_ctx, context_capacity) - input_size;
            output = n_predict == -1 ? remaining : (uint64_t) n_predict;
            if (output > remaining) {
                reason = "output_limit_exceeds_effective_context";
                error = ERROR_TYPE_KV_CAPACITY;
                return false;
            }
        } else if (task.type != SERVER_TASK_TYPE_EMBEDDING && task.type != SERVER_TASK_TYPE_RERANK) {
            reason = "task_type_unsupported";
            error = ERROR_TYPE_KV_UNSUPPORTED;
            return false;
        }
        const uint64_t prompt = (uint64_t) input_size;
        if (output > std::numeric_limits<uint64_t>::max() - prompt) {
            reason = "peak_arithmetic_overflow";
            error = ERROR_TYPE_KV_CAPACITY;
            return false;
        }
        out.owner = owner;
        out.parent = parent;
        llama_memory_resource_capacity raw = {};
        if (!llama_memory_get_resource_capacity(llama_get_memory(context), slot.id, &raw, sizeof(raw))) {
            reason = "unknown_memory_resource_snapshot";
            error = ERROR_TYPE_KV_UNSUPPORTED;
            return false;
        }
        uint64_t lookahead = 0;
        if (kv_admission_mtp_hybrid() && task.type == SERVER_TASK_TYPE_COMPLETION && output > 1) {
            // MTP draft and target verification consume tokens already bounded by output.
            // Clamp to both configured n_max and remaining output; verifier adds the sampled token.
            const uint64_t configured = (uint64_t) std::max<int32_t>(0, params_base.speculative.draft.n_max);
            lookahead = std::min<uint64_t>(configured, output - 1);
            lookahead = std::min<uint64_t>(lookahead, std::min<uint64_t>(slot.n_ctx, context_capacity) - prompt - 1);
        }
        const uint64_t verifier = lookahead ? lookahead + 1 : 0;
        const uint64_t attention_peak = std::max<uint64_t>(prompt + output, prompt + verifier);
        if (attention_peak > std::min<uint64_t>(slot.n_ctx, context_capacity)) {
            reason = "mtp_verifier_peak_exceeds_effective_context";
            error = ERROR_TYPE_KV_CAPACITY;
            return false;
        }
        for (const auto & resource : resources) {
            uint64_t peak = attention_peak;
            if (resource.kind == kvplan::Kind::recurrent_rows) {
                if (!raw.recurrent_known || !raw.recurrent_rollback_groups ||
                        raw.recurrent_rollback_groups != llama_n_rs_seq(context) + 1 ||
                        raw.recurrent_capacity_rows % raw.recurrent_rollback_groups ||
                        raw.recurrent_used_rows % raw.recurrent_rollback_groups) {
                    reason = "recurrent_rollback_group_derivation_missing";
                    error = ERROR_TYPE_KV_UNSUPPORTED;
                    return false;
                }
                // Preserve the root conservative transient bound for every
                // recurrent reservation, including no-draft, parent, and MTP.
                peak = 2;
            }
            if (!out.peak_by_resource.emplace(resource, peak).second) {
                reason = "duplicate_resource_binding";
                error = ERROR_TYPE_KV_UNSUPPORTED;
                return false;
            }
        }
        error = ERROR_TYPE_KV_CAPACITY;
        return true;
    }

#ifdef LLAMA_KV_ADMISSION_CPU_TEST_DRIVER
    void kv_admission_test_observe(const char * point, int slot_id, size_t n_draft, size_t n_accepted) const {
        if (!kv_admission_cpu_test_env("LLAMA_KV_ADMISSION_TEST_TRACE") || !kv_admission_mtp_hybrid()) return;
        // Test-only event boundaries are synchronous; the production dispatcher does not
        // gain extra fences or observers from this function.
        llama_synchronize(ctx_tgt);
        llama_synchronize(ctx_dft);
        llama_memory_resource_capacity target = {}, draft = {};
        const bool target_ok = llama_memory_get_resource_capacity(llama_get_memory(ctx_tgt), slot_id, &target, sizeof(target));
        const bool draft_ok = llama_memory_get_resource_capacity(llama_get_memory(ctx_dft), slot_id, &draft, sizeof(draft));
        SRV_INF("KV_ADMISSION_TEST mtp_event point=%s slot=%d draft=%zu accepted=%zu target_ctx=%p target_mem=%p target_ok=%d attn=%u/%u/%u rec=%u/%u/%u groups=%u draft_ctx=%p draft_mem=%p draft_ok=%d attn=%u/%u/%u rec_known=%d shared_ctx_other=%p\n",
                point, slot_id, n_draft, n_accepted, (void *)ctx_tgt, (void *)llama_get_memory(ctx_tgt), (int)target_ok,
                target.attention_used_cells, target.attention_capacity_cells, target.attention_seq_exclusive_cells,
                target.recurrent_used_rows, target.recurrent_capacity_rows, target.recurrent_seq_exclusive_rows,
                target.recurrent_rollback_groups, (void *)ctx_dft, (void *)llama_get_memory(ctx_dft), (int)draft_ok,
                draft.attention_used_cells, draft.attention_capacity_cells, draft.attention_seq_exclusive_cells,
                (int)draft.recurrent_known, (void *)llama_get_ctx_other(ctx_dft));
    }
#endif

    bool kv_admission_preflight(server_task & task, server_slot & parent_slot,
                                const std::vector<server_slot *> & child_slots,
                                std::string & reason, error_type & error) {
        if (!params_base.kv_admission) return true;
        if (common_kv_admission_plain_bypass(params_base.kv_admission_scope)) {
            // Keep L3 fail-closed: parent/child groups must not reach unreserved pools.
            if (!child_slots.empty()) {
                reason = "capacity exceeded";
                error = ERROR_TYPE_KV_CAPACITY;
                return false;
            }
            if (kv_admission_release_failed) {
                reason = "native_release_ledger_unavailable";
                error = ERROR_TYPE_KV_UNSUPPORTED;
                return false;
            }
            return true;
        }
        // A memory getter is metadata only. Fence both real engines at the task boundary
        // before observing either context, and keep dispatch serialized through commit.
        llama_synchronize(ctx_tgt);
        if (ctx_dft) llama_synchronize(ctx_dft);
        if (kv_admission_release_failed) {
            reason = "native_release_ledger_unavailable";
            error = ERROR_TYPE_KV_UNSUPPORTED;
            return false;
        }
        reason = kv_admission_static_reason();
        if (reason.empty() && !kv_admission_layout_supported) reason = kv_admission_layout_reason;
        if (!reason.empty()) {
            error = ERROR_TYPE_KV_UNSUPPORTED;
            return false;
        }
        if (task.type == SERVER_TASK_TYPE_COMPLETION && !child_slots.empty() && task.tokens.has_media()) {
            reason = "media_parent_group_unsupported";
            error = ERROR_TYPE_KV_UNSUPPORTED;
            return false;
        }

        std::vector<kvplan::Snapshot> snapshots;
        kv_admission_slot_resources resource_by_slot;
        if (!kv_admission_snapshots(snapshots, resource_by_slot, reason)) {
            error = ERROR_TYPE_KV_UNSUPPORTED;
            return false;
        }

        if (kv_admission_cpu_test_env("LLAMA_KV_ADMISSION_TEST_TRACE")) {
#ifdef LLAMA_KV_ADMISSION_CPU_TEST_DRIVER
            SRV_INF("KV_ADMISSION_TEST object_map task=%d target_ctx=%p target_mem=%p target_model=%p draft_ctx=%p draft_mem=%p draft_model=%p ctx_other=%p target_groups=%u draft_groups=%u mtp_layers=%d n_max=%d\n",
                    task.id, (void *) ctx_tgt, (void *) llama_get_memory(ctx_tgt), (const void *) llama_get_model(ctx_tgt),
                    (void *) ctx_dft, (void *) llama_get_memory(ctx_dft), (const void *) llama_get_model(ctx_dft),
                    (void *) (ctx_dft ? llama_get_ctx_other(ctx_dft) : nullptr),
                    llama_n_rs_seq(ctx_tgt) + 1, ctx_dft ? llama_n_rs_seq(ctx_dft) + 1 : 0,
                    llama_model_n_layer_nextn(model_tgt), params_base.speculative.draft.n_max);
#endif
            for (const auto & snapshot : snapshots) {
                SRV_INF("KV_ADMISSION_TEST snapshot task=%d role=%s stream=%u capacity=%" PRIu64 " used=%" PRIu64
                        " exclusive=%" PRIu64 " owners=%zu\n", task.id,
                        snapshot.key.context == kvplan::Context::target ? "target" : "draft",
                        snapshot.key.stream, snapshot.capacity, snapshot.used_real, snapshot.seq_exclusive_real,
                        snapshot.exclusive_by_owner.size());
            }
            for (size_t i = 0; i < kv_admission_slot_owners.size(); ++i) {
                if (kv_admission_slot_owners[i]) {
                    const auto & owner = *kv_admission_slot_owners[i];
                    SRV_INF("KV_ADMISSION_TEST ledger task=%d slot=%zu owner_task=%" PRIu64 " epoch=%" PRIu64 "\n",
                            task.id, i, owner.task, owner.epoch);
                }
            }
            const auto target = std::find_if(snapshots.begin(), snapshots.end(), [](const kvplan::Snapshot & s) {
                return s.key.context == kvplan::Context::target;
            });
            const auto draft = std::find_if(snapshots.begin(), snapshots.end(), [](const kvplan::Snapshot & s) {
                return s.key.context == kvplan::Context::draft;
            });
            if (target != snapshots.end() && draft != snapshots.end()) {
                SRV_INF("KV_ADMISSION_TEST feasibility task=%d target_free=%" PRIu64 " draft_free=%" PRIu64 " launches=%" PRIu64 "\n",
                        task.id, target->capacity - std::min(target->capacity, target->used_real),
                        draft->capacity - std::min(draft->capacity, draft->used_real), kv_admission_test_launch_count);
            }
        }

        const size_t count = 1 + child_slots.size();
        std::vector<kvplan::Reservation> reservations;
        std::vector<kvplan::Owner> owners;
        std::vector<server_slot *> all_slots;
        reservations.reserve(count);
        owners.reserve(count);
        all_slots.reserve(count);
        const auto parent_owner = kvplan::Owner{kv_admission_epoch, (uint64_t) task.id, (uint64_t) parent_slot.id};
        auto parent_resource = resource_by_slot.find({kvplan::Context::target, parent_slot.id});
        if (parent_resource == resource_by_slot.end()) {
            reason = "parent_stream_snapshot_missing";
            error = ERROR_TYPE_KV_UNSUPPORTED;
            return false;
        }
        if (kv_admission_slot_owners[(size_t) parent_slot.id]) {
            reason = "parent_slot_has_active_reservation";
            error = ERROR_TYPE_KV_UNSUPPORTED;
            return false;
        }
        kvplan::Reservation parent_reservation;
        if (!kv_admission_peak(task, task.tokens, parent_slot, parent_owner, std::nullopt,
                               ctx_tgt, parent_resource->second, parent_reservation, reason, error)) return false;
        if (ctx_dft) {
            const auto draft_resource = resource_by_slot.find({kvplan::Context::draft, parent_slot.id});
            if (draft_resource == resource_by_slot.end() || !kv_admission_supported_draft()) {
                reason = "draft_resource_or_simple_binding_missing";
                error = ERROR_TYPE_KV_UNSUPPORTED;
                return false;
            }
            if (!kv_admission_peak(task, task.tokens, parent_slot, parent_owner, std::nullopt,
                                   ctx_dft, draft_resource->second, parent_reservation, reason, error)) return false;
        }
        // Only the selected parent slot is credited (children: none), computed from the pre-move reservation.
        kv_admission_resident_credit(task, parent_slot, parent_owner, resource_by_slot,
                                     parent_reservation, snapshots);
        reservations.push_back(std::move(parent_reservation));
        owners.push_back(parent_owner);
        all_slots.push_back(&parent_slot);

        if (child_slots.size() != task.child_tasks.size()) {
            reason = "parent_child_slot_count_mismatch";
            error = ERROR_TYPE_KV_UNSUPPORTED;
            return false;
        }
        for (size_t i = 0; i < child_slots.size(); ++i) {
            server_slot * child_slot = child_slots[i];
            const server_task & child_task = task.child_tasks[i];
            if (child_slot->id < 0 || (size_t) child_slot->id >= kv_admission_slot_owners.size() || kv_admission_slot_owners[(size_t) child_slot->id]) {
                reason = "child_slot_has_active_reservation";
                error = ERROR_TYPE_KV_UNSUPPORTED;
                return false;
            }
            auto child_resource = resource_by_slot.find({kvplan::Context::target, child_slot->id});
            if (child_resource == resource_by_slot.end()) {
                reason = "child_stream_snapshot_missing";
                error = ERROR_TYPE_KV_UNSUPPORTED;
                return false;
            }
            if (child_task.id < 0 || child_task.id_parent != task.id) {
                reason = "invalid_child_task_identity";
                error = ERROR_TYPE_KV_UNSUPPORTED;
                return false;
            }
            const auto child_owner = kvplan::Owner{kv_admission_epoch, (uint64_t) child_task.id, (uint64_t) child_slot->id};
            kvplan::Reservation child_reservation;
            if (!kv_admission_peak(child_task, task.tokens, *child_slot, child_owner, parent_owner,
                                   ctx_tgt, child_resource->second, child_reservation, reason, error)) return false;
            if (ctx_dft) {
                const auto draft_resource = resource_by_slot.find({kvplan::Context::draft, child_slot->id});
                if (draft_resource == resource_by_slot.end() || !kv_admission_supported_draft()) {
                    reason = "child_draft_resource_or_simple_binding_missing";
                    error = ERROR_TYPE_KV_UNSUPPORTED;
                    return false;
                }
                if (!kv_admission_peak(child_task, task.tokens, *child_slot, child_owner, parent_owner,
                                       ctx_dft, draft_resource->second, child_reservation, reason, error)) return false;
            }
            reservations.push_back(std::move(child_reservation));
            owners.push_back(child_owner);
            all_slots.push_back(child_slot);
        }

        if (kv_admission_cpu_test_env("LLAMA_KV_ADMISSION_TEST_TRACE")) {
            for (const auto & reservation : reservations) {
                for (const auto & peak : reservation.peak_by_resource) {
                    const auto snapshot = std::find_if(snapshots.begin(), snapshots.end(), [&peak](const kvplan::Snapshot & s) {
                        return s.key == peak.first;
                    });
                    if (snapshot != snapshots.end()) {
                        const uint64_t free = snapshot->capacity - std::min(snapshot->capacity, snapshot->used_real);
                        SRV_INF("KV_ADMISSION_TEST demand task=%d slot=%" PRIu64 " role=%s stream=%u peak=%" PRIu64
                                " free_real=%" PRIu64 " independently_fits=%d\n", task.id, reservation.owner.slot,
                                peak.first.context == kvplan::Context::target ? "target" : "draft",
                                peak.first.stream, peak.second, free, (int) (peak.second <= free));
                    }
                }
            }
        }

        kvplan::Result admitted{kvplan::Status::rejected, "planner not called"};
        admitted = kv_admission_planner.admit(snapshots, reservations);
        if (admitted.status != kvplan::Status::accepted) {
            reason = admitted.reason;
            error = ERROR_TYPE_KV_CAPACITY;
            return false;
        }
        if (kv_admission_cpu_test_env("LLAMA_KV_ADMISSION_TEST_TRACE")) {
            ++kv_admission_test_commit_count;
            SRV_INF("KV_ADMISSION_TEST commit count=%" PRIu64 " task=%d reservations=%zu\n",
                    kv_admission_test_commit_count, task.id, reservations.size());
        }
        for (size_t i = 0; i < all_slots.size(); ++i) {
            kv_admission_slot_owners[(size_t) all_slots[i]->id] = owners[i];
        }
        if (kv_admission_cpu_test_env("LLAMA_KV_ADMISSION_TEST_TRACE")) {
            SRV_INF("KV_ADMISSION_TEST admitted task=%d reservations=%zu ledger_size=%zu launches=%" PRIu64 "\n",
                    task.id, all_slots.size(), kv_admission_planner.size(), kv_admission_test_launch_count);
        }
        SRV_INF("KV_ADMISSION_ACCEPTED task=%d count=%zu epoch=%" PRIu64 "\n", task.id, all_slots.size(), kv_admission_epoch);
        return true;
    }

    void kv_admission_release_unlaunched(const std::vector<server_slot *> & candidates) noexcept {
        for (server_slot * slot : candidates) {
            if (slot && !slot->is_processing()) release_kv_admission_slot(slot->id);
        }
    }

    void process_single_task(server_task && task) {
        if (params_base.kv_admission && !kv_admission_plan_restore_enabled() &&
                task.type == SERVER_TASK_TYPE_SLOT_RESTORE) {
            if (!task.slot_action.operation_id.empty()) {
                slot_operations.finish(task.slot_action.operation_id, "failed", {}, "KV admission B1 blocks slot restore writers");
            }
            send_error(task, "KV admission B1 blocks slot restore path", ERROR_TYPE_KV_UNSUPPORTED);
            return;
        }
        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_task = task.id;
                    if (params_base.kv_admission && (task.id < 0 || task.id_slot < -1 ||
                            (task.id_slot >= 0 && (size_t) task.id_slot >= slots.size()))) {
                        send_error(task, "KV admission refused: invalid_native_task_or_slot_id", ERROR_TYPE_KV_UNSUPPORTED);
                        for (const auto & child : task.child_tasks) {
                            send_error(child, "KV admission refused: invalid_native_task_or_slot_id", ERROR_TYPE_KV_UNSUPPORTED);
                        }
                        break;
                    }

                    server_slot * slot = get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    std::vector<server_slot *> child_slots;
                    if (task.is_parent()) {
                        // Reserve the whole parent group before any child is launched.
                        const size_t n_child_tasks = task.child_tasks.size();
                        child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                    }

                    // Prepare launch bookkeeping before committing reservations.
                    std::vector<server_slot *> attempted;
                    try {
                        attempted.reserve(1 + child_slots.size());
                        attempted.push_back(slot);
                        attempted.insert(attempted.end(), child_slots.begin(), child_slots.end());
                    } catch (const std::bad_alloc &) {
                        send_error(task, "KV admission refused: launch_bookkeeping_allocation_failure", ERROR_TYPE_KV_CAPACITY);
                        for (const auto & child : task.child_tasks) {
                            send_error(child, "KV admission refused: launch_bookkeeping_allocation_failure", ERROR_TYPE_KV_CAPACITY);
                        }
                        break;
                    }

                    if (params_base.kv_admission) {
                        std::string reason;
                        error_type error = ERROR_TYPE_KV_UNSUPPORTED;
                        bool admitted = false;
                        try {
                            admitted = kv_admission_preflight(task, *slot, child_slots, reason, error);
                        } catch (const std::bad_alloc &) {
                            reason = "admission_allocation_failure";
                            error = ERROR_TYPE_KV_CAPACITY;
                        }
                        if (!admitted) {
                            // A preflight exception after commit must not strand unlaunched owners.
                            kv_admission_release_unlaunched(attempted);
                            if (kv_admission_cpu_test_env("LLAMA_KV_ADMISSION_TEST_TRACE")) {
                                SRV_INF("KV_ADMISSION_TEST refused task=%d reason=%s launches=%" PRIu64 " ledger_release_failed=%d\n",
                                        task.id, reason.c_str(), kv_admission_test_launch_count,
                                        (int) kv_admission_release_failed);
                                std::vector<kvplan::Snapshot> after;
                                kv_admission_slot_resources resources_after;
                                std::string after_reason;
                                if (kv_admission_snapshots(after, resources_after, after_reason)) {
                                    for (const auto & snapshot : after) {
                                        SRV_INF("KV_ADMISSION_TEST after_refusal task=%d role=%s stream=%u capacity=%" PRIu64 " used=%" PRIu64
                                                " exclusive=%" PRIu64 " owners=%zu\n", task.id,
                                                snapshot.key.context == kvplan::Context::target ? "target" : "draft",
                                                snapshot.key.stream, snapshot.capacity, snapshot.used_real, snapshot.seq_exclusive_real,
                                                snapshot.exclusive_by_owner.size());
                                    }
                                } else {
                                    SRV_INF("KV_ADMISSION_TEST after_refusal_snapshot_unavailable task=%d reason=%s\n",
                                            task.id, after_reason.c_str());
                                }
                                for (size_t i = 0; i < kv_admission_slot_owners.size(); ++i) {
                                    if (kv_admission_slot_owners[i]) {
                                        const auto & owner = *kv_admission_slot_owners[i];
                                        SRV_INF("KV_ADMISSION_TEST after_refusal_ledger task=%d slot=%zu owner_task=%" PRIu64 " epoch=%" PRIu64 "\n",
                                                task.id, i, owner.task, owner.epoch);
                                    }
                                }
                                SRV_INF("KV_ADMISSION_TEST after_refusal_ledger_size task=%d size=%zu\n", task.id,
                                        kv_admission_planner.size());
                            }
                            send_error(task, "KV admission refused: " + reason, error);
                            for (const auto & child : task.child_tasks) {
                                send_error(child, "KV admission refused: " + reason, error);
                            }
                            break;
                        }
                    }

                    bool launched = false;
                    try {
                        if (task.is_parent()) {
                            launched = launch_slots_with_parent_task(*slot, child_slots, std::move(task));
                        } else {
                            launched = launch_slot_with_task(*slot, std::move(task));
                        }
                    } catch (const std::exception & e) {
                        for (server_slot * attempted_slot : attempted) {
                            if (attempted_slot && attempted_slot->is_processing()) attempted_slot->release();
                        }
                        kv_admission_release_unlaunched(attempted);
                        send_error(id_task, std::string("Failed to launch task: ") + e.what(), ERROR_TYPE_SERVER);
                        break;
                    }
                    if (!launched) {
                        for (server_slot * attempted_slot : attempted) {
                            if (attempted_slot && attempted_slot->is_processing()) attempted_slot->release();
                        }
                        kv_admission_release_unlaunched(attempted);
                        SRV_ERR("failed to launch slot group, id_task = %d\n", id_task);
                        break;
                    }

                    if (params_base.cache_idle_slots) {
                        for (auto & slot : slots) {
                            if (!slot.is_processing()) {
                                SLT_TRC(slot, "%s", "saving idle slot to prompt cache\n");

                                if (slot.prompt_save(*prompt_cache)) {
                                    SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
                                    prompt_cache->update();
                                }

                                if (params_base.kv_unified) {
                                    // [TAG_IDLE_SLOT_CLEAR]
                                    slot.prompt_clear();
                                }
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        SRV_WRN("control %s on unknown completion id=%s, no live slot\n",
                                task.params.control_action.c_str(), task.params.control_cmpl_id.c_str());
                        res->success = false;
                        res->message = "no active completion for this id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (task.params.control_action == "reasoning_end") {
                        // the budget sampler only exists when reasoning control was armed
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            queue_results.send(std::move(res));
                            break;
                        }
                        // act on the live slot mid generation, never defer
                        common_sampler_reasoning_budget_force(slot->smpl.get());
                        res->success = true;
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    json slots_data = json::array();

                    int n_idle_slots       = 0;
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        json slot_data = slot.to_json(slots_debug == 0);

                        if (slot.is_processing()) {
                            n_processing_slots++;
                        } else {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot_data);
                    }
                    SRV_DBG("n_idle_slots = %d, n_processing_slots = %d\n", n_idle_slots, n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->slots_data          = std::move(slots_data);
                    res->n_idle_slots        = n_idle_slots;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->t_start             = metrics.t_start;

                    res->n_prompt_tokens_processed_total = metrics.n_prompt_tokens_processed_total;
                    res->t_prompt_processing_total       = metrics.t_prompt_processing_total;
                    res->n_tokens_predicted_total        = metrics.n_tokens_predicted_total;
                    res->t_tokens_generation_total       = metrics.t_tokens_generation_total;

                    res->n_tokens_max = metrics.n_tokens_max;

                    res->n_prompt_tokens_processed = metrics.n_prompt_tokens_processed;
                    res->t_prompt_processing       = metrics.t_prompt_processing;
                    res->n_tokens_predicted        = metrics.n_tokens_predicted;
                    res->t_tokens_generation       = metrics.t_tokens_generation;

                    res->n_decode_total          = metrics.n_decode_total;
                    res->n_busy_slots_total      = metrics.n_busy_slots_total;

                    // The opt-in integration gate observes mutable admission state
                    // only here, on the same loop that owns slots and the ledger.
                    if (kv_admission_cpu_test_env("LLAMA_KV_ADMISSION_TEST_DRIVER") ||
                            kv_admission_cpu_test_env("LLAMA_KV_ADMISSION_TEST_OBSERVE")) {
                        json owners = json::array();
                        for (size_t i = 0; i < kv_admission_slot_owners.size(); ++i) {
                            if (kv_admission_slot_owners[i]) {
                                const auto & owner = *kv_admission_slot_owners[i];
                                owners.push_back({{"epoch", owner.epoch}, {"task", owner.task}, {"slot", owner.slot}});
                            }
                        }
                        res->kv_admission_test_snapshot = {
                            {"launch_count", kv_admission_test_launch_count},
                            {"commit_count", kv_admission_test_commit_count},
                            {"ledger_size", kv_admission_planner.size()},
                            {"release_failed", kv_admission_release_failed},
                            {"owners", std::move(owners)},
                        };
                    }

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        slot_operations.finish(task.slot_action.operation_id, "failed", nullptr, "Invalid slot ID");
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (!check_slot_no_media(*slot, task.id)) {
                        slot_operations.finish(task.slot_action.operation_id, "failed", nullptr, "Slot contains unsupported media state");
                        send_error(task, "Unable to save slot state containing unsupported media", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        slot_operations.set_status(task.slot_action.operation_id, "deferred");
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    slot_operations.set_status(task.slot_action.operation_id, "running");

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    const llama_tokens tokens = slot->prompt.tokens.get_text_tokens();
                    const size_t token_count = tokens.size();
                    const size_t nwrite = llama_state_seq_save_file(ctx_tgt, filepath.c_str(), slot->id, tokens.data(), token_count);
                    if (nwrite == 0) {
                        slot_operations.finish(task.slot_action.operation_id, "failed", nullptr, "write, flush, close, or stat failed");
                        send_error(task, "Unable to save slot state: write, flush, close, or stat failed", ERROR_TYPE_SERVER);
                        break;
                    }
                    const size_t n_ck = nwrite > 0 ? slot_ckpt_save(filepath + ".ckpt", tokens, slot->prompt.checkpoints) : 0;

#ifdef LLAMA_EXP_SLOT_DFT
                    size_t nwrite_dft = 0;

                    if (slot->ctx_dft) {
                        // experimental: persist the draft (MTP) sequence state + the speculative carry (pending_h)
                        // next to the main slot file, so that a restore on another engine does not leave the
                        // draft context stale (v2 layout: see slot_dft_save)
                        const std::string filepath_dft = filepath + ".dft";

                        size_t n_spec = 0;
                        nwrite_dft = slot_dft_save(slot->ctx_dft, spec.get(), slot->id, filepath_dft, tokens.data(), token_count, n_spec);

                        if (nwrite_dft == 0) {
                            SLT_WRN((*slot), "failed to save draft state to '%s' - a restore of this slot will warm up the draft cold\n", filepath_dft.c_str());
                        } else {
                            SLT_TRC((*slot), "saved draft state to '%s', %zu bytes (spec carry: %zu bytes)\n", filepath_dft.c_str(), nwrite_dft, n_spec);
                            if (n_spec == 0) {
                                SLT_WRN((*slot), "%s", "draft state saved without the speculative carry (pending_h not valid yet) - the first draft after a restore pairs with zeros\n");
                            }
                        }
                    }
#endif

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = token_count;
                    res->n_bytes  = nwrite;
#ifdef LLAMA_EXP_SLOT_DFT
                    res->n_bytes_dft = nwrite_dft;
#endif
#ifdef LLAMA_EXP_SLOT_DIAG
                    slot_diag_state("save", ctx_tgt, slot->ctx_dft, spec.get(), slot->id, tokens);
#endif
                    res->n_checkpoints = n_ck;
                    res->t_ms     = t_save_ms;
                    slot_operations.finish(task.slot_action.operation_id, "completed", res->to_json());
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        slot_operations.finish(task.slot_action.operation_id, "failed", nullptr, "Invalid slot ID");
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        slot_operations.set_status(task.slot_action.operation_id, "deferred");
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    slot_operations.set_status(task.slot_action.operation_id, "running");

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    llama_tokens tokens;
                    tokens.resize(slot->n_ctx);
                    size_t token_count = 0;

                    // When kv_admission is active, plan_restore reserves resources BEFORE
                    // clearing prior slot state. Clear occurs only after an admitted plan.
                    std::unique_ptr<kvplan::RestorePlan> rplan;
                    std::optional<kvplan::Owner> restore_owner;
                    kvplan::RestoreKeyMap tgt_keys, dft_keys; // (kind,stream) -> key; no single-key fallback
                    std::vector<kvplan::ResourceKey> declared;
                    struct RestoreLease {
                        kvplan::Planner * planner = nullptr;
                        kvplan::RestorePlan * plan = nullptr;
                        kvplan::Owner * owner = nullptr;
                        llama_context * target = nullptr;
                        llama_context * draft = nullptr;
                        bool released = false;
                        std::atomic<uint64_t> * fences = nullptr;
                        std::atomic<uint64_t> * releases = nullptr;
                        void release() noexcept {
                            if (released || !planner || !plan || !owner) return;
                            llama_synchronize(target);
                            if (draft) llama_synchronize(draft);
                            const bool fenced = planner->fence_restore(plan->token);
                            if (fenced && fences) ++*fences;
                            const auto result = planner->native_release(*owner);
                            if (result.status == kvplan::Status::accepted && releases) ++*releases;
                            SRV_INF("B3_RESTORE fence=%d release=%d ledger=%zu\n", fenced,
                                    result.status == kvplan::Status::accepted, planner->size());
                            released = true;
                        }
                        ~RestoreLease() { release(); }
                    } lease{&kv_admission_planner, nullptr, nullptr, ctx_tgt, ctx_dft, false, &b3_fenced, &b3_released};

                    if (kv_admission_plan_restore_enabled()) {
                        llama_synchronize(ctx_tgt);
                        if (ctx_dft) llama_synchronize(ctx_dft);

                        std::string snap_reason;
                        std::vector<kvplan::Snapshot> snapshots;
                        kv_admission_slot_resources resource_by_slot;
                        if (!kv_admission_snapshots(snapshots, resource_by_slot, snap_reason)) {
                            llama_synchronize(ctx_tgt);
                            if (ctx_dft) llama_synchronize(ctx_dft);
                            slot_operations.finish(task.slot_action.operation_id, "failed", nullptr, "Snapshot derivation failed: " + snap_reason);
                            send_error(task, "KV admission snapshot failed", ERROR_TYPE_KV_UNSUPPORTED);
                            break;
                        }

                        const auto tgt_it = resource_by_slot.find({kvplan::Context::target, slot->id});
                        if (tgt_it == resource_by_slot.end()) {
                            llama_synchronize(ctx_tgt);
                            if (ctx_dft) llama_synchronize(ctx_dft);
                            slot_operations.finish(task.slot_action.operation_id, "failed", nullptr, "Missing target resource key");
                            send_error(task, "Missing target resource key", ERROR_TYPE_KV_UNSUPPORTED);
                            break;
                        }
                        const bool tgt_rec = llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt);
                        const bool tgt_att = !tgt_rec || llama_model_is_hybrid(model_tgt);
                        auto refuse_restore = [&](const std::string & why) {
                            ++b3_refused;
                            SRV_INF("B3_RESTORE plan_refused=%s clears=%" PRIu64 "\n", why.c_str(), b3_clears.load());
                            llama_synchronize(ctx_tgt);
                            if (ctx_dft) llama_synchronize(ctx_dft);
                            slot_operations.finish(task.slot_action.operation_id, "failed", nullptr, "KV admission restore refused: " + why);
                            send_error(task, "KV admission restore refused: " + why, ERROR_TYPE_KV_CAPACITY);
                        };
                        {
                            // (kind,stream)->key map of the destination slot: hybrid = attention + recurrent. Any key of an
                            // unknown kind, duplicate pair or missing declared kind is refused_unsupported_layout, zero clear.
                            bool ok = true;
                            for (const auto & k : tgt_it->second) ok = tgt_keys.add(k) && ok;
                            ok = ok && tgt_keys.keys.size() == size_t(tgt_att) + size_t(tgt_rec) &&
                                 tgt_keys.has(kvplan::Kind::attention_cells) == tgt_att && tgt_keys.has(kvplan::Kind::recurrent_rows) == tgt_rec;
                            if (ctx_dft) {
                                const auto dft_it = resource_by_slot.find({kvplan::Context::draft, slot->id});
                                if (dft_it != resource_by_slot.end()) {
                                    for (const auto & k : dft_it->second) {
                                        if (k.kind == kvplan::Kind::attention_cells) ok = dft_keys.add(k) && ok;
                                    }
                                }
                            }
                            if (!ok || !kvplan::restore_declared_keys(resource_by_slot, tgt_att, tgt_rec, ctx_dft != nullptr, declared)) {
                                refuse_restore("refused_unsupported_layout: target key map differs from declared layout");
                                break;
                            }
                        }

                        restore_owner = kvplan::Owner{kv_admission_epoch, (uint64_t) task.id, (uint64_t) slot->id};
                        // CPU-only negative fixture: exercise a genuine typed-plan refusal.
#ifdef LLAMA_B3_CPU_FIXTURES
                        const char * refuse = std::getenv("LLAMA_B3_CPU_REFUSE_PLAN_ONCE");
                        static std::atomic<bool> refused_once{false};
                        if (refuse && std::string(refuse) == "1" && !refused_once.exchange(true) && !snapshots.empty()) {
                            snapshots.front().known = false;
                            SRV_WRN("%s\n", "B3_CPU_UNKNOWN_SNAPSHOT injected");
                        }
#endif
                        if (tgt_rec) {
                            // D1: replacement credit of the destination's own recurrent cells, from its physical view.
                            llama_memory_resource_capacity raw = {};
                            std::vector<kvplan::PhysicalView> dest;
                            std::string credit_reason;
                            if (!llama_memory_get_resource_capacity(llama_get_memory(ctx_tgt), slot->id, &raw, sizeof(raw)) ||
                                    !kvplan::normalize_resources(kvplan::Context::target, kv_admission_epoch, raw, dest) ||
                                    !kvplan::apply_restore_credit(dest, *restore_owner, snapshots, credit_reason)) {
                                refuse_restore("refused_unsupported_layout: " + (credit_reason.empty() ? std::string("destination view unknown") : credit_reason));
                                break;
                            }
                        }
#ifdef LLAMA_B3_LIMIT_FIXTURE
                        // TEST-ONLY (never compiled in production): force the restore limit I of the target resource
                        // (attention / recurrent) to an exact value by faking the physical occupancy seen by the planner.
                        // Planner, ledger and loader logic are untouched.
                        for (const auto & fx : {std::make_pair("LLAMA_B3_FIXTURE_RESTORE_LIMIT", kvplan::Kind::attention_cells),
                                                std::make_pair("LLAMA_B3_FIXTURE_RESTORE_LIMIT_REC", kvplan::Kind::recurrent_rows)}) {
                            if (const char * force = std::getenv(fx.first)) {
                                char * end = nullptr;
                                const unsigned long long n = std::strtoull(force, &end, 10);
                                if (end != force && *end == 0 && kvplan::apply_limit_fixture(snapshots, fx.second, n)) {
                                    SRV_WRN("B3_LIMIT_FIXTURE %s forced limit_I=%llu\n", fx.first, n);
                                }
                            }
                        }
#endif
                        rplan = std::make_unique<kvplan::RestorePlan>(kv_admission_planner.plan_restore(snapshots, *restore_owner, declared));
                        if (rplan->kind != kvplan::RestorePlan::Kind::ok) {
                            ++b3_refused;
                            SRV_INF("B3_RESTORE plan_refused=%s clears=%" PRIu64 "\n", rplan->reason.c_str(), b3_clears.load());
                            // Budget or admission refused: ZERO clear performed, prior slot state remains intact.
                            llama_synchronize(ctx_tgt);
                            if (ctx_dft) llama_synchronize(ctx_dft);
                            slot_operations.finish(task.slot_action.operation_id, "failed", nullptr, "KV admission restore refused: " + rplan->reason);
                            send_error(task, "KV admission restore refused: " + rplan->reason, ERROR_TYPE_KV_CAPACITY);
                            break;
                        }
                        ++b3_reserved;
                        b3_limit = 0; b3_limit_rec = 0;
                        for (const auto & item : tgt_keys.keys) {
                            const uint64_t lim = rplan->token.limit(item.second).value_or(0);
                            (item.second.kind == kvplan::Kind::recurrent_rows ? b3_limit_rec : b3_limit) = lim;
                        }
                        SRV_INF("B3_RESTORE reserved limit_I=%" PRIu64 " limit_rec=%" PRIu64 " ledger=%zu\n", b3_limit.load(), b3_limit_rec.load(), kv_admission_planner.size());
                        lease.plan = rplan.get();
                        lease.owner = &*restore_owner;
                    }

                    // A restore is a replacement transaction. Remove prior target, draft,
                    // checkpoint and speculative carry state only after bounded snapshot validation.
                    bool restore_mutated = false;
                    if (!rplan) slot->clear_after_restore_failure();

                    size_t nread = 0;
                    if (rplan != nullptr) {
                        struct RestoreConsumerBridge {
                            kvplan::Planner * planner;
                            kvplan::RestoreToken * token;
                            kvplan::RestoreKeyMap map;
                            server_slot * slot;
                            std::atomic<uint64_t> * clears;
                            uint64_t unmapped = 0;
                            static void begin_cb(void * user) {
                                auto * bridge = static_cast<RestoreConsumerBridge *>(user);
                                ++*bridge->clears;
                                bridge->slot->clear_after_restore_failure();
                            }
                            static bool consume_cb(void * user, uint32_t kind, uint32_t stream, uint64_t count) {
                                auto * bridge = static_cast<RestoreConsumerBridge *>(user);
                                return kvplan::restore_consume(*bridge->planner, *bridge->token, bridge->map, kind, stream, count, bridge->unmapped);
                            }
                        } bridge{&kv_admission_planner, &rplan->token, tgt_keys, slot, &b3_clears};

                        struct llama_state_seq_limits limits{};
                        limits.user = &bridge;
                        limits.consume = RestoreConsumerBridge::consume_cb;
                        limits.before_replace = RestoreConsumerBridge::begin_cb;
                        limits.max_tensor_bytes = llama_state_seq_restore_tensor_capacity(ctx_tgt);
                        limits.max_snapshot_bytes = llama_state_seq_restore_file_capacity(ctx_tgt, slot->id);

                        nread = llama_state_seq_load_file_bounded(ctx_tgt, filepath.c_str(), slot->id,
                                                                 tokens.data(), tokens.size(), &token_count, &limits);
                        restore_mutated = limits.mutation_started;
                        if (bridge.unmapped) SRV_WRN("B3_RESTORE refused charge for pair without key count=%" PRIu64 "\n", bridge.unmapped);
                    } else {
                        nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, tokens.data(), tokens.size(), &token_count);
                    }

#ifdef LLAMA_B3_CPU_FIXTURES
                    if (nread > 0 && kv_admission_plan_restore_enabled()) {
                        const char * fault = std::getenv("LLAMA_B3_CPU_FAIL_AFTER_RESTORE_READ");
                        static std::atomic<bool> injected{false};
                        if (fault != nullptr && std::string(fault) == "1" && !injected.exchange(true)) {
                            SRV_WRN("%s\n", "B3_CPU_FAULT_AFTER_RESTORE_READ injected");
                            nread = 0;
                        }
                    }
#endif
                    if (nread == 0) {
                        if (!rplan || restore_mutated) slot->clear_after_restore_failure();
                        lease.release();
                        SRV_INF("%s\n", "B3_RESTORE terminal=failed after_fence");
                        slot_operations.finish(task.slot_action.operation_id, "failed", nullptr, "Unable to restore slot state");
                        send_error(task, "Unable to restore slot, no available space in KV cache or invalid slot save file", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }

#ifdef LLAMA_EXP_SLOT_DFT
                    size_t nread_dft = 0;

                    if (slot->ctx_dft) {
                        // experimental: restore the draft (MTP) sequence state and the speculative carry (pending_h)
                        // saved next to the main slot file. v1 files (KV only) are still accepted.
                        // the draft state is optional: on any failure we clear the draft sequence and keep
                        // the current behavior - the draft warms up as the target decodes
                        const std::string filepath_dft = filepath + ".dft";

                        tokens.resize(token_count);

                        int  version_dft   = 0;
                        bool spec_restored = false;

                        nread_dft = slot_dft_load(slot->ctx_dft, spec.get(), slot->id, filepath_dft, tokens, version_dft, spec_restored, kv_admission_plan_restore_enabled(), rplan ? &kv_admission_planner : nullptr, rplan ? &rplan->token : nullptr, rplan ? &dft_keys : nullptr, &b3_clears);

                        if (nread_dft == 0) {
                            SLT_WRN((*slot), "no usable draft state in '%s' (n_tokens = %zu) - the draft will warm up cold\n",
                                    filepath_dft.c_str(), token_count);

                            slot->clear_optional_draft_restore();
                        } else {
                            SLT_TRC((*slot), "restored draft state from '%s', v%d, %zu bytes, spec carry %s\n",
                                    filepath_dft.c_str(), version_dft, nread_dft,
                                    spec_restored ? "restored" : "absent (first draft pairs with a stale/zero h row)");
                        }
                    }
#endif

                    tokens.resize(token_count);
#ifdef LLAMA_EXP_SLOT_DIAG
                    slot_diag_state("restore", ctx_tgt, slot->ctx_dft, spec.get(), slot->id, tokens);
#endif
                    slot->prompt.clear();
                    slot->prompt.tokens.insert(tokens);
                    const size_t n_ck = slot_ckpt_load(filepath + ".ckpt", tokens, (size_t) std::max(0, params_base.n_ctx_checkpoints), slot->prompt.checkpoints, params_base.kv_admission_restore);
                    if (n_ck > 0) {
                        SLT_INF(*slot, "restored %zu context checkpoints from side file\n", n_ck);
                    }

                    // Native output fence (R3): synchronizes target and draft contexts
                    // before releasing admission token and publishing completed status.
                    llama_synchronize(ctx_tgt);
                    if (ctx_dft) llama_synchronize(ctx_dft);

                    lease.release();

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = token_count;
                    res->n_bytes  = nread;
#ifdef LLAMA_EXP_SLOT_DFT
                    res->n_bytes_dft = nread_dft;
#endif
                    res->n_checkpoints = n_ck;
                    res->t_ms     = t_restore_ms;
                    slot_operations.finish(task.slot_action.operation_id, "completed", res->to_json());
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        slot_operations.finish(task.slot_action.operation_id, "failed", nullptr, "Invalid slot ID");
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    // Gate on slot content, consistent with save/restore.
                    if (!check_slot_no_media(*slot, task.id)) {
                        slot_operations.finish(task.slot_action.operation_id, "failed", nullptr, "Slot contains unsupported media state");
                        send_error(task, "Unable to erase slot containing unsupported media", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        slot_operations.set_status(task.slot_action.operation_id, "deferred");
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    slot_operations.set_status(task.slot_action.operation_id, "running");

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear();

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    slot_operations.finish(task.slot_action.operation_id, "completed", res->to_json());
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_TRC("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }
    }

    void iterate(std::vector<server_slot> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(slot);
            } catch (const std::exception & e) {
                SLT_ERR(slot, "got exception: %s\n", e.what());
                send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    void iterate(std::vector<server_slot *> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(*slot);
            } catch (const std::exception & e) {
                SLT_ERR(*slot, "got exception: %s\n", e.what());
                send_error(*slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot->release();
            }
        }
    }

    void abort_all_slots(const std::string & reason) {
        for (auto & slot : slots) {
            if (slot.is_processing()) {
                send_error(slot, reason, ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    // @ngxson : for debugging only
    int64_t t_pre_decode  = 0;
    int64_t t_decode      = 0;
    int64_t t_post_decode = 0;
    int64_t t_sampl       = 0;
    int64_t n_pre_decode  = 0;
    int64_t n_decode      = 0;
    int64_t n_post_decode = 0;
    int64_t n_sampl       = 0;
// #define DEBUG_TIMINGS
#ifdef DEBUG_TIMINGS
    struct scoped_timer {
        int64_t & t;
        int64_t & n;
        int64_t t_start;
        scoped_timer(int64_t & t_, int64_t & n_) : t(t_), n(n_) {
            t_start = ggml_time_us();
        }
        ~scoped_timer() {
            t += ggml_time_us() - t_start;
            n++;
        }
    };
#else
    struct scoped_timer {
        scoped_timer(int64_t &, int64_t &) {}
        ~scoped_timer() {}
    };
#endif

    void update_slots() {
#ifdef DEBUG_TIMINGS
        static int64_t t_prev = 0;
        int64_t t_start = ggml_time_us();
        if (t_start - t_prev > 5 * 1000 * 1000) { // every 5 seconds
            t_prev = t_start;
            SRV_INF("n_pre_decode      = %" PRId64 "\n", n_pre_decode);
            SRV_INF("avg t_pre_decode  = %f ms\n", (double) t_pre_decode / n_pre_decode / 1000.0);
            SRV_INF("avg t_decode      = %f ms\n", (double) t_decode / n_decode / 1000.0);
            SRV_INF("avg t_post_decode = %f ms\n", (double) t_post_decode / n_post_decode / 1000.0);
            SRV_INF("avg t_sampl       = %f ms\n", (double) t_sampl / n_sampl / 1000.0);
        }
#endif

        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_TRC("%s", "all slots are idle\n");
                return; // skip further processing

            } else {
                SRV_DBG("%s", "posting NEXT_RESPONSE\n");

                server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
                task.id = queue_tasks.get_new_id();
                queue_tasks.post(std::move(task));
            }
        }

        try {
            scoped_timer t(t_pre_decode, n_pre_decode);
            pre_decode();
            batch.render();
        } catch (const std::exception & e) {
            SRV_ERR("pre_decode() failed: %s\n", e.what());
            abort_all_slots("pre_decode() failed: " + std::string(e.what()));
        }

        GGML_ASSERT(batch.slot_batched || batch.size() == 0);

        if (batch.slot_batched) {
            auto & slot_batched      = batch.slot_batched;
            auto & alora_scale       = batch.alora_scale;
            auto & alora_disabled_id = batch.alora_disabled_id;

            // TODO @ngxson : alora handling is too messy, need to refactor it to be more clear and maintainable
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());
        }

        llama_batch batch_view;
        int32_t off_next = 0;
        int32_t n_batch = llama_n_batch(ctx_tgt);
        for (int32_t off = 0; off < batch.size(); off = off_next) {
            const int32_t n_tokens = std::min(n_batch, batch.size() - off);
            try {
                scoped_timer t(t_decode, n_decode);
                // TODO @ngxson : maybe handle n_batch == 1 here instead of inside decode()

                batch_view = batch.get_view(off, n_tokens);
                bool ok = decode(n_batch, off, batch_view);
#ifdef DEBUG_TIMINGS
                llama_synchronize(ctx_tgt);
#endif

                if (ok) {
                    // move the head of the batch forward with the number of tokens we just processed
                    off_next = off + n_tokens;

                    // on successful decode, restore the original batch size
                    n_batch = llama_n_batch(ctx_tgt);
                } else {
                    // try again with the updated n_batch
                    continue;
                }
            } catch (const std::exception & e) {
                SRV_ERR("decode() failed: %s\n", e.what());
                abort_all_slots("decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }

            try {
                scoped_timer t(t_post_decode, n_post_decode);
                post_decode(n_tokens, off, batch_view);
            } catch (const std::exception & e) {
                SRV_ERR("post_decode() failed: %s\n", e.what());
                abort_all_slots("post_decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }
        }
    }

    void pre_decode() {
        // apply context-shift if needed
        // TODO: simplify and improve
        iterate(slots, [&](server_slot & slot) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                int       n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                // ref: https://github.com/ggml-org/llama.cpp/pull/24786
                n_discard = std::clamp(n_discard, 0, std::max(0, n_left - 1));

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                slot.mem.seq_rm (slot.id, n_keep            , n_keep + n_discard);
                slot.mem.seq_add(slot.id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
            }
        });

        // start populating the batch for this iteration
        batch.clear();

        // track if given slot can be batched with slots already in the batch
        auto & slot_batched = batch.slot_batched;

        std::vector<server_slot *> generating;
        std::vector<server_slot *> drafting;

        // determine which slots are generating and drafting
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                return;
            }

            generating.push_back(&slot);

            if (spec) {
                common_speculative_get_draft_params(spec.get(), slot.id).drafting = false;

                const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                const int n_draft_max = slot.get_n_draft_max();

                if (n_draft_max > 0) {
                    GGML_ASSERT(slot.can_speculate());

                    if (!slot.spec_draft.empty()) {
                        // we have a previous (partial) draft to reuse
                        if (use_ckpt_tgt) {
                            GGML_ASSERT(!slot.spec_ckpt.empty());
                        }
                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        slot.spec_ckpt.update_pos(
                                slot.prompt.n_tokens(),
                                llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id),
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id));

                        if (use_ckpt_dft) {
                            slot.spec_ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }

                        slot.spec_prompt = slot.prompt.tokens.get_text_tokens();

                        common_speculative_get_draft_params(spec.get(), slot.id) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            /* .n_past   = */ slot.prompt.n_tokens(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                        };

                        drafting.push_back(&slot);
                    }
                }
            }
        });

        // generate the actual drafts (if any)
        {
#ifdef LLAMA_KV_ADMISSION_CPU_TEST_DRIVER
            for (server_slot * slot : drafting) this->kv_admission_test_observe("before_draft", slot->id, slot->get_n_draft_max(), 0);
#endif
            common_speculative_draft(spec.get());
#ifdef LLAMA_KV_ADMISSION_CPU_TEST_DRIVER
            for (server_slot * slot : drafting) this->kv_admission_test_observe("after_draft", slot->id, slot->spec_draft.size(), 0);
#endif
        }

        // make checkpoints if needed
        iterate(drafting, [&](server_slot & slot) {
            auto & draft = slot.spec_draft;
            auto & ckpt  = slot.spec_ckpt;

            slot.n_draft_total += draft.size();

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

            if (ctx_dft) {
                if (use_ckpt_dft) {
                    ckpt.load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }

                if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, ckpt.pos_max + 1, -1)) {
                    GGML_ABORT("failed to remove sequence %d\n", slot.id);
                }
            }

            if (!draft.empty()) {
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt));

                const bool use_ckpt_dft =
                   (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_dft));

                if (use_ckpt_tgt) {
                    //const int64_t t_start = ggml_time_us();

                    ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                    //const int64_t t_total = ggml_time_us() - t_start;
                    //printf("checkpoint total: %f ms\n", t_total / 1000.0);

                    SLT_DBG(slot, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = %.3f MiB)\n",
                            ckpt.pos_min, ckpt.pos_max, slot.prompt.n_tokens(),
                            (float) ckpt.size() / 1024 / 1024,
                            (float) ckpt.data_dft.size() / 1024 / 1024);
                }

                if (use_ckpt_dft) {
                    ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }
        });

        // update the batch with the sampled/drafted tokens
        iterate(generating, [&](server_slot & slot) {
            slot.handle_last_sampled_token(batch);
        });

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx_tgt);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);

        auto & alora_scale       = batch.alora_scale;
        auto & alora_disabled_id = batch.alora_disabled_id;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.size() == 0) {
            bool add_ok = true; // false means the batch is full, skip remaining slots

            iterate(slots, [&](server_slot & slot) {
                if (!add_ok || batch.size() >= n_batch) {
                    return; // batch is full, skip remaining slots
                }

                if (!slot.is_processing()) {
                    return;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    return;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    return;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.size();

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        // Reserve room before the first prompt allocation. Purging only
                        // after decode fails can leave early batches in the cache tail,
                        // pinning the attention span at capacity for the whole prompt.
                        // This runs on the serialized server task thread; never evict
                        // a processing slot. Count our exclusive cells as reclaimable
                        // so replacing/reusing a resident prompt does not evict peers.
                        llama_memory_seq_capacity capacity = {};
                        auto mem = llama_get_memory(ctx_tgt);
                        while (params_base.kv_unified &&
                               llama_memory_get_seq_capacity(mem, slot.id, &capacity) &&
                               (uint64_t) slot.task->n_tokens() >
                                   capacity.capacity_cells - capacity.used_cells + capacity.seq_exclusive_cells) {
                            bool cleared = false;
                            for (auto & idle : slots) {
                                if (idle.is_processing() || idle.prompt.n_tokens() == 0) {
                                    continue;
                                }
                                SLT_WRN(slot, "pre-purging idle slot %d before prompt allocation\n", idle.id);
                                idle.prompt_clear();
                                cleared = true;
                                break;
                            }
                            if (!cleared) {
                                break; // active sequences retain the existing retry path
                            }
                        }

                        slot.t_start_process_prompt = ggml_time_us();
                        slot.t_start_generation = 0;

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            return;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt)) {
                            send_error(slot, "the current context does not logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            return;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                return;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }

                            if (slot.task->params.cache_prompt) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_TRC(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            slot.mem.seq_rm (slot.id, head_p, head_c);
                                            slot.mem.seq_add(slot.id, head_c, head_c + n_match, kv_shift);

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // ref: https://github.com/ggml-org/llama.cpp/pull/24110
                            const bool has_new_tokens = (n_past < slot.task->n_tokens());

                            // the largest pos_min required for a checkpoint to be useful
                            const auto pos_min_thold = std::max(0, pos_next - n_swa - (has_new_tokens ? 0 : 1));

                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - 4, 0);
                                    const int np1 = std::min<int>(n_past + 6, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                if (pos_min >= pos_min_thold) {
                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&](const auto & cur) {
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            SLT_TRC(slot, "checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
                                            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
                                            if (cur.pos_max > pos_next) {
                                                return false;
                                            }
                                            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        it->load_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        it->load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        // restore the draft's speculative state
                                        common_speculative_set_state(spec.get(), slot.id, it->data_spec);

                                        pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                                        n_past   = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) it->n_tokens);
                                        SLT_TRC(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, n_past, (float) it->size() / 1024 / 1024);
                                    }

                                    if (do_reset) {
                                        SLT_TRC(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past = 0;
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur.pos_max > pos_next) {
                                        SLT_TRC(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.n_prompt_tokens_cache = n_past;
                        slot.n_prompt_tokens_processed = 0;

                        slot.prompt.tokens.keep_first(n_past);

                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream) {
                            if (slot.task->params.return_progress) {
                                // send initial 0% progress update if needed
                                send_partial_response(slot, {}, true);
                            } else {
                                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                                send_partial_response(slot, {}, false, true);
                            }
                        }
                    } // end of SLOT_STATE_STARTED

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.size() + slot.task->n_tokens() > n_batch) {
                            return;
                        }
                    }

                    const int64_t t_now = ggml_time_us();
                    slot.t_prompt_processing = (t_now - slot.t_start_process_prompt) / 1e3;
                    slot.print_timings_pp();

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    slot.mem.seq_rm(slot.id, p0, -1);

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    // - the model supports partial sequence removal but only up to a fixed bound
                    do_checkpoint = do_checkpoint && (
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                            n_swa > 0);

                    bool has_mtmd = false;

                    // check if we should process the image
                    while (true) {
                        auto cur_token_idx = slot.prompt.n_tokens();
                        if (
                            cur_token_idx >= slot.task->n_tokens() ||
                            input_tokens[cur_token_idx] != LLAMA_TOKEN_NULL // encountered a text token
                        ) {
                            break;
                        }

                        // process the image
                        size_t n_tokens_out = 0;
                        int32_t res = slot.process_mtmd_chunk(cur_token_idx, n_tokens_out);
                        if (res != 0) {
                            SLT_ERR(slot, "failed to process image, res = %d\n", res);
                            send_error(slot, "failed to process image", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        slot.n_prompt_tokens_processed += n_tokens_out;

                        // add the image chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(cur_token_idx);
                            slot.prompt.tokens.push_back(chunk.get()); // copy
                        }

                        has_mtmd = true;
                    }

                    const auto & spans = slot.task->params.message_spans;
                    const auto last_user_pos = spans.last_user_message_pos();

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output;
                        // MTP also wants logits at every prompt position so the
                        // streaming hook can mirror t_h_nextn into ctx_dft.
                        add_ok &= batch.add(slot.id,
                            cur_tok,
                            slot.prompt.tokens.pos_next(),
                            slot.need_embd());
                        slot.prompt.tokens.push_back(cur_tok);

                        slot.n_prompt_tokens_processed++;

                        // break at the last user message, or at user messages at least min step past the last checkpoint
                        if (do_checkpoint && spans.is_user_start(slot.prompt.n_tokens())) {
                            const auto pos = slot.prompt.n_tokens();
                            const auto & checkpoints = slot.prompt.checkpoints;

                            if (pos == last_user_pos || checkpoints.empty() || pos > checkpoints.back().n_tokens + params_base.checkpoint_min_step) {
                                break;
                            }
                        }

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            static const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.size() - n_tokens_prev;

                    const auto n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                    const bool near_prompt_end = slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch;

                    const bool is_user_start = spans.is_user_start(n_tokens_start);
                    const bool is_last_user_message = n_tokens_start == last_user_pos;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.size() > 0);

                        // extract the logits only for the last token
                        batch.set_output(batch.size() - 1, true);

                        slot.n_decoded = 0;
                        slot.i_batch   = batch.size() - 1;

                        slot.init_sampler();
                    } else {
                        // skip ordinary mid-prompt checkpoints, unless the batch starts a user
                        // message or we are near the end of the prompt
                        if (!is_user_start && !near_prompt_end) {
                            do_checkpoint = false;
                        }
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);

                    // nothing to checkpoint yet
                    // TODO: is this check needed?
                    if (do_checkpoint && pos_min < 0) {
                        do_checkpoint = false;
                    }

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together, unless it's the last user message
                    do_checkpoint = do_checkpoint && (
                            slot.prompt.checkpoints.empty() ||
                            is_last_user_message || near_prompt_end ||
                            n_tokens_start > slot.prompt.checkpoints.back().n_tokens + params_base.checkpoint_min_step);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }
            });
        }
    }

    // returns true = success ; false = retry with smaller batch size
    // throw std::runtime_error on fatal error
    bool decode(int32_t & n_batch, int32_t off, llama_batch & batch_view) {
        SRV_DBG("n_batch (effective) = %d, off = %d\n", n_batch, off);

        if (batch.size() == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }

            return true; // nothing to decode
        } else {
            n_empty_consecutive = 0;
        }

        // TODO @ngxson : dft model may have different n_embd than the tgt model, so we check & reject if that's the case
        // this case is not currently used by any models, but may need to be supported in the future
        if (spec && batch.has_embd) {
            if (llama_model_n_embd_inp(model_dft) != llama_model_n_embd_inp(model_tgt)) {
                SRV_ERR("%s", "unsupported batch.has_embd + spec case\n");
                throw std::runtime_error("unsupported batch.has_embd + spec case");
            }
        }

        const int ret = llama_decode(ctx_tgt, batch_view);

        metrics.on_decoded(slots);

        if (ret != 0) {
            {
                std::string err;

                if (n_batch == 1 && ret == 1) {
                    // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                    //       need to remove the tokens from the current batch too
                    err = "Context size has been exceeded.";
                }

                if (ret == -1) {
                    err = "Invalid input batch.";
                }

                if (ret < -1) {
                    // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                    err = "Compute error.";
                }

                // TODO: handle ret == 2 (abort) when we start aborting

                if (!err.empty()) {
                    SRV_ERR("%s off = %d, n_batch = %d, ret = %d\n", err.c_str(), off, n_batch, ret);

                    for (auto & slot : slots) {
                        if (slot.is_processing()) {
                            send_error(slot, err);
                            slot.release();

                            // note: it's complicated to keep track of how much of the current batch has been
                            //       processed before the error occurred, so we simply clear the entire context
                            slot.prompt_clear();
                        }
                    }

                    // stop, do not retry with smaller batch size
                    throw std::runtime_error(err);
                }
            }

            // retry with half the batch size to try to find a free slot in the KV cache
            if (!try_clear_idle_slots()) {
                n_batch /= 2;
            }

            SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, off = %d, n_batch = %d, ret = %d\n", off, n_batch, ret);

            return false; // retry with the updated n_batch
        }

        // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
        //       for now, always re-evaluate for simplicity
        //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
        if (!common_speculative_process(spec.get(), batch_view)) {
            SRV_ERR("%s", "failed to process speculative batch\n");

            // TODO: handle error
            throw std::runtime_error("failed to process speculative batch");
        }

        // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                std::vector<server_slot *> children;
                for (auto & other : slots) {
                    if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                        children.push_back(&other);
                    }
                }

                // all children slots should already launched by launch_slots_with_parent_task()
                // copy state to the child slots
                for (auto & child : children) {
                    SLT_TRC(slot, " - copying state to child %d\n", child->id);

                    GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);

                    slot.copy_state_to(*child);
                    child->state = SLOT_STATE_DONE_PROMPT;
                }
            }
        }

        return true;
    }

    void post_decode(int32_t n_batch_tokens, int32_t off, llama_batch & batch_view) {
        // for checking if a given batch index is inside batch_view
        auto is_inside_view = [&](int32_t idx) {
            return idx >= off && idx < off + n_batch_tokens;
        };

        // TODO @ngxson : it's tricky to make sub-batch compatible with common_sampler_sample_and_accept_n,
        // so for now we will throw an error in this case: https://github.com/ggml-org/llama.cpp/issues/24840
        iterate(slots, [&](server_slot & slot) {
            for (auto & i : slot.spec_i_batch) {
                if (!is_inside_view(i)) {
                    throw std::runtime_error(string_format("speculative batch index %d is not inside the current sub-batch [%d, %d)", i, off, off + n_batch_tokens));
                }
            }
        });

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        iterate(slots, [&](server_slot & slot) {
            // optionally send prompt processing progress
            if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->params.stream && slot.task->params.return_progress) {
                    send_partial_response(slot, {}, true);
                }
            }

            if (!is_inside_view(slot.i_batch)) {
                // the required token not in this sub-batch, skip
                return;
            }

            if (slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                    // prompt evaluated for embedding
                    send_embedding(slot, batch_view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                    send_rerank(slot, batch_view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                GGML_ASSERT(slot.task->need_sampling());

                // prompt evaluated for next-token prediction
                slot.state = SLOT_STATE_GENERATING;

                if (slot.can_speculate()) {
                    common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                }
#ifdef LLAMA_EXP_SLOT_DIAG
                // same point for a native prefill and for restore + tail: n_prompt_tokens_cache = tokens reused (not reprocessed)
                if (slot_diag_on()) {
                    fprintf(stderr, "DIAG post-prefill slot=%d n_prompt=%d n_reused=%d n_processed=%d can_speculate=%d\n", slot.id,
                            (int) slot.prompt.n_tokens(), (int) slot.n_prompt_tokens_cache, (int) slot.n_prompt_tokens_processed,
                            (int) slot.can_speculate());
                    g_diag_gen_hash[slot.id & 255] = 1469598103934665603ull;
                    g_diag_round[slot.id & 255]    = 0;
                    slot_diag_state("post-prefill", ctx_tgt, slot.ctx_dft, spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                }
#endif
            } else if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            if (slot.can_speculate() && !slot.spec_draft.empty()) {
                return; // sample using speculative decoding
            }

            // shifted according to the current sub-batch
            const int tok_idx = slot.i_batch - off;

            llama_token id;
            {
                scoped_timer timer(t_sampl, n_sampl);
                id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);
            }

            slot.i_batch = -1;

            common_sampler_accept(slot.smpl.get(), id, true);

            // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
            const int64_t t_now = ggml_time_us();

            slot.n_decoded += 1;

            if (slot.n_decoded == 1) {
                slot.t_start_generation = t_now;
                slot.t_print_last = t_now;
                slot.n_decoded_last = 0;
                slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                metrics.on_prompt_eval(slot);
            }

            slot.t_token_generation = std::max<int64_t>(1, t_now - slot.t_start_generation) / 1e3;

            completion_token_output result;
            result.tok          = id;
            result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
            result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

            if (slot.task->params.sampling.n_probs > 0) {
                populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
            }

            if (!process_token(result, slot)) {
                // release slot because of stop condition
                slot.print_timings();
                send_final_response(slot);
                metrics.on_prediction(slot);
                slot.release();

                return;
            }

            slot.print_timings_tg();
        });

        // speculative decoding - main model sample and accept
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() || slot.spec_draft.empty()) {
                return;
            }

            // save the original draft size
            const size_t n_draft = slot.spec_draft.size();

            GGML_ASSERT(n_draft > 0);

            // verify and try to accept the draft
            {
                // save the sampler sampler state in case we need to restore it
                common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));

                GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
                auto accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft);
                slot.spec_i_batch.clear();

#ifdef LLAMA_KV_ADMISSION_CPU_TEST_DRIVER
                this->kv_admission_test_observe("after_verify_accept", slot.id, n_draft, accepted.size() - 1);
#endif

                GGML_ASSERT(accepted.size() >= 1);

                const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();

                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_rollback > llama_n_rs_seq(ctx_tgt));

#ifdef LLAMA_KV_ADMISSION_CPU_TEST_DRIVER
                this->kv_admission_test_observe("before_rollback", slot.id, n_draft, accepted.size() - 1);
                SRV_INF("KV_ADMISSION_TEST mtp_rollback slot=%d draft=%zu accepted=%zu rollback=%u recurrent_groups=%u n_rs_seq=%u checkpoint_tgt=%d\n",
                        slot.id, n_draft, accepted.size() - 1, n_rollback,
                        llama_n_rs_seq(ctx_tgt) + 1, llama_n_rs_seq(ctx_tgt), (int) use_ckpt_tgt);
#endif

                // check for partial draft acceptance
                if (n_rollback > 0) {
                    if (use_ckpt_tgt) {
#ifdef LLAMA_EXP_SLOT_DIAG
                        slot_diag_round("rollback", slot.id, slot.prompt.n_tokens(), n_draft, accepted.size() - 1, slot.spec_is_replay,
                                llama_tokens(accepted.begin(), accepted.end()));
#endif
                        if (trace > 0) {
                            SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", accepted.size() - 1, slot.spec_draft.size());
                        }

                        // partial acceptance is not supported by the context -> truncate the draft and restore the state
                        slot.spec_is_replay = true;
                        slot.spec_draft = std::move(accepted);

                        const auto & ckpt = slot.spec_ckpt;

                        SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", ckpt.pos_min, ckpt.pos_max, ckpt.size());

                        ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                        if (slot.ctx_dft) {
                            ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }

                        slot.mem.seq_rm(slot.id, ckpt.pos_max + 1, -1);

                        slot.prompt.tokens.keep_first(ckpt.n_tokens);
                        slot.smpl = std::move(smpl_save);

                        return;
                    }
                }

                if (trace > 0) {
                    SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
                }

                common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);

                slot.spec_draft = std::move(accepted);
            }

            const int64_t t_now = ggml_time_us();

            const auto ids = std::move(slot.spec_draft);

#ifdef LLAMA_EXP_SLOT_DIAG
            slot_diag_round("round", slot.id, slot.prompt.n_tokens(), n_draft, ids.size() - 1, slot.spec_is_replay,
                    llama_tokens(ids.begin(), ids.end()));
#endif
            size_t n_accepted = ids.size() - 1;
            if (slot.spec_is_replay && n_accepted > 0) {
                n_accepted--;
            }
            slot.spec_is_replay = false;

            slot.t_token_generation = std::max<int64_t>(1, t_now - slot.t_start_generation) / 1e3;

            // update how many tokens out of those tested were accepted
            slot.n_draft_accepted += n_accepted;
            slot.n_draft_verif_steps += 1;

            if (slot.n_accepted_per_pos.empty()) {
                slot.n_accepted_per_pos.resize(common_speculative_n_max(&params_base.speculative), 0);
            }
            for (size_t i = 0; i < n_accepted && i < slot.n_accepted_per_pos.size(); ++i) {
                slot.n_accepted_per_pos[i]++;
            }

            // add accepted tokens to the prompt
            slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
            slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

            slot.sampled = ids.back(); // last accepted token
            SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

            slot.mem.seq_rm(slot.id, slot.prompt.tokens.pos_next(), -1);

            for (size_t i = 0; i < ids.size(); ++i) {
                completion_token_output result;

                result.tok          = ids[i];
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // set later

                // TODO: set result.probs

                slot.n_decoded += 1;

                if (!process_token(result, slot)) {
                    slot.print_timings();
                    send_final_response(slot);
                    metrics.on_prediction(slot);
                    slot.release();

                    return;
                }
            }

            slot.print_timings_tg();

            SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) n_accepted, (int) n_draft, slot.prompt.n_tokens());
        });
    }

    int get_slot_n_ctx() {
        return slots.back().n_ctx;
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

#ifdef LLAMA_KV_ADMISSION_CPU_TEST_DRIVER
bool server_context::test_only_populate_draft_sequence(int32_t seq_id, int32_t n_tokens) {
    if (seq_id < 0) return false;
    if (!impl->ctx_dft) {
        if (!impl->model_tgt || !llama_model_is_hybrid(impl->model_tgt)) return false;
        llama_memory_resource_capacity raw = {};
        if (!llama_memory_get_resource_capacity(llama_get_memory(impl->ctx_tgt), seq_id, &raw, sizeof(raw)) ||
                !raw.attention_known || !raw.recurrent_known || raw.recurrent_rollback_groups == 0 ||
                raw.recurrent_capacity_rows % raw.recurrent_rollback_groups != 0) return false;
        SRV_INF("KV_ADMISSION_TEST fixture role=target_hybrid seq=%d attention_capacity=%u recurrent_capacity_bundles=%u groups=%u pressure=live_owner_only\n",
                seq_id, raw.attention_capacity_cells, raw.recurrent_capacity_rows / raw.recurrent_rollback_groups,
                raw.recurrent_rollback_groups);
        return raw.recurrent_capacity_rows / raw.recurrent_rollback_groups == 2;
    }
    auto * memory = llama_get_memory(impl->ctx_dft);
    if (!llama_memory_seq_rm(memory, seq_id, -1, -1)) return false;
    if (n_tokens < 0) {
        llama_memory_resource_capacity raw = {};
        if (!llama_memory_get_resource_capacity(memory, seq_id, &raw, sizeof(raw)) ||
                !raw.attention_known || raw.attention_used_cells >= raw.attention_capacity_cells) return false;
        n_tokens = (int32_t) std::min<uint64_t>(INT32_MAX,
                raw.attention_capacity_cells - raw.attention_used_cells - 1);
    }
    if (n_tokens <= 0) return false;
    constexpr int32_t chunk_size = 128;
    for (int32_t base = 0; base < n_tokens; base += chunk_size) {
        const int32_t count = std::min(chunk_size, n_tokens - base);
        llama_batch batch = llama_batch_init(count, 0, 1);
        if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
            llama_batch_free(batch);
            return false;
        }
        batch.n_tokens = count;
        for (int32_t i = 0; i < count; ++i) {
            const int32_t index = base + i;
            batch.token[index - base] = 19;
            batch.pos[index - base] = index;
            batch.n_seq_id[index - base] = 1;
            batch.seq_id[index - base][0] = seq_id;
            batch.logits[index - base] = false;
        }
        const int32_t status = llama_decode(impl->ctx_dft, batch);
        llama_batch_free(batch);
        if (status != 0) return false;
    }
    llama_synchronize(impl->ctx_dft);
    if (kv_admission_cpu_test_env("LLAMA_KV_ADMISSION_TEST_TRACE")) {
        SRV_INF("KV_ADMISSION_TEST fixture role=draft seq=%d decoded_tokens=%d\n", seq_id, n_tokens);
    }
    return true;
}

bool server_context::test_only_hybrid_draft_guard() {
    if (!impl->model_tgt || !impl->ctx_tgt || !llama_model_is_hybrid(impl->model_tgt)) return false;
    const auto saved_types = impl->params_base.speculative.types;
    impl->params_base.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE};
    const std::string draft_reason = impl->kv_admission_static_reason();
    impl->params_base.speculative.types = {COMMON_SPECULATIVE_TYPE_NONE};
    const std::string no_draft_reason = impl->kv_admission_static_reason();
    impl->params_base.speculative.types = saved_types;
    const bool pass = draft_reason == "hybrid_target_with_draft_unsupported" && no_draft_reason.empty();
    std::fprintf(stderr, "KV_ADMISSION_TEST gate=A branch=hybrid_target_with_draft_unsupported expected=hybrid_target_with_draft_unsupported actual=%s control=hybrid_no_draft expected=accepted actual=%s pass=%d\n",
        draft_reason.c_str(), no_draft_reason.empty() ? "accepted" : no_draft_reason.c_str(), (int) pass);
    return pass;
}

bool server_context::test_only_mtp_hybrid_path() {
    const bool actual_context_map = impl->kv_admission_mtp_hybrid();
    const auto saved_types = impl->params_base.speculative.types;
    impl->params_base.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE};
    const std::string simple_reason = impl->kv_admission_static_reason();
    const auto saved_ctx_dft = impl->ctx_dft;
    auto saved_spec = std::move(impl->spec);
    const auto saved_model_dft = impl->model_dft;
    impl->ctx_dft = nullptr;
    impl->model_dft = nullptr;
    impl->params_base.speculative.types = {COMMON_SPECULATIVE_TYPE_NONE};
    const std::string no_draft_reason = impl->kv_admission_static_reason();
    impl->ctx_dft = saved_ctx_dft;
    impl->spec = std::move(saved_spec);
    impl->model_dft = saved_model_dft;
    impl->params_base.speculative.types = saved_types;
    const bool simple_guard = simple_reason == "hybrid_target_with_draft_unsupported";
    const bool no_draft_control = no_draft_reason.empty();
    const bool saved_enabled = impl->params_base.kv_admission;
    impl->params_base.kv_admission = false;
    server_task off_task;
    server_slot off_slot;
    std::string off_reason;
    error_type off_error = ERROR_TYPE_SERVER;
    const bool off_bypass = impl->kv_admission_preflight(off_task, off_slot, {}, off_reason, off_error);
    impl->params_base.kv_admission = saved_enabled;
    const std::string reason = impl->kv_admission_static_reason();
    const bool supported = actual_context_map && reason.empty() && impl->kv_admission_layout_supported &&
        simple_guard && no_draft_control && off_bypass;
    std::fprintf(stderr, "KV_ADMISSION_TEST gate=MTP-layout target_ctx=%p target_mem=%p target_model=%p draft_ctx=%p draft_mem=%p draft_model=%p ctx_other=%p target_groups=%u draft_groups=%u supported=%d reason=%s\n",
            (void *)impl->ctx_tgt, (void *)llama_get_memory(impl->ctx_tgt), (const void *)llama_get_model(impl->ctx_tgt),
            (void *)impl->ctx_dft, (void *)llama_get_memory(impl->ctx_dft), (const void *)llama_get_model(impl->ctx_dft),
            (void *)llama_get_ctx_other(impl->ctx_dft), llama_n_rs_seq(impl->ctx_tgt) + 1,
            llama_n_rs_seq(impl->ctx_dft) + 1, (int)supported, reason.c_str());
    std::fprintf(stderr, "KV_ADMISSION_TEST regression=eligibility simple_refused=%d no_draft_accepted=%d admission_off_bypass=%d\n",
            (int)simple_guard, (int)no_draft_control, (int)off_bypass);
    return supported;
}

bool server_context::test_only_recurrent_getter_cases() {
    if (!impl->ctx_tgt || !impl->model_tgt || !llama_model_is_hybrid(impl->model_tgt)) return false;
    auto * memory = llama_get_memory(impl->ctx_tgt);
    llama_batch batch = llama_batch_init(1, 0, 1);
    if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
        llama_batch_free(batch);
        return false;
    }
    batch.n_tokens = 1;
    batch.token[0] = 19;
    batch.pos[0] = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = false;
    const int decode = llama_decode(impl->ctx_tgt, batch);
    llama_batch_free(batch);
    if (decode != 0) return false;
    llama_synchronize(impl->ctx_tgt);
    llama_memory_resource_capacity exclusive = {};
    if (!llama_memory_get_resource_capacity(memory, 0, &exclusive, sizeof(exclusive)) ||
            !exclusive.recurrent_known || exclusive.recurrent_rollback_groups == 0 ||
            exclusive.recurrent_seq_exclusive_rows != exclusive.recurrent_rollback_groups) return false;
    llama_memory_seq_cp(memory, 0, 1, 0, -1);
    llama_memory_resource_capacity shared0 = {}, shared1 = {};
    const bool shared_ok = llama_memory_get_resource_capacity(memory, 0, &shared0, sizeof(shared0)) &&
        llama_memory_get_resource_capacity(memory, 1, &shared1, sizeof(shared1)) &&
        shared0.recurrent_known && shared1.recurrent_known &&
        shared0.recurrent_seq_exclusive_rows == 0 && shared1.recurrent_seq_exclusive_rows == 0;
    const bool removed = llama_memory_seq_rm(memory, 1, -1, -1);
    llama_memory_resource_capacity exclusive_again = {};
    const bool exclusive_ok = removed && llama_memory_get_resource_capacity(memory, 0, &exclusive_again, sizeof(exclusive_again)) &&
        exclusive_again.recurrent_known &&
        exclusive_again.recurrent_seq_exclusive_rows == exclusive_again.recurrent_rollback_groups;
    // CPU fixture only, before the dispatcher starts: duplicate a sequence owner
    // across two cells. No valid public API can create this inconsistent tail map.
    const size_t state_size = llama_state_seq_get_size(impl->ctx_tgt, 0);
    std::vector<uint8_t> before_state(state_size);
    if (llama_state_seq_get_data(impl->ctx_tgt, before_state.data(), state_size, 0) != state_size) return false;
    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory);
    if (!hybrid) return false;
    auto * recurrent = hybrid->get_mem_recr();
    if (!recurrent) return false;
    size_t victim = recurrent->cells.size();
    for (size_t i = 0; i < recurrent->cells.size(); ++i) {
        if (recurrent->cells[i].is_empty() && (int32_t)i != recurrent->cells[0].tail) { victim = i; break; }
    }
    if (victim == recurrent->cells.size()) return false;
    const auto saved_cell = recurrent->cells[victim];
    const uint32_t saved_used = recurrent->used;
    const auto saved_ledger_size = impl->kv_admission_planner.size();
    recurrent->cells[victim].pos = 0;
    recurrent->cells[victim].seq_id.insert(0);
    ++recurrent->used;
    llama_memory_resource_capacity inconsistent = {};
    const bool inconsistent_metadata_rejected =
        !llama_memory_get_resource_capacity(memory, 0, &inconsistent, sizeof(inconsistent)) &&
        !inconsistent.attention_known && !inconsistent.recurrent_known;
    std::vector<kvplan::Snapshot> bad_snapshots;
    server_context_impl::kv_admission_slot_resources bad_resources;
    std::string bad_reason;
    const bool producer_unknown_rejected = !impl->kv_admission_snapshots(bad_snapshots, bad_resources, bad_reason) &&
        bad_reason == "unknown_memory_resource_snapshot" && bad_snapshots.empty() && bad_resources.empty() &&
        impl->kv_admission_planner.size() == saved_ledger_size;
    recurrent->cells[victim] = saved_cell;
    recurrent->used = saved_used;
    std::vector<uint8_t> after_state(llama_state_seq_get_size(impl->ctx_tgt, 0));
    const bool state_restored = after_state.size() == before_state.size() &&
        llama_state_seq_get_data(impl->ctx_tgt, after_state.data(), after_state.size(), 0) == after_state.size() &&
        before_state == after_state;
    const bool cleared = llama_memory_seq_rm(memory, 0, -1, -1);
    llama_memory_resource_capacity bad_query = {};
    const bool invalid_id_rejected = !llama_memory_get_resource_capacity(memory, INT32_MAX, &bad_query, sizeof(bad_query));
    std::fprintf(stderr, "KV_ADMISSION_TEST gate=B-native shared=%d exclusive=%d invalid_query_rejected=%d inconsistent_metadata_rejected=%d producer_unknown_rejected=%d state_restored=%d groups=%u shared_rows=%u\n",
        (int)shared_ok, (int)exclusive_ok, (int)invalid_id_rejected, (int)inconsistent_metadata_rejected, (int)producer_unknown_rejected, (int)state_restored, exclusive.recurrent_rollback_groups,
        shared0.recurrent_seq_rows);
    return shared_ok && exclusive_ok && invalid_id_rejected && inconsistent_metadata_rejected && producer_unknown_rejected && state_restored && cleared;
}

bool server_context::test_only_clear_draft_sequence(int32_t seq_id) {
    if (seq_id < 0) return false;
    if (!impl->ctx_dft) return impl->model_tgt && llama_model_is_hybrid(impl->model_tgt);
    return llama_memory_seq_rm(llama_get_memory(impl->ctx_dft), seq_id, -1, -1);
}

void server_context::test_only_synchronize_contexts() const {
    if (impl->ctx_tgt) llama_synchronize(impl->ctx_tgt);
    if (impl->ctx_dft) llama_synchronize(impl->ctx_dft);
}
#endif // LLAMA_KV_ADMISSION_CPU_TEST_DRIVER

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    const char * ftype_name = llama_ftype_name(llama_model_ftype(impl->model_tgt));

    return server_context_meta {
        /* build_info             */ std::string(llama_build_info()),
        /* model_name             */ impl->model_name,
        /* model_aliases          */ impl->model_aliases,
        /* model_tags             */ impl->model_tags,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr,
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* has_inp_video          */ impl->chat_params.allow_video,
        /* json_ui_settings       */ impl->json_ui_settings,
        /* slot_n_ctx             */ impl->get_slot_n_ctx(),
        /* pooling_type           */ llama_pooling_type(impl->ctx_tgt),
        /* kv_admission           */ impl->kv_admission_descriptor(),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token          */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token          */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token          */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog         */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params         */ llama_model_n_params(impl->model_tgt),
        /* model_size             */ llama_model_size(impl->model_tgt),
        /* model_ftype            */ ftype_name,
    };
}

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_res_spipe {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::set_state_callback(server_state_callback_t callback) {
    impl->callback_state = std::move(callback);
    impl->queue_tasks.on_sleeping_state([this](bool sleeping) {
        if (sleeping) {
            impl->callback_state(SERVER_STATE_SLEEPING, {});
        }
        // for sleeping == false, event is emitted by load_model()
    });
}

//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    res->set_req(&req); // will also set spipe if needed

    int32_t sse_ping_interval = params.sse_ping_interval;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) / string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        // message delimiters for checkpointing
        auto delimiters = common_chat_msg_delimiters_parse(json_value(data, "message_delimiters", json::array()));
        delimiters.tokenize(ctx_server.vocab);

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_schema::eval_llama_cmpl_schema(
                    ctx_server.vocab,
                    params,
                    meta->slot_n_ctx,
                    meta->logit_bias_eog,
                    data);

            task.params.message_spans = task.tokens.find_message_spans(delimiters);

            task.id_slot = json_value(data, "id_slot", -1);
            sse_ping_interval = task.params.sse_ping_interval;

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->set_next([res_this = res.get(), res_type, sse_ping_interval](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            auto effective_should_stop = [&res_this]() {
                return res_this->should_stop();
            };

            try {
                if (effective_should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &start_time, sse_ping_interval, &effective_should_stop]() {
                    if (effective_should_stop()) {
                        return true; // should_stop condition met
                    } else if (sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)sse_ping_interval * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(effective_should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);
        if (!res_task->kv_admission_test_snapshot.is_null()) {
            res->headers["X-KV-Admission-Test-Snapshot"] = res_task->kv_admission_test_snapshot.dump();
        }

        // metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
        json all_metrics_def = json {
            {"counter", {{
                    {"name",  "prompt_tokens_total"},
                    {"help",  "Number of prompt tokens processed."},
                    {"value",  (uint64_t) res_task->n_prompt_tokens_processed_total}
            }, {
                    {"name",  "prompt_seconds_total"},
                    {"help",  "Prompt process time"},
                    {"value",  (uint64_t) res_task->t_prompt_processing_total / 1.e3}
            }, {
                    {"name",  "tokens_predicted_total"},
                    {"help",  "Number of generation tokens processed."},
                    {"value",  (uint64_t) res_task->n_tokens_predicted_total}
            }, {
                    {"name",  "tokens_predicted_seconds_total"},
                    {"help",  "Predict process time"},
                    {"value",  (uint64_t) res_task->t_tokens_generation_total / 1.e3}
            }, {
                    {"name",  "n_decode_total"},
                    {"help",  "Total number of llama_decode() calls"},
                    {"value",  res_task->n_decode_total}
            }, {
                    {"name",  "n_tokens_max"},
                    {"help",  "Largest observed n_tokens."},
                    {"value",  res_task->n_tokens_max}
            }}},
            {"gauge", {{
                    {"name",  "prompt_tokens_seconds"},
                    {"help",  "Average prompt throughput in tokens/s."},
                    {"value",  res_task->n_prompt_tokens_processed ? 1.e3 / res_task->t_prompt_processing * res_task->n_prompt_tokens_processed : 0.}
            },{
                    {"name",  "predicted_tokens_seconds"},
                    {"help",  "Average generation throughput in tokens/s."},
                    {"value",  res_task->n_tokens_predicted ? 1.e3 / res_task->t_tokens_generation * res_task->n_tokens_predicted : 0.}
            },{
                    {"name",  "requests_processing"},
                    {"help",  "Number of requests processing."},
                    {"value",  (uint64_t) res_task->n_processing_slots}
            },{
                    {"name",  "requests_deferred"},
                    {"help",  "Number of requests deferred."},
                    {"value",  (uint64_t) res_task->n_tasks_deferred}
            },{
                    {"name",  "n_busy_slots_per_decode"},
                    {"help",  "Average number of busy slots per llama_decode() call"},
                    {"value",  (float) res_task->n_busy_slots_total / std::max((float) res_task->n_decode_total, 1.f)}
            }}}
        };

        std::stringstream prometheus;

        for (const auto & el : all_metrics_def.items()) {
            const auto & type        = el.key();
            const auto & metrics_def = el.value();

            for (const auto & metric_def : metrics_def) {
                const std::string name = metric_def.at("name");
                const std::string help = metric_def.at("help");

                auto value = json_value(metric_def, "value", 0.);
                prometheus << "# HELP llamacpp:" << name << " " << help  << "\n"
                            << "# TYPE llamacpp:" << name << " " << type  << "\n"
                            << "llamacpp:"        << name << " " << value << "\n";
            }
        }

        res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->t_start);
        res->content_type = "text/plain; version=0.0.4";
        res->status = 200;
        res->data = prometheus.str();
        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto * res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);
        if (!res_task->kv_admission_test_snapshot.is_null()) {
            res->headers["X-KV-Admission-Test-Snapshot"] = res_task->kv_admission_test_snapshot.dump();
        }

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->slots_data);
        return res;
    };

    this->get_slot_operations = [this](const server_http_req &) {
        auto res = create_response(true);
        if (params.slot_save_path.empty()) {
            res->error(json{{"code", 501}, {"message", "slot operation protocol is disabled"}});
            return res;
        }
        if (params.slot_operation_owner.empty() || params.slot_operation_engine_id.empty()) {
            res->error(json{{"code", 501}, {"message", "slot operation owner or engine ID is not configured"}});
            return res;
        }
        auto capability = ctx_server.slot_operations.capability();
        capability["b3_restore"] = ctx_server.b3_restore_metrics();
        res->ok(capability);
        return res;
    };

    this->get_slot_operation = [this](const server_http_req & req) {
        auto res = create_response(true);
        slot_operation_record op;
        if (!ctx_server.slot_operations.get(req.get_param("op_id"), op)) {
            res->error(json{{"code", 404}, {"message", "unknown slot operation"}});
            return res;
        }
        res->ok(op.to_json());
        return res;
    };

    this->post_slot_operation_ack = [this](const server_http_req & req) {
        auto res = create_response(true);
        try {
            const json body = json::parse(req.body);
            const std::string engine_id = body.at("engine_id").get<std::string>();
            const std::string owner = body.at("owner").get<std::string>();
            const std::string epoch = body.at("engine_epoch").get<std::string>();
            const uint64_t seq = body.at("owner_seq").get<uint64_t>();
            if (seq == 0) {
                res->error(json{{"code", 400}, {"message", "owner_seq must be positive"}});
                return res;
            }
            if (owner != ctx_server.slot_operations.owner_name() || engine_id != ctx_server.slot_operations.configured_engine_id) {
                res->error(json{{"code", 409}, {"message", "engine_id or owner does not match this configured endpoint"}});
                return res;
            }
            const auto code = ctx_server.slot_operations.ack(owner, req.get_param("op_id"), epoch, seq);
            if (code == slot_operation_registry::ack_code::accepted || code == slot_operation_registry::ack_code::repeated || code == slot_operation_registry::ack_code::old_epoch) {
                res->status = 204;
                res->data.clear();
            } else if (code == slot_operation_registry::ack_code::nonterminal) {
                res->error(json{{"code", 409}, {"message", "operation is not terminal"}});
            } else {
                res->error(json{{"code", 404}, {"message", "unknown slot operation"}});
            }
        } catch (const std::exception & e) {
            res->error(json{{"code", 400}, {"message", e.what()}});
        }
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        task_params tparams;
        tparams.sampling = params.sampling;
        json default_generation_settings_for_props = json {
            { "params", tparams.to_json(true) },
            { "n_ctx",  meta->slot_n_ctx },
        };

        std::string tmpl_default = common_chat_templates_source(meta->chat_params.tmpls.get(), "");
        std::string tmpl_tools   = common_chat_templates_source(meta->chat_params.tmpls.get(), "tool_use");

        json props = {
            { "default_generation_settings", default_generation_settings_for_props },
            { "total_slots",                 params.n_parallel },
            { "model_alias",                 meta->model_name },
            { "model_ftype",                 meta->model_ftype },
            { "model_path",                  meta->model_path },
            { "modalities",                  json {
                {"vision", meta->has_inp_image},
                {"video",  meta->has_inp_video},
                {"audio",  meta->has_inp_audio},
            } },
            { "media_marker",                get_media_marker() },
            { "endpoint_slots",              params.endpoint_slots },
            { "endpoint_props",              params.endpoint_props },
            { "endpoint_metrics",            params.endpoint_metrics },
            { "ui",                          params.ui },
            { "ui_settings",                 meta->json_ui_settings },
            { "chat_template",               tmpl_default },
            { "chat_template_caps",          meta->chat_template_caps },
            { "bos_token",                   meta->bos_token_str },
            { "eos_token",                   meta->eos_token_str },
            { "build_info",                  meta->build_info },
            { "kv_admission",                meta->kv_admission },
            { "is_sleeping",                 queue_tasks.is_sleeping() },
            { "cors_proxy_enabled",          params.ui_mcp_proxy },
        };
        if (params.use_jinja) {
            if (!tmpl_tools.empty()) {
                props["chat_template_tool_use"] = tmpl_tools;
            }
        }
        res->ok(props);
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = action;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        json models = {
            {"models", {
                {
                    {"name",  meta->model_name},
                    {"model", meta->model_name},
                    {"modified_at", ""},
                    {"size", ""},
                    {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                    {"type", "model"},
                    {"description", ""},
                    {"tags", {""}},
                    {"capabilities", meta->has_mtmd ? json({"completion","multimodal"}) : json({"completion"})},
                    {"parameters", ""},
                    {"details", {
                        {"parent_model", ""},
                        {"format", "gguf"},
                        {"family", ""},
                        {"families", {""}},
                        {"parameter_size", ""},
                        {"quantization_level", ""}
                    }}
                }
            }},
            {"object", "list"},
            {"data", {
                get_model_info(),
            }}
        };

        res->ok(models);
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens");
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query, documents[i]);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

json server_routes::get_model_info() const {
    return json {
        {"id",       meta->model_name},
        {"aliases",  meta->model_aliases},
        {"tags",     meta->model_tags},
        {"object",   "model"},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta->model_vocab_type},
            {"n_vocab",     meta->model_vocab_n_tokens},
            {"n_ctx",       meta->slot_n_ctx},
            {"n_ctx_train", meta->model_n_ctx_train},
            {"n_embd",      meta->model_n_embd_inp},
            {"n_params",    meta->model_n_params},
            {"size",        meta->model_size},
            {"ftype",       meta->model_ftype},
        }},
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    json request_data;
    std::string filename, op_id, engine_epoch, owner, engine_id;
    uint64_t owner_seq = 0;
    try {
        request_data = json::parse(req.body);
        filename = request_data.at("filename").get<std::string>();
        op_id = request_data.at("op_id").get<std::string>();
        owner = request_data.at("owner").get<std::string>();
        engine_id = request_data.at("engine_id").get<std::string>();
        engine_epoch = request_data.at("engine_epoch").get<std::string>();
        owner_seq = request_data.at("owner_seq").get<uint64_t>();
    } catch (const std::exception & e) {
        res->error(json{{"code", 400}, {"message", e.what()}});
        return res;
    }
    if (owner != ctx_server.slot_operations.owner_name() || engine_id != ctx_server.slot_operations.configured_engine_id) {
        res->error(json{{"code", 409}, {"message", "engine_id or owner does not match this configured endpoint"}});
        return res;
    }
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    slot_operation_record op;
    const std::string payload = json{{"action", "save"}, {"slot", id_slot}, {"filename", filename}}.dump();
    const auto submit = ctx_server.slot_operations.submit(op_id, owner_seq, engine_epoch, engine_id, "save", id_slot, filename, payload, op);
    if (submit == slot_operation_registry::submit_code::conflict) {
        res->error(json{{"code", 409}, {"message", "stale sequence, epoch, owner, op_id, or payload conflict"}});
        return res;
    }
    if (submit == slot_operation_registry::submit_code::overloaded) {
        res->error(json{{"code", 503}, {"message", "slot operation capacity reached; retry after ACK"}});
        return res;
    }
    if (submit == slot_operation_registry::submit_code::exact_retry) {
        res->ok(op.to_json());
        return res;
    }

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        task.slot_action.operation_id = op_id;
        task.slot_action.owner_seq = owner_seq;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    slot_operation_record final_op;
    if (!ctx_server.slot_operations.get(op_id, final_op)) {
        res->error(json{{"code", 500}, {"message", "slot operation result was lost"}});
        return res;
    }
    res->ok(final_op.to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    json request_data;
    std::string filename, op_id, engine_epoch, owner, engine_id;
    uint64_t owner_seq = 0;
    try {
        request_data = json::parse(req.body);
        filename = request_data.at("filename").get<std::string>();
        op_id = request_data.at("op_id").get<std::string>();
        owner = request_data.at("owner").get<std::string>();
        engine_id = request_data.at("engine_id").get<std::string>();
        engine_epoch = request_data.at("engine_epoch").get<std::string>();
        owner_seq = request_data.at("owner_seq").get<uint64_t>();
    } catch (const std::exception & e) {
        res->error(json{{"code", 400}, {"message", e.what()}});
        return res;
    }
    if (owner != ctx_server.slot_operations.owner_name() || engine_id != ctx_server.slot_operations.configured_engine_id) {
        res->error(json{{"code", 409}, {"message", "engine_id or owner does not match this configured endpoint"}});
        return res;
    }
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    slot_operation_record op;
    const std::string payload = json{{"action", "restore"}, {"slot", id_slot}, {"filename", filename}}.dump();
    const auto submit = ctx_server.slot_operations.submit(op_id, owner_seq, engine_epoch, engine_id, "restore", id_slot, filename, payload, op);
    if (submit == slot_operation_registry::submit_code::conflict) {
        res->error(json{{"code", 409}, {"message", "stale sequence, epoch, owner, op_id, or payload conflict"}});
        return res;
    }
    if (submit == slot_operation_registry::submit_code::overloaded) {
        res->error(json{{"code", 503}, {"message", "slot operation capacity reached; retry after ACK"}});
        return res;
    }
    if (submit == slot_operation_registry::submit_code::exact_retry) {
        res->ok(op.to_json());
        return res;
    }

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        task.slot_action.operation_id = op_id;
        task.slot_action.owner_seq = owner_seq;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    slot_operation_record final_op;
    if (!ctx_server.slot_operations.get(op_id, final_op)) {
        res->error(json{{"code", 500}, {"message", "slot operation result was lost"}});
        return res;
    }
    res->ok(final_op.to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    std::string op_id, engine_epoch, owner, engine_id;
    uint64_t owner_seq = 0;
    try {
        const json body = json::parse(req.body);
        op_id = body.at("op_id").get<std::string>();
        owner = body.at("owner").get<std::string>();
        engine_id = body.at("engine_id").get<std::string>();
        engine_epoch = body.at("engine_epoch").get<std::string>();
        owner_seq = body.at("owner_seq").get<uint64_t>();
    } catch (const std::exception & e) {
        res->error(json{{"code", 400}, {"message", e.what()}});
        return res;
    }
    if (owner != ctx_server.slot_operations.owner_name() || engine_id != ctx_server.slot_operations.configured_engine_id) {
        res->error(json{{"code", 409}, {"message", "engine_id or owner does not match this configured endpoint"}});
        return res;
    }
    slot_operation_record op;
    const std::string payload = json{{"action", "erase"}, {"slot", id_slot}}.dump();
    const auto submit = ctx_server.slot_operations.submit(op_id, owner_seq, engine_epoch, engine_id, "erase", id_slot, "", payload, op);
    if (submit == slot_operation_registry::submit_code::conflict) {
        res->error(json{{"code", 409}, {"message", "stale sequence, epoch, owner, op_id, or payload conflict"}});
        return res;
    }
    if (submit == slot_operation_registry::submit_code::overloaded) {
        res->error(json{{"code", 503}, {"message", "slot operation capacity reached; retry after ACK"}});
        return res;
    }
    if (submit == slot_operation_registry::submit_code::exact_retry) {
        res->ok(op.to_json());
        return res;
    }
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        task.slot_action.operation_id = op_id;
        task.slot_action.owner_seq = owner_seq;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    slot_operation_record final_op;
    if (!ctx_server.slot_operations.get(op_id, final_op)) {
        res->error(json{{"code", 500}, {"message", "slot operation result was lost"}});
        return res;
    }
    res->ok(final_op.to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize");
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const llama_vocab * vocab, mtmd_context * mctx, const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = json::parse(req.body);
    bool is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            } break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                body = server_chat_convert_responses_to_chatcmpl(body);
            } break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = server_chat_convert_anthropic_to_oai(body);
            } break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
    json prompt = body_parsed.at("prompt");
    // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

    // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
    size_t n_tokens;
    if (mctx != nullptr) {
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        n_tokens = process_mtmd_prompt(mctx, prompt.get<std::string>(), files, true).size();
    } else {
        n_tokens = tokenize_mixed(vocab, prompt, true, true).size();
    }

    json response = {{"input_tokens", static_cast<int64_t>(n_tokens)}};
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}
