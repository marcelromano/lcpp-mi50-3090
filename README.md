# lcpp-mi50-3090

Fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) tuned for two GPUs: **NVIDIA RTX 3090 (sm86)** and **AMD Instinct MI50 (gfx906)**. Licensed under MIT, like upstream. Part of the kernels was developed with the help of AI agents.

## Patches on top of upstream

| Patch | Targets | Origin |
|---|---|---|
| gfx906 `mul_mat_vec_q` decode kernels (lane-packed rows, DPP reductions, i-quant dot products) | MI50 | Original |
| Fused decode-time glue kernels (norm + q8_1 quantize, residual add, gates, SwiGLU, top-k MoE) | MI50 | Original |
| Conv-state step fused into one kernel (concat + conv + SiLU) | MI50 / 3090 | Inspired by the fused conv-state step in [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) |
| GQA-shared flash-attention decode kernel for quantized KV | MI50 | Original |
| Quantized-KV GQA flash-attention with int8/f16 MMA, split-K | 3090 | Original |
| Tensor-core chunked gated delta net (four-term precision option) | 3090 | Helpers adapted from upstream PRs [#29353](https://github.com/ggml-org/llama.cpp/pull/29353) and [#29612](https://github.com/ggml-org/llama.cpp/pull/29612); kernel original |
| Wave64 decode gated delta net kernel | MI50 | Original |
| BF16-wire copy-only AllReduce for two-GPU tensor split | MI50 | Original, built on the upstream AllReduce |
| Penalty + greedy top-k sampling fast path | both | Original |
| Fast logits path in the sampler (skips full-vocabulary work for the draft and greedy sampling steps) | both | Original |
| Uninitialized speculative checkpoint buffers | both | Original |
| Memory resource capacity API, per-sequence KV window, state I/O limits | both | Original |
| KV admission, restore under admission, slot operations, proxy-cancel fix (server) | both | Original |

Everything is opt-in at build time (`GGML_EXP_*` macros) or at runtime (environment variables); see the comments at the top of each kernel file. Upstream behavior is the default. The exact build flags and runtime arguments used for the numbers below are in [docs/BUILD.md](docs/BUILD.md); experiments that were measured and rejected are in [docs/rejected-experiments.md](docs/rejected-experiments.md).

## Results

All numbers were measured on one host (X99 board, 2x Xeon E5-2673 v3, PCIe 3.0) with an RTX 3090 24 GB (270 W limit) and two MI50 32 GB cards (PCIe x8 / x16, gfx906, ROCm 7.2.4). Models: Qwen3.8-27B, a hybrid (Gated-DeltaNet + attention) model, as an IQ3_S-based custom GGUF mix at 3.44 bpw with an MTP (multi-token prediction) head ("27B" below); and Qwen3.6-35B-A3B, a MoE, as IQ4_XS GGUF ("35B-A3B" below). KV cache is K q8_0 / V q4_0 unless stated. Quality gates compare against the same engine without the patch. "ABBA" means interleaved A/B/B/A runs in fresh processes; intervals are 95% CIs.

### Stock vs fork at the published tip

Fork = `publish/master` at `sampling : fast logits path`; stock = upstream llama.cpp at the fork's base commit (`de699957b`). Server runs (`/completion`, `/v1/chat/completions`), 131072-token context allocated, 2 slots, batch/ubatch 512, flash attention on, unified KV, K q8_0 / V q4_0. PP and TG are medians of 3 runs (prompt of exact synthetic tokens; TG is 256 generated tokens); TTFT is for an 8,192-token prompt through the chat endpoint, first call cold and median of the warm calls (warm calls hit the prompt cache, so they measure cache reuse, not prefill). Both builds use the same model files. Full build flags: [docs/BUILD.md](docs/BUILD.md). MTP = the model's multi-token-prediction head used for speculative decoding (n_max=2); stock has no MTP-OFF/ON rows because MTP was not run on it.

**RTX 3090 (single GPU, CUDA sm86).** Stock and fork are on the same topology (one card).

| Model | Build | PP 512 (tok/s) | PP 8192 (tok/s) | TG 256 (tok/s) | TTFT 8k cold | TTFT 8k warm |
|---|---|---:|---:|---:|---:|---:|
| 27B | stock, MTP off | 1,020.1 | 1,141.9 | 36.74 | 8.08 s | 111 ms |
| 27B | fork, MTP off | 951.2 | 1,234.3 | 41.40 | 7.56 s | 107 ms |
| 27B | fork, MTP on | 888.9 | 1,132.8 | 56.72 | 8.16 s | 106 ms |
| 35B-A3B | stock, MTP off | 2,343.7 | 3,474.0 | 150.25 | 2.56 s | 23.9 ms |
| 35B-A3B | fork, MTP off | 2,217.1 | 3,735.9 | 172.88 | 2.36 s | 23.4 ms |

27B decode: +12.7% fork vs stock without MTP, +54.4% with MTP (56.72 vs 36.74 tok/s). 35B-A3B decode +15.1%, PP 8192 +7.5%. Short-prompt prefill (PP 512) is 6.8% (27B) and 5.4% (35B-A3B) lower on the fork than on stock; the fork's prefill gains show up at 8k. These are single runs of 3 per cell; no confidence intervals. The 35B-A3B was not run with MTP (no MTP head in that file).

**2x MI50 (gfx906, ROCm 7.2.4), 27B only. Not the production topology.** Upstream tensor split aborts on gfx906 (a backend assertion), so the stock baseline cannot run the topology the fork is deployed with (tensor split 0.5/0.5). The runs below use row split (`-sm row -ts 1,1`), the configuration the report records for this table; the stock HIP build used the same ROCm toolchain image as the fork (it includes the community rocBLAS gfx906 patches). Treat the numbers as "stock row split vs fork row split", not as stock vs the deployed fork.

| Model | Build | PP 512 (tok/s) | PP 8192 (tok/s) | TG 256 (tok/s) | TTFT 8k cold | TTFT 8k warm |
|---|---|---:|---:|---:|---:|---:|
| 27B | stock, row split, MTP off | 183.8 | 208.3 | 19.70 | 40.7 s | 432 ms |
| 27B | fork, row split, MTP off | 338.5 | 379.0 | 40.41 | 22.8 s | 244 ms |
| 27B | fork, row split, MTP on | 318.3 | 361.9 | 46.61 | 24.0 s | 295 ms |

Fork vs stock on the same row split: decode +105% without MTP and +137% with MTP, PP 8192 +82%, cold 8k TTFT 40.7 s to 22.8 s. The 35B-A3B was not benchmarked against stock on the MI50 cards. The MI50 numbers here come from server runs with 2 slots and 131k context; the layer-split `llama-bench` tables further down use a different method and a different baseline and are not directly comparable to this table.

**Parity with the deployed builds.** Before the matrix, the fork build was checked against the deployed binary (27B, MTP on, TG 256, 3 runs): RTX 3090 61.76 vs 62.61 tok/s (-1.36%), 2x MI50 46.70 vs 47.16 tok/s (-0.99%), both inside a +-3% gate. The earlier -3.5% gap on the 3090 came from the missing sampling fast path.

**Hardware and build summary.** One host: X99 board, 2x Xeon E5-2673 v3, PCIe 3.0; RTX 3090 24 GB (270 W limit); 2x MI50 32 GB. CUDA build: GCC 13.3, CUDA 12.8, sm86, `-use_fast_math`, CUDA graphs, FA for all KV quant types, plus the `GGML_EXP_*` opt-ins. HIP build: GCC 13.3, ROCm 7.2.4 (Clang 22), gfx906, FA for all KV quant types, plus the `GGML_EXP_*` opt-ins. Full lists in [docs/BUILD.md](docs/BUILD.md).

The depth sweeps and per-patch tables below were measured earlier, on production binaries of their time, and are kept for the per-patch gains and the fidelity data.

### RTX 3090, 27B: decode vs stock at depth (earlier production binary)

Qwen3.8-27B (IQ3_S-based mix, MTP head). Single slot, 131072-token context allocated, MTP n_max=2, 48 output tokens, greedy, batch/ubatch 512, one run per cell (no CI). Fork = the production binary of the time (FA quantized-KV GQA kernel and the other 3090 kernels, without the GDN tensor-core prefill patch, measured separately below), not the published tip. Stock is the same base commit with MTP on.

| Prompt tokens | Decode, stock | Decode, fork | Gain | TTFT, stock | TTFT, fork |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 68.3 tok/s | 73.8 tok/s | +8.0% | 1.96 s | 2.06 s |
| 8,192 | 65.7 | 72.7 | +10.7% | 7.67 s | 7.89 s |
| 32,768 | 57.9 | 68.3 | +18.0% | 34.2 s | 34.7 s |
| 129,500 | 38.1 | 53.4 | +39.9% | 206 s | 208 s |

The quantized-KV GQA flash-attention kernel does not change prefill (TTFT within about 1-5% of stock, no gain). On the current upstream master with only that patch applied and no speculative decoding (3 ABBA quartets per cell, same Qwen3.8-27B IQ3_S-based mix and KV types):

| Prompt tokens | Decode, master | Decode, master + patch | Gain (95% CI) |
|---:|---:|---:|---|
| 8,192 | 36.1 tok/s | 39.3 tok/s | +8.9% [8.5, 9.2] |
| 32,768 | 27.4 | 36.8 | +34.2% [33.4, 35.1] |
| 129,024 | 13.9 | 30.0 | +115% [113, 117] |

### RTX 3090: prefill with the tensor-core gated delta net

Production binary with/without the patch (four-term precision option), ABBA, 3 pairs, KV q8_0/q4_0, ubatch 512. Models: Qwen3.8-27B (IQ3_S-based mix) and Qwen3.6-35B-A3B (IQ4_XS).

| Model | Prompt | Prefill before | Prefill after | Time saved (95% CI) |
|---|---:|---:|---:|---|
| 27B | 8,192 | 1,047 tok/s | 1,153 tok/s | 9.2% [8.8, 9.6] |
| 27B | 32,768 | 953 | 1,038 | 8.2% [7.8, 8.6] |
| 35B-A3B | 8,192 | 2,751 | 3,052 | 9.9% [9.2, 10.5] |
| 35B-A3B | 32,768 | 2,505 | 2,767 | 9.5% [9.2, 9.7] |

### 2x MI50, 27B: vs the community gfx906 build (layer split, llama-bench)

Qwen3.8-27B (IQ3_S-based mix, 3.44 bpw). `llama-bench`, layer split 0.5/0.5, batch 2048 / ubatch 512, FA on, K q8_0 / V q4_0, same model, ABBA with 2 processes per arm, both GPUs confirmed busy. TG is 32 tokens generated at the stated depth, without speculative decoding. Perplexity on a 2-chunk corpus: 6.2736 (community) vs 6.2748 (fork), +0.019%.

| Context | Prefill, community | Prefill, fork | Gain | Decode, community | Decode, fork | Gain |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 331.5 tok/s | 389.7 | +17.6% | 18.9 tok/s | 30.5 | +61.6% |
| 8,192 | 359.8 | 411.1 | +14.3% | 18.0 | 29.9 | +66.0% |
| 32,768 | 320.7 | 363.9 | +13.5% | 15.1 | 28.5 | +88.6% |
| 131,072 | 225.6 | 249.8 | +10.7% | 9.3 | 23.4 | +151% |

Where the decode gain comes from (ablation at 8k, ratio of means, N=2): disabling the gfx906 i-quant / sub-4-bit `mul_mat_vec_q` kernels costs 21% of decode; disabling the fused glue kernels about 4%; the wave64 GDN decode kernel about 5%. The vocabulary projection (Q4_K, 682 MiB, 248k-token vocabulary) already reads at 875 GB/s, 98% of the measured read ceiling of the card, so there is no kernel headroom left there.

Upstream llama.cpp at the base commit aborts on this model with tensor split on gfx906 (an assertion in the backend meta layer), so there is no stock tensor-split baseline.

### 2x MI50, 27B: BF16 AllReduce for tensor split

Qwen3.8-27B (IQ3_S-based mix), two-GPU tensor split, MTP n_max=2, 2 slots, 262,144-token context, ABBA, 3 to 6 pairs. Baseline is the F32-wire copy-only AllReduce already in this fork.

| Measure | F32 wire | BF16 wire | Change (95% CI) |
|---|---:|---:|---|
| Prefill, 6,144 tokens | 19.05 s | 17.03 s | -10.6% time [9.3, 11.9] |
| Prefill, 7,168 tokens | 21.9 s (327 tok/s) | 19.5 s (367 tok/s) | -12.2% time [11.9, 12.5] |
| Decode | | | no effect (+0.3% [-3.5, +4.1]) |

Mean KLD vs the F32 wire 0.0015, same-top-p 98.6%. A 20-minute soak (50 requests, 355k prompt tokens) and a 62-minute soak (142 requests, 1.0M prompt tokens) ran without errors. Qwen3.6-35B-A3B (IQ4_XS) on the same cards uses a layer split, where the all-reduce is not exercised; it is unchanged within noise (-0.06% [-0.8, +0.7]).

### 2x MI50: KV window and cancellation

* Per-sequence KV window (two slots sharing one unified KV pool, one slot in a 28.5k-token request): the decode cost imposed by the other slot's cache drops from 8.7% to 1.9%; direct decode gain +4.7% [4.3, 5.1] at that point and +9.5% at 57k tokens of neighbour context. With no neighbour there is no regression (56.96 vs 56.58 tok/s). Output is byte-identical in 12/12 samples and the KLD sits at the run-to-run noise floor.
* Proxy-cancel fix: after a client aborts in the middle of a prefill, the next request waits 5.4 s instead of 34.6 s (-84%, n=3); the abandoned prefill no longer runs to the end.
* Speculative decoding (MTP head, n_max=2) on Qwen3.8-27B: +27.7% decode over no speculation (natural text, draft acceptance 73-75%); n_max=1 is 10% slower, n_max=3 is only +3.1% and not worth the concurrency cost.

### Fidelity

| Patch | Test | Result |
|---|---|---|
| FA quantized-KV GQA (3090) | Teacher-forced full-vocabulary KLD vs stock FA, 8 positions | 32k: 0.0003 mean, 129k: 0.0006 mean; top-1 100% |
| FA quantized-KV GQA (3090) | 20 greedy continuations x 32 tokens, 27B | 17/20 identical (97.3% of tokens) in-domain, 20/20 diverse; a plain q8_0/q8_0 KV gives 15/20 and 19/20 |
| GDN tensor-core (3090) | Same, 27B | 14-16/20 identical in-domain (88-89% of tokens), 19/20 diverse; KLD 0.0015, top-1 8/8. Run-to-run floor of the unpatched build on the 27B: 20/20 |
| GDN tensor-core (3090) | Code generation, Qwen3.6-35B-A3B, 120 problems x 2 repeats, 8192-token cap | 119-120/120 with the patch vs 120/120 without; 0 truncated answers |
| BF16 AllReduce (MI50) | KLD vs F32 wire, 27B | mean 0.0015 (35B: 0, byte-identical) |
| gfx906 kernels (MI50) | Perplexity vs community build | +0.019% |
| KV window (MI50) | Greedy output, 12 samples | byte-identical; KLD at A/A floor |

The 35B MoE is not bit-deterministic under concurrent slots and speculation even against itself (16/20 identical greedy sequences between two instances of the same build), so "identical output" is not a usable gate there; KLD and task accuracy are.

### Tried and rejected

Experiments that were measured and rejected (fidelity gate, gain within noise, or regression) are listed with numbers in [docs/rejected-experiments.md](docs/rejected-experiments.md), so nobody repeats them.

## What it enables

* **Long context on consumer cards.** A 27B at 129k-token context decodes at 53 tok/s on one RTX 3090 (24 GB, K q8_0 / V q4_0 + MTP), against 38 tok/s on the stock build; a 35B-A3B MoE with 400k-token context runs on two MI50 cards.
* **Cheap old datacenter GPUs as real inference nodes.** Two MI50 cards (gfx906) serve the 27B at 41 tok/s decode (47 tok/s with MTP) and about 340 tok/s prefill in the row-split benchmark, against 20 tok/s and 184 tok/s on stock under the same split, up to 131k tokens of context; tensor split also works where the stock build aborts.
* **Two slots on one card without the usual penalty.** The per-sequence KV window removes most of the attention cost one slot imposes on the other when they share a unified KV pool.
* **Safe interactive serving.** Aborting a request mid-prefill frees the slot within a batch instead of after the whole prompt; KV admission and restore limits give a server a hard memory budget per sequence.
* **Prefill on one card, decode on another.** KV state files can be moved between engines with identical model identity; see the pool proxy for a 3.6x time-to-first-token improvement for a 79k-token prompt (192 s to 53 s).
* **Upstreamable pieces.** The patches are separated by feature and opt-in at build or run time, so the kernels, the BF16 AllReduce, the quantized-KV GQA attention and the KV-window API can be taken independently.

---

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp)](https://github.com/ggml-org/llama.cpp/releases)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3A0cc4m%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev branches](https://github.com/ggml-org/llama.cpp-dev/blob/master/README-features.md) / [compile times](https://github.com/ggml-org/llama.cpp-dev/blob/master/README-compile-times.md) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
