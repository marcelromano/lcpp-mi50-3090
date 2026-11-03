# Build flags used for the published results

These are the build options and runtime arguments behind the numbers in the README. Every `GGML_EXP_*` macro is an opt-in; leaving them out gives upstream behavior for that code path.

## RTX 3090 (CUDA, sm86)

Toolchain: GCC 13.3, CUDA 12.8.

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_CUDA_FORCE_CUBLAS=OFF \
  -DGGML_CPU_ALL_VARIANTS=OFF \
  -DGGML_NATIVE=OFF \
  -DGGML_CUDA_COMPRESSION_MODE=size \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DGGML_CCACHE=OFF \
  -DCMAKE_CUDA_FLAGS="-use_fast_math -extended-lambda -compress-mode=size <defines below>" \
  -DCMAKE_CXX_FLAGS="<defines below>"
```

Defines (passed to both the C++ and CUDA compilers):

```text
-DGGML_EXP_CKPT_FAST -DGGML_EXP_MMVQ_V5 -DGGML_EXP_MMVF_BF16_N8 -DGGML_EXP_GDN_NOGATHER
-DGGML_EXP_Q8_SHARE -DGGML_EXP_FUSE2 -DGGML_EXP_SK_MMVF -DGGML_EXP_SK_FFN_TAIL
-DGGML_EXP_SK_ADD_RMS -DGGML_EXP_SK_RMS_GATE -DGGML_EXP_KV_RESTORE_RUNS -DLLAMA_EXP_SLOT_DFT
-DGGML_EXP_CUDA_FA_KVQ_GQA -DGGML_CUDA_GDN_MMA_FOUR_TERM -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128
-DGGML_SCHED_MAX_COPIES=4
```

Runtime: environment variable `GGML_CUDA_GDN_MMA=1`, and the NUMA node of the card's PCIe root bound with `numactl --membind`.

Server arguments, 27B (with MTP):

```text
--ctx-size 131072 --batch-size 512 --ubatch-size 512 --parallel 2
--cache-type-k q8_0 --cache-type-v q4_0 --cache-ram 8192 -ctxcp 8
--flash-attn on --kv-unified --reasoning on --reasoning-format deepseek --no-cache-idle-slots
--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.0
```

35B-A3B: the same without the `--spec-*` options.

## 2x MI50 (HIP, gfx906)

Toolchain: GCC 13.3, ROCm 7.2.4 (Clang 22).

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DAMDGPU_TARGETS=gfx906 \
  -DGGML_HIP=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_CPU_ALL_VARIANTS=OFF \
  -DGGML_CUDA_FORCE_CUBLAS=OFF \
  -DGGML_NATIVE=OFF \
  -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DGGML_CCACHE=OFF \
  -DCMAKE_CXX_FLAGS="<defines below>" -DCMAKE_HIP_FLAGS="<defines below>"
```

Defines:

```text
-DGGML_EXP_IQ_LDS -DGGML_EXP_IQ_LDS2 -DGGML_EXP_VCMP -DGGML_EXP_VSUB -DGGML_EXP_VSUBSS
-DGGML_EXP_GDN_NOGATHER -DGGML_EXP_Q8_SHARE -DGGML_EXP_CKPT_FAST -DGGML_EXP_GCN_MMVQ_GEOM
-DGGML_EXP_MMVQ_GCN2 -DGGML_EXP_MMVQ_GCN2_NO_IQ3 -DGGML_EXP_MMVQ_GCN2_IQ3V2 -DGGML_EXP_GDN_DEC2
-DGGML_EXP_SK_TOPK -DGGML_EXP_SK_MMVF -DGGML_EXP_SK_FFN_TAIL -DGGML_EXP_SK_ADD_RMS
-DGGML_EXP_SK_RMS_GATE -DGGML_EXP_FUSE2 -DGGML_EXP_FA_LC -DGGML_EXP_MMQ_LDSTAB
-DGGML_EXP_MMQ_J128_IQ -DGGML_EXP_MMQ_MOE_J=16 -DGGML_EXP_COLA -DGGML_EXP_MMVQ_REUSE3
-DGGML_EXP_MMVQ_DEC_MUL24 -DGGML_EXP_MMVQ_DEC_Q2K -DGGML_EXP_MMVQ_DEC_NW=4
-DGGML_EXP_MMVQ_GCN2_IQ2S_LDS -DGGML_EXP_KV_RESTORE_RUNS -DLLAMA_EXP_SLOT_DFT
```

Runtime: environment variable `GGML_COLA=251`, and `numactl --membind` to the NUMA node of the cards.

Server arguments used for the benchmark, 27B (with MTP):

```text
--ctx-size 131072 --batch-size 512 --ubatch-size 512 --parallel 2
--cache-type-k q8_0 --cache-type-v q4_0 -sm row -ts 1,1
--flash-attn on --kv-unified --reasoning on --reasoning-format deepseek --no-cache-idle-slots
--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.0
```

`-sm row` is the benchmark configuration because upstream tensor split aborts on gfx906 and would leave no stock baseline; the fork's deployed configuration uses tensor split with a 0.5/0.5 split.

## Notes

* The stock builds in the benchmark are upstream at the fork's base commit with the same toolchain image and the same CMake options that exist upstream; they contain none of the `GGML_EXP_*` code.
* The flag lists were collected from the build recipes of the deployed binaries; the exact way the defines are passed to the compilers (CMake variable names) depends on your build wrapper, which is why they are shown as one list.
