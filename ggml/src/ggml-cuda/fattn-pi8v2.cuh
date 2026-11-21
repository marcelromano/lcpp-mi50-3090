#pragma once

// FlashAttention de PROMPT int8 v2 para a RTX 3090 (sm_86), D = 256, GQA 6.
//
// Le o KV de producao no formato em que ele esta no cache (K = q8_0, V = q4_0) e nunca
// materializa uma copia f16 do KV inteiro:
//   QK^T : mma.sync m16n8k32 s8 — Q quantizado por bloco de 32 em REGISTRADORES (a escala do bloco
//          leva o `scale` da atencao), os qs de K q8_0 sao usados como estao com a escala d do
//          proprio bloco aplicada em fp32 na saida do mma (s32 -> fp32).
//   PV   : mma.sync m16n8k16 f16 (acumulacao f32) com V q4_0 decodificado UMA vez por tile para
//          shared transposto [d][pos].
//   softmax: fp32, 100% em registradores (o fragmento C do QK e' exatamente o fragmento A do PV).
//
// O que muda em relacao a v1 da revisão anterior (medida: -5,3% PP a 32k, -15,1% a 120k) — as duas causas
// apontadas no relatorio daquela tarefa:
//   (b) 1,6x mais blocos: a v1 tinha 5 queries (30 linhas) por bloco porque 4 warps dividiam os
//       256 dims de D. Aqui cada warp e' dono de 16 LINHAS x 256 dims (o acumulador D256 fica em
//       registradores, como no kernel de referencia do NInfer): 128 linhas por CTA, i.e. 21,33
//       queries. Blocos: 4,3x menos que a v1 e 2,67x menos que a producao (48 linhas), logo o KV
//       e' relido 4,3x/2,67x menos vezes por token de prompt.
//   (a) decodificacao q4_0->f16 do V por tile: continua por tile, mas agora (i) e' amortizada por
//       128 linhas em vez de 30 e (ii) o tile seguinte e' copiado com `cp.async` em double buffer
//       (K e V brutos em shared), entao a decodificacao roda sobre dado ja' residente e nao
//       segura o pipeline. Custo por token de prompt da decodificacao: 4,3x menor que na v1.
//   Alem disso S e P nao passam mais por shared (v1 gravava scores 128x33 f32 e P em meio) e o
//   mascaramento so' e' lido nos 1-2 tiles da diagonal (o resto e' totalmente visivel).
//
// SPLIT-KV (merge FP32 por (max, soma)).
// O perfil da revisão anterior mediu POR QUE o kernel perde 13% por lancamento: com 128 linhas por CTA o
// chunk de prefill de 512 tokens da' 24 blocos x 4 cabecas de KV = 96 CTAs contra 82 SMs, i.e.
// 58% de onda (a producao lanca 256 CTAs de 48 linhas e fica em 78%). Nada disso e' codec: e'
// paralelismo DENTRO do lancamento. Aqui o eixo z da grade divide a faixa causal em `nsplit`
// fatias contiguas de tiles de KV; cada CTA faz a sua fatia com softmax local (M_s, L_s) e grava
// O_s NAO normalizado + (M_s, L_s); um segundo kernel combina
//     O = sum_s O_s * exp(M_s - M) / sum_s L_s * exp(M_s - M),   M = max_s M_s
// em FP32 (nada de precisao perdida em relacao ao caminho sem split). Com 2 splits: 192 CTAs =
// 78% de onda; com 3: 288 CTAs = 88%. O numero de splits e' escolhido por lancamento pelo
// comprimento de KV e pelo numero de SMs do dispositivo (utilizacao de onda), com um piso de
// tiles por fatia para nao partir demais KV curto; e um limiar de KV minimo
// (GGML_CUDA_FA_PI8V2_MIN_KV) manda o lancamento de volta para a FA de producao.
//
// Opt-in por env, default OFF: GGML_CUDA_FA_PI8V2 (+ _MIN_Q, _MAX_Q, _VERBOSE, _SPLIT, _MIN_KV,
// _SPLIT_MIN_TILES). O decode nunca entra no predicado (n_q <= 4).
//
// CODEC DO QK selecionavel em compilacao (o F3 da revisão anterior reprovou o A8: KLD 0,063, e a
// atribuicao mediu que o erro e' do Q int8, nao do split). Split-KV, geometria (128 linhas x
// 256 threads x nsplit) e PV ficam identicos. Macro GGML_EXP_CUDA_FA_PI8V2_QK:
//   0 = A8: Q int8 por linha x bloco de 32, escala amax/127. Erro de Q ate'
//            amax/254 por elemento.
//   1 = F16  : Q em f16 (pre-multiplicado pelo scale, como a FA de producao) nos registradores,
//            K q8_0 decodificado UMA vez por tile para f16 em shared [pos][d] (mesma rotina do V),
//            QK em mma m16n8k16 f16 com acumulacao f32 — aritmetica da FA de producao. Custa
//            +16,9 KB de shared (64000 B no total -> shared DINAMICA, opt-in acima de 48 KB) e o QK
//            vai para o throughput f16/f32-acc do GA102 (1/4 do int8).
//   2 = A8x2 (DEFAULT): escala dupla por residuo. q_hi = rn(x*127/amax) como no A8 e
//            q_lo = rn((x*127/amax - q_hi)*254) em [-127,127]; os dois mma s8 reaproveitam os MESMOS
//            fragmentos B de K e somam exato em int32: c = 254*c_hi + c_lo (|c| < 1,32e8 < 2^31).
//            Q efetivo = (254*q_hi + q_lo)*amax/(127*254): erro ate' amax/64516 por elemento,
//            254x menor que o A8 e abaixo do arredondamento f16 do Q da producao. Os fragmentos
//            de q_lo (+32 regs/lane) ficam em registradores (255, spill so' fora do laco); opcional
//            em shared (GGML_EXP_CUDA_FA_PI8V2_QLO_SMEM=1). 2 mma s8 por bloco reaproveitando o B:
//            QK a 1/2 do throughput int8, ainda 2x o f16/f32-acc do GA102.

#if !defined(PI8V2_DEVICE_ONLY)
#include "common.cuh"
#else
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#define GGML_UNUSED(x) (void) (x)
#define GGML_UNUSED_VARS(...) (void) 0
#endif

#if defined(GGML_EXP_CUDA_FA_PI8V2) || defined(PI8V2_DEVICE_ONLY)

#define V2_D         256
#define V2_T          32                       // posicoes de KV por tile
#define V2_NW          8                       // warps por CTA
#define V2_NTHREADS  (V2_NW*32)                // 256
#define V2_ROWS      (V2_NW*16)                // 128 linhas por CTA = 21,33 queries
#define V2_GQA         6                       // razao GQA da producao (27B)
#define V2_KROW      272                       // bytes de uma linha q8_0 de 256
#define V2_VROW      144                       // bytes de uma linha q4_0 de 256
#define V2_VT_STR     40                       // halves por linha d do V transposto (32 + 8)
#define V2_KCH       (V2_T*V2_KROW/16)         // 544 chunks de 16 B
#define V2_VCH       (V2_T*V2_VROW/16)         // 288
#define V2_NPF_K     ((V2_KCH + V2_NTHREADS - 1)/V2_NTHREADS)   // 3
#define V2_NPF_V     ((V2_VCH + V2_NTHREADS - 1)/V2_NTHREADS)   // 2
#define V2_SMEM      (2*V2_T*V2_KROW + 2*V2_T*V2_VROW + 2*V2_D*V2_VT_STR)  // 47104 B
#define V2_MAX_SPLIT   6                       // fatias maximas do split-KV (eixo z da grade)

// codec do QK (ver cabecalho). Default = 2 (A8x2).
#ifndef GGML_EXP_CUDA_FA_PI8V2_QK
#define GGML_EXP_CUDA_FA_PI8V2_QK 2
#endif
#define V2_QK GGML_EXP_CUDA_FA_PI8V2_QK
#if V2_QK != 0 && V2_QK != 1 && V2_QK != 2
#error "GGML_EXP_CUDA_FA_PI8V2_QK deve ser 0 (A8), 1 (F16) ou 2 (A8x2)"
#endif
#define V2_KH_STR    (V2_D + 8)                // halves por linha de posicao do K f16 (QK=1)
// QK=2: onde ficam os fragmentos de q_lo. 0 (default) = registradores: ptxas 12.8 da' 255 regs com
// 16 B de spill, mas o SASS mostra os STL so' no prologo e os LDL so' no epilogo (fora do laco de
// tiles) — custo nulo, shared estatica 47104 B como na revisão anterior. 1 = shared (+32 KB dinamica, um int4
// por (warp, bloco, lane), lido/gravado SO' pela propria lane, sem barreira): reserva para o caso
// de um ptxas futuro derramar dentro do laco.
#ifndef GGML_EXP_CUDA_FA_PI8V2_QLO_SMEM
#define GGML_EXP_CUDA_FA_PI8V2_QLO_SMEM 0
#endif
#define V2_QLO_SMEM (V2_QK == 2 && GGML_EXP_CUDA_FA_PI8V2_QLO_SMEM)
#if V2_QK == 1
#define V2_SMEM_EXTRA (V2_T*V2_KH_STR*2)       // sKh [V2_T][V2_KH_STR] f16 = 16896 B
#elif V2_QLO_SMEM
#define V2_SMEM_EXTRA (V2_NW*8*32*16)          // sQl [warp][bloco][lane] int4 = 32768 B
#else
#define V2_SMEM_EXTRA 0
#endif
#if V2_SMEM_EXTRA > 0
// shared DINAMICA (acima dos 48 KB estaticos): sK, sV, sVt como antes + o extra.
// QK=1: 64000 B; QK=2 com QLO_SMEM: 79872 B. 1 CTA/SM (launch_bounds ..., 1) cabe nos 100 KB do sm_86.
#define V2_SMEM_DYN  (V2_SMEM + V2_SMEM_EXTRA)
#else
#define V2_SMEM_DYN  0                         // shared estatica (47104 B), como na revisão anterior
#endif

// DISCRIMINADORES de aritmetica do PV para ATRIBUIR o KLD do F3 (diagnostico; o default 0
// e' o kernel da revisão anterior, bit-identico). Escolha por env, resolvida no host e passada como template:
//   GGML_CUDA_FA_PI8V2_ACC16=1  -> EMU bit 0: acumulador do PV em f16, como a FA mma_f16 de
//         producao (T_C_VKQ = tile<16,8,half2>, mma.f16.f16.f16.f16) + rescale de alfa em f16.
//   GGML_CUDA_FA_PI8V2_OFF3=1   -> EMU bit 1: max do softmax com o off-set de producao
//         (FATTN_KQ_MAX_OFFSET = 3*0.6931, P <= e^-2.08, rescales mais raros) e FTZ em -20 no
//         fator de rescale (SOFTMAX_FTZ_THRESHOLD).
//   GGML_CUDA_FA_PI8V2_P32=1    -> EMU bit 2: P efetivo em f32 (hi+lo f16, DOIS mma f16 sobre os
//         MESMOS fragmentos de V): isola o arredondamento f16 de P.
//   GGML_CUDA_FA_PI8V2_MASKALL=1 -> EMU bit 3: aplica a MASCARA EM TODOS OS TILES (a producao faz
//         isso sempre). O kernel das revisões anteriores pula a mascara nos tiles que julga "totalmente
//         visiveis" a partir de n_past = ne11 - ne01 — e isso e' FALSO quando o KV chega PADDED ao
//         FATTN_KQ_STRIDE (256): nas posicoes 64..255 de um prompt de 64 tokens, ou 3900..4095 de um
//         de 3900, n_past sai inflado em (256 - L%256) e o kernel atende linhas de padding que a
//         producao mascara. Com o bit 3 o mascaramento e' incondicional (a mascara e' a autoridade).
// EMU != 0 e' SO' para medicao de fidelidade (custa throughput); nenhum braco destes e' candidato
// a promocao.
//
// MASCARA COMO AUTORIDADE, VERSAO BARATA (EMU bit 4 = 16, "KVMAX"; proposta §7.2 da revisão anterior).
// Antes do kernel, uma varredura da mascara (pi8v2_mask_scan_kernel, mesma ideia do
// flash_attn_mask_to_KV_max da producao, fattn-common.cuh:666) classifica cada par
// (bloco de 128 linhas do CTA, tile de 32 posicoes de KV) olhando TODAS as queries do bloco:
//   0 = todo -inf  -> o tile e' pulado (nem decodifica V, nem mma);
//   1 = todo +-0   -> totalmente visivel, sem ler a mascara (somar +0 e' identidade);
//   2 = misto      -> a mascara e' lida e somada, como no MASKALL.
// e grava kv_max[bloco] = 1 + ultima posicao visivel (atomicMax). No kernel:
//   kv_hi = min(ne11, kv_max[bloco])  (reativa o clamp do ultimo tile; n_past = ne11-ne01 NAO e'
//   mais usado), need_mask = (flag == 2), skip = (flag == 0).
// Isto e' EXATO para mascara arbitraria (inclusive KV unificado com celulas de outra sequencia no
// meio, onde um predicado causal derivado de kv_max nao seria): pular um tile todo -inf e nao somar
// uma mascara toda zero sao bit-identicos ao MASKALL (o F1 ops da revisão anterior confere bit a bit).
// Selecao: GGML_CUDA_FA_PI8V2_KVMAX (default 1 quando o lever esta ligado); GGML_CUDA_FA_PI8V2_MASKALL=1
// tem precedencia (fallback medido da revisão anterior); KVMAX=0 e MASKALL=0 = kernel legado da revisão anterior (defeituoso
// com KV padded; so' para reproducao).
#ifndef V2_EMU_DEFAULT
#define V2_EMU_DEFAULT 0                       // modo forcado em compilacao (so' para o f1ops)
#endif
#define V2_KQ_MAX_OFFSET (3.0f*0.6931f)        // FATTN_KQ_MAX_OFFSET da producao
#define V2_SOFTMAX_FTZ  (-20.0f)               // SOFTMAX_FTZ_THRESHOLD da producao

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800)
#define V2_AVAILABLE
#endif

static __device__ __forceinline__ void v2_mma_s8(int (&c)[4], const int (&a)[4], const int b0, const int b1) {
#if defined(V2_AVAILABLE) && defined(__CUDA_ARCH__)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    GGML_UNUSED(c); GGML_UNUSED(a); GGML_UNUSED(b0); GGML_UNUSED(b1);
#endif
}

static __device__ __forceinline__ void v2_mma_f16(float (&c)[4], const uint32_t (&a)[4],
                                                  const uint32_t b0, const uint32_t b1) {
#if defined(V2_AVAILABLE) && defined(__CUDA_ARCH__)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    GGML_UNUSED(c); GGML_UNUSED(a); GGML_UNUSED(b0); GGML_UNUSED(b1);
#endif
}

// (EMU bit 0): acumulador do PV em f16 — o mma que a FA mma_f16 de producao usa para o VKQ
// (tile<16,8,half2>: c[0] = (linha r0, 2 dims), c[1] = (linha r1, 2 dims)).
static __device__ __forceinline__ void v2_mma_f16c(uint32_t (&c)[2], const uint32_t (&a)[4],
                                                   const uint32_t b0, const uint32_t b1) {
#if defined(V2_AVAILABLE) && defined(__CUDA_ARCH__)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    GGML_UNUSED(c); GGML_UNUSED(a); GGML_UNUSED(b0); GGML_UNUSED(b1);
#endif
}

static __device__ __forceinline__ uint32_t v2_h2mul(const uint32_t x, const uint32_t a) {
#if defined(V2_AVAILABLE) && defined(__CUDA_ARCH__)
    const __half2 r = __hmul2(*(const __half2 *) &x, *(const __half2 *) &a);
    return *(const uint32_t *) &r;
#else
    GGML_UNUSED(x); GGML_UNUSED(a);
    return 0u;
#endif
}

static __device__ __forceinline__ float2 v2_h2f2(const uint32_t x) {
#if defined(V2_AVAILABLE) && defined(__CUDA_ARCH__)
    return __half22float2(*(const __half2 *) &x);
#else
    GGML_UNUSED(x);
    return make_float2(0.0f, 0.0f);
#endif
}

// 4 bytes consecutivos de shared (endereco alinhado a 2 bytes)
static __device__ __forceinline__ int v2_ld2(const char * p) {
    const uint16_t * q = (const uint16_t *) p;
    return (int) ((uint32_t) q[0] | ((uint32_t) q[1] << 16));
}

static __device__ __forceinline__ uint32_t v2_pack_half2(const float x0, const float x1) {
    const __half2 h = __floats2half2_rn(x0, x1);
    return *(const uint32_t *) &h;
}

// (EMU bit 2): residuo de P em f16 — P = f16(x) + f16(x - f16(x)), 2 mma sobre o mesmo V.
static __device__ __forceinline__ uint32_t v2_pack_half2_lo(const float x0, const float x1,
                                                           const uint32_t hi) {
#if defined(V2_AVAILABLE) && defined(__CUDA_ARCH__)
    const float2 b = __half22float2(*(const __half2 *) &hi);
    return v2_pack_half2(x0 - b.x, x1 - b.y);
#else
    GGML_UNUSED(x0); GGML_UNUSED(x1); GGML_UNUSED(hi);
    return 0u;
#endif
}

static __device__ __forceinline__ void v2_cp16(void * dst_s, const void * src_g) {
#if defined(V2_AVAILABLE) && defined(__CUDA_ARCH__)
    const unsigned s = (unsigned) __cvta_generic_to_shared(dst_s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(src_g));
#else
    *(uint4 *) dst_s = *(const uint4 *) src_g;
#endif
}

static __device__ __forceinline__ void v2_cp_commit() {
#if defined(V2_AVAILABLE) && defined(__CUDA_ARCH__)
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

// espera todos os grupos menos `n` pendentes
template <int n> static __device__ __forceinline__ void v2_cp_wait() {
#if defined(V2_AVAILABLE) && defined(__CUDA_ARCH__)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
#endif
}

// Um CTA = 1 cabeca de KV (blockIdx.y) x 128 linhas (blockIdx.x, invertido) x uma fatia de KV
// (blockIdx.z, quando ha split).
template <int EMU>
__launch_bounds__(V2_NTHREADS, 1)
static __global__ void flash_attn_prompt_pi8v2(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V,
        const half * __restrict__ mask, float * __restrict__ dst,
        const int * __restrict__ kv_rows, const int n_cells,
        const float scale, const int ne01, const int ne11, const int nheads_q, const int gqa,
        const int64_t nbq1, const int64_t nbq2, const int64_t nbk1, const int64_t nbk2,
        const int64_t nbv1, const int64_t nbv2, const int64_t nbm1,
        float * __restrict__ Opart, float2 * __restrict__ ML, const int nsplit,
        const uint8_t * __restrict__ tflags, const int * __restrict__ kvmax,
        float * __restrict__ dbg_vt, float * __restrict__ dbg_s, float * __restrict__ dbg_lm) {
#if defined(V2_AVAILABLE)
    constexpr bool V2_ACC16 = (EMU & 1) != 0;
    constexpr bool V2_OFF3  = (EMU & 2) != 0;
    constexpr bool V2_P32   = (EMU & 4) != 0;
    constexpr bool V2_MASKALL = (EMU & 8) != 0;
    constexpr bool V2_KVMAX   = (EMU & 16) != 0 && !V2_MASKALL;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int w    = tid >> 5;
    const int g    = lane >> 2;              // 0..7
    const int t    = lane & 3;               // 0..3
    const int h    = blockIdx.y;
    const int si   = blockIdx.z;             // fatia do split-KV (0 quando nsplit == 1)

    // ordem invertida: o tile de linhas com a maior faixa causal sai primeiro
    const int nrows_total = ne01*gqa;
    const int rbase = (gridDim.x - 1 - blockIdx.x)*V2_ROWS;
    const int nrows = min(V2_ROWS, nrows_total - rbase);

    const int r0 = rbase + w*16 + g;         // linha do lane (m-tile do warp = 16 linhas)
    const int r1 = r0 + 8;
    const bool v0 = r0 < rbase + nrows;      // linha valida (nao e' padding)
    const bool v1 = r1 < rbase + nrows;

    const int q0 = r0/V2_GQA;                // posicao de query da linha r0
    const int q1 = r1/V2_GQA;
    const int i0 = r0 - q0*V2_GQA;           // cabeca dentro do grupo
    const int i1 = r1 - q1*V2_GQA;

    // faixa causal do CTA: a ultima query do bloco ve' ate' n_past + q_last
    const int q_first = rbase/V2_GQA;
    const int q_last  = min(ne01 - 1, (rbase + nrows - 1)/V2_GQA);
    const int n_past  = ne11 - ne01;
    // com KVMAX o limite vem da MASCARA (kv_max do bloco), nao de n_past (inflado com padding)
    const int blk     = rbase/V2_ROWS;
    const int kv_hi   = V2_KVMAX ? min(ne11, kvmax[blk]) : (V2_MASKALL ? ne11 : min(ne11, n_past + q_last + 1));
    const int ntiles  = (kv_hi + V2_T - 1)/V2_T;

    // faixa de tiles desta fatia do split-KV, recortada pelo kv_hi desta CTA (as blocos de linha
    // mais "jovens" tem faixa causal menor: as fatias de cima ficam vazias, o merge ignora)
    const int tile_all = (ne11 + V2_T - 1)/V2_T;
    const uint8_t * tfl = V2_KVMAX ? tflags + (int64_t) blk*tile_all : nullptr;
    const int tb = min((si*tile_all)/nsplit, ntiles);
    const int te = min(((si + 1)*tile_all)/nsplit, ntiles);

#if V2_SMEM_DYN > 0
    extern __shared__ __align__(16) char v2_smem[];
    char (*sK)[V2_T*V2_KROW] = (char (*)[V2_T*V2_KROW]) (v2_smem);
    char (*sV)[V2_T*V2_VROW] = (char (*)[V2_T*V2_VROW]) (v2_smem + 2*V2_T*V2_KROW);
    half * sVt = (half *) (v2_smem + 2*V2_T*V2_KROW + 2*V2_T*V2_VROW);
#if V2_QK == 1
    half * sKh = (half *) (v2_smem + V2_SMEM);   // K f16 [pos][V2_KH_STR]
#else
    int4 * sQl = (int4 *) (v2_smem + V2_SMEM) + w*8*32 + lane;   // q_lo desta lane: sQl[b*32]
#endif
#else
    __shared__ __align__(16) char sK[2][V2_T*V2_KROW];
    __shared__ __align__(16) char sV[2][V2_T*V2_VROW];
    __shared__ __align__(16) half sVt[V2_D*V2_VT_STR];
#endif

    const char * Kh = K + (int64_t) h*nbk2;
    const char * Vh = V + (int64_t) h*nbv2;

#if V2_QK == 1
    // ---- Q -> fragmentos A f16 (m16n8k16) em REGISTRADORES, pre-multiplicados pelo scale (como a
    // FA de producao). k-step j = dims [16j, 16j+16): a0 = (r0, 16j+2t..+1), a1 = (r1, idem),
    // a2 = (r0, 16j+2t+8..+9), a3 = (r1, idem). 16 x 4 = 64 registradores.
    uint32_t aQh[16][4];
    {
        const float * qp0 = (const float *) (Q + (int64_t) q0*nbq1 + (int64_t) (h*gqa + i0)*nbq2);
        const float * qp1 = (const float *) (Q + (int64_t) q1*nbq1 + (int64_t) (h*gqa + i1)*nbq2);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int o0 = j*16 + 2*t;
            float2 x0 = make_float2(0.0f, 0.0f), y0 = x0, x1 = x0, y1 = x0;
            if (v0) { x0 = *(const float2 *) (qp0 + o0); y0 = *(const float2 *) (qp0 + o0 + 8); }
            if (v1) { x1 = *(const float2 *) (qp1 + o0); y1 = *(const float2 *) (qp1 + o0 + 8); }
            aQh[j][0] = v2_pack_half2(x0.x*scale, x0.y*scale);
            aQh[j][1] = v2_pack_half2(x1.x*scale, x1.y*scale);
            aQh[j][2] = v2_pack_half2(y0.x*scale, y0.y*scale);
            aQh[j][3] = v2_pack_half2(y1.x*scale, y1.y*scale);
        }
    }
#else
    // ---- Q -> fragmentos A int8 em REGISTRADORES (escala por bloco de 32, com o scale da atencao)
    int   aQ[8][4];
#if V2_QK == 2 && !V2_QLO_SMEM
    int   aQl[8][4];                         // residuo (escala dupla): q_lo = rn((x*id - q_hi)*254)
#endif
    float dq0[8], dq1[8];
    {
        const float * qp0 = (const float *) (Q + (int64_t) q0*nbq1 + (int64_t) (h*gqa + i0)*nbq2);
        const float * qp1 = (const float *) (Q + (int64_t) q1*nbq1 + (int64_t) (h*gqa + i1)*nbq2);
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            const int o0 = b*32 + 4*t;
            float x0[4], x1[4], y0[4], y1[4];
            if (v0) {
                const float4 a = *(const float4 *) (qp0 + o0);
                const float4 c = *(const float4 *) (qp0 + o0 + 16);
                x0[0] = a.x; x0[1] = a.y; x0[2] = a.z; x0[3] = a.w;
                y0[0] = c.x; y0[1] = c.y; y0[2] = c.z; y0[3] = c.w;
            } else {
                for (int k = 0; k < 4; ++k) { x0[k] = 0.0f; y0[k] = 0.0f; }
            }
            if (v1) {
                const float4 a = *(const float4 *) (qp1 + o0);
                const float4 c = *(const float4 *) (qp1 + o0 + 16);
                x1[0] = a.x; x1[1] = a.y; x1[2] = a.z; x1[3] = a.w;
                y1[0] = c.x; y1[1] = c.y; y1[2] = c.z; y1[3] = c.w;
            } else {
                for (int k = 0; k < 4; ++k) { x1[k] = 0.0f; y1[k] = 0.0f; }
            }
            // amax do bloco de 32 do Q, POR LINHA: 8 valores por lane, reduzido entre as 4 lanes
            // do grupo (a escala do q8_0 e' por linha e por bloco de 32)
            float amax0 = 0.0f, amax1 = 0.0f;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                amax0 = fmaxf(amax0, fmaxf(fabsf(x0[k]), fabsf(y0[k])));
                amax1 = fmaxf(amax1, fmaxf(fabsf(x1[k]), fabsf(y1[k])));
            }
            amax0 = fmaxf(amax0, __shfl_xor_sync(0xFFFFFFFF, amax0, 1));
            amax0 = fmaxf(amax0, __shfl_xor_sync(0xFFFFFFFF, amax0, 2));
            amax1 = fmaxf(amax1, __shfl_xor_sync(0xFFFFFFFF, amax1, 1));
            amax1 = fmaxf(amax1, __shfl_xor_sync(0xFFFFFFFF, amax1, 2));
            const float id0 = amax0 > 0.0f ? 127.0f/amax0 : 0.0f;
            const float id1 = amax1 > 0.0f ? 127.0f/amax1 : 0.0f;
            int p0[4], p1[4], s0[4], s1[4];
#if V2_QK == 2
            int l0[4], l1[4], m0[4], m1[4];
#endif
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float ux0 = x0[k]*id0, ux1 = x1[k]*id1, uy0 = y0[k]*id0, uy1 = y1[k]*id1;
                p0[k] = __float2int_rn(ux0);
                p1[k] = __float2int_rn(ux1);
                s0[k] = __float2int_rn(uy0);
                s1[k] = __float2int_rn(uy1);
#if V2_QK == 2
                // residuo em unidades do passo do A8: |u - rn(u)| <= 0,5 (exato em fp32), logo
                // rn(254*(u - q_hi)) cabe em [-127, 127]
                l0[k] = __float2int_rn((ux0 - (float) p0[k])*254.0f);
                l1[k] = __float2int_rn((ux1 - (float) p1[k])*254.0f);
                m0[k] = __float2int_rn((uy0 - (float) s0[k])*254.0f);
                m1[k] = __float2int_rn((uy1 - (float) s1[k])*254.0f);
#endif
            }
            aQ[b][0] = (p0[0] & 0xFF) | ((p0[1] & 0xFF) << 8) | ((p0[2] & 0xFF) << 16) | ((p0[3] & 0xFF) << 24);
            aQ[b][1] = (p1[0] & 0xFF) | ((p1[1] & 0xFF) << 8) | ((p1[2] & 0xFF) << 16) | ((p1[3] & 0xFF) << 24);
            aQ[b][2] = (s0[0] & 0xFF) | ((s0[1] & 0xFF) << 8) | ((s0[2] & 0xFF) << 16) | ((s0[3] & 0xFF) << 24);
            aQ[b][3] = (s1[0] & 0xFF) | ((s1[1] & 0xFF) << 8) | ((s1[2] & 0xFF) << 16) | ((s1[3] & 0xFF) << 24);
#if V2_QK == 2
            {
                const int ql0 = (l0[0] & 0xFF) | ((l0[1] & 0xFF) << 8) | ((l0[2] & 0xFF) << 16) | ((l0[3] & 0xFF) << 24);
                const int ql1 = (l1[0] & 0xFF) | ((l1[1] & 0xFF) << 8) | ((l1[2] & 0xFF) << 16) | ((l1[3] & 0xFF) << 24);
                const int ql2 = (m0[0] & 0xFF) | ((m0[1] & 0xFF) << 8) | ((m0[2] & 0xFF) << 16) | ((m0[3] & 0xFF) << 24);
                const int ql3 = (m1[0] & 0xFF) | ((m1[1] & 0xFF) << 8) | ((m1[2] & 0xFF) << 16) | ((m1[3] & 0xFF) << 24);
#if V2_QLO_SMEM
                sQl[b*32] = make_int4(ql0, ql1, ql2, ql3);   // so' esta lane le' este slot
#else
                aQl[b][0] = ql0; aQl[b][1] = ql1; aQl[b][2] = ql2; aQl[b][3] = ql3;
#endif
            }
            // c = 254*c_hi + c_lo esta' em unidades de amax/(127*254)
            dq0[b] = (amax0/(127.0f*254.0f))*scale;
            dq1[b] = (amax1/(127.0f*254.0f))*scale;
#else
            dq0[b] = (amax0/127.0f)*scale;
            dq1[b] = (amax1/127.0f)*scale;
#endif
        }
    }
#endif // V2_QK == 1

    // ---- acumulador D256 em registradores: 32 n-tiles de 8 dims x (linha r0, r1) x 2 dims
    float O[32][4];          // EMU sem o bit 0 (acumulacao f32, como na revisão anterior)
    uint32_t O16[32][2];     // EMU com o bit 0: 2 half2 por n-tile (linha r0, linha r1) — o array
                             // nao referenciado no instante e' eliminado pelo compilador
#pragma unroll
    for (int n = 0; n < 32; ++n) {
        if constexpr (V2_ACC16) {
            O16[n][0] = 0u; O16[n][1] = 0u;
        } else {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                O[n][k] = 0.0f;
            }
        }
    }
    float M0 = -INFINITY, M1 = -INFINITY, L0 = 0.0f, L1 = 0.0f;

    // ---- staging do KV: cp.async em double buffer
    auto issue = [&](const int tile, const int bit) {
#pragma unroll
        for (int k = 0; k < V2_NPF_K; ++k) {
            const int c = tid + k*V2_NTHREADS;
            if (c < V2_KCH) {
                const int pos = c / 17;
                const int off = c - pos*17;
                const int gp  = min(tile*V2_T + pos, ne11 - 1);   // ultima linha duplicada no fim
                const int row = kv_rows ? kv_rows[gp] : gp;
                v2_cp16(sK[bit] + c*16, Kh + (int64_t) (row >= 0 && row < n_cells ? row : 0)*nbk1 + off*16);
            }
        }
#pragma unroll
        for (int k = 0; k < V2_NPF_V; ++k) {
            const int c = tid + k*V2_NTHREADS;
            if (c < V2_VCH) {
                const int pos = c / 9;
                const int off = c - pos*9;
                const int gp  = min(tile*V2_T + pos, ne11 - 1);
                const int row = kv_rows ? kv_rows[gp] : gp;
                v2_cp16(sV[bit] + c*16, Vh + (int64_t) (row >= 0 && row < n_cells ? row : 0)*nbv1 + off*16);
            }
        }
        v2_cp_commit();
    };

    if (tb < te) {
        issue(tb, tb & 1);
    }

    for (int tile = tb; tile < te; ++tile) {
        const int bit = tile & 1;
        if (tile + 1 < te) {
            issue(tile + 1, bit ^ 1);
            v2_cp_wait<1>();
        } else {
            v2_cp_wait<0>();
        }
        __syncthreads();

        // classe do tile vinda da varredura da mascara (uniforme no CTA)
        const int tflag = V2_KVMAX ? (int) tfl[tile] : 2;
        if (!V2_KVMAX || tflag != 0) {   // tile todo -inf: pula (bit-identico a somar -inf e nao mudar M/L/O)

        // ---- V q4_0 -> f16 transposto [d][pos]: thread = (par de posicoes, bloco de 32, metade)
        {
            const int pp = tid & 15;                       // 0..15 (par de posicoes)
            const int b  = (tid >> 4) & 7;                 // 0..7  (bloco de 32 dims)
            const int hh = tid >> 7;                       // 0..1  (metade dos 8 bytes)
            const char * b0p = sV[bit] + (2*pp    )*V2_VROW + b*18;
            const char * b1p = sV[bit] + (2*pp + 1)*V2_VROW + b*18;
            const float d0 = __half2float(*(const half *) b0p);
            const float d1 = __half2float(*(const half *) b1p);
            const uint16_t * q0p = (const uint16_t *) (b0p + 2);
            const uint16_t * q1p = (const uint16_t *) (b1p + 2);
            half2 * col = (half2 *) (sVt + (b*32)*V2_VT_STR + 2*pp);
#pragma unroll
            for (int k4 = 0; k4 < 4; ++k4) {
                const int k2 = 4*hh + k4;
                const uint32_t x0 = q0p[k2];
                const uint32_t x1 = q1p[k2];
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int k  = 2*k2 + e;
                    const int y0 = (x0 >> (8*e)) & 0xFF;
                    const int y1 = (x1 >> (8*e)) & 0xFF;
                    col[(k     )*(V2_VT_STR/2)] = __floats2half2_rn(((y0 & 0xF) - 8)*d0, ((y1 & 0xF) - 8)*d1);
                    col[(k + 16)*(V2_VT_STR/2)] = __floats2half2_rn(((y0 >>  4) - 8)*d0, ((y1 >>  4) - 8)*d1);
                }
            }
        }
#if V2_QK == 1
        // ---- K q8_0 -> f16 [pos][d] (UMA vez por tile, compartilhado pelos 8 warps): thread =
        // (posicao, bloco de 32). Valor = f16(q*d), o mesmo que a dequantizacao da FA de producao.
        {
            const int pos = tid >> 3;                      // 0..31
            const int b   = tid & 7;                       // 0..7
            const char * kb = sK[bit] + pos*V2_KROW + b*34;
            const float d = __half2float(*(const half *) kb);
            const int8_t * qs = (const int8_t *) (kb + 2);
            half2 * out = (half2 *) (sKh + pos*V2_KH_STR + b*32);
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                out[k] = __floats2half2_rn((float) qs[2*k]*d, (float) qs[2*k + 1]*d);
            }
        }
#endif
        __syncthreads();
        if (dbg_vt != nullptr && tile == 0) {
            for (int idx = tid; idx < V2_D*V2_VT_STR; idx += V2_NTHREADS) {
                dbg_vt[idx] = __half2float(sVt[idx]);
            }
        }

        // ---- S = Q K^T (int8) com as escalas por bloco aplicadas em fp32
        float sc[4][4];
        // com V2_MASKALL a mascara e' sempre aplicada (o unico jeito correto quando o KV
        // chega padded: n_past = ne11 - ne01 fica inflado em (256 - L%256) posicoes de padding).
        const bool need_mask = V2_KVMAX ? (tflag == 2)
                             : (V2_MASKALL || ((tile*V2_T + V2_T - 1) > (n_past + q_first)));
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int pb = nt*8;
            float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#if V2_QK == 1
            // B (k16 x n8, col): b0 = K[pos pb+g][16j+2t..+1], b1 = K[pb+g][16j+2t+8..+9];
            // linha de 264 halves = 132 palavras -> banco (4g + t) mod 32: sem conflito
            const half * khr = sKh + (pb + g)*V2_KH_STR + 2*t;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const uint32_t b0 = *(const uint32_t *) (khr + j*16);
                const uint32_t b1 = *(const uint32_t *) (khr + j*16 + 8);
                v2_mma_f16(s, aQh[j], b0, b1);
            }
#else
            const char * kr  = sK[bit] + (pb + g)*V2_KROW;
            const char * kd0 = sK[bit] + (pb + 2*t    )*V2_KROW;
            const char * kd1 = sK[bit] + (pb + 2*t + 1)*V2_KROW;
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                int c[4] = {0, 0, 0, 0};
                const int b0 = v2_ld2(kr + b*34 + 2 + 4*t);
                const int b1 = v2_ld2(kr + b*34 + 2 + 16 + 4*t);
                v2_mma_s8(c, aQ[b], b0, b1);
#if V2_QK == 2
                // escala dupla: c = 254*c_hi + c_lo, exato em int32 (|c_hi| <= 32*127*127)
                c[0] *= 254; c[1] *= 254; c[2] *= 254; c[3] *= 254;
#if V2_QLO_SMEM
                {
                    const int4 ql = sQl[b*32];      // LDS.128, lanes contiguas: sem conflito
                    const int aql[4] = {ql.x, ql.y, ql.z, ql.w};
                    v2_mma_s8(c, aql, b0, b1);
                }
#else
                v2_mma_s8(c, aQl[b], b0, b1);
#endif
#endif
                const float dk0 = __half2float(*(const half *) (kd0 + b*34));
                const float dk1 = __half2float(*(const half *) (kd1 + b*34));
                s[0] += (float) c[0] * (dq0[b]*dk0);
                s[1] += (float) c[1] * (dq0[b]*dk1);
                s[2] += (float) c[2] * (dq1[b]*dk0);
                s[3] += (float) c[3] * (dq1[b]*dk1);
            }
#endif // V2_QK == 1
            if (tile == ntiles - 1) {
                // posicoes >= kv_hi (duplicadas pelo clamp) nao existem para nenhuma linha
                const int kb = tile*V2_T + nt*8 + 2*t;
                if (kb     >= kv_hi) { s[0] = -INFINITY; s[2] = -INFINITY; }
                if (kb + 1 >= kv_hi) { s[1] = -INFINITY; s[3] = -INFINITY; }
            }
            if (kv_rows) {
                const int kb = tile*V2_T + nt*8 + 2*t;
                const int a = kb < ne11 ? kv_rows[kb] : -1;
                const int b = kb + 1 < ne11 ? kv_rows[kb + 1] : -1;
                if (a < 0 || a >= n_cells) { s[0] = -INFINITY; s[2] = -INFINITY; }
                if (b < 0 || b >= n_cells) { s[1] = -INFINITY; s[3] = -INFINITY; }
            }
            if (v0) { sc[nt][0] = s[0]; sc[nt][1] = s[1]; } else { sc[nt][0] = -INFINITY; sc[nt][1] = -INFINITY; }
            if (v1) { sc[nt][2] = s[2]; sc[nt][3] = s[3]; } else { sc[nt][2] = -INFINITY; sc[nt][3] = -INFINITY; }
        }
        // mascaramento: so' os tiles que tocam a diagonal (o resto e' totalmente visivel)
        if (need_mask) {
            const int kk = tile*V2_T + 2*t;
            const int pb0 = kk;
            const int pb1 = kk + 8;
            const int pb2 = kk + 16;
            const int pb3 = kk + 24;
            // (KVMAX): indice limitado a ne11-1 no ultimo tile parcial — as posicoes >= kv_hi ja'
            // sao -inf pelo clamp, entao o valor lido nao importa; so' nao le' fora da linha da mascara.
            auto mv = [&](const half * m, const int p) -> float {
                return __half2float(m[min(p, ne11 - 1)]);
            };
            if (v0) {
                const half * m = mask + (int64_t) q0*nbm1;
                sc[0][0] += mv(m, pb0); sc[0][1] += mv(m, pb0 + 1);
                sc[1][0] += mv(m, pb1); sc[1][1] += mv(m, pb1 + 1);
                sc[2][0] += mv(m, pb2); sc[2][1] += mv(m, pb2 + 1);
                sc[3][0] += mv(m, pb3); sc[3][1] += mv(m, pb3 + 1);
            }
            if (v1) {
                const half * m = mask + (int64_t) q1*nbm1;
                sc[0][2] += mv(m, pb0); sc[0][3] += mv(m, pb0 + 1);
                sc[1][2] += mv(m, pb1); sc[1][3] += mv(m, pb1 + 1);
                sc[2][2] += mv(m, pb2); sc[2][3] += mv(m, pb2 + 1);
                sc[3][2] += mv(m, pb3); sc[3][3] += mv(m, pb3 + 1);
            }
        }

        if (dbg_s != nullptr && tile == 0) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    dbg_s[tid*16 + nt*4 + j] = sc[nt][j];
                }
            }
        }

        // ---- softmax fp32 por linha, em registradores (4 lanes por linha)
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            mx0 = fmaxf(mx0, fmaxf(sc[nt][0], sc[nt][1]));
            mx1 = fmaxf(mx1, fmaxf(sc[nt][2], sc[nt][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFF, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFF, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFF, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFF, mx1, 2));
        // (EMU bit 1 = OFF3): a producao soma FATTN_KQ_MAX_OFFSET ao maximo de cada tile
        // (o maximo corrente fica ~3 logits alto: P <= e^-2.08) e aplica FTZ em -20 no fator de
        // rescale. Sem OFF3 fica identico a revisão anterior (max exato do tile).
        const float off3 = V2_OFF3 ? V2_KQ_MAX_OFFSET : 0.0f;
        const float Mn0 = fmaxf(M0, mx0 + off3);
        const float Mn1 = fmaxf(M1, mx1 + off3);
        float a0 = (Mn0 == -INFINITY) ? 1.0f : expf(M0 - Mn0);
        float a1 = (Mn1 == -INFINITY) ? 1.0f : expf(M1 - Mn1);
        if constexpr (V2_OFF3) {
            a0 = (M0 - Mn0) >= V2_SOFTMAX_FTZ ? a0 : 0.0f;
            a1 = (M1 - Mn1) >= V2_SOFTMAX_FTZ ? a1 : 0.0f;
        }
        float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            sc[nt][0] = (Mn0 == -INFINITY) ? 0.0f : expf(sc[nt][0] - Mn0);
            sc[nt][1] = (Mn0 == -INFINITY) ? 0.0f : expf(sc[nt][1] - Mn0);
            sc[nt][2] = (Mn1 == -INFINITY) ? 0.0f : expf(sc[nt][2] - Mn1);
            sc[nt][3] = (Mn1 == -INFINITY) ? 0.0f : expf(sc[nt][3] - Mn1);
            sum0 += sc[nt][0] + sc[nt][1];
            sum1 += sc[nt][2] + sc[nt][3];
        }
        sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 1);
        sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 2);
        sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 1);
        sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 2);
        L0 = L0*a0 + sum0;
        L1 = L1*a1 + sum1;
        M0 = Mn0;
        M1 = Mn1;

        // ---- O = O*alpha + P V (f16, acumulacao f32 ou f16 conforme EMU bit 0)
        if constexpr (V2_ACC16) {
            const uint32_t ah0 = v2_pack_half2(a0, a0);   // rescale de alfa em f16 (producao)
            const uint32_t ah1 = v2_pack_half2(a1, a1);
#pragma unroll
            for (int n = 0; n < 32; ++n) {
                O16[n][0] = v2_h2mul(O16[n][0], ah0);
                O16[n][1] = v2_h2mul(O16[n][1], ah1);
            }
        } else {
#pragma unroll
            for (int n = 0; n < 32; ++n) {
                O[n][0] *= a0; O[n][1] *= a0;
                O[n][2] *= a1; O[n][3] *= a1;
            }
        }
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            uint32_t aP[4], aPl[4];
            aP[0] = v2_pack_half2(sc[2*ks][0], sc[2*ks][1]);
            aP[1] = v2_pack_half2(sc[2*ks][2], sc[2*ks][3]);
            aP[2] = v2_pack_half2(sc[2*ks + 1][0], sc[2*ks + 1][1]);
            aP[3] = v2_pack_half2(sc[2*ks + 1][2], sc[2*ks + 1][3]);
            if constexpr (V2_P32) {
                aPl[0] = v2_pack_half2_lo(sc[2*ks][0], sc[2*ks][1], aP[0]);
                aPl[1] = v2_pack_half2_lo(sc[2*ks][2], sc[2*ks][3], aP[1]);
                aPl[2] = v2_pack_half2_lo(sc[2*ks + 1][0], sc[2*ks + 1][1], aP[2]);
                aPl[3] = v2_pack_half2_lo(sc[2*ks + 1][2], sc[2*ks + 1][3], aP[3]);
            }
#pragma unroll
            for (int n = 0; n < 32; ++n) {
                const half * vt = sVt + (n*8 + g)*V2_VT_STR + ks*16 + 2*t;
                const uint32_t b0 = *(const uint32_t *) (vt);
                const uint32_t b1 = *(const uint32_t *) (vt + 8);
                if constexpr (V2_ACC16) {
                    v2_mma_f16c(O16[n], aP, b0, b1);
                    if constexpr (V2_P32) { v2_mma_f16c(O16[n], aPl, b0, b1); }
                } else {
                    v2_mma_f16(O[n], aP, b0, b1);
                    if constexpr (V2_P32) { v2_mma_f16(O[n], aPl, b0, b1); }
                }
            }
        }
        } // fim do "tile nao pulado"
        __syncthreads();
    }

    if (dbg_lm != nullptr) {
        dbg_lm[tid*4 + 0] = L0; dbg_lm[tid*4 + 1] = L1;
        dbg_lm[tid*4 + 2] = M0; dbg_lm[tid*4 + 3] = M1;
    }

    // ---- normalizacao e escrita (split-KV: parciais crus + (M, L) por linha, merge em FP32)
    // leitura do acumulador no formato do modo (f32 na revisão anterior; f16 com EMU bit 0).
    auto orow = [&](const int n, const int row) -> float2 {
        if constexpr (V2_ACC16) {
            return v2_h2f2(O16[n][row]);
        } else {
            return row == 0 ? make_float2(O[n][0], O[n][1]) : make_float2(O[n][2], O[n][3]);
        }
    };
    const int64_t rows_out = (int64_t) ne01*nheads_q;
    if (Opart != nullptr) {
        float * op = Opart + (int64_t) si*rows_out*V2_D;
        const int64_t o0 = (int64_t) q0*nheads_q + (int64_t) (h*gqa + i0);
        const int64_t o1 = (int64_t) q1*nheads_q + (int64_t) (h*gqa + i1);
#pragma unroll
        for (int n = 0; n < 32; ++n) {
            const int d = n*8 + 2*t;
            if (v0) {
                *(float2 *) (op + o0*V2_D + d) = orow(n, 0);
            }
            if (v1) {
                *(float2 *) (op + o1*V2_D + d) = orow(n, 1);
            }
        }
        if (t == 0) {
            if (v0) {
                ML[(int64_t) si*rows_out + o0] = make_float2(M0, L0);
            }
            if (v1) {
                ML[(int64_t) si*rows_out + o1] = make_float2(M1, L1);
            }
        }
        return;
    }

    const float i0f = v0 ? (L0 > 0.0f ? 1.0f/L0 : 0.0f) : 0.0f;
    const float i1f = v1 ? (L1 > 0.0f ? 1.0f/L1 : 0.0f) : 0.0f;
    const int64_t o0 = (int64_t) q0*nheads_q + (int64_t) (h*gqa + i0);
    const int64_t o1 = (int64_t) q1*nheads_q + (int64_t) (h*gqa + i1);
#pragma unroll
    for (int n = 0; n < 32; ++n) {
        const int d = n*8 + 2*t;
        if (v0) {
            const float2 o = orow(n, 0);
            *(float2 *) (dst + o0*V2_D + d) = make_float2(o.x*i0f, o.y*i0f);
        }
        if (v1) {
            const float2 o = orow(n, 1);
            *(float2 *) (dst + o1*V2_D + d) = make_float2(o.x*i1f, o.y*i1f);
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, dst, scale, ne01, ne11, nheads_q, gqa,
                     nbq1, nbq2, nbk1, nbk2, nbv1, nbv2, nbm1, Opart, ML, nsplit, tflags, kvmax);
#endif // defined(V2_AVAILABLE)
}

// Merge FP32 do split-KV: O = sum_s O_s*exp(M_s - M) / sum_s L_s*exp(M_s - M), M = max_s M_s.
// Uma CTA por linha de destino (ne01*nheads_q linhas), 256 threads = as 256 dims de D.
__launch_bounds__(V2_D)
static __global__ void pi8v2_merge_kernel(
        const float * __restrict__ Opart, const float2 * __restrict__ ML,
        float * __restrict__ dst, const int64_t nrows, const int nsplit) {
    const int64_t r = blockIdx.x;
    if (r >= nrows) {
        return;
    }
    float M = -INFINITY;
    for (int s = 0; s < nsplit; ++s) {
        M = fmaxf(M, ML[(int64_t) s*nrows + r].x);
    }
    const int d = threadIdx.x;
    float acc = 0.0f, den = 0.0f;
    if (M != -INFINITY) {
        for (int s = 0; s < nsplit; ++s) {
            const float2 ml = ML[(int64_t) s*nrows + r];
            const float w = (ml.x == -INFINITY) ? 0.0f : expf(ml.x - M);
            den += ml.y*w;
            acc += Opart[(int64_t) (s*nrows + r)*V2_D + d]*w;
        }
    }
    dst[r*V2_D + d] = den > 0.0f ? acc/den : 0.0f;
}

// varredura da mascara (EMU bit 4 / KVMAX). Grade (ceil(tile_all/8), nblocks): cada warp
// classifica 1 par (bloco de linhas do CTA, tile de 32 posicoes) lendo as ~22 queries do bloco
// (lane = posicao; 64 B coalescidos por query). Classe: 0 = todo -inf, 1 = todo +-0, 2 = misto.
// kvmax[bloco] = 1 + maior posicao visivel (precisa de memset 0 antes). Mesma ideia do
// flash_attn_mask_to_KV_max da producao, mas por tile do PI8V2 e exata para mascara arbitraria.
#define V2_SCAN_WARPS 8
__launch_bounds__(V2_SCAN_WARPS*32)
static __global__ void pi8v2_mask_scan_kernel(
        const half * __restrict__ mask, const int64_t nbm1, const int ne01, const int ne11,
        const int nrows_total, const int tile_all,
        uint8_t * __restrict__ tflags, int * __restrict__ kvmax) {
    const int lane = threadIdx.x & 31;
    const int tile = blockIdx.x*V2_SCAN_WARPS + (threadIdx.x >> 5);
    const int blk  = blockIdx.y;
    if (tile >= tile_all) {
        return;                                // warp inteiro sai junto (sem barreira de bloco)
    }
    const int rbase = blk*V2_ROWS;
    const int nrows = min(V2_ROWS, nrows_total - rbase);
    const int qf    = rbase/V2_GQA;
    const int ql    = min(ne01 - 1, (rbase + nrows - 1)/V2_GQA);
    const int pos   = tile*V2_T + lane;
    bool vis = false, nz = false;
    if (pos < ne11) {
        const uint16_t * m = (const uint16_t *) mask + pos;
        for (int q = qf; q <= ql; ++q) {
            const uint16_t b = m[(int64_t) q*nbm1];
            vis = vis || (b != 0xFC00u);           // -inf em f16
            nz  = nz  || ((b & 0x7FFFu) != 0u);    // qualquer coisa diferente de +-0
        }
    }
    const unsigned bv = __ballot_sync(0xFFFFFFFFu, vis);
    const unsigned bn = __ballot_sync(0xFFFFFFFFu, nz);
    if (lane == 0) {
        tflags[(int64_t) blk*tile_all + tile] = bv == 0u ? 0 : (bn == 0u ? 1 : 2);
        if (bv != 0u) {
            atomicMax(kvmax + blk, tile*V2_T + (32 - __clz((int) bv)));
        }
    }
}

// memset + varredura no stream (capturavel em grafo CUDA). rows = ne01*gqa.
static void pi8v2_launch_mask_scan(const half * mask, const int64_t nbm1, const int ne01, const int ne11,
                                   const int rows, const int nblocks, const int tile_all,
                                   uint8_t * tflags, int * kvmax, cudaStream_t stream) {
    cudaMemsetAsync(kvmax, 0, (size_t) nblocks*sizeof(int), stream);
    const dim3 grid((unsigned) ((tile_all + V2_SCAN_WARPS - 1)/V2_SCAN_WARPS), (unsigned) nblocks, 1);
    pi8v2_mask_scan_kernel<<<grid, V2_SCAN_WARPS*32, 0, stream>>>(mask, nbm1, ne01, ne11, rows, tile_all,
                                                                  tflags, kvmax);
}

#endif // GGML_EXP_CUDA_FA_PI8V2 || PI8V2_DEVICE_ONLY

#if !defined(PI8V2_DEVICE_ONLY) && defined(GGML_EXP_CUDA_FA_PI8V2)

static int pi8v2_env_int(const char * name, const int def) {
    const char * s = getenv(name);
    return s && *s ? atoi(s) : def;
}

// Forma de producao (D = 256, GQA 6, K q8_0 / V q4_0, mascara f16). Qualquer outra coisa fica no
// seletor padrao.
static bool ggml_cuda_fa_pi8v2_applies(const int device, const ggml_tensor * dst) {
    static const int enabled = pi8v2_env_int("GGML_CUDA_FA_PI8V2",        0);
    static const int min_q   = pi8v2_env_int("GGML_CUDA_FA_PI8V2_MIN_Q", 32);
    static const int max_q   = pi8v2_env_int("GGML_CUDA_FA_PI8V2_MAX_Q", 4096);
    static const int min_kv  = pi8v2_env_int("GGML_CUDA_FA_PI8V2_MIN_KV", 0);
    if (!enabled) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE ||
        ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return false;
    }
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sink = dst->src[4];
    const ggml_tensor * r = dst->src[5];
    const int64_t n_kv = r ? r->ne[0] : K->ne[1];
    if (ggml_get_op_params_i32(dst, 4) != 0 ||
        (r && (r->type != GGML_TYPE_I32 || !ggml_is_contiguous(r) || r->ne[1] != 1 || r->ne[2] != 1 || r->ne[3] != 1)) || n_kv <= 0) {
        return false;
    }
    // Gather needs an authoritative mask even if legacy diagnostic knobs are set.
    if (r && pi8v2_env_int("GGML_CUDA_FA_PI8V2_MASKALL", 0) == 0 && pi8v2_env_int("GGML_CUDA_FA_PI8V2_KVMAX", 1) == 0) return false;

    float max_bias = 0.0f, softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));

    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_Q8_0 || V->type != GGML_TYPE_Q4_0 ||
        dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (Q->ne[0] != V2_D || K->ne[0] != V2_D || V->ne[0] != V2_D) {
        return false;
    }
    if (K->ne[2] <= 0 || Q->ne[2] != V2_GQA*K->ne[2] || V->ne[2] != K->ne[2]) {
        return false;
    }
    if (Q->ne[1] < min_q || Q->ne[1] > max_q) {
        return false;
    }
    // limiar de KV minimo: abaixo disto um lancamento curto nao paga a geometria do kernel e volta
    // para a FA de producao (foi o que comeu o -0,8% a 32k na revisão anterior)
    if (min_kv > 0 && n_kv < min_kv) {
        return false;
    }
    if (Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1) {
        return false;
    }
    if (K->ne[1] != V->ne[1] || K->ne[1] <= 0 || n_kv < Q->ne[1]) {
        return false;
    }
    if (max_bias != 0.0f || softcap != 0.0f || sink != nullptr) {
        return false;
    }
    if (mask == nullptr || mask->type != GGML_TYPE_F16 || mask->ne[0] < n_kv ||
        mask->ne[1] < Q->ne[1] || mask->ne[2] != 1 || mask->ne[3] != 1 ||
        mask->nb[1] % sizeof(half) != 0) {
        return false;
    }
    if (Q->nb[0] != sizeof(float) || Q->nb[1] % 16 != 0 || Q->nb[2] % 16 != 0 ||
        (uintptr_t) Q->data % 16 != 0) {
        return false;
    }
    if (K->nb[0] != ggml_type_size(GGML_TYPE_Q8_0) || K->nb[1] % 16 != 0 || K->nb[2] % 16 != 0 ||
        (uintptr_t) K->data % 16 != 0) {
        return false;
    }
    if (V->nb[0] != ggml_type_size(GGML_TYPE_Q4_0) || V->nb[1] % 16 != 0 || V->nb[2] % 16 != 0 ||
        (uintptr_t) V->data % 16 != 0) {
        return false;
    }
    if (!ggml_is_contiguous(dst)) {
        return false;
    }
    return true;
}

static void ggml_cuda_fa_pi8v2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int gqa      = (int) (Q->ne[2]/K->ne[2]);
    const int rows     = (int) (Q->ne[1]*gqa);
    const int nblocks  = (rows + V2_ROWS - 1)/V2_ROWS;
    const int64_t nbm1 = mask->nb[1] / sizeof(half);
    const int64_t rows_out = (int64_t) Q->ne[1]*Q->ne[2];
    const int heads_kv = (int) K->ne[2];
    const ggml_tensor * r = dst->src[5];
    const int n_kv = (int) (r ? r->ne[0] : K->ne[1]);

    // ---- quantas fatias de KV (eixo z) — decisao por lancamento: comprimento de KV x numero de SMs
    static const int split_cfg  = pi8v2_env_int("GGML_CUDA_FA_PI8V2_SPLIT", 0);  // 0 = auto
    static const int min_tiles  = pi8v2_env_int("GGML_CUDA_FA_PI8V2_SPLIT_MIN_TILES", 8);
    const int       sms         = (int) ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int       ctas1       = nblocks*heads_kv;
    const int       tile_all    = (n_kv + V2_T - 1)/V2_T;
    auto util = [&](const int c) -> float {
        const int waves = (c + sms - 1)/sms;
        return waves > 0 ? (float) c/(float) (waves*sms) : 0.0f;
    };
    int nsplit = 1;
    if (split_cfg > 0) {
        nsplit = std::min(split_cfg, std::max(1, tile_all/std::max(1, min_tiles)));
        nsplit = std::min(nsplit, V2_MAX_SPLIT);
    } else {
        float best = util(ctas1);
        for (int s = 2; s <= V2_MAX_SPLIT; ++s) {
            if (tile_all < s*std::max(1, min_tiles)) {
                break;
            }
            const float u = util(ctas1*s);
            if (u > best*1.02f) {
                best = u;
                nsplit = s;
            }
        }
    }

    const dim3 grid(nblocks, (unsigned) heads_kv, (unsigned) nsplit);
    const dim3 block(V2_NTHREADS, 1, 1);

    static const int verbose = pi8v2_env_int("GGML_FA_B4_VERBOSE", 0) || pi8v2_env_int("GGML_CUDA_FA_PI8V2_VERBOSE", 0);
    static long n_dispatch = 0;
    // discriminador de aritmetica do PV (default 0 = kernel da revisão anterior)
    // modo da mascara — MASKALL=1 (fallback da revisão anterior, precedencia) > KVMAX (default 1) > legado
    static const int maskall_env = pi8v2_env_int("GGML_CUDA_FA_PI8V2_MASKALL", 0);
    static const int kvmax_env   = pi8v2_env_int("GGML_CUDA_FA_PI8V2_KVMAX",   1);
    static const int emu_req = (pi8v2_env_int("GGML_CUDA_FA_PI8V2_ACC16", 0) ? 1 : 0)
                         | (pi8v2_env_int("GGML_CUDA_FA_PI8V2_OFF3",  0) ? 2 : 0)
                         | (pi8v2_env_int("GGML_CUDA_FA_PI8V2_P32",   0) ? 4 : 0)
                         | (maskall_env ? 8 : (kvmax_env ? 16 : 0));
    // combinacao sem instanciacao -> KVMAX com aritmetica da revisão anterior (nunca cai no legado em silencio)
    static const int emu = [] (const int e) {
        switch (e) {
            case 0: case 1: case 2: case 3: case 4: case 5: case 6: case 7: case 8: case 11:
            case 16: case 17: case 18: case 19: case 20: return e;
            default: fprintf(stderr, "[fa-pi8v2] emu=%d sem instanciacao -> 16 (kvmax)\n", e); return 16;
        }
    }(emu_req);
    // prova de despacho (F1d) sem inundar o log: as 4 primeiras e 1 a cada 2000 lancamentos
    if (verbose && (n_dispatch < 4 || n_dispatch % 2000 == 0)) {
        fprintf(stderr, "[fa-b4] kernel=pi8v2 n=%ld qk=%d emu=%d q=%d kv=%d heads_kv=%d gqa=%d rows=%d blocks=%d "
                        "split=%d ctas=%d sms=%d util=%.3f mask=%s gather=%d cells=%lld\n",
                n_dispatch, (int) V2_QK, emu, (int) Q->ne[1], n_kv, heads_kv, gqa, rows, nblocks,
                nsplit, ctas1*nsplit, sms, util(ctas1*nsplit),
                (emu & 8) ? "all" : ((emu & 16) ? "kvmax" : "legacy"), r != nullptr, (long long) K->ne[1]);
    }
    ++n_dispatch;

    cudaStream_t stream = ctx.stream();

    // parciais do split-KV (2 lancamentos de 12,6 MB a 120k; alocacao pelo pool do backend, como o
    // mmq faz por chamada — nao ha malloc cru, entao captura de grafo CUDA continua valida)
    ggml_cuda_pool_alloc<float>  o_part(ctx.pool());
    ggml_cuda_pool_alloc<float2> ml_part(ctx.pool());
    float *  Opart = nullptr;
    float2 * ML    = nullptr;
    if (nsplit > 1) {
        Opart = o_part.alloc(ctx.pool(), (size_t) nsplit*rows_out*V2_D);
        ML    = ml_part.alloc(ctx.pool(), (size_t) nsplit*rows_out);
    }

    // (KVMAX): classes de tile + kv_max por bloco de linhas, a partir da propria mascara
    ggml_cuda_pool_alloc<uint8_t> tfl_buf(ctx.pool());
    ggml_cuda_pool_alloc<int>     kvmax_buf(ctx.pool());
    const uint8_t * tflags = nullptr;
    const int     * kvmax  = nullptr;
    if (emu & 16) {
        uint8_t * fl = tfl_buf.alloc(ctx.pool(), (size_t) nblocks*tile_all);
        int     * km = kvmax_buf.alloc(ctx.pool(), (size_t) nblocks);
        pi8v2_launch_mask_scan((const half *) mask->data, nbm1, (int) Q->ne[1], n_kv,
                               rows, nblocks, tile_all, fl, km, stream);
        CUDA_CHECK(cudaGetLastError());
        tflags = fl;
        kvmax  = km;
    }

#if V2_SMEM_DYN > 0
    // QK=1 (64000 B) / QK=2+QLO_SMEM (79872 B): shared dinamica acima dos 48 KB padrao — opt-in uma
    // vez por device, como a FA mma de producao faz; 1 CTA/SM cabe nos 100 KB do SM sm_86
    {
        static bool smem_raised[64] = {false};
        const int dev = ggml_cuda_get_device();
        if (dev >= 0 && dev < 64 && !smem_raised[dev]) {
            // cada instanciacao de EMU tem o seu kernel — levanta o limite em todas as
            // que o launcher pode usar.
#define V2_RAISE(E) CUDA_CHECK(cudaFuncSetAttribute(flash_attn_prompt_pi8v2<E>,                 \
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, V2_SMEM_DYN))
            V2_RAISE(0); V2_RAISE(1); V2_RAISE(2); V2_RAISE(3); V2_RAISE(4); V2_RAISE(5); V2_RAISE(6);
            V2_RAISE(7); V2_RAISE(8); V2_RAISE(11); V2_RAISE(16); V2_RAISE(17); V2_RAISE(18);
            V2_RAISE(19); V2_RAISE(20);
#undef V2_RAISE
            smem_raised[dev] = true;
        }
    }
#endif
#define V2_LAUNCH(E)                                                                          \
    flash_attn_prompt_pi8v2<E><<<grid, block, V2_SMEM_DYN, stream>>>(                         \
        (const char *) Q->data, (const char *) K->data, (const char *) V->data,               \
        (const half *) mask->data, (float *) dst->data, r ? (const int *) r->data : nullptr, (int) K->ne[1], scale, (int) Q->ne[1], n_kv,\
        (int) Q->ne[2], gqa, Q->nb[1], Q->nb[2], K->nb[1], K->nb[2], V->nb[1], V->nb[2], nbm1,\
        Opart, ML, nsplit, tflags, kvmax, nullptr, nullptr, nullptr)
    switch (emu) {
        case 1: V2_LAUNCH(1); break;
        case 2: V2_LAUNCH(2); break;
        case 3: V2_LAUNCH(3); break;
        case 4: V2_LAUNCH(4); break;
        case 5: V2_LAUNCH(5); break;
        case 6: V2_LAUNCH(6); break;
        case 7: V2_LAUNCH(7); break;
        case 8: V2_LAUNCH(8); break;
        case 11: V2_LAUNCH(11); break;
        case 16: V2_LAUNCH(16); break;
        case 17: V2_LAUNCH(17); break;
        case 18: V2_LAUNCH(18); break;
        case 19: V2_LAUNCH(19); break;
        case 20: V2_LAUNCH(20); break;
        default: V2_LAUNCH(0); break;
    }
#undef V2_LAUNCH
    CUDA_CHECK(cudaGetLastError());
    if (nsplit > 1) {
        pi8v2_merge_kernel<<<(unsigned) rows_out, V2_D, 0, stream>>>(
            Opart, ML, (float *) dst->data, rows_out, nsplit);
        CUDA_CHECK(cudaGetLastError());
    }
}

#endif // !PI8V2_DEVICE_ONLY && GGML_EXP_CUDA_FA_PI8V2
