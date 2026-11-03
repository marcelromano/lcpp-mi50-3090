# Fork para RTX 3090 e AMD MI50

## Base e escopo

Esta linha deriva do upstream `ggml-org/llama.cpp` em
`de699957b92f490efebad149665b0dccf127eaff` (MIT). O histórico upstream
permanece intacto. Os commits posteriores separam os patches por finalidade;
a autoria dos commits de integração não substitui a atribuição aos autores
do código de origem.

A combinação preserva os patches de duas árvores de produção históricas,
sem importar seus históricos de snapshots, receitas privadas ou artefatos.
Não representa um rebase ao upstream mais recente. Nenhum resultado de
performance ou qualidade de uma árvore isolada deve ser atribuído à combinação.

## Origem dos patches

| Patch | Origem e adaptação |
|---|---|
| Recursos de memória, checkpoint limitado e restore híbrido | Original deste fork, sobre serialização e memória upstream. Mantém limites de leitura antes de escritas, contabilidade por recurso e snapshots recorrentes. |
| Estado especulativo | Original: preservação/limpeza da linha oculta MTP e estado Eagle3 nas operações de slot. |
| Operações de slot e admissão KV | Original: protocolo transacional, fencing, limites por recurso e escopo de admissão. |
| Sampling esparso e greedy rápido | Original sobre samplers upstream; mantém fallback para cadeias não elegíveis. Não inclui experimentos de prefilter. |
| Fusões GPU de decode e quantização compartilhada | Original sobre as primitivas upstream; macros opt-in preservam os caminhos comuns. |
| Tabela MMQ GCN | Backport do [PR #27841](https://github.com/ggml-org/llama.cpp/pull/27841), por thelittlefireman, commit `2df4b9551bbc2e1a4600fb85cf180c91aa3d7a2f`. |
| Extensões gfx906 | Extensões locais originais: geometria wave64, MMVQ/IQ, LDS/SWAR, reutilização de cargas e tabelas de prefill, sobre o MMQ/vecdot upstream e a tabela GCN acima. A pilha histórica também é usada com o runtime [ML-gfx906](https://github.com/mixa3607/ML-gfx906); não se importa aqui a receita de empacotamento desse projeto. |
| COLA/estado de convolução | Implementação local. A inspiração explícita no código para a fusão do estado de convolução é [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), não uma importação de seu backend inteiro. |
| Flash Attention KVQ/GQA | Original: K q8_0, V q4_0, dimensão 256, query quantizada e reuse de tiles GQA. [PR #30792](https://github.com/ggml-org/llama.cpp/pull/30792) e [issue #29935](https://github.com/ggml-org/llama.cpp/issues/29935) são trabalho relacionado, não doadores. |
| GDN MMA | Backport de [#29353](https://github.com/ggml-org/llama.cpp/pull/29353), commit upstream `62df1815bc053066761f4eac58e288f23778e164`, com origem anterior em qvac-fabric PR257, commit `53f0027d8070715fe9ad2131f3201bb0001d4eed`. Helpers isolados de swizzle/ldmatrix, adaptação local de IDs de estado, replay escalar dos últimos snapshots e produto BF16 low-low opcional. Aviso de copyright ggml preservado. |
| AllReduce HIP BF16 | Original: staging pinned/copy-only de dois dispositivos, tensores F32 e transporte BF16. Suporte HIP upstream [#27825](https://github.com/ggml-org/llama.cpp/pull/27825) é contexto relacionado; CUDA conserva seu pipeline existente. O arredondamento do parcial remoto não é numericamente exato em F32. |
| KV span por sequência | Original: seleção de janela, colocação por sequência, máscara e views relativas à janela. |
| Pré-liberação especulativa e cancelamento de proxy | Originais: liberar cauda obsoleta antes do novo prompt; abortar upstream somente enquanto a requisição está em voo. |
| Testes e perfis de build | Integração original deste fork. |

Parte dos kernels foi desenvolvida com auxílio de agentes de IA. Não se
reivindica exclusividade sobre código upstream ou sobre as ideias citadas.
A licença MIT em `LICENSE` e os avisos de terceiros existentes foram mantidos.

## Compilação

Perfis explícitos; use diretórios de build separados. Sem perfil, as macros
experimentais não são habilitadas em bloco.

RTX 3090, CUDA 12.8 e compilador host compatível:

```sh
cmake -S . -B build-3090 -DLLAMA_FORK_TARGET=rtx3090 \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_GDN_MMA_FOUR_TERM=ON \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-3090 --target llama-server test-backend-ops -j3
```

O perfil seleciona sm_86 e FA all-quants. `GGML_CUDA_GDN_MMA=1` habilita
GDN MMA em runtime; o padrão continua desligado. O backport exige um prefixo
suficiente e uma sequência elegível; decode pequeno, KDA e shapes não
suportados conservam fallback. Quatro termos recompõem o produto das duas
componentes BF16, não todos os bits dos operandos F32 originais.

MI50, ROCm 7.2.4 com bibliotecas que suportem gfx906:

```sh
cmake -S . -B build-mi50 -DLLAMA_FORK_TARGET=mi50 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-mi50 --target llama-server test-backend-ops -j3
```

O perfil seleciona gfx906, FA all-quants e as macros da pilha histórica.
O arquivo GDN MMA CUDA não é compilado pelo backend HIP. COLA, GCN e os
predicados de dispatch mantêm seus controles de runtime. Usar `-D` CMake
sem uma opção conectada não habilita uma macro C/C++; os perfis acima fazem
essa ligação explicitamente, inclusive entre common, llama e backend.

NCCL/RCCL são opcionais e dependem da instalação. Os builds de integração
podem desabilitar NCCL; isso não testa sua comunicação nem seu desempenho.
Não misture bibliotecas de árvores ou perfis diferentes: o estado e os
layouts dos recursos fazem parte do contrato binário desta linha.

## Validação e limites

A revisão de publicação exige build completo de `llama-server` e
`test-backend-ops` para os dois perfis, além dos testes CPU de recursos e
admissão sem modelos externos. Os casos de restore com modelo devem receber
um GGUF de teste; seus caminhos de exemplo não representam instalações reais.
Os geradores em `tests/b3` produzem pesos sintéticos, não modelos treinados.

Build em máquinas sem GPU não prova execução de kernels, fidelidade,
serving, AllReduce de dois dispositivos ou segurança de checkpoints
não confiáveis. Revalidar numericalidade, snapshots, concorrência e
performance na revisão exata antes de promover a combinação. Não estão
incluídos experimentos rejeitados de GDN wave64, overlap INT8, compact-mask,
MTP adaptativo/launch ou prefilter de sampling.
