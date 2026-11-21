# Snapshots de produção — MI50 (gfx906) e RTX 3090 (sm86)

## Licença e atribuição

Esta linha deriva do upstream [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
(MIT), a partir da release **v1.0.0** (`b1b770d90`). O histórico upstream e os
arquivos `LICENSE`, `AUTHORS` e `CONTRIBUTING` permanecem intactos; a atribuição
aos autores do código de origem não é substituída pela autoria dos commits de
integração deste fork.

Parte dos kernels e da instrumentação foi desenvolvida com auxílio de agentes de
IA. Não se reivindica exclusividade sobre código upstream nem sobre as ideias
citadas. Avisos de terceiros já presentes no upstream (por exemplo o copyright
ggml do backport GCN) foram preservados.

## Rótulos internos

Os nomes de macro/env `GGML_EXP_*_T20xx`, a env `T2012_TRACE` e os prefixos de
log no formato `[t2012-*]` / `tNNNN ...` são **rótulos internos de experimento**.
Eles foram mantidos de propósito: são a receita de build e a saída de diagnóstico
do binário de produção, e renomeá-los invalidaria a correspondência exata entre
árvore, flags e binário. Não têm semântica pública, não indicam versão nem
estado de suporte. O mesmo vale para os nomes de tags e referências `tNNNN` que
aparecem em comentários como marcadores de linhagem interna.

## Linhagem das branches

| ref | base | conteúdo | binário `llama-server` (sha256) |
|---|---|---|---|
| `master` | upstream `de699957b` | série curada de backports por finalidade (perfis de build explícitos, testes com geradores sintéticos) | — |
| `prod/mi50-v1` + tag `prod/mi50-t2101` | `v1.0.0` | 7 commits por área sobre a árvore viva da produção MI50 (4 198 arquivos) | `1354bf687bd5f12aac4efcf33a325f8790291d61ba9b4c5cd10fcad115a410dd` |
| `prod/3090-v1` + tag `prod/3090-t2087` | `v1.0.0` | baseline do snapshot upstream da produção reconstruído + 5 áreas + correção de cache de prompt | `4d317d3b84e24b35716e452149c9bfeb67d19e3c1e9671a5c67f693ebc912d5c` |
| `prod/multiarch-t2112` | `prod/mi50-v1` | só receita: `-DAMDGPU_TARGETS/-DGPU_TARGETS="gfx906;gfx1100"`; fonte idêntica à `prod/mi50-v1` | `6725b4e53f71f34df6dfff2cfc4b51d07bd3669dd123b7b9be0b5d7d092cbaf6` |
| tag `archive/publish-20261104` | `de699957b` | ponta pública anterior (`33800c21`), preservada como arquivo | — |

O commit de docs desta série não altera código: acrescenta este arquivo e, na
`prod/multiarch-t2112`, a receita em `docs/BUILD-MULTIARCH.md`. A
`prod/multiarch-t2112` restaura ainda os três arquivos de upstream que a árvore
importada havia perdido (`build-xcframework.sh`, um log de bench e um
`xcshareddata/`), de modo que ela não parece apagar arquivos do upstream.

## Proveniência e limites

Cada branch é a **fonte exata** da árvore que gerou o binário indicado acima
(comparação `diff -r` vazia contra a árvore de produção correspondente, exceto
por linhas de comentário e por este `docs/FORK.md`). A correspondência
binário↔árvore↔flags é o que os rótulos internos preservam.

Estes são snapshots pessoais de produção, publicados como registro técnico, **sem
suporte** e **sem validação numérica** nesta publicação. Não são uma release do
upstream, não representam um rebase ao upstream mais recente e não devem ser
lidos como recomendação de flags para outras máquinas. Reproduzir os números
exige a mesma GPU, runtime, modelo e janela de medição.
