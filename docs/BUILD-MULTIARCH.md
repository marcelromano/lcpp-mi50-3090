# Build multi-arch: gfx906 e gfx1100 na mesma árvore

Esta branch não muda um byte de código em relação a `prod/mi50-v1`. O patch de
fonte do experimento multi-arch é vazio; o que muda é a receita de compilação.

## Receita

```sh
cmake -S . -B build-multiarch \
  -DGPU_TARGETS="gfx906;gfx1100" -DAMDGPU_TARGETS="gfx906;gfx1100" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-multiarch --target llama-server -j3
```

Runtime: ROCm 7.2.4 com bibliotecas que suportem gfx906, em um builder que aceite
os dois alvos. O artefato tem 286 code objects (143 gfx906 + 143 gfx1100).

`llama-server` resultante (sha256):
`6725b4e53f71f34df6dfff2cfc4b51d07bd3669dd123b7b9be0b5d7d092cbaf6`.

## Resultado e limites

O portão de arquitetura reprovou por bloat: 4 kernels wave64 aparecem no code
object do gfx1100 e nenhum deles é despachável fora de gfx906. Por isso esta
branch é a publicação da **receita**, não de um build promovido. Não há medição
numérica do binário multi-arch nesta publicação; ver `docs/FORK.md` para o aviso
geral de limites.
