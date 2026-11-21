// teste do plano físico do checkpoint device (CPU puro, sem backend).
//
// Oráculo independente: exige cobertura exata do intervalo (cada byte lógico exatamente uma vez),
// ausência de sobreposição física por rank, e confere o mapeamento em casos conhecidos. Cobre:
// split do eixo 0 com 2 ranks, múltiplos segmentos com repetição, split do eixo 1, espelho,
// quantizado no eixo 0, recorte parcial e entradas inválidas.
#include "../src/llama-checkpoint-plan.h"

#include <cstdio>
#include <map>
#include <set>
#include <vector>

static int failures = 0;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); ++failures; } } while (0)

static void verify(const llama_ckpt_split & sp, size_t offset, size_t size,
                   const std::vector<llama_ckpt_piece> & pieces) {
    std::set<size_t> logical;
    std::map<size_t, std::set<size_t>> physical;
    size_t total = 0;
    for (const auto & p : pieces) {
        CHECK(p.bytes > 0);
        CHECK(p.rank < sp.n_ranks);
        for (size_t b = 0; b < p.bytes; ++b) {
            CHECK(logical.insert(p.logical + b).second);
            CHECK(physical[p.rank].insert(p.local + b).second);
        }
        total += p.bytes;
    }
    CHECK(total == size);
    CHECK(logical.size() == size);
    if (!logical.empty()) {
        CHECK(*logical.begin() == 0);
        CHECK(*logical.rbegin() == size - 1);
    }
    (void) offset;
}

int main() {
    const int type_size = 4; // f32
    const int blck      = 1;

    // 1) eixo 0 com 2 ranks: ne0 = 8 (4 por rank), 3 linhas, 128 bytes de linha
    {
        llama_ckpt_split sp;
        sp.axis = 0; sp.n_ranks = 2; sp.n_segments = 1;
        sp.nr[0] = 1; sp.ne[0] = 4; sp.ne[1] = 4;
        const size_t total = 3 * 32; // 3 linhas de 8 elementos x 4 bytes
        std::vector<llama_ckpt_piece> pieces;
        CHECK(llama_ckpt_plan(sp, total, type_size, blck, 0, total, true, pieces));
        verify(sp, 0, total, pieces);
        // os ranks alternam dentro de cada linha: 6 peças de 16 bytes
        CHECK(pieces.size() == 6);
        CHECK(pieces[0].bytes == 16 && pieces[0].local == 0 && pieces[0].logical == 0 && pieces[0].rank == 0);
        CHECK(pieces[1].bytes == 16 && pieces[1].local == 0 && pieces[1].logical == 16 && pieces[1].rank == 1);
        CHECK(pieces[2].bytes == 16 && pieces[2].local == 16 && pieces[2].logical == 32 && pieces[2].rank == 0);
    }

    // 2) recorte: intervalo que começa e termina no meio de uma linha
    {
        llama_ckpt_split sp;
        sp.axis = 0; sp.n_ranks = 2; sp.n_segments = 1;
        sp.nr[0] = 1; sp.ne[0] = 4; sp.ne[1] = 4;
        const size_t total = 3 * 32;
        std::vector<llama_ckpt_piece> pieces;
        CHECK(llama_ckpt_plan(sp, total, type_size, blck, 8, 48, true, pieces));
        verify(sp, 8, 48, pieces);
        CHECK(pieces.front().logical == 0 && pieces.back().logical + pieces.back().bytes == 48);
        // peça que começa no meio da linha: rank 0 em local 8
        CHECK(pieces.front().rank == 0 && pieces.front().local == 8 && pieces.front().bytes == 8);
    }

    // 3) múltiplos segmentos com repetição (QKV fundido): segmento 0 repete 2x
    {
        llama_ckpt_split sp;
        sp.axis = 0; sp.n_ranks = 2; sp.n_segments = 2;
        sp.nr[0] = 2; sp.nr[1] = 1;
        sp.ne[0*2 + 0] = 2; sp.ne[0*2 + 1] = 2;   // 2 ranks x 2 elementos, 2 repetições -> 8
        sp.ne[1*2 + 0] = 4; sp.ne[1*2 + 1] = 4;   // 2 ranks x 4 elementos -> 8
        const size_t total = 2 * 64;               // linha = 16 elementos x 4 bytes
        std::vector<llama_ckpt_piece> pieces;
        CHECK(llama_ckpt_plan(sp, total, type_size, blck, 0, total, true, pieces));
        verify(sp, 0, total, pieces);
        CHECK(pieces.size() == 12);                // 6 peças por linha (segmentos x repetições x ranks) x 2 linhas
    }

    // 4) eixo 1 (células) com 2 ranks: ne1 = 8 (4 por rank), 4 elementos por célula
    {
        llama_ckpt_split sp;
        sp.axis = 1; sp.n_ranks = 2; sp.n_segments = 1;
        sp.nr[0] = 1; sp.ne[0] = 4; sp.ne[1] = 4;
        const size_t total = 4 * 8 * 4; // 4 elementos x 8 células x 4 bytes
        std::vector<llama_ckpt_piece> pieces;
        CHECK(llama_ckpt_plan(sp, total, type_size, blck, 0, total, true, pieces));
        verify(sp, 0, total, pieces);
        CHECK(pieces.size() == 2);
        CHECK(pieces[0].bytes == total / 2 && pieces[1].bytes == total / 2);
        CHECK(pieces[0].local == 0 && pieces[1].local == 0);
    }

    // 5) espelhado: todos os ranks recebem a mesma peça
    {
        llama_ckpt_split sp;
        sp.axis = -1; sp.n_ranks = 2;
        std::vector<llama_ckpt_piece> pieces;
        CHECK(llama_ckpt_plan(sp, 64, type_size, blck, 4, 40, true, pieces));
        CHECK(pieces.size() == 2);
        CHECK(pieces[0].rank == 0 && pieces[1].rank == 1);
        CHECK(pieces[0].local == 4 && pieces[0].bytes == 40 && pieces[0].logical == 0);
        CHECK(pieces[1].local == 4 && pieces[1].bytes == 40 && pieces[1].logical == 0);
        // modo save (all_ranks = false): uma única peça
        CHECK(llama_ckpt_plan(sp, 64, type_size, blck, 4, 40, false, pieces));
        CHECK(pieces.size() == 1 && pieces[0].rank == 0);
    }

    // 6) quantizado no eixo 0 (bloco de 32 elementos, 18 bytes por bloco)
    {
        llama_ckpt_split sp;
        sp.axis = 0; sp.n_ranks = 2; sp.n_segments = 1;
        sp.nr[0] = 1; sp.ne[0] = 64; sp.ne[1] = 64; // 2 blocos por rank por linha
        const size_t total = 2 * 2 * 36;             // 2 linhas x 2 ranks x 36 bytes
        std::vector<llama_ckpt_piece> pieces;
        CHECK(llama_ckpt_plan(sp, total, 18, 32, 0, total, true, pieces));
        verify(sp, 0, total, pieces);
        CHECK(pieces.size() == 4);                   // 2 ranks x 2 linhas
        CHECK(pieces[0].bytes == 36 && pieces[0].local == 0);
    }

    // 7) entradas inválidas/fora do intervalo: recusa, sem plano parcial
    {
        llama_ckpt_split sp;
        sp.axis = 0; sp.n_ranks = 2; sp.n_segments = 1;
        sp.nr[0] = 1; sp.ne[0] = 4; sp.ne[1] = 4;
        const size_t total = 3 * 32;
        std::vector<llama_ckpt_piece> pieces;
        CHECK(!llama_ckpt_plan(sp, total, type_size, blck, 90, 16, true, pieces));
        CHECK(!llama_ckpt_plan(sp, total, type_size, blck, 0, total + 1, true, pieces));
        sp.n_ranks = 0;
        CHECK(!llama_ckpt_plan(sp, total, type_size, blck, 0, 8, true, pieces));
        sp.n_ranks = 2; sp.axis = 9;
        CHECK(!llama_ckpt_plan(sp, total, type_size, blck, 0, 8, true, pieces));
        sp.axis = 0; sp.ne[0] = 3; // soma dos segmentos deixa de dividir o total
        CHECK(!llama_ckpt_plan(sp, total, type_size, blck, 0, 8, true, pieces));
        CHECK(pieces.empty());
    }

    // 8) casamento do armazenamento salvo com as peças que o destino pede (layout por
    // dispositivo). O defeito da revisão anterior §3 foi uma chave AUSENTE aqui: `mbufs.at(buft)` lançava
    // std::out_of_range dentro de um destrutor noexcept → std::terminate → SIGABRT do servidor.
    {
        // chaves opacas: 0x100 = buffer Meta do tensor lógico, 0x200/0x201 = buffers dos ranks
        const std::map<uintptr_t, std::vector<size_t>> saved = {
            {0x200, {1024, 512, 1024}},
            {0x201, {1024, 512, 1024}},
        };
        std::vector<size_t> idx;

        // (a) mesmo layout, mesma ordem: casa e devolve os índices
        CHECK(llama_ckpt_restore_map(saved, {{0x200, 1024}, {0x201, 1024}, {0x200, 512},
                                             {0x201, 512}, {0x200, 1024}, {0x201, 1024}}, idx));
        CHECK(idx == std::vector<size_t>({0, 0, 1, 1, 2, 2}));

        // (b) destino num único buffer Meta: o save foi por dispositivo — recusa (não exceção)
        CHECK(!llama_ckpt_restore_map(saved, {{0x100, 2048}, {0x100, 2048}, {0x100, 2048}}, idx));

        // (c) chave presente mas ordem trocada (bytes diferentes na posição): recusa
        CHECK(!llama_ckpt_restore_map(saved, {{0x200, 512}, {0x201, 1024}}, idx));

        // (d) peça salva sobrando (destino pede menos do que foi gravado): recusa
        CHECK(!llama_ckpt_restore_map(saved, {{0x200, 1024}, {0x201, 1024}}, idx));

        // (e) destino pede mais peças do que existem: recusa
        std::vector<std::pair<uintptr_t, size_t>> too_many;
        for (int i = 0; i < 4; ++i) { too_many.push_back({0x200, 1024}); }
        too_many.push_back({0x201, 1024}); too_many.push_back({0x201, 1024}); too_many.push_back({0x201, 1024});
        CHECK(!llama_ckpt_restore_map(saved, too_many, idx));

        // (f) armazenamento vazio com destino não vazio: recusa (e o caso b vira o oposto de `at`)
        CHECK(!llama_ckpt_restore_map({}, {{0x200, 16}}, idx));
        CHECK(llama_ckpt_restore_map({}, {}, idx) && idx.empty());

        // (g) espelhado: cada rank tem a mesma peça salva, na mesma ordem
        const std::map<uintptr_t, std::vector<size_t>> mirror = {{0x300, {128}}, {0x301, {128}}};
        CHECK(llama_ckpt_restore_map(mirror, {{0x301, 128}, {0x300, 128}}, idx));
        CHECK(idx == std::vector<size_t>({0, 0}));
    }

    // 9) layout legado: o casamento é de totais por chave (o destino consome o byte-stream
    // em ordem, então a divisão em views não é 1:1 com as peças salvas)
    {
        const std::map<uintptr_t, std::vector<size_t>> saved = {
            {0x200, {1024, 512}},
            {0x201, {1536}},
        };
        CHECK(llama_ckpt_restore_totals_match(saved, {{0x200, 1536}, {0x201, 1536}}));
        // chave ausente no armazenamento salvo: recusa (era o `map::at`)
        CHECK(!llama_ckpt_restore_totals_match(saved, {{0x100, 1536}}));
        // chave a mais no destino
        CHECK(!llama_ckpt_restore_totals_match(saved, {{0x200, 1536}, {0x201, 1024}, {0x202, 512}}));
        // mesmas chaves, bytes diferentes (destino menor/maior que o salvo)
        CHECK(!llama_ckpt_restore_totals_match(saved, {{0x200, 1536}, {0x201, 1024}}));
        CHECK(!llama_ckpt_restore_totals_match(saved, {{0x200, 2048}, {0x201, 1536}}));
        // vazio dos dois lados casa; armazenamento vazio com destino não vazio não
        CHECK(llama_ckpt_restore_totals_match({}, {}));
        CHECK(!llama_ckpt_restore_totals_match({}, {{0x200, 16}}));
    }

    if (failures) {
        std::fprintf(stderr, "FAILED: %d\n", failures);
        return 1;
    }
    std::printf("PASS checkpoint plan: cobertura exata, recorte, segmentos/repetições, eixo 1, "
                "espelho, quantizado, casamento do restore (por peça e por total)\n");
    return 0;
}
