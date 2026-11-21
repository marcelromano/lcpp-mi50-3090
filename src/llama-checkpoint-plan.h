#pragma once
// plano físico (layout por dispositivo) de um intervalo lógico de um tensor dividido.
//
// Motivação: o save do checkpoint device copiava o byte-stream lógico para um destino
// *espelhado* (o mesmo stream em todas as GPUs). Metade das cópias era, portanto, peer
// (GPU→GPU, ~3 GB/s na MI50, medido na revisão anterior) e o dobro de bytes era movido. Aqui o intervalo
// é decomposto nas peças contíguas que cada dispositivo realmente possui: cópias D2D locais.
//
// O plano é uma função pura do descritor de split + bitola do tipo + intervalo, para poder ser
// testado em CPU sem backend (tests/test-checkpoint-plan.cpp).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

// descritor mínimo do split de um tensor (espelha ggml_backend_meta_split_state)
struct llama_ckpt_split {
    int      axis       = -1; // < 0: espelhado/idêntico em todos os ranks
    size_t   n_ranks    = 1;
    size_t   n_segments = 0;
    uint32_t nr[16]     = {};  // repetições por segmento
    size_t   ne[16*16]  = {};  // elementos ao longo do eixo dividido, por (segmento, rank)
};

struct llama_ckpt_piece {
    size_t rank    = 0; // índice do dispositivo simples
    size_t local   = 0; // offset no tensor simples desse rank
    size_t bytes   = 0; // bytes contíguos
    size_t logical = 0; // offset dentro do intervalo pedido
};

// Decompõe o intervalo lógico [offset, offset + size) de um tensor dividido com `total_bytes`.
// `all_ranks = false` (save): tensor espelhado produz a peça do rank 0 apenas.
// `all_ranks = true`  (restore): tensor espelhado produz uma peça por rank.
// Peças consecutivas contíguas (mesmo rank, local e lógico vizinhos) são fundidas.
// Retorna false quando o layout não é suportado: o chamador mantém o caminho legado.
inline bool llama_ckpt_plan(
        const llama_ckpt_split & split,
        size_t                   total_bytes,
        int                      type_size,
        int                      blck_size,
        size_t                   offset,
        size_t                   size,
        bool                     all_ranks,
        std::vector<llama_ckpt_piece> & out) {
    out.clear();
    if (split.n_ranks == 0 || split.n_ranks > 16) { return false; }
    if (type_size <= 0 || blck_size <= 0) { return false; }
    if (offset > total_bytes || size > total_bytes - offset) { return false; }
    if (size == 0) { return true; }

    if (split.axis < 0) {
        // espelhado: layout físico == lógico em cada rank
        const size_t ranks = all_ranks ? split.n_ranks : 1;
        for (size_t j = 0; j < ranks; ++j) {
            out.push_back({j, offset, size, 0});
        }
        return true;
    }
    if (split.axis >= 4 || split.n_segments == 0 || split.n_segments > 16) { return false; }

    const size_t n     = split.n_ranks;
    const bool   axis0 = split.axis == 0;

    // elementos ao longo do eixo dividido (soma dos segmentos, com repetições)
    size_t axis_elems = 0;
    for (size_t s = 0; s < split.n_segments; ++s) {
        for (size_t j = 0; j < n; ++j) { axis_elems += split.ne[s*n + j] * split.nr[s]; }
    }
    if (axis_elems == 0) { return false; }

    // bytes por elemento ao longo do eixo: eixo 0 é o bloco do tipo; os demais eixos têm
    // extensão inteira (tensor contíguo), derivável do total.
    size_t inner_bytes = 0;
    if (!axis0) {
        if (total_bytes % axis_elems != 0) { return false; }
        inner_bytes = total_bytes / axis_elems;
        if (inner_bytes == 0) { return false; }
    }

    size_t local_row_sum = 0;
    std::vector<size_t> local_row(n, 0);
    for (size_t s = 0; s < split.n_segments; ++s) {
        for (size_t j = 0; j < n; ++j) {
            const size_t el = split.ne[s*n + j];
            size_t bytes;
            if (axis0) {
                if (el % (size_t) blck_size != 0) { return false; }
                bytes = (el / (size_t) blck_size) * (size_t) type_size;
            } else {
                bytes = el * inner_bytes;
            }
            local_row[j] += bytes * split.nr[s];
        }
    }
    for (size_t j = 0; j < n; ++j) { local_row_sum += local_row[j]; }
    if (local_row_sum == 0 || total_bytes % local_row_sum != 0) { return false; }
    const size_t row     = local_row_sum;
    const size_t n_outer = total_bytes / row;

    // percorre apenas as linhas cobertas pelo intervalo
    const size_t first_o = offset / row;
    const size_t last_o  = (offset + size - 1) / row;
    if (last_o >= n_outer) { return false; }

    std::vector<size_t> local(n, 0);
    for (size_t j = 0; j < n; ++j) { local[j] = first_o * local_row[j]; }

    const size_t end = offset + size;
    for (size_t o = first_o; o <= last_o; ++o) {
        const size_t base = o * row;
        size_t pos = 0;
        for (size_t s = 0; s < split.n_segments; ++s) {
            for (size_t r = 0; r < split.nr[s]; ++r) {
                for (size_t j = 0; j < n; ++j) {
                    const size_t el = split.ne[s*n + j];
                    const size_t bytes = axis0
                        ? (el / (size_t) blck_size) * (size_t) type_size
                        : el * inner_bytes;
                    const size_t lcur = local[j];
                    local[j] += bytes;
                    const size_t l0 = base + pos;
                    pos += bytes;
                    const size_t l1 = l0 + bytes;
                    const size_t i0 = std::max(l0, offset);
                    const size_t i1 = std::min(l1, end);
                    if (i1 <= i0) { continue; }
                    llama_ckpt_piece p{j, lcur + (i0 - l0), i1 - i0, i0 - offset};
                    // funde quando o físico e o lógico continuam vizinhos
                    if (!out.empty()) {
                        llama_ckpt_piece & q = out.back();
                        if (q.rank == p.rank && q.local + q.bytes == p.local &&
                                q.logical + q.bytes == p.logical) {
                            q.bytes += p.bytes;
                            continue;
                        }
                    }
                    out.push_back(p);
                }
            }
        }
        if (pos != row) { out.clear(); return false; }
    }

    // cobertura exata do intervalo, sem buraco nem sobreposição
    size_t covered = 0;
    for (const auto & p : out) { covered += p.bytes; }
    if (covered != size) { out.clear(); return false; }
    return true;
}

// casamento entre o armazenamento salvo do checkpoint device e as peças que o destino pede.
//
// Puro (sem backend) para poder ser testado em CPU: o defeito da revisão anterior §3 foi exatamente uma
// chave ausente neste casamento (`mbufs.at(buft)` → `std::out_of_range` dentro de um destrutor
// noexcept → std::terminate → SIGABRT do servidor inteiro). Aqui a recusa é um `false`.
//
// `saved`: chave de armazenamento (`ggml_backend_buffer_type_t` do rank, opaca neste header) →
//          bytes de cada peça salva, na ordem em que foi gravada.
// `dest` : peças que o destino pede, na ordem em que o stream as apresenta (chave, bytes).
// `out`  : para cada peça do destino, o índice da peça salva correspondente (mesma chave).
// Retorna false para chave ausente, bytes diferentes ou peça salva sobrando. Nunca lança.
inline bool llama_ckpt_restore_map(
        const std::map<uintptr_t, std::vector<size_t>> & saved,
        const std::vector<std::pair<uintptr_t, size_t>> & dest,
        std::vector<size_t> & out) {
    out.assign(dest.size(), 0);
    std::map<uintptr_t, size_t> cursors;
    for (size_t i = 0; i < dest.size(); ++i) {
        const auto it = saved.find(dest[i].first);
        if (it == saved.end()) { return false; }
        auto & cursor = cursors[dest[i].first];
        if (cursor >= it->second.size()) { return false; }
        if (it->second[cursor] != dest[i].second) { return false; }
        out[i] = cursor++;
    }
    for (const auto & entry : saved) {
        const auto it = cursors.find(entry.first);
        const size_t used = it == cursors.end() ? 0 : it->second;
        if (used != entry.second.size()) { return false; }
    }
    return true;
}

// Layout legado (byte-stream lógico espelhado): o destino consome os bytes em ordem por chave, então
// o casamento é de totais por chave. Mesma regra: recusa não é exceção.
inline bool llama_ckpt_restore_totals_match(
        const std::map<uintptr_t, std::vector<size_t>> & saved,
        const std::map<uintptr_t, size_t> & dest) {
    std::map<uintptr_t, size_t> totals;
    for (const auto & entry : saved) {
        size_t sum = 0;
        for (const size_t bytes : entry.second) { sum += bytes; }
        totals[entry.first] = sum;
    }
    return totals == dest;
}
