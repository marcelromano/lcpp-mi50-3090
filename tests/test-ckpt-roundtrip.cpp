// roundtrip concorrente do checkpoint device (save/restore pelo layout por dispositivo).
//
// Prova, no modelo real: (a) o save device é byte-exato contra o save host nativo; (b) a
// materialização para host concorrente (duas threads sobre leases distintos) devolve os mesmos
// bytes; (c) o restore no segundo contexto reproduz o estado (save host do alvo == nativo);
// (d) re-materializar depois do restore continua byte-exato. Uso: test-ckpt-roundtrip MODEL [ngl]
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); return 1; } } while (0)

static bool decode(llama_context * ctx, int start, int n, int seq) {
    auto b = llama_batch_init(n, 0, 1);
    b.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        b.token[i]     = 19 + (start + i) % 10;
        b.pos[i]       = start + i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = seq;
        b.logits[i]    = i + 1 == n;
    }
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc == 0;
}

static std::vector<uint8_t> host_bytes(llama_context * ctx, llama_seq_id seq) {
    std::vector<uint8_t> buf(llama_state_seq_get_size(ctx, seq));
    if (buf.empty()) { return {}; }
    if (llama_state_seq_get_data(ctx, buf.data(), buf.size(), seq) != buf.size()) { return {}; }
    return buf;
}

using lease_t = std::unique_ptr<llama_state_seq_device, decltype(&llama_state_seq_device_free)>;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL [ngl] [tensor|layer]\n", argv[0]);
        return 1;
    }
    const int ngl = argc > 2 ? std::atoi(argv[2]) : 0;
    llama_backend_init();

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    // a topologia TEM de ser a do preset de produção (`split-mode = tensor`). Com
    // `split-mode = layer` cada camada mora inteira numa GPU, o KV não vira um tensor Meta e o
    // save cai no caminho legado — o teste passa sem nunca exercitar o layout por dispositivo, que
    // é o que aborta no servidor (a revisão anterior rodou exatamente assim e por isso não viu o defeito).
    // `argv[3]` = tensor (default) | layer, para publicar a diferença.
    const char * split = argc > 3 ? argv[3] : "tensor";
    const bool split_layer = strcmp(split, "layer") == 0;
    if (ngl > 0) {
        mp.split_mode = split_layer ? LLAMA_SPLIT_MODE_LAYER : LLAMA_SPLIT_MODE_TENSOR;
        mp.main_gpu   = 0;
        static const float ts[2] = {0.5f, 0.5f};
        mp.tensor_split = ts;
    }
    std::fprintf(stderr, "ckpt-roundtrip: ngl=%d split-mode=%s\n", ngl, split_layer ? "layer" : "tensor");
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], mp), llama_model_free);
    REQUIRE(model);

    auto cp = llama_context_default_params();
    cp.n_ctx      = 1024;
    cp.n_batch    = 64;
    cp.n_ubatch   = 64;
    cp.n_seq_max  = 2;
    cp.n_rs_seq   = 4;
    cp.kv_unified = true;
    cp.n_threads       = 2;
    cp.n_threads_batch = 2;

    std::unique_ptr<llama_context, decltype(&llama_free)> source(llama_init_from_model(model.get(), cp), llama_free);
    std::unique_ptr<llama_context, decltype(&llama_free)> target(llama_init_from_model(model.get(), cp), llama_free);
    REQUIRE(source && target);

    REQUIRE(decode(source.get(), 0, 16, 0));
    REQUIRE(decode(source.get(), 0,  8, 1));

    const auto native0 = host_bytes(source.get(), 0);
    const auto native1 = host_bytes(source.get(), 1);
    REQUIRE(!native0.empty() && !native1.empty());

    lease_t a(llama_state_seq_device_acquire(source.get(), 0, 2), llama_state_seq_device_free);
    lease_t b(llama_state_seq_device_acquire(source.get(), 1, 2), llama_state_seq_device_free);
    REQUIRE(a && b);

    // (a) save device concorrente dos dois leases + (b) materialização concorrente byte-exata
    std::vector<uint8_t> meta0(llama_state_seq_get_size_ext(source.get(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE));
    std::vector<uint8_t> meta1(llama_state_seq_get_size_ext(source.get(), 1, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE));
    REQUIRE(!meta0.empty() && !meta1.empty());
    REQUIRE(llama_state_seq_get_size_ext(source.get(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) == meta0.size());

    // (a) save device concorrente dos dois leases
    int rc0 = 0, rc1 = 0;
    std::thread t0([&] {
        if (llama_state_seq_device_get(source.get(), meta0.data(), meta0.size(), 0, 0, a.get()) != meta0.size()) { rc0 = 1; }
    });
    std::thread t1([&] {
        if (llama_state_seq_device_get(source.get(), meta1.data(), meta1.size(), 1, 0, b.get()) != meta1.size()) { rc1 = 1; }
    });
    t0.join();
    t1.join();
    REQUIRE(rc0 == 0);
    REQUIRE(rc1 == 0);

    // o buffer de metadados é só o staging do byte-stream: o total lógico vem dos segmentos
    const size_t logical0 = llama_state_seq_device_to_host(a.get(), nullptr, 0, nullptr, 0);
    const size_t logical1 = llama_state_seq_device_to_host(b.get(), nullptr, 0, nullptr, 0);
    REQUIRE(logical0 == native0.size());
    REQUIRE(logical1 == native1.size());

    // (b) materialização concorrente byte-exata
    std::vector<uint8_t> host0(logical0), host1(logical1);
    std::thread m0([&] {
        if (llama_state_seq_device_to_host(a.get(), meta0.data(), meta0.size(), host0.data(), host0.size()) != host0.size()) { rc0 = 2; return; }
        if (host0 != native0) { rc0 = 3; }
    });
    std::thread m1([&] {
        if (llama_state_seq_device_to_host(b.get(), meta1.data(), meta1.size(), host1.data(), host1.size()) != host1.size()) { rc1 = 2; return; }
        if (host1 != native1) { rc1 = 3; }
    });
    m0.join();
    m1.join();
    REQUIRE(rc0 == 0);
    REQUIRE(rc1 == 0);

    // materialização repetida (segunda thread sobre o mesmo lease já materializado)
    std::vector<uint8_t> again(logical0);
    REQUIRE(llama_state_seq_device_to_host(a.get(), meta0.data(), meta0.size(), again.data(), again.size()) == again.size());
    REQUIRE(again == native0);

    // (c) restore concorrente pelo armazenamento device no `target` e pelo byte-stream host no
    //     `target_host` (contexto novo). Os dois caminhos devem produzir o mesmo estado: o save
    //     host do próprio contexto é o oráculo que não depende de canonicalidade de índice de célula.
    std::unique_ptr<llama_context, decltype(&llama_free)> target_host(llama_init_from_model(model.get(), cp), llama_free);
    REQUIRE(target_host);
    int rr0 = 0, rr1 = 0;
    // dois contextos distintos: restore pelo armazenamento device (target) em paralelo com o
    // restore pelo byte-stream host (target_host). Restaurar dois seqs no MESMO contexto não é
    // seguro (o restore muta o KV cache), então a concorrência é entre contextos.
    std::thread r0([&] {
        if (llama_state_seq_device_set(target.get(), meta0.data(), meta0.size(), 0, 0, a.get()) != meta0.size()) { rr0 = 1; return; }
        if (llama_state_seq_device_set(target.get(), meta1.data(), meta1.size(), 1, 0, b.get()) != meta1.size()) { rr0 = 2; }
    });
    std::thread r1([&] {
        if (llama_state_seq_set_data(target_host.get(), host0.data(), host0.size(), 0) != host0.size()) { rr1 = 1; return; }
        if (llama_state_seq_set_data(target_host.get(), host1.data(), host1.size(), 1) != host1.size()) { rr1 = 2; }
    });
    r0.join();
    r1.join();
    REQUIRE(rr0 == 0);
    REQUIRE(rr1 == 0);
    const auto dev0 = host_bytes(target.get(), 0);
    const auto dev1 = host_bytes(target.get(), 1);
    const auto host_ref0 = host_bytes(target_host.get(), 0);
    const auto host_ref1 = host_bytes(target_host.get(), 1);
    REQUIRE(!dev0.empty() && !dev1.empty());
    REQUIRE(dev0.size() == host_ref0.size());
    REQUIRE(dev1.size() == host_ref1.size());
    REQUIRE(dev0 == host_ref0);
    REQUIRE(dev1 == host_ref1);

    // (d) re-materialização depois do restore continua byte-exata
    REQUIRE(llama_state_seq_device_to_host(a.get(), meta0.data(), meta0.size(), again.data(), again.size()) == again.size());
    REQUIRE(again == native0);

    // (e) restore DEPOIS de truncar a sequência, o padrão S4→S5→S6 da revisão anterior §3: o cache do
    //     slot fica MAIOR que o pedido, o servidor trunca e restaura o checkpoint. Antes da revisão anterior
    //     este caminho matava o processo (`std::out_of_range: map::at` no destrutor do leitor de
    //     checkpoint device, porque o destino era um tensor do buffer Meta e o armazenamento salvo
    //     estava indexado pelos buffers de cada GPU). Repete 5×, que é a contagem de requisições
    //     truncantes que a produção atendeu na revisão anterior.
    {
        auto * mem = llama_get_memory(source.get());
        for (int rep = 0; rep < 5; ++rep) {
            REQUIRE(decode(source.get(), 16, 4, 0));       // cache cresce além do checkpoint (posições 16..19)
            // truncatura: parcial quando o cache suporta (atenção); total no híbrido/recurrente, que é
            // justamente onde o servidor usa o checkpoint para voltar ao estado anterior
            if (!llama_memory_seq_rm(mem, 0, 8, -1)) {
                REQUIRE(llama_memory_seq_rm(mem, 0, -1, -1));
            }
            REQUIRE(llama_state_seq_device_set(source.get(), meta0.data(), meta0.size(), 0, 0, a.get()) == meta0.size());
            const auto restored = host_bytes(source.get(), 0);
            REQUIRE(!restored.empty());
            REQUIRE(restored == native0);                  // restore byte-exato
        }
        std::fprintf(stderr, "PASS ckpt truncation: 5/5 restores apos truncar, byte-exatos\n");
    }

    std::fprintf(stderr, "PASS ckpt roundtrip (ngl=%d): save/restore device concorrente byte-exato, "
                         "materializacao repetida e restore do host\n", ngl);
    return 0;
}
