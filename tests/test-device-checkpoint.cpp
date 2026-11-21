// Structural lease and lossless materialization tests, independent of backend arithmetic.
#include "llama.h"
#include <algorithm>
#include <cstdio>
#include <memory>
#include <vector>
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x); return 1; } } while (0)
static bool decode(llama_context * ctx, int start, int n, int seq) {
    auto b = llama_batch_init(n, 0, 1); b.n_tokens = n;
    for (int i=0;i<n;++i) {
        b.token[i]=19+i%10; b.pos[i]=start+i; b.n_seq_id[i]=1; b.seq_id[i][0]=seq; b.logits[i]=i+1==n;
    }
    int rc=llama_decode(ctx,b); llama_batch_free(b); return rc==0;
}
static std::vector<uint8_t> bytes(llama_context * c,int seq) {
    std::vector<uint8_t> b(llama_state_seq_get_size(c,seq));
    if (llama_state_seq_get_data(c,b.data(),b.size(),seq)!=b.size()) b.clear();
    return b;
}
using lease=std::unique_ptr<llama_state_seq_device,decltype(&llama_state_seq_device_free)>;
int main(int argc,char ** argv) {
    REQUIRE(argc==2); llama_backend_init();
    auto mp=llama_model_default_params(); mp.n_gpu_layers=0;
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);
    REQUIRE(model);
    auto cp=llama_context_default_params(); cp.n_ctx=512; cp.n_batch=64; cp.n_ubatch=64;
    cp.n_seq_max=2; cp.n_rs_seq=4; cp.kv_unified=true; cp.n_threads=2; cp.n_threads_batch=2;
    std::unique_ptr<llama_context,decltype(&llama_free)> source(llama_init_from_model(model.get(),cp),llama_free);
    std::unique_ptr<llama_context,decltype(&llama_free)> target(llama_init_from_model(model.get(),cp),llama_free);
    REQUIRE(source && target); REQUIRE(decode(source.get(),0,16,0)); REQUIRE(decode(source.get(),0,8,1));
    auto original=bytes(source.get(),0); REQUIRE(!original.empty());
    lease a(llama_state_seq_device_acquire(source.get(),0,2),llama_state_seq_device_free);
    lease b(llama_state_seq_device_acquire(source.get(),0,2),llama_state_seq_device_free);
    REQUIRE(a && b); REQUIRE(!llama_state_seq_device_acquire(source.get(),0,2));
    auto flags=LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
    std::vector<uint8_t> metadata(llama_state_seq_get_size_ext(source.get(),0,flags));
    REQUIRE(llama_state_seq_device_get(source.get(),metadata.data(),metadata.size(),0,0,a.get())==metadata.size());
    REQUIRE(decode(source.get(),16,5,0));
    const auto size=llama_state_seq_device_to_host(a.get(),metadata.data(),metadata.size(),nullptr,0);
    std::vector<uint8_t> host(size);
    REQUIRE(size==original.size());
    REQUIRE(llama_state_seq_device_to_host(a.get(),metadata.data(),metadata.size(),host.data(),host.size())==host.size());
    REQUIRE(host==original);
    REQUIRE(llama_state_seq_device_set(target.get(),metadata.data(),metadata.size(),0,0,a.get())==metadata.size());
    REQUIRE(bytes(target.get(),0)==original);
    // Release then reacquire: bounded pool may reuse storage only after the last lease.
    b.reset(); b.reset(llama_state_seq_device_acquire(source.get(),0,2)); REQUIRE(b);
    source.reset(); // an outstanding lease remains valid after context/stream destruction
    REQUIRE(llama_state_seq_device_to_host(a.get(),metadata.data(),metadata.size(),host.data(),host.size())==host.size());
    REQUIRE(host==original);
    REQUIRE(llama_state_seq_set_data(target.get(),host.data(),host.size(),0)==host.size());
    REQUIRE(bytes(target.get(),0)==original);
    std::fprintf(stderr,"PASS device checkpoint: bounded leases, D2D restore, byte-exact host materialization, context lifetime\n");
    return 0;
}
