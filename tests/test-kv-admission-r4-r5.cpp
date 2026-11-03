#include "llama.h"
#include "../tools/server/kv_planner.hpp"
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#define CHECK(x) do { if(!(x)){std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);return 1;} } while(0)
using namespace kvplan;
// Where the restore is held while a competitor is probed, and where a failure is injected.
enum class Mode { control, fail_tensor_write, fail_consume };
struct Writer {
    Planner * planner{}; RestoreToken * token{}; ResourceKey key{};
    Mode mode{Mode::control}; bool inject{true};
    std::mutex mutex; std::condition_variable cv; bool entered=false,release=false,block=false;
    std::atomic<unsigned> begins{0},writes{0},consume_calls{0},injected{0}; std::atomic<uint64_t> consumed{0},fail_bytes{0};
};
static bool consume(void * user,uint32_t kind,uint32_t stream,uint64_t count) {
    (void)kind;(void)stream;auto * w=static_cast<Writer *>(user);
    const unsigned call=w->consume_calls.fetch_add(1);
    if(w->mode==Mode::fail_consume&&call==0&&w->inject){
        // mid-consume: half of the cells are charged to the token, then the consumer fails while a competitor is probed
        const uint64_t half=count/2;
        if(!w->planner->consume_restore(*w->token,w->key,half))return false;
        w->consumed.fetch_add(half);
        {std::unique_lock<std::mutex> lock(w->mutex);w->entered=true;w->cv.notify_all();w->cv.wait(lock,[&]{return w->release;});}
        w->injected.fetch_add(1);return false;
    }
    if(!w->planner->consume_restore(*w->token,w->key,count))return false;
    w->consumed.fetch_add(count);
    if(w->mode==Mode::fail_consume&&call==0&&w->block){std::unique_lock<std::mutex> lock(w->mutex);w->entered=true;w->cv.notify_all();w->cv.wait(lock,[&]{return w->release;});}
    return true;
}
static void before_replace(void * user){static_cast<Writer *>(user)->begins.fetch_add(1);}
static bool before_tensor_write(void * user,uint64_t bytes){
    auto * w=static_cast<Writer *>(user);const unsigned idx=w->writes.fetch_add(1);
    if(w->mode!=Mode::fail_consume&&idx==0&&w->block){std::unique_lock<std::mutex> lock(w->mutex);w->entered=true;w->cv.notify_all();w->cv.wait(lock,[&]{return w->release;});}
    if(w->mode==Mode::fail_tensor_write&&idx==1&&w->inject){w->fail_bytes.store(bytes);w->injected.fetch_add(1);return false;} // mid-writer: first tensor already written
    return true;
}
static bool decode(llama_context * ctx,int seq,int start,int n){
    auto batch=llama_batch_init(n,0,1);
    for(int i=0;i<n;++i){batch.token[i]=19;batch.pos[i]=start+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=seq;batch.logits[i]=i==n-1;}
    batch.n_tokens=n;const int rc=llama_decode(ctx,batch);llama_batch_free(batch);llama_synchronize(ctx);return rc==0;
}
static std::vector<uint8_t> state(llama_context * ctx,int seq){
    std::vector<uint8_t> b(llama_state_seq_get_size(ctx,seq));if(llama_state_seq_get_data(ctx,b.data(),b.size(),seq)!=b.size())std::abort();return b;
}
static Snapshot snap(ResourceKey key,uint64_t cap){Snapshot s{};s.key=key;s.known=true;s.capacity=cap;return s;}
// Terminal record of the slot operation: after a post-submit failure the operation finishes "failed" exactly once and no HTTP reply is involved.
struct Finish { std::atomic<unsigned> failed{0},completed{0}; };
struct Totals { unsigned cycles=0,reservations=0,fences=0,releases=0,failed=0,completed=0,refused=0,retry_ok=0; uint64_t calls=0,writes=0,fail_bytes=0,consumed_at_fail=0; };
struct Fixture { llama_context * ctx; Planner * planner; std::string path; std::vector<llama_token> * out; std::vector<uint8_t> prior; size_t count; };
static int cycle(Fixture & f,Mode mode,bool inject,uint64_t n,Totals & t){
    const ResourceKey target{Context::target,0,7,Kind::attention_cells};
    const Owner accepted{7,n+1,0},competitor{7,n+1001,1};Writer w;w.planner=f.planner;w.key=target;w.block=true;w.mode=mode;w.inject=inject;Finish fin;
    std::atomic<bool> ok{false},fenced{false},released{false},reserved{false},mutated{false};std::atomic<uint64_t> clears{0};
    std::thread writer([&]{auto plan=f.planner->plan_restore({snap(target,64)},accepted);if(plan.kind!=RestorePlan::Kind::ok)return;reserved.store(true);w.token=&plan.token;
        llama_state_seq_limits limits{};limits.user=&w;limits.consume=consume;limits.before_replace=before_replace;limits.before_tensor_write=before_tensor_write;
        limits.max_tensor_bytes=llama_state_seq_restore_tensor_capacity(f.ctx);limits.max_snapshot_bytes=llama_state_seq_restore_file_capacity(f.ctx,0);
        ok.store(llama_state_seq_load_file_bounded(f.ctx,f.path.c_str(),0,f.out->data(),f.out->size(),&f.count,&limits)>0);
        mutated.store(limits.mutation_started);clears.store(limits.clear_count);
        if(!ok.load()&&limits.mutation_started)llama_memory_seq_rm(llama_get_memory(f.ctx),0,-1,-1); // slot->clear_after_restore_failure analogue
        llama_synchronize(f.ctx); // lease.release(): synchronize, fence, native release
        fenced.store(f.planner->fence_restore(plan.token));released.store(f.planner->native_release(accepted).status==Status::accepted);
        if(ok.load())fin.completed.fetch_add(1);else fin.failed.fetch_add(1);}); // terminal "failed", no send_error path in this harness
    {std::unique_lock<std::mutex> lock(w.mutex);w.cv.wait(lock,[&]{return w.entered;});}
    const uint64_t consumed_before=w.consumed.load();const unsigned begins_before=w.begins.load();
    auto denied=f.planner->plan_restore({snap(target,64)},competitor);
    CHECK(denied.kind==RestorePlan::Kind::refused_overcommit);
    CHECK(f.planner->size()==1&&!f.planner->committed_peak(competitor,target).has_value()&&f.planner->committed_peak(accepted,target).has_value());
    CHECK(w.consumed.load()==consumed_before&&w.begins.load()==begins_before&&w.begins.load()==(mode==Mode::fail_consume?0u:1u)); // competitor touched neither balance nor slot
    {std::lock_guard<std::mutex> lock(w.mutex);w.release=true;}w.cv.notify_all();writer.join();
    CHECK(reserved.load()&&fenced.load()&&released.load());CHECK(f.planner->size()==0);
    if(mode==Mode::control){
        CHECK(w.injected.load()==0&&ok.load()&&w.consumed.load()==16&&mutated.load());CHECK(fin.completed.load()==1&&fin.failed.load()==0);CHECK(state(f.ctx,0)==f.prior);
    } else {
        CHECK(w.injected.load()==1);                               // the injection really fired
        CHECK(!ok.load());                                         // ...and made the native load fail
        CHECK(fin.failed.load()==1&&fin.completed.load()==0);      // post-submit failure => finish failed, once
        if(mode==Mode::fail_tensor_write){
            CHECK(w.writes.load()==2&&w.fail_bytes.load()>0&&w.consumed.load()==16&&mutated.load()&&w.begins.load()==1);
            CHECK(llama_memory_seq_pos_max(llama_get_memory(f.ctx),0)==-1&&llama_state_seq_get_size(f.ctx,0)<f.prior.size()); // half-written destination cleared, not left dirty
            CHECK(llama_state_seq_set_data(f.ctx,f.prior.data(),f.prior.size(),0)>0); // coherent: slot is reusable
        } else {
            CHECK(w.consume_calls.load()==1&&w.consumed.load()==8&&!mutated.load()&&w.begins.load()==0&&w.writes.load()==0&&clears.load()==0);
        }
        CHECK(llama_memory_seq_pos_max(llama_get_memory(f.ctx),1)==-1); // nothing leaked into another sequence
    }
    CHECK(state(f.ctx,0)==f.prior);
    // The reservation balance is fully back: the previously refused competitor is now admitted and cleanly released.
    {auto retry=f.planner->plan_restore({snap(target,64)},competitor);CHECK(retry.kind==RestorePlan::Kind::ok);CHECK(f.planner->fence_restore(retry.token));CHECK(f.planner->native_release(competitor).status==Status::accepted);++t.retry_ok;}
    CHECK(f.planner->size()==0);
    ++t.cycles;t.reservations+=reserved.load();t.fences+=fenced.load();t.releases+=released.load();t.failed+=fin.failed.load();t.completed+=fin.completed.load();++t.refused;
    t.calls+=w.consume_calls.load();t.writes+=w.writes.load();t.fail_bytes+=w.fail_bytes.load();t.consumed_at_fail+=w.consumed.load();
    return 0;
}
static int phase(Fixture & f,const char * name,Mode mode,bool inject,unsigned cycles){
    Totals t;for(uint64_t c=0;c<cycles;++c){if(cycle(f,mode,inject,static_cast<uint64_t>(mode)*100000+c,t))return 1;}
    std::printf("%s: cycles=%u injected_failure=%d reservations/fences/releases=%u/%u/%u ledger=0 finish_failed=%u finish_completed=%u competitor_refused=%u retry_admitted=%u consume_calls=%llu tensor_writes=%llu consumed_total=%llu fail_bytes_total=%llu\n",
        name,t.cycles,inject&&mode!=Mode::control,t.reservations,t.fences,t.releases,t.failed,t.completed,t.refused,t.retry_ok,
        (unsigned long long)t.calls,(unsigned long long)t.writes,(unsigned long long)t.consumed_at_fail,(unsigned long long)t.fail_bytes);
    return 0;
}
int main(int argc,char **argv){
    // argv[3]=="--disable-injection": negative control, injected modes keep expecting failure so the test MUST fail.
    CHECK(argc==3||(argc==4&&std::strcmp(argv[3],"--disable-injection")==0));const bool inject=argc==3;
    llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=0;
    auto * model=llama_model_load_from_file(argv[1],mp);CHECK(model);
    auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=128;cp.n_ubatch=128;cp.n_seq_max=2;cp.kv_unified=true;cp.n_threads=2;cp.n_threads_batch=2;
    auto * ctx=llama_init_from_model(model,cp);CHECK(ctx);CHECK(decode(ctx,0,0,16));const auto prior=state(ctx,0);
    const std::string path=std::string(argv[2])+"/r4-r5-state.dft";std::vector<llama_token> ids(16,19),out(512);size_t count=0;
    CHECK(llama_state_seq_save_file(ctx,path.c_str(),0,ids.data(),ids.size())>0);
    Planner planner;Fixture f{ctx,&planner,path,&out,prior,0};
    if(phase(f,"R4 control (no injection, harness must succeed)",Mode::control,true,100))return 1;
    if(phase(f,"R4 injected failure: before_tensor_write=false mid-writer",Mode::fail_tensor_write,inject,100))return 1;
    if(phase(f,"R4 injected failure: consume=false mid-restore",Mode::fail_consume,inject,100))return 1;
    std::puts("R4 PASS injected post-submit failures with Planner: tensor-write and consume failures end finish=failed; reserve/fence/release=1/1/1; ledger=0; destination cleared or untouched and coherent; competitor refused without touching slot/balance");
    const ResourceKey draft{Context::draft,0,8,Kind::attention_cells};
    {const Owner owner{8,1,0};auto plan=planner.plan_restore({snap(draft,15)},owner);CHECK(plan.kind==RestorePlan::Kind::ok);
        Writer w;w.planner=&planner;w.token=&plan.token;w.key=draft;llama_state_seq_limits limits{};limits.user=&w;limits.consume=consume;limits.before_replace=before_replace;
        limits.max_tensor_bytes=llama_state_seq_restore_tensor_capacity(ctx);limits.max_snapshot_bytes=llama_state_seq_restore_file_capacity(ctx,0);
        CHECK(llama_state_seq_load_file_bounded(ctx,path.c_str(),0,out.data(),out.size(),&count,&limits)==0);
        CHECK(w.begins.load()==0&&!limits.mutation_started&&state(ctx,0)==prior);CHECK(planner.fence_restore(plan.token));CHECK(planner.native_release(owner).status==Status::accepted);CHECK(planner.size()==0);}
    std::puts("R5 PASS draft-v1 bounded reader limit+1: before_replace=0; destination intact; ledger=0");
    {const Owner owner{8,2,0};auto plan=planner.plan_restore({snap(draft,16)},owner);CHECK(plan.kind==RestorePlan::Kind::ok);
        Writer w;w.planner=&planner;w.token=&plan.token;w.key=draft;llama_state_seq_limits limits{};limits.user=&w;limits.consume=consume;limits.before_replace=before_replace;
        limits.max_tensor_bytes=llama_state_seq_restore_tensor_capacity(ctx);limits.max_snapshot_bytes=llama_state_seq_restore_file_capacity(ctx,0);
        CHECK(llama_state_seq_load_file_bounded(ctx,path.c_str(),0,out.data(),out.size(),&count,&limits)>0);
        CHECK(w.begins.load()==1&&w.consumed.load()==16&&limits.mutation_started);CHECK(planner.fence_restore(plan.token));CHECK(planner.native_release(owner).status==Status::accepted);CHECK(planner.size()==0);}
    std::puts("R5 PASS draft-v1 bounded reader exact limit: consumed=16; before_replace=1; ledger=0");
    llama_free(ctx);llama_model_free(model);llama_backend_free();return 0;
}
