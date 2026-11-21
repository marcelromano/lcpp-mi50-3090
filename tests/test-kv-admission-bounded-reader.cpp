#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <vector>
#define CHECK(x) do { if(!(x)){std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);return 1;} }while(0)
struct Budget { uint64_t limit{},consumed{}; uint32_t calls{},writes{},begins{}; };
static bool consume(void* ptr,uint32_t kind,uint32_t stream,uint64_t count) {
 auto & b=*static_cast<Budget*>(ptr);++b.calls;
 if(kind!=0 || stream!=0 || count>std::numeric_limits<uint64_t>::max()-b.consumed || b.consumed+count>b.limit)return false;
 b.consumed+=count;return true;
}
static bool decode(llama_context* ctx,int seq,int start,int n,int token=19) {
 auto batch=llama_batch_init(n,0,1);
 for(int i=0;i<n;++i){batch.token[i]=token;batch.pos[i]=start+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=seq;batch.logits[i]=i==n-1;}
 batch.n_tokens=n;const int rc=llama_decode(ctx,batch);llama_batch_free(batch);llama_synchronize(ctx);return rc==0;
}
static std::vector<uint8_t> state(llama_context* c,int seq) {
 std::vector<uint8_t> v(llama_state_seq_get_size(c,seq));
 if(llama_state_seq_get_data(c,v.data(),v.size(),seq)!=v.size()) std::abort();
 return v;
}
int main(int argc,char**argv) {
 CHECK(argc==3);llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=0;
 auto *model=llama_model_load_from_file(argv[1],mp);CHECK(model);
 auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=128;cp.n_ubatch=128;cp.n_seq_max=2;cp.kv_unified=true;cp.n_threads=2;cp.n_threads_batch=2;cp.n_rs_seq=0;
 auto *ctx=llama_init_from_model(model,cp);CHECK(ctx);
 CHECK(decode(ctx,0,0,16));CHECK(decode(ctx,1,0,16,20));
 const auto before0=state(ctx,0),before1=state(ctx,1);
 std::vector<llama_token> tokens(16,19),loaded(512);size_t count=0;
 const std::string path=std::string(argv[2])+"/reader-state.bin";
 CHECK(llama_state_seq_save_file(ctx,path.c_str(),0,tokens.data(),tokens.size())>0);
 Budget budget{15,0,0};llama_state_seq_limits limits{};limits.user=&budget;limits.consume=consume;
 limits.before_replace=[](void * p){++static_cast<Budget*>(p)->begins;};
 limits.max_tensor_bytes=llama_state_seq_restore_tensor_capacity(ctx);CHECK(limits.max_tensor_bytes>0);
 CHECK(llama_state_seq_load_file_bounded(ctx,path.c_str(),0,loaded.data(),loaded.size(),&count,&limits)==0);
 CHECK(limits.clear_count==0 && limits.allocation_count==0 && limits.limit_refused && limits.fence_count==1 && budget.begins==0);
 CHECK(state(ctx,0)==before0 && state(ctx,1)==before1);
 std::puts("G4 actual reader limit+1: clear=0 allocations=0 target_equal=1 other_equal=1 fence=1");
 budget={16,0,0};
 CHECK(llama_state_seq_load_file_bounded(ctx,path.c_str(),0,loaded.data(),loaded.size(),&count,&limits)>0);
 CHECK(count==16 && budget.consumed==16 && budget.begins==1 && limits.clear_count==1 && limits.allocation_count==2 && limits.fence_count==1);
 CHECK(state(ctx,0)==before0 && state(ctx,1)==before1);
 std::puts("G4 actual reader exact limit: consumed=16 clear=1 allocations=2 target_equal=1 other_equal=1 fence=1");
 budget={16,1,0};
 CHECK(llama_state_seq_load_file_bounded(ctx,path.c_str(),0,loaded.data(),loaded.size(),&count,&limits)==0);
 CHECK(limits.clear_count==0 && limits.allocation_count==0 && state(ctx,0)==before0);
 std::puts("G4 cumulative prior consumption: prior=1 incoming=16 I=16 refused before clear");
 budget={std::numeric_limits<uint64_t>::max(),std::numeric_limits<uint64_t>::max()-10,0};
 CHECK(llama_state_seq_load_file_bounded(ctx,path.c_str(),0,loaded.data(),loaded.size(),&count,&limits)==0);
 CHECK(limits.clear_count==0 && limits.allocation_count==0 && limits.limit_refused && limits.fence_count==1 && state(ctx,0)==before0);
 std::puts("G3 actual reader UINT64_MAX cumulative overflow: refused before clear=0 allocations=0");
 std::ifstream f(path,std::ios::binary);std::vector<char> bytes((std::istreambuf_iterator<char>(f)),{});CHECK(bytes.size()>128);
 const std::string truncated=path+".truncated";{std::ofstream bad(truncated,std::ios::binary);bad.write(bytes.data(),bytes.size()/2);}
 budget={16,0,0};
 CHECK(llama_state_seq_load_file_bounded(ctx,truncated.c_str(),0,loaded.data(),loaded.size(),&count,&limits)==0);
 CHECK(budget.begins==0 && !limits.mutation_started && limits.clear_count==0 && limits.fence_count==1 && state(ctx,0)==before0 && state(ctx,1)==before1);
 budget={16,0,0};limits.before_tensor_write=[](void* p,uint64_t)->bool{return ++static_cast<Budget*>(p)->writes<=1;};
 CHECK(llama_state_seq_load_file_bounded(ctx,path.c_str(),0,loaded.data(),loaded.size(),&count,&limits)==0);
 limits.before_tensor_write=nullptr;
 CHECK(limits.mutation_started && limits.fence_count==1 && state(ctx,1)==before1);
 CHECK(llama_memory_seq_pos_max(llama_get_memory(ctx),0)==-1);
 auto *cold=llama_init_from_model(model,cp);CHECK(cold);
 CHECK(decode(ctx,0,0,16));CHECK(decode(cold,0,0,16));
 CHECK(state(ctx,0)==state(cold,0));
 auto *sampler=llama_sampler_init_greedy();
 std::vector<llama_token> ids;
 for(int i=0;i<16;++i){auto a=llama_sampler_sample(sampler,ctx,-1);auto b=llama_sampler_sample(sampler,cold,-1);CHECK(a==b);ids.push_back(a);CHECK(decode(ctx,0,16+i,1,a));CHECK(decode(cold,0,16+i,1,b));}
 llama_sampler_free(sampler);
 std::puts("G6 actual reader truncated tensors: target_cold=1 other_bit_identical=1 cold_control_ids_equal=16/16 output_fence=1");
 // An impossible host scratch cap must be checked before tensor vector resize.
 budget={16,0,0};limits.max_tensor_bytes=0;
 CHECK(llama_state_seq_load_file_bounded(ctx,path.c_str(),0,loaded.data(),loaded.size(),&count,&limits)==0);
 CHECK(limits.limit_refused && limits.fence_count==1 && state(ctx,1)==before1);
 std::puts("G10 actual reader host tensor cap=0 refused, native fence=1 other_equal=1");
 // The unchanged API is still callable with its original signature.
 CHECK(llama_state_seq_load_file(ctx,path.c_str(),0,loaded.data(),loaded.size(),&count)>0);
 CHECK(state(ctx,1)==before1);
 std::puts("G11 legacy API restore: accepted other_equal=1");
 llama_free(cold);llama_free(ctx);llama_model_free(model);llama_backend_free();return 0;
}
