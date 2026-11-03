#include "ggml-backend.h"
#include "llama-memory-recurrent.h"
#include "llama-memory-hybrid.h"
#include "llama-model.h"
#include "llama-io.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#define CHECK(x) do {if(!(x)){std::fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);return 1;}}while(0)
struct fixture_model:llama_model {
 fixture_model():llama_model(llama_model_params{}){}
 void load_stats(llama_model_loader&)override{} void load_hparams(llama_model_loader&)override{}
 void load_vocab(llama_model_loader&)override{} bool load_tensors(llama_model_loader&)override{return false;}
 void load_arch_hparams(llama_model_loader&)override{} void load_arch_tensors(llama_model_loader&)override{}
 std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params&)const override{return {};}
};
struct Writer:llama_io_write_i {
 std::vector<uint8_t> bytes;
 void write(const void* p,size_t n)override{auto v=(const uint8_t*)p;bytes.insert(bytes.end(),v,v+n);}
 void write_tensor(ggml_tensor* t,size_t offset,size_t n)override{size_t old=bytes.size();bytes.resize(old+n);ggml_backend_tensor_get(t,bytes.data()+old,offset,n);}
 size_t n_bytes()override{return bytes.size();}
};
static std::vector<uint8_t> dump(llama_memory_i &m,int seq,llama_state_seq_flags flags=0){Writer w;m.state_write(w,seq,flags);return w.bytes;}
struct Authority{llama_memory_recurrent* memory;uint64_t limit,consumed{};uint32_t groups;};
static bool charge(void* p,uint32_t kind,uint32_t stream,uint64_t count){
 auto &a=*static_cast<Authority*>(p);llama_memory_resource_capacity raw{};
 if(kind!=1 || stream!=0 || !a.memory->get_resource_capacity(0,raw) || !raw.recurrent_known || raw.recurrent_rollback_groups!=a.groups ||
    raw.recurrent_capacity_rows%a.groups || a.consumed>a.limit || count>a.limit-a.consumed)return false;
 a.consumed+=count;return true;
}
struct Reader:llama_io_read_i{
 const std::vector<uint8_t>&bytes;size_t position=0;
 Reader(const std::vector<uint8_t>&b,llama_state_seq_limits*l):llama_io_read_i(l),bytes(b){}
 void read(void* p,size_t n)override{if(n>bytes.size()-position)throw std::runtime_error("EOF");std::memcpy(p,bytes.data()+position,n);position+=n;}
 void read_tensor(ggml_tensor*t,size_t offset,size_t n)override{if(n>bytes.size()-position || n>limits->max_tensor_bytes || offset>ggml_nbytes(t) || n>ggml_nbytes(t)-offset)throw std::runtime_error("tensor EOF/bounds");ggml_backend_tensor_set(t,bytes.data()+position,offset,n);position+=n;}
 size_t n_bytes()override{return position;}
};
static void seed(llama_memory_recurrent&m,int pattern,uint32_t rollback){
 for(int seq=0;seq<2;++seq){m.cells[seq].pos=10+seq;m.cells[seq].seq_id.insert(seq);m.cells[seq].tail=seq;}
 m.used=2;m.rs_idx[0]=rollback;
 for(auto*t:m.r_l)if(t){std::vector<float> v(ggml_nbytes(t)/sizeof(float));for(size_t i=0;i<v.size();++i)v[i]=float(pattern+i);ggml_backend_tensor_set(t,v.data(),0,ggml_nbytes(t));}
 for(auto*t:m.s_l)if(t){std::vector<float> v(ggml_nbytes(t)/sizeof(float));for(size_t i=0;i<v.size();++i)v[i]=float(pattern+100+i);ggml_backend_tensor_set(t,v.data(),0,ggml_nbytes(t));}
}
static int hybrid_checkpoint(){
 fixture_model model;model.hparams.n_layer_all=2;model.hparams.n_embd=2;
 model.hparams.n_head_arr.fill(1);model.hparams.n_head_kv_arr.fill(1);
 model.hparams.n_embd_head_k_full=2;model.hparams.n_embd_head_v_full=2;
 model.hparams.ssm_d_conv=2;model.hparams.ssm_d_inner=1;model.hparams.ssm_n_group=1;model.hparams.ssm_d_state=1;
 llama_memory_hybrid memory(model,GGML_TYPE_F32,GGML_TYPE_F32,false,16,1,0,LLAMA_SWA_TYPE_NONE,
     GGML_TYPE_F32,GGML_TYPE_F32,4,2,1,false,true,[](int32_t i){return i==0;},[](int32_t i){return i==1;});
 auto *attention=memory.get_mem_attn();auto *recurrent=memory.get_mem_recr();
 for(int seq=0;seq<2;++seq){
  const int n=seq?2:8;llama_batch_allocr allocator(1);auto batch=allocator.ubatch_reserve(n,1);batch.seq_id_unq[0]=seq;
  for(int i=0;i<n;++i){batch.pos[i]=i;batch.n_seq_id[i]=1;batch.seq_id[i]=&seq;}
  auto slot=attention->find_slot(batch,false);CHECK(!slot.empty());attention->apply_ubatch(slot,batch);
 }
 seed(*recurrent,100,0);recurrent->cells[0].pos=3;recurrent->cells[1].pos=1;
 const auto checkpoint=dump(memory,0,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
 const auto before_attention=dump(*attention,0),other=dump(memory,1);
 recurrent->cells[0].pos=7;
 llama_memory_resource_capacity before{},partial{},pruned{};CHECK(memory.get_resource_capacity(0,before));
 Authority authority{recurrent,1,0,2};llama_state_seq_limits limits{};limits.user=&authority;limits.consume=charge;
 for(const auto & buffer:memory.memory_breakdown()) limits.max_tensor_bytes+=buffer.second;
 Reader reader(checkpoint,&limits);memory.state_read(reader,0,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
 CHECK(memory.get_resource_capacity(0,partial));
 CHECK(before.attention_used_cells==partial.attention_used_cells && dump(*attention,0)==before_attention && recurrent->cells[0].pos==3);
 CHECK(memory.seq_rm(0,4,-1));CHECK(memory.get_resource_capacity(0,pruned));
 CHECK(partial.attention_used_cells==10 && pruned.attention_used_cells==6 && partial.recurrent_used_rows==pruned.recurrent_used_rows);
 CHECK(dump(memory,1)==other);
 std::printf("native3.2 hybrid PARTIAL_ONLY attention_delta=0; subsequent native seq_rm attention_delta=-4; physical_getter=10->6 recurrent_rows=%u->%u other_equal=1\n",partial.recurrent_used_rows,pruned.recurrent_used_rows);
 return 0;
}
int main(){
 fixture_model model;model.hparams.n_layer_all=1;model.hparams.n_embd=1;model.hparams.ssm_d_conv=2;model.hparams.ssm_d_inner=1;model.hparams.ssm_n_group=1;model.hparams.ssm_d_state=1;
 llama_memory_i::layer_filter_cb filter=[](int32_t){return true;};
 for(uint32_t groups=1;groups<=3;++groups){
  llama_memory_recurrent source(model,GGML_TYPE_F32,GGML_TYPE_F32,false,4,2,groups-1,filter),target(model,GGML_TYPE_F32,GGML_TYPE_F32,false,4,2,groups-1,filter);
  seed(source,10,groups-1);seed(target,200,0);
  const auto blob=dump(source,0),old=dump(target,0),other=dump(target,1);uint32_t count=0;std::memcpy(&count,blob.data(),sizeof(count));CHECK(count==1);
  llama_memory_resource_capacity raw{};CHECK(target.get_resource_capacity(0,raw));CHECK(raw.recurrent_capacity_rows==4*groups && raw.recurrent_used_rows==2*groups && raw.recurrent_seq_rows==groups);
  Authority authority{&target,0,0,groups};llama_state_seq_limits limits{};limits.user=&authority;limits.consume=charge;for(const auto & buffer:target.memory_breakdown()) limits.max_tensor_bytes+=buffer.second;
  bool refused=false;try{Reader reader(blob,&limits);target.state_read(reader,0,0);}catch(const std::exception&){refused=true;}
  CHECK(refused && limits.clear_count==0 && limits.allocation_count==0 && dump(target,0)==old && dump(target,1)==other);
  authority.limit=1;Reader reader(blob,&limits);target.state_read(reader,0,0);
  CHECK(authority.consumed==1 && dump(target,0)==blob && dump(target,1)==other && target.rs_idx[0]==0);
  CHECK(target.get_resource_capacity(0,raw) && raw.recurrent_used_rows==2*groups);
  std::printf("native g=%u CELL_COUNT=1 normalized_bundles=1 physical_rows=%u rollback_plane_restored=1 other_equal=1 negative_clear=0 negative_alloc=0\n",groups,groups);
 }
 CHECK(hybrid_checkpoint()==0);
 return 0;
}
