
#pragma once
#include "kv_planner.hpp"
#include "llama.h"
namespace kvplan {
struct PhysicalView {
 ResourceKey key;
 uint64_t capacity{},used{},sequence{},exclusive{};
 uint32_t raw_rows_per_bundle{};
 uint32_t stream_count{};
};
// No model-name inference, no reserves, no context/kernel fence here.
// Consumer synchronizes real target/draft at task boundary BEFORE observing.
inline bool normalize_resources(Context role,uint64_t epoch,const llama_memory_resource_capacity &raw,
                                std::vector<PhysicalView> &out) {
 out.clear();
 if(!epoch || (!raw.attention_known&&!raw.recurrent_known))return false;
 std::vector<PhysicalView> staged;
 if(raw.attention_known){
  if(!raw.attention_stream_count||raw.attention_stream_id>=raw.attention_stream_count||
     raw.attention_used_cells>raw.attention_capacity_cells||
     raw.attention_seq_cells>raw.attention_used_cells||
     raw.attention_seq_exclusive_cells>raw.attention_seq_cells)return false;
  staged.push_back({{role,raw.attention_stream_id,epoch,Kind::attention_cells},
      raw.attention_capacity_cells,raw.attention_used_cells,raw.attention_seq_cells,
      raw.attention_seq_exclusive_cells,0,raw.attention_stream_count});
 } else if(raw.attention_stream_id||raw.attention_stream_count||raw.attention_capacity_cells||
           raw.attention_used_cells||raw.attention_seq_cells||raw.attention_seq_exclusive_cells)return false;
 if(raw.recurrent_known){
  const uint32_t g=raw.recurrent_rollback_groups;
  if(!g||raw.recurrent_capacity_rows%g||raw.recurrent_used_rows%g||
     raw.recurrent_seq_rows%g||raw.recurrent_seq_exclusive_rows%g||
     raw.recurrent_used_rows>raw.recurrent_capacity_rows||
     raw.recurrent_seq_rows>raw.recurrent_used_rows||
     raw.recurrent_seq_exclusive_rows>raw.recurrent_seq_rows)return false;
  staged.push_back({{role,0,epoch,Kind::recurrent_rows},
      raw.recurrent_capacity_rows/g,raw.recurrent_used_rows/g,raw.recurrent_seq_rows/g,
      raw.recurrent_seq_exclusive_rows/g,g,0});
 } else if(raw.recurrent_capacity_rows||raw.recurrent_used_rows||raw.recurrent_seq_rows||
           raw.recurrent_seq_exclusive_rows||raw.recurrent_rollback_groups)return false;
 out.swap(staged);
 return true;
}
}
