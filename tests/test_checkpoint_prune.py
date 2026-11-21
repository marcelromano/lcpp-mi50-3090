"""CPU regression: compile the actual server pruning loop, including shared ownership."""
import pathlib,subprocess,tempfile,sys
source=pathlib.Path(sys.argv[1]).read_text()
start=source.index('        int64_t last = -1;',source.index('    void create_checkpoint('))
end=source.index('        while (slot.prompt.checkpoints.size()',start)
loop=source[start:end]
cpp=r'''
#include <cassert>
#include <cstdint>
#include <memory>
#include <list>
#include <initializer_list>
#include <iostream>
#define SLT_TRC(...) ((void)0)
struct checkpoint { int id_task; int64_t n_tokens; };
struct prompt_t { std::list<std::shared_ptr<checkpoint>> checkpoints; };
struct slot_t { prompt_t prompt; };
struct params_t { int n_ctx_checkpoints=8; int64_t checkpoint_min_step=8192; } params_base;
void prune(slot_t & slot, int id_task) {
LOOP
}
slot_t make(std::initializer_list<checkpoint> c) { slot_t s; for(auto &v:c)s.prompt.checkpoints.push_back(std::make_shared<checkpoint>(v));return s; }
int main() {
 auto s=make({{1,32252},{1,32764}});prune(s,2);assert(s.prompt.checkpoints.size()==1 && s.prompt.checkpoints.front()->n_tokens==32252);
 auto own=make({{2,32252},{2,32764}});prune(own,2);assert(own.prompt.checkpoints.size()==2);
 auto far=make({{1,0},{1,16384},{1,32768}});prune(far,2);assert(far.prompt.checkpoints.size()==3);
 auto shared=make({{1,32252},{1,32764}});auto cached=shared.prompt.checkpoints.back();prune(shared,2);assert(shared.prompt.checkpoints.size()==1 && cached->n_tokens==32764 && cached.use_count()==1);
 auto full=make({{1,0},{1,100},{1,10000},{1,10100},{1,20000},{1,20100},{1,30000},{1,30100}});prune(full,2);assert(full.prompt.checkpoints.size()==4 && params_base.n_ctx_checkpoints==8);
 std::cout << "checkpoint-prune: 5 passed (below-capacity, current-task, distance, shared-owner, capacity)\n";
}
'''.replace('LOOP',loop)
with tempfile.TemporaryDirectory(prefix='t1999-prune-') as tmp:
 p=pathlib.Path(tmp);(p/'test.cpp').write_text(cpp)
 subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
