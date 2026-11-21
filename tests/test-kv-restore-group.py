#!/usr/bin/env python3
"""CPU unit harness: compile the actual preflight prefix and error formatter.
The scope-all planner tail is deliberately a sentinel, not a mocked planner proof.
HTTP dispatch/slot integration is covered separately by T4/T5.
"""
from pathlib import Path
import subprocess
root = Path(__file__).resolve().parents[1]
def extract(text, start):
    a = text.index(start); b = text.index('{', a); depth = 1; i = b + 1
    while depth:
        if text[i] == '{': depth += 1
        if text[i] == '}': depth -= 1
        i += 1
    return text[a:i]
context = (root/'tools/server/server-context.cpp').read_text()
start = context.index('        if (!params_base.kv_admission) return true;', context.index('    bool kv_admission_preflight('))
end = context.index('        // A memory getter is metadata only.', start)
prefix = context[start:end]
formatter = extract((root/'tools/server/server-common.cpp').read_text(), 'json format_error_response(')
enum = extract((root/'tools/server/server-common.h').read_text(), 'enum error_type {') + ';'
cpp = '''#include "common/kv-admission-scope.h"
#include "vendor/nlohmann/json.hpp"
#include <cassert>
#include <vector>
using json = nlohmann::ordered_json;
''' + enum + '\n' + formatter + '''
struct Fixture {
 struct { bool kv_admission = true; common_kv_admission_scope kv_admission_scope; } params_base;
 bool kv_admission_release_failed = false;
 bool reached_all = false;
 bool preflight(const std::vector<int> & child_slots, std::string & reason, error_type & error) {
''' + prefix + '''
 reached_all = true; return true; // sentinel: unchanged all-policy tail
 }
};
int main() {
 for (auto scope : {common_kv_admission_scope::all, common_kv_admission_scope::restore}) {
  for (int n : {1, 2}) {
   Fixture f; f.params_base.kv_admission_scope = scope;
   std::string reason; error_type error = ERROR_TYPE_SERVER;
   bool ok = f.preflight(std::vector<int>(n-1), reason, error);
   if (scope == common_kv_admission_scope::restore && n == 2) {
    assert(!ok); assert(!f.reached_all);
    auto result = format_error_response("KV admission refused: " + reason, error);
    assert(result["code"] == 503);
    assert(result["type"] == "kv_capacity_error");
    assert(result["message"] == "KV admission refused: capacity exceeded");
   } else {
    assert(ok);
    assert(f.reached_all == (scope == common_kv_admission_scope::all));
   }
  }
 }
}
'''
out = root/'build-cpu/test-kv-restore-group-extracted.cpp'
out.write_text(cpp)
bin = out.with_suffix('')
subprocess.run(['nice','-n','19','c++','-std=c++17','-O0','-I',str(root),str(out),'-o',str(bin)],check=True)
subprocess.run([str(bin)],check=True)
print('PASS: restore n=2 => 503 kv_capacity_error exact capacity message; restore n=1 bypass; all n=1/n=2 reaches unchanged tail')
