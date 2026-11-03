#include "../tools/server/kv_planner.hpp"
#include "../tools/server/kv_resource_view.hpp"
#include <cstdio>
#include <limits>
#include <type_traits>
#include <utility>
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x); return 1; } } while(0)
using namespace kvplan;
static int g_fail=0;
#define EXPECT(x) do { if (!(x)) { ++g_fail; std::fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x); } } while(0)
using PK=RestorePlan::Kind;
static Snapshot snap(ResourceKey k,uint64_t c,uint64_t u=0) { Snapshot s{}; s.key=k;s.known=true;s.capacity=c;s.used_real=u;return s; }
static void recurrent_tests() {
 ResourceKey at{Context::target,0,7,Kind::attention_cells}, rc{Context::target,0,7,Kind::recurrent_rows};
 Owner a{7,1,0}, b{7,2,1};
 auto rsnap=[&](uint64_t cap,uint64_t used,uint64_t excl,uint64_t credit,Owner who){
   Snapshot s{}; s.key=rc; s.known=true; s.capacity=cap; s.used_real=used; s.seq_exclusive_real=excl;
   if(credit) { s.reusable_recurrent_by_incoming_owner.emplace(who,credit); }
   return s; };
 // T3: limit +-1 of the recurrent pool (np2 => 2 cells), no credit
 { Planner p; auto r=p.plan_restore({rsnap(2,1,0,0,a)},a); EXPECT(r.kind==PK::ok); EXPECT(r.token.limit(rc)==1);
   EXPECT(p.consume_restore(r.token,rc,1)); EXPECT(!p.consume_restore(r.token,rc,1));
   EXPECT(p.fence_restore(r.token)); EXPECT(p.native_release(a).status==Status::accepted); EXPECT(p.size()==0); }
 { Planner p; auto r=p.plan_restore({rsnap(2,2,0,0,a)},a); EXPECT(r.kind==PK::refused_overcommit); EXPECT(p.size()==0); EXPECT(!p.committed_peak(a,rc)); }
 // T3: replacement credit D1 (pool full, destination owns 1 exclusive cell)
 { Planner p; auto r=p.plan_restore({rsnap(2,2,1,1,a)},a); EXPECT(r.kind==PK::ok); EXPECT(r.token.limit(rc)==1);
   EXPECT(!p.consume_restore(r.token,rc,2)); EXPECT(p.consume_restore(r.token,rc,1)); EXPECT(!p.consume_restore(r.token,rc,1)); }
 { Planner p; auto r=p.plan_restore({rsnap(2,1,1,1,a)},a); EXPECT(r.kind==PK::ok); EXPECT(r.token.limit(rc)==2);
   EXPECT(!p.consume_restore(r.token,rc,3)); EXPECT(p.consume_restore(r.token,rc,2)); }
 { Planner p; EXPECT(p.plan_restore({rsnap(2,2,1,2,a)},a).kind==PK::refused_identity); // credit > exclusive
   EXPECT(p.plan_restore({rsnap(2,2,0,1,a)},a).kind==PK::refused_identity);            // no exclusive cell
   EXPECT(p.plan_restore({rsnap(2,2,1,1,b)},a).kind==PK::refused_identity);            // other owner credit
   EXPECT(p.size()==0); }
 { Planner p; Snapshot s=snap(at,10,2); s.seq_exclusive_real=2; s.reusable_recurrent_by_incoming_owner.emplace(a,1);
   EXPECT(p.plan_restore({s},a).kind==PK::refused_identity); EXPECT(p.size()==0); } // attention takes no credit
 { Planner p; EXPECT(p.admit({rsnap(2,0,0,0,a)},{{b,{},{{rc,1}}}}).status==Status::accepted);
   EXPECT(p.plan_restore({rsnap(2,1,0,0,a)},a).kind==PK::refused_overcommit); }      // peer commitment counts
 // hybrid: declared layout requires attention + recurrent
 { std::vector<ResourceKey> decl{at,rc};
   Planner p; auto r=p.plan_restore({snap(at,10),rsnap(2,1,0,0,a)},a,decl);
   EXPECT(r.kind==PK::ok); EXPECT(r.token.limit(at)==10 && r.token.limit(rc)==1); }
 { std::vector<ResourceKey> decl{at,rc}; Planner p;
   EXPECT(p.plan_restore({snap(at,10)},a,decl).kind==PK::refused_unsupported_layout);          // missing rec
   EXPECT(p.plan_restore({rsnap(2,1,0,0,a)},a,decl).kind==PK::refused_unsupported_layout);     // missing att
   ResourceKey dr{Context::draft,0,7,Kind::attention_cells};
   EXPECT(p.plan_restore({snap(at,10),rsnap(2,1,0,0,a),snap(dr,10)},a,decl).kind==PK::refused_unsupported_layout); // undeclared
   EXPECT(p.size()==0); }
 { std::vector<ResourceKey> decl{rc}; Planner p; // recurrent-only declared model
   EXPECT(p.plan_restore({rsnap(2,1,0,0,a)},a,decl).kind==PK::ok); }
 // T5: unknown layouts, all refuse with an empty ledger (nothing committed => nothing cleared)
 { Planner p; auto bad=rc; bad.kind=static_cast<Kind>(7);
   EXPECT(p.plan_restore({snap(bad,10)},a).kind==PK::refused_unsupported_layout);
   auto drec=rc; drec.context=Context::draft;
   EXPECT(p.plan_restore({snap(drec,10)},a).kind==PK::refused_unsupported_layout);
   auto s1=rc; s1.stream=1;
   EXPECT(p.plan_restore({snap(s1,10)},a).kind==PK::refused_unsupported_layout);
   auto ks=rsnap(2,1,0,0,a); ks.known=false;
   EXPECT(p.plan_restore({ks},a).kind==PK::refused_unknown);
   EXPECT(p.plan_restore({snap(at,10),snap(bad,10)},a).kind==PK::refused_unsupported_layout);
   EXPECT(p.size()==0); EXPECT(!p.committed_peak(a,rc)); EXPECT(!p.committed_peak(a,at)); }
 { // g not dividing the rows: normalization refuses, so no known recurrent snapshot can exist
   llama_memory_resource_capacity raw{}; raw.recurrent_known=true; raw.recurrent_capacity_rows=6;
   raw.recurrent_used_rows=3; raw.recurrent_seq_rows=3; raw.recurrent_seq_exclusive_rows=0; raw.recurrent_rollback_groups=2;
   std::vector<PhysicalView> v; EXPECT(!normalize_resources(Context::target,7,raw,v)&&v.empty());
   raw.recurrent_rollback_groups=3; raw.recurrent_used_rows=2; raw.recurrent_seq_rows=2;
   EXPECT(!normalize_resources(Context::target,7,raw,v)&&v.empty());
   raw.recurrent_rollback_groups=0; EXPECT(!normalize_resources(Context::target,7,raw,v)); }
}
int main() {
 recurrent_tests(); if (g_fail) { std::fprintf(stderr,"planner tests failed: %d\n",g_fail); return 1; }
 static_assert(!std::is_copy_constructible<RestoreToken>::value,"engine token must not copy");
 static_assert(!std::is_move_assignable<RestoreToken>::value,"cannot overwrite a live commitment");
 ResourceKey t{Context::target,0,7,Kind::attention_cells}, d{Context::draft,0,7,Kind::attention_cells};
 Owner a{7,1,0}, b{7,2,1};
 { Planner p; auto r=p.plan_restore({snap(t,0)},a); CHECK(r.kind==RestorePlan::Kind::ok); CHECK(r.token.limit(t)==0);
   CHECK(p.consume_restore(r.token,t,0)); CHECK(!p.consume_restore(r.token,t,1));
   CHECK(p.native_release(a).status==Status::rejected); CHECK(p.size()==1);
   p.cancel_by_caller(a); CHECK(p.size()==1); CHECK(p.fence_restore(r.token));
   CHECK(p.native_release(a).status==Status::accepted); CHECK(p.native_release(a).status==Status::already_released);
   CHECK(!p.consume_restore(r.token,t,0)); }
 { Planner p; auto s=snap(t,10);s.known=false; CHECK(p.plan_restore({s},a).kind==RestorePlan::Kind::refused_unknown);CHECK(p.size()==0); }
 { Planner p; const auto m=std::numeric_limits<uint64_t>::max();
   CHECK(p.admit({snap(t,m)},{{a,{},{{t,100}}}}).status==Status::accepted);
   CHECK(p.plan_restore({snap(t,m,m-50)},b).kind==RestorePlan::Kind::refused_overflow); CHECK(p.size()==1); }
 { Planner p; CHECK(p.admit({snap(t,1000)},{{a,{},{{t,300}}}}).status==Status::accepted);
   CHECK(p.plan_restore({snap(t,1000,800)},b).kind==RestorePlan::Kind::refused_overcommit);CHECK(p.size()==1); }
 { Planner p; auto r=p.plan_restore({snap(t,10),snap(d,20)},a);CHECK(r.kind==RestorePlan::Kind::ok);
   CHECK(r.token.limit(t)==10 && r.token.limit(d)==20); CHECK(p.consume_restore(r.token,t,4));
   CHECK(!p.consume_restore(r.token,t,7)); CHECK(p.consume_restore(r.token,t,6)); CHECK(!p.consume_restore(r.token,t,1));
   CHECK(p.consume_restore(r.token,d,20)); CHECK(!p.consume_restore(r.token,d,1));
   CHECK(p.admit({snap(t,10)},{{b,{},{{t,1}}}}).status==Status::rejected);
   Planner alien; CHECK(!alien.fence_restore(r.token)); CHECK(!alien.consume_restore(r.token,t,0));
   RestoreToken moved(std::move(r.token)); CHECK(!p.fence_restore(r.token)); CHECK(p.fence_restore(moved));CHECK(!p.consume_restore(moved,t,0));
   CHECK(p.native_release(a).status==Status::accepted); }
 { Planner p; CHECK(p.admit({snap(d,10)},{{a,{},{{d,5}}}}).status==Status::accepted);
   CHECK(p.plan_restore({snap(t,10),snap(d,10,6)},b).kind==RestorePlan::Kind::refused_overcommit);
   CHECK(p.size()==1);CHECK(!p.committed_peak(b,t)); }
 { Planner p; auto wrong=d;wrong.epoch=8;
   CHECK(p.plan_restore({snap(t,10),snap(wrong,10)},a).kind==RestorePlan::Kind::refused_identity);
   CHECK(p.plan_restore({snap(t,10),snap(t,10)},a).kind==RestorePlan::Kind::refused_identity);
   auto r=d;r.kind=Kind::recurrent_rows; // draft recurrent stays unsupported (target only)
   CHECK(p.plan_restore({snap(r,10)},a).kind==RestorePlan::Kind::refused_unsupported_layout);CHECK(p.size()==0); }
 { Planner p; auto s=snap(t,100,90);s.seq_exclusive_real=90;auto r=p.plan_restore({s},a);
   CHECK(r.kind==RestorePlan::Kind::ok);CHECK(r.token.limit(t)==10);CHECK(!p.consume_restore(r.token,t,50));
   CHECK(p.fence_restore(r.token));CHECK(p.native_release(a).status==Status::accepted); }
 { Planner p; { auto r=p.plan_restore({snap(t,10)},a);CHECK(r.kind==RestorePlan::Kind::ok); }
   CHECK(p.size()==1);CHECK(p.native_release(a).status==Status::rejected); }
 std::puts("PASS planner-only: typed-zero unknown overflow overcommit cumulative compound identity move fence cancellation false-refusal destructor-retention");
}
