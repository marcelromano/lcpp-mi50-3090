from pathlib import Path

root = Path(__file__).resolve().parents[3]
common = (root / "common/common.h").read_text()
args = (root / "common/arg.cpp").read_text()
context = (root / "tools/server/server-context.cpp").read_text()
planner = (root / "tools/server/kv_planner.cpp").read_text()

checks = {
    "default off": "bool    kv_admission        = false;" in common,
    "explicit CLI opt-in": '{"--kv-admission"}' in args,
    "recurrent and hybrid components are architecture checked": "expected_recurrent = llama_model_is_recurrent(model) || llama_model_is_hybrid(model)" in context and '"incomplete_native_resource_set"' in context,
    "recurrent peak uses normalized cell bundle": "case kvplan::Kind::recurrent_rows:" in context and "peak = 1;" in context,
    "target and draft SWA rejected": '"target_swa_memory_unsupported"' in context and '"draft_swa_memory_unsupported"' in context,
    "only native SIMPLE draft accepted": '"unsupported_speculative_context"' in context and "COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE" in context,
    "unknown runtime component layout rejected": '"unknown_or_invalid_memory_resource_layout"' in context,
    "all native resource keys are retained per context and slot": "std::vector<kvplan::ResourceKey>" in context and "slot_keys.push_back(key)" in context,
    "target and draft contexts fenced before snapshot": context.index("llama_synchronize(ctx_tgt)") < context.index("kv_admission_snapshots(snapshots, resource_by_slot, reason)"),
    "draft resource gets its own reservation key": "kvplan::Context::draft" in context and "ctx_dft, draft_resource->second" in context,
    "media rejected": '"media_cell_layout_unsupported"' in context,
    "output arithmetic checked": 'output > std::numeric_limits<uint64_t>::max() - prompt' in context,
    "task and slot zero remain valid": "if (!r.owner.epoch || r.peak_by_resource.empty())" in planner and "owner.task" not in planner,
    "cache disabled before model setup": context.index("params_base.cache_ram_mib = 0;") < context.index("params_base.n_ctx_checkpoints = 0;"),
    "slot selection cannot mutate prompt cache": "update_cache = update_cache && prompt_cache && !params_base.kv_admission;" in context,
    "restore actions fail before switch": context.index("KV admission B1 blocks slot restore path") < context.index("switch (task.type)", context.index("void process_single_task")),
    "parent preflight precedes grouped launch": context.index("kv_admission_preflight(task, *slot, child_slots") < context.index("launch_slots_with_parent_task(*slot, child_slots", context.index("void process_single_task")),
    "native release uses stored owner": "release_kv_admission_slot(id_slot);" in context and "kv_admission_planner.native_release(*owner)" in context,
    "release callback does not recover task after reset": "slot.callback_on_release = [this](int id_slot)" in context and "slot.task" not in context[context.index("slot.callback_on_release = [this](int id_slot)"):context.index("slot.reset();", context.index("slot.callback_on_release = [this](int id_slot)"))],
}
failed = [name for name, passed in checks.items() if not passed]
for name, passed in checks.items():
    print(("PASS " if passed else "FAIL ") + name)
if failed:
    raise SystemExit(1)
print(f"ALL {len(checks)} B1 source-order/unsupported gates passed")
