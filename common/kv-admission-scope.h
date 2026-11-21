#pragma once

#include <stdexcept>
#include <string>

// Shared policy for dispatch and native snapshots. Physical usage is never bypassed.
enum class common_kv_admission_scope { all, restore };
inline common_kv_admission_scope common_parse_kv_admission_scope(const std::string & value) {
    if (value == "all") return common_kv_admission_scope::all;
    if (value == "restore") return common_kv_admission_scope::restore;
    throw std::invalid_argument("kv-admission-scope must be all or restore");
}
inline const char * common_kv_admission_scope_name(common_kv_admission_scope scope) {
    return scope == common_kv_admission_scope::all ? "all" : "restore";
}
inline bool common_kv_admission_plain_bypass(common_kv_admission_scope scope) {
    return scope == common_kv_admission_scope::restore;
}
inline bool common_kv_admission_snapshot_owner_valid(common_kv_admission_scope scope, bool active, bool owner) {
    return scope == common_kv_admission_scope::restore ? (!owner || active) : active == owner;
}
