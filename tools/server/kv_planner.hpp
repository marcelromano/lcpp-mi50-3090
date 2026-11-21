#pragma once
#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>
#include <optional>
#include <string>
#include <vector>
#include <utility>

namespace kvplan {
enum class Context : uint8_t { target, draft };
enum class Kind : uint8_t { attention_cells, recurrent_rows };
struct ResourceKey {
    Context context{}; uint32_t stream{}; uint64_t epoch{}; Kind kind{Kind::attention_cells};
    friend bool operator<(const ResourceKey &a, const ResourceKey &b) {
        return std::tie(a.context,a.stream,a.epoch,a.kind)<std::tie(b.context,b.stream,b.epoch,b.kind);
    }
    friend bool operator==(const ResourceKey &a, const ResourceKey &b) {
        return a.context==b.context && a.stream==b.stream && a.epoch==b.epoch && a.kind==b.kind;
    }
};
struct Owner {
    uint64_t epoch{}, task{}, slot{};
    friend bool operator<(const Owner &a, const Owner &b) { return std::tie(a.epoch,a.task,a.slot)<std::tie(b.epoch,b.task,b.slot); }
    friend bool operator==(const Owner &a, const Owner &b) { return a.epoch==b.epoch && a.task==b.task && a.slot==b.slot; }
};
struct Snapshot {
    ResourceKey key; bool known{}; uint64_t capacity{}, used_real{}, seq_exclusive_real{};
    std::map<Owner,uint64_t> exclusive_by_owner;
    // Only authentic exclusive recurrent tail rows retained by this incoming actor.
    // Native producer must bind seq_id to owner.slot and guarantee in-place reuse.
    std::map<Owner,uint64_t> reusable_recurrent_by_incoming_owner;
    // Exclusive cells/rows of the IDLE target slot selected for the incoming owner. The native
    // transition (seq_rm of the resident before any growth) retires them while the peak counts
    // the whole sequence, so they are credited once, bound to owner/epoch/resource.
    std::map<Owner,uint64_t> resident_credit_by_incoming_owner;
};
struct Reservation {
    Owner owner; std::optional<Owner> parent;
    std::map<ResourceKey,uint64_t> peak_by_resource;
};
enum class Status { accepted, rejected, already_released };
struct Result { Status status; std::string reason; };

class Planner;
// Engine-owned: destruction never releases a native commitment. The executor
// must call fence_restore only AFTER target and draft native synchronization.
class RestoreToken {
public:
    RestoreToken() = default;
    RestoreToken(const RestoreToken&) = delete;
    RestoreToken& operator=(const RestoreToken&) = delete;
    RestoreToken& operator=(RestoreToken&&) = delete;
    RestoreToken(RestoreToken&& other) noexcept : issuer_(other.issuer_), owner_(other.owner_),
        limits_(std::move(other.limits_)), consumed_(std::move(other.consumed_)) { other.issuer_=nullptr; }
    std::optional<uint64_t> limit(const ResourceKey& key) const noexcept {
        const auto it=limits_.find(key);
        return it==limits_.end()?std::nullopt:std::optional<uint64_t>(it->second);
    }
private:
    friend class Planner;
    const Planner* issuer_{};
    Owner owner_{};
    std::map<ResourceKey,uint64_t> limits_,consumed_;
};
struct RestorePlan {
    enum class Kind { ok, refused_unknown, refused_overflow, refused_overcommit,
        refused_identity, refused_unsupported_layout };
    Kind kind{Kind::refused_unknown};
    std::string reason;
    RestoreToken token;
};

class Planner {
public:
    Result admit(const std::vector<Snapshot>& snapshots, const std::vector<Reservation>& batch);
    Result native_release(const Owner& owner);
    RestorePlan plan_restore(const std::vector<Snapshot>& snapshots, const Owner& owner,
                             const std::vector<ResourceKey>& declared={});
    bool consume_restore(RestoreToken& token, const ResourceKey& key, uint64_t count) noexcept;
    bool fence_restore(const RestoreToken& token) noexcept;
    void pause(const Owner&) const noexcept {}
    void cancel_by_caller(const Owner&) const noexcept {}
    uint64_t restore_available(const Snapshot& snapshot) const noexcept;
    std::optional<uint64_t> committed_peak(const Owner &owner, const ResourceKey &key) const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto entry=ledger_.find(owner);
        if(entry==ledger_.end() || owner.epoch!=key.epoch) return std::nullopt;
        const auto peak=entry->second.peaks.find(key);
        return peak==entry->second.peaks.end()?std::nullopt:std::optional<uint64_t>(peak->second);
    }
    size_t size() const noexcept { std::lock_guard<std::mutex> lock(mutex_); return ledger_.size(); }
private:
    friend struct PlannerTestAccess;
    bool active_total(const Snapshot& snapshot, uint64_t& total) const noexcept;
    struct Entry { Owner owner; std::optional<Owner> parent; std::map<ResourceKey,uint64_t> peaks; bool restore{}; bool fenced{}; };
    mutable std::mutex mutex_;
    std::map<Owner,Entry> ledger_;
    std::map<Owner,bool> released_;
};
} // namespace kvplan
