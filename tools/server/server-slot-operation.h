#pragma once

#include <nlohmann/json.hpp>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>

using slot_operation_json = nlohmann::ordered_json;

static std::string slot_operation_new_epoch() {
    std::random_device rd;
    const uint64_t a = ((uint64_t) rd() << 32) ^ rd() ^ (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
    const uint64_t b = ((uint64_t) rd() << 32) ^ rd();
    char buf[37];
    snprintf(buf, sizeof(buf), "%08x-%04x-%04x-%04x-%012llx",
            (uint32_t) (a >> 32), (uint16_t) (a >> 16), (uint16_t) a,
            (uint16_t) (b >> 48), (unsigned long long) (b & 0x0000ffffffffffffULL));
    return buf;
}

static bool slot_operation_is_uuid(const std::string & value) {
    if (value.size() != 36) return false;
    for (size_t i = 0; i < value.size(); ++i) {
        if (i == 8 || i == 13 || i == 18 || i == 23) {
            if (value[i] != '-') return false;
        } else if (!std::isxdigit((unsigned char) value[i])) {
            return false;
        }
    }
    return true;
}

struct slot_operation_record {
    std::string op_id;
    std::string engine_id;
    std::string owner = "proxy"; // server-configured proxy principal; never client supplied
    std::string engine_epoch;
    uint64_t owner_seq = 0;
    std::string action;
    std::string payload;
    int id_slot = -1;
    std::string filename;
    std::string status = "queued";
    slot_operation_json result = nullptr;
    std::string error;

    bool terminal() const {
        return status == "completed" || status == "failed" || status == "cancelled_before_start";
    }
    slot_operation_json to_json() const {
        return slot_operation_json{{"op_id", op_id}, {"engine_id", engine_id}, {"owner", owner}, {"owner_seq", owner_seq}, {"engine_epoch", engine_epoch},
                    {"action", action}, {"status", status}, {"slot", id_slot}, {"filename", filename},
                    {"result", result}, {"error", error}};
    }
};

struct slot_operation_registry {
    enum class submit_code { created, exact_retry, conflict, overloaded };
    enum class ack_code { accepted, repeated, old_epoch, nonterminal, unknown };
    static constexpr size_t capacity = 256;

    struct owner_state {
        uint64_t last_acked_seq = 0;
        std::string active_op_id;
    };

    explicit slot_operation_registry(size_t max_operations = capacity) : engine_epoch(slot_operation_new_epoch()), max_operations(max_operations) {}

    void configure_owner(const std::string & value) {
        std::lock_guard<std::mutex> lock(mutex);
        if (value.empty() || !operations.empty()) return;
        configured_owner = value;
    }

    void configure_engine_id(const std::string & value) {
        std::lock_guard<std::mutex> lock(mutex);
        if (value.empty() || !operations.empty()) return;
        configured_engine_id = value;
    }

    const std::string & owner_name() const { return configured_owner; }

    slot_operation_json capability(const std::string & owner_name = "") const {
        std::lock_guard<std::mutex> lock(mutex);
        const std::string & principal = owner_name.empty() ? configured_owner : owner_name;
        const auto it = owners.find(owner_key(principal));
        const uint64_t next_seq = (it == owners.end() ? 0 : it->second.last_acked_seq) + 1;
        return slot_operation_json{{"capability", "op_id-terminal-v1"}, {"engine_id", configured_engine_id}, {"engine_epoch", engine_epoch},
                    {"owner", principal}, {"next_seq", next_seq}};
    }

    submit_code submit(const std::string & owner_name, const std::string & id, uint64_t seq, const std::string & epoch, const std::string & requested_engine_id,
            const std::string & action, int slot, const std::string & filename, const std::string & payload,
            slot_operation_record & out) {
        std::lock_guard<std::mutex> lock(mutex);
        if (owner_name.empty() || configured_engine_id.empty() || requested_engine_id != configured_engine_id || !slot_operation_is_uuid(id) || epoch != engine_epoch || seq == 0) return submit_code::conflict;
        auto & owner = owners[owner_key(owner_name)];
        auto found = operations.find(id);
        if (found != operations.end()) {
            out = found->second;
            return found->second.owner == owner_name && found->second.owner_seq == seq && found->second.payload == payload ? submit_code::exact_retry : submit_code::conflict;
        }
        if (seq <= owner.last_acked_seq || seq != owner.last_acked_seq + 1 || !owner.active_op_id.empty()) return submit_code::conflict;
        if (operations.size() >= max_operations) return submit_code::overloaded;
        slot_operation_record op;
        op.op_id = id;
        op.engine_id = configured_engine_id;
        op.owner = owner_name;
        op.engine_epoch = engine_epoch;
        op.owner_seq = seq;
        op.payload = payload;
        op.action = action;
        op.id_slot = slot;
        op.filename = filename;
        operations.emplace(id, op);
        owner.active_op_id = id;
        out = std::move(op);
        return submit_code::created;
    }

    submit_code submit(const std::string & id, uint64_t seq, const std::string & epoch, const std::string & requested_engine_id,
            const std::string & action, int slot, const std::string & filename, const std::string & payload,
            slot_operation_record & out) {
        return submit(configured_owner, id, seq, epoch, requested_engine_id, action, slot, filename, payload, out);
    }

    bool get(const std::string & owner_name, const std::string & id, slot_operation_record & out) const {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = operations.find(id);
        if (it == operations.end() || it->second.owner != owner_name) return false;
        out = it->second;
        return true;
    }

    bool get(const std::string & id, slot_operation_record & out) const {
        return get(configured_owner, id, out);
    }

    void set_status(const std::string & id, const std::string & status) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = operations.find(id);
        if (it != operations.end() && !it->second.terminal()) it->second.status = status;
    }

    void finish(const std::string & id, const std::string & status, const slot_operation_json & result, const std::string & error = {}) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = operations.find(id);
        if (it == operations.end() || it->second.terminal()) return;
        it->second.status = status;
        it->second.result = result;
        it->second.error = error;
    }

    void cancel_before_start(const std::string & id) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = operations.find(id);
        if (it != operations.end() && (it->second.status == "queued" || it->second.status == "deferred")) {
            it->second.status = "cancelled_before_start";
            it->second.error = "cancelled before callback start";
        }
    }

    ack_code ack(const std::string & owner_name, const std::string & id, const std::string & epoch, uint64_t seq) {
        std::lock_guard<std::mutex> lock(mutex);
        if (epoch != engine_epoch) return ack_code::old_epoch;
        auto & owner = owners[owner_key(owner_name)];
        auto it = operations.find(id);
        if (it == operations.end()) return seq > 0 && seq <= owner.last_acked_seq ? ack_code::repeated : ack_code::unknown;
        if (it->second.owner != owner_name || it->second.owner_seq != seq) return ack_code::unknown;
        if (!it->second.terminal()) return ack_code::nonterminal;
        owner.last_acked_seq = seq;
        owner.active_op_id.clear();
        operations.erase(it);
        return ack_code::accepted;
    }

    ack_code ack(const std::string & id, uint64_t seq) {
        return ack(configured_owner, id, engine_epoch, seq);
    }

    ack_code ack(const std::string & owner_name, const std::string & id, uint64_t seq) {
        return ack(owner_name, id, engine_epoch, seq);
    }

    std::string configured_owner = "proxy";
    std::string configured_engine_id;
    std::string engine_epoch;
private:
    std::string owner_key(const std::string & owner_name) const {
        return configured_engine_id + '\0' + owner_name;
    }
    size_t max_operations;
    mutable std::mutex mutex;
    std::unordered_map<std::string, owner_state> owners;
    std::unordered_map<std::string, slot_operation_record> operations;
};
