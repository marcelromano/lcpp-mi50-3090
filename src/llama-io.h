#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <stdexcept>
#include "llama.h"

struct ggml_tensor;

class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes written so far
    virtual size_t n_bytes() = 0;

    void write_string(const std::string & str);
};

class llama_io_read_i {
public:
    explicit llama_io_read_i(llama_state_seq_limits * authority = nullptr) : limits(authority) {}
    virtual void restore_cells(uint32_t kind, uint32_t stream, uint64_t count) {
        if (limits && (!limits->consume || !limits->consume(limits->user,kind,stream,count))) {
            limits->limit_refused=true;
            throw std::runtime_error("cumulative restore limit refused");
        }
    }
    void restore_clear() {
        if (limits) { limits->mutation_started=true; ++limits->clear_count; }
    }
    void restore_allocate() { if (limits) ++limits->allocation_count; }
    bool restore_should_cleanup() const { return !validation_only() && (!limits || limits->mutation_started); }
    virtual bool validation_only() const { return false; }
    virtual bool destination_precleared() const { return false; }
protected:
    llama_state_seq_limits * limits;
public:
    virtual ~llama_io_read_i() = default;

    virtual void read(void * dst, size_t size) = 0;
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string & str);
};
