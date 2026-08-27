#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>

struct llama_context;
struct llama_model;

// DS4-only packed rolling bank for host-backed FRONT tensors. The model and
// context must outlive this object, and deactivate() must run before either is
// destroyed.
class siliang_ds4_front_slab {
public:
    enum failure_code : int32_t {
        FAILURE_NONE              = 0,
        FAILURE_INVALID_MODEL     = -13001,
        FAILURE_INVALID_STATE     = -13002,
        FAILURE_CUDA_BRIDGE       = -13003,
        FAILURE_FRONT_PLACEMENT   = -13004,
        FAILURE_ALLOCATION        = -13005,
        FAILURE_CONTEXT_BIND      = -13006,
        FAILURE_MARKER_TOPOLOGY   = -13007,
        FAILURE_MARKER_SEQUENCE   = -13008,
        FAILURE_CUDA_RUNTIME      = -13009,
    };

    struct metrics {
        bool prepared = false;
        bool bound = false;
        bool active = false;
        size_t bank_bytes = 0;
        size_t host_store_bytes = 0;
        int32_t resident_layer = -1;
        uint64_t tokens = 0;
        uint64_t copies = 0;
        uint64_t waits = 0;
        uint64_t payload_h2d_bytes = 0;
        uint64_t wire_h2d_bytes = 0;
        uint64_t submission_host_ns = 0;
        uint64_t wait_enqueue_host_ns = 0;
    };

    siliang_ds4_front_slab();
    ~siliang_ds4_front_slab();

    siliang_ds4_front_slab(const siliang_ds4_front_slab &) = delete;
    siliang_ds4_front_slab & operator=(const siliang_ds4_front_slab &) = delete;
    siliang_ds4_front_slab(siliang_ds4_front_slab &&) = delete;
    siliang_ds4_front_slab & operator=(siliang_ds4_front_slab &&) = delete;

    bool prepare(const llama_model & model);
    bool bind(llama_context & ctx);
    bool activate();
    void deactivate();

    ggml_backend_sched_split_observer_callback observer_callback() const noexcept;
    void * observer_user_data() noexcept;

    metrics snapshot() const noexcept;
    int32_t failure() const noexcept;
    const char * failure_message() const noexcept;

private:
    static bool observer_thunk(
            int split_index,
            ggml_backend_t backend,
            const ggml_cgraph * graph,
            void * user_data);

    struct impl;
    std::unique_ptr<impl> impl_;
};
