#include "siliang-moe-runtime.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int32_t SILIANG_RUNTIME_FAILURE_INIT       = -4401;
constexpr int32_t SILIANG_RUNTIME_FAILURE_ROUTE      = -4402;
constexpr int32_t SILIANG_RUNTIME_FAILURE_L2         = -4403;
constexpr int32_t SILIANG_RUNTIME_FAILURE_STAGING    = -4404;
constexpr int32_t SILIANG_RUNTIME_FAILURE_H2D        = -4405;
constexpr int32_t SILIANG_RUNTIME_FAILURE_EVENT      = -4406;
constexpr int32_t SILIANG_RUNTIME_FAILURE_PREFILL    = -4407;
constexpr std::array<uint32_t, 8> SILIANG_ROUTE_STATS_TOKEN_CHECKPOINTS = {32, 64, 128, 256, 512, 1024, 1512, 2048};

using siliang_moe_policy::cache_slot;
using siliang_moe_policy::slot_segment;

struct runtime_part {
    llama_siliang_moe_arena_part_role role = LLAMA_SILIANG_MOE_ARENA_GATE_WEIGHT;
    const ggml_tensor * source = nullptr;
    ggml_tensor * arena = nullptr;
    size_t bytes = 0;
    size_t source_stride = 0;
    size_t staging_offset = 0;
    int32_t l2_part = -1;
};

struct runtime_layer {
    std::array<runtime_part, LLAMA_SILIANG_MOE_ARENA_PART_ROLE_COUNT> parts = {};
    std::array<uint8_t, LLAMA_SILIANG_MOE_ARENA_PART_ROLE_COUNT> present = {};
    size_t part_count = 0;
    size_t schema_index = 0;
    uint32_t policy_first = 0;
    uint32_t policy_count = 0;
    uint32_t physical_first = 0;
};

struct route_stats_metrics {
    uint64_t routes = 0;
    uint64_t exact_routes = 0;
    uint64_t unknown_routes = 0;
    uint64_t selections = 0;
    uint64_t state_l1 = 0;
    uint64_t state_l2 = 0;
    uint64_t state_uncached = 0;
    uint64_t state_unknown = 0;
    uint64_t exec_gpu_k_hit = 0;
    uint64_t exec_gpu_k_admit = 0;
    uint64_t exec_gpu_r = 0;
    uint64_t exec_cpu = 0;
    uint64_t exec_unknown = 0;
    std::vector<uint64_t> compositions;
    std::vector<uint64_t> execution_compositions;
};

struct l2_prepare_counts {
    uint32_t hits = 0;
    uint32_t misses = 0;
    bool exact = true;
    std::vector<int32_t> hit_experts;
    std::vector<int32_t> miss_experts;
};

struct pending_k_transition {
    int32_t layer = -1;
    int32_t candidate_expert = -1;
    int64_t candidate_key = -1;
    int32_t candidate_r_physical = -1;
    int32_t policy_slot = -1;
    int32_t k_physical = -1;
    int64_t victim_key = -1;
    uint32_t released_l2_slot = std::numeric_limits<uint32_t>::max();
};

// Splits a set of host copies evenly across the calling thread and a few
// persistent workers. Used for the L2 -> P staging copy, which is otherwise a
// single-threaded memcpy of each expert.
class staging_copier {
public:
    struct segment {
        uint8_t * destination = nullptr;
        const uint8_t * source = nullptr;
        size_t bytes = 0;
    };

    ~staging_copier() {
        stop();
    }

    void start(size_t threads) {
        stop();
        stopping = false;
        for (size_t index = 1; index < threads; ++index) {
            workers.emplace_back([this, index] { worker_loop(index); });
        }
        thread_count = threads;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        start_cv.notify_all();
        for (auto & worker : workers) {
            worker.join();
        }
        workers.clear();
        thread_count = 1;
    }

    size_t threads() const {
        return thread_count;
    }

    void copy(const std::vector<segment> & segments) {
        size_t total = 0;
        for (const auto & item : segments) {
            total += item.bytes;
        }
        if (workers.empty() || total == 0) {
            copy_range(segments, 0, total);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            job = &segments;
            job_bytes = total;
            pending = workers.size();
            ++generation;
        }
        start_cv.notify_all();
        copy_range(segments, 0, share(total, 0));
        std::unique_lock<std::mutex> lock(mutex);
        done_cv.wait(lock, [this] { return pending == 0; });
        job = nullptr;
    }

private:
    size_t share(size_t total, size_t index) const {
        // 4 KiB aligned boundaries; the last thread takes the remainder
        const size_t per = ((total / thread_count) + 4095) & ~size_t {4095};
        const size_t first = std::min(total, per * index);
        const size_t last = index + 1 == thread_count ? total : std::min(total, per * (index + 1));
        return index == 0 ? last : last - first;
    }

    static void copy_range(const std::vector<segment> & segments, size_t first, size_t count) {
        size_t skip = first;
        for (const auto & item : segments) {
            if (count == 0) {
                break;
            }
            if (skip >= item.bytes) {
                skip -= item.bytes;
                continue;
            }
            const size_t bytes = std::min(item.bytes - skip, count);
            std::memcpy(item.destination + skip, item.source + skip, bytes);
            count -= bytes;
            skip = 0;
        }
    }

    void worker_loop(size_t index) {
        uint64_t seen = 0;
        for (;;) {
            const std::vector<segment> * segments = nullptr;
            size_t total = 0;
            {
                std::unique_lock<std::mutex> lock(mutex);
                start_cv.wait(lock, [&] { return stopping || generation != seen; });
                if (stopping) {
                    return;
                }
                seen = generation;
                segments = job;
                total = job_bytes;
            }
            const size_t per = ((total / thread_count) + 4095) & ~size_t {4095};
            const size_t first = std::min(total, per * index);
            copy_range(*segments, first, share(total, index));
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (--pending == 0) {
                    done_cv.notify_one();
                }
            }
        }
    }

    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable start_cv;
    std::condition_variable done_cv;
    const std::vector<segment> * job = nullptr;
    size_t job_bytes = 0;
    size_t pending = 0;
    size_t thread_count = 1;
    uint64_t generation = 0;
    bool stopping = false;
};

struct runtime_metrics {
    uint64_t map_calls = 0;
    uint64_t route_choices = 0;
    uint64_t k_hits = 0;
    uint64_t k_misses = 0;
    uint64_t k_admissions = 0;
    uint64_t k_evictions = 0;
    uint64_t k_rejections = 0;
    uint64_t k_reuse_fences = 0;
    uint64_t r_experts = 0;
    uint64_t r_bank_uses = 0;
    uint64_t r_bank_reuses = 0;
    uint64_t p_bank_uses = 0;
    uint64_t p_bank_reuse_waits = 0;
    uint64_t staged_bytes = 0;
    uint64_t h2d_ops = 0;
    uint64_t h2d_bytes = 0;
    uint64_t ready_events = 0;
    uint64_t compute_waits = 0;
    uint64_t pending_recoveries = 0;
    uint64_t l2_async_prepares = 0;
    uint64_t l2_sync_prepares = 0;
    uint64_t l2_hits = 0;
    uint64_t l2_misses = 0;
    uint64_t l2_waits = 0;
    uint64_t l2_releases = 0;
    uint64_t l2_release_failures = 0;
    uint64_t l2_demotions = 0;
    uint64_t l2_demotion_failures = 0;
    uint64_t k_deferred_promotions = 0;
    uint64_t k_transition_commits = 0;
    uint64_t k_transition_cancels = 0;
    uint64_t demotion_d2h_ops = 0;
    uint64_t demotion_d2h_bytes = 0;
    uint64_t promotion_d2d_ops = 0;
    uint64_t promotion_d2d_bytes = 0;
    uint64_t transition_submit_ns = 0;
    uint64_t transition_device_wait_ns = 0;
    uint64_t transition_host_copy_ns = 0;
    uint64_t transition_exposed_wait_ns = 0;
    uint64_t demotion_reuse_l2 = 0;
    uint64_t demotion_reuse_cold = 0;
    uint64_t demotion_reuse_unknown = 0;
    uint64_t demotion_reuse_pending = 0;
    std::array<uint64_t, 5> demotion_reuse_l2_round_buckets = {};
    std::array<uint64_t, 5> demotion_reuse_cold_round_buckets = {};
    uint64_t slfu_cold_bypasses = 0;
    uint64_t hybrid_cpu_l2 = 0;
    uint64_t hybrid_cpu_miss = 0;
    uint64_t hybrid_gpu_admissions = 0;
    uint64_t hybrid_gpu_bypasses = 0;
    uint64_t hybrid_gpu_anchors = 0;
    uint64_t decode_host_ns = 0;
    uint64_t decode_prepare_ns = 0;
    uint64_t decode_staged_l2 = 0;
    uint64_t decode_staged_miss = 0;
    uint64_t direct_experts = 0;
    uint64_t direct_bytes = 0;
    uint64_t direct_fallbacks = 0;
    uint64_t lease_returns = 0;
    uint64_t deferred_releases = 0;
    uint64_t deferred_release_failures = 0;
    uint64_t pin_register_ns = 0;
    uint64_t pin_warmup_ns = 0;
    uint32_t pinned_slots_armed = 0;
    uint64_t staging_copy_ns = 0;
    uint64_t staging_copy_bytes = 0;
    uint64_t staging_transient_copies = 0;
    uint64_t prefill_prepare_ns = 0;
    uint64_t prefill_read_wait_ns = 0;
    uint64_t prefill_copy_ns = 0;
    uint64_t p_bank_wait_ns = 0;
    uint64_t probe_checks = 0;
    uint64_t probe_pending = 0;
    uint64_t verify_entries = 0;
    uint64_t verify_windows = 0;
    uint64_t verify_mismatches = 0;
    uint64_t verify_skipped = 0;
    uint64_t phase_invalidations = 0;
    uint64_t k_ext_clears = 0;
    uint64_t k_ext_admissions = 0;
    uint64_t k_ext_plan_max = 0;
    uint64_t prefill_maps = 0;
    uint64_t prefill_tokens = 0;
    uint64_t prefill_choices = 0;
    uint64_t prefill_unique = 0;
    uint64_t prefill_unique_max = 0;
    uint64_t prefill_k_hits = 0;
    uint64_t prefill_k_misses = 0;
    uint64_t prefill_k_admissions = 0;
    uint64_t prefill_k_evictions = 0;
    uint64_t prefill_p_waves = 0;
    uint64_t prefill_h2d_ops = 0;
    uint64_t prefill_h2d_bytes = 0;
    uint64_t prefill_compute_waits = 0;
    uint64_t prefill_bitmap_sweeps = 0;
    uint64_t prefill_bitmap_sweep_tokens = 0;
    uint64_t prefill_bitmap_pairs = 0;
    uint64_t prefill_bitmap_seeded = 0;
    uint64_t prefill_bitmap_needed = 0;
    uint64_t prefill_bitmap_overlap = 0;
    uint64_t prefill_bitmap_new = 0;
    uint64_t prefill_bitmap_unused = 0;
    uint64_t prefill_bitmap_resets = 0;
    uint64_t prefill_bitmap_incomplete = 0;
};

struct route_bitmap_metrics {
    uint64_t pairs = 0;
    uint64_t seeded = 0;
    uint64_t needed = 0;
    uint64_t overlap = 0;
};

static bool checked_add(size_t lhs, size_t rhs, size_t & result) {
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

static bool checked_mul(size_t lhs, size_t rhs, size_t & result) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

static bool checked_align(size_t value, size_t alignment, size_t & result) {
    if (alignment == 0) {
        alignment = 1;
    }
    const size_t remainder = value % alignment;
    if (remainder == 0) {
        result = value;
        return true;
    }
    return checked_add(value, alignment - remainder, result);
}

static bool is_weight_role(llama_siliang_moe_arena_part_role role) {
    return role == LLAMA_SILIANG_MOE_ARENA_GATE_WEIGHT ||
           role == LLAMA_SILIANG_MOE_ARENA_UP_WEIGHT ||
           role == LLAMA_SILIANG_MOE_ARENA_DOWN_WEIGHT ||
           role == LLAMA_SILIANG_MOE_ARENA_GATE_UP_WEIGHT;
}

static const char * role_name(llama_siliang_moe_arena_part_role role) {
    switch (role) {
        case LLAMA_SILIANG_MOE_ARENA_GATE_WEIGHT:   return "gate_weight";
        case LLAMA_SILIANG_MOE_ARENA_UP_WEIGHT:     return "up_weight";
        case LLAMA_SILIANG_MOE_ARENA_DOWN_WEIGHT:   return "down_weight";
        case LLAMA_SILIANG_MOE_ARENA_GATE_UP_WEIGHT:return "gate_up_weight";
        case LLAMA_SILIANG_MOE_ARENA_GATE_BIAS:     return "gate_bias";
        case LLAMA_SILIANG_MOE_ARENA_UP_BIAS:       return "up_bias";
        case LLAMA_SILIANG_MOE_ARENA_DOWN_BIAS:     return "down_bias";
        case LLAMA_SILIANG_MOE_ARENA_GATE_UP_BIAS:  return "gate_up_bias";
        case LLAMA_SILIANG_MOE_ARENA_GATE_SCALE:    return "gate_scale";
        case LLAMA_SILIANG_MOE_ARENA_UP_SCALE:      return "up_scale";
        case LLAMA_SILIANG_MOE_ARENA_DOWN_SCALE:    return "down_scale";
        case LLAMA_SILIANG_MOE_ARENA_PART_ROLE_COUNT: break;
    }
    return "unknown";
}

static int32_t source_name_role(const std::string & name) {
    if (name == "gate") {
        return LLAMA_SILIANG_MOE_ARENA_GATE_WEIGHT;
    }
    if (name == "up") {
        return LLAMA_SILIANG_MOE_ARENA_UP_WEIGHT;
    }
    if (name == "down") {
        return LLAMA_SILIANG_MOE_ARENA_DOWN_WEIGHT;
    }
    if (name == "gate_up") {
        return LLAMA_SILIANG_MOE_ARENA_GATE_UP_WEIGHT;
    }
    return -1;
}

static std::vector<std::string> split_names(const std::string & value) {
    std::vector<std::string> result;
    size_t begin = 0;
    while (begin <= value.size()) {
        const size_t end = value.find(',', begin);
        const size_t count = (end == std::string::npos ? value.size() : end) - begin;
        if (count == 0) {
            return {};
        }
        result.push_back(value.substr(begin, count));
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return result;
}

template <typename T>
static bool load_proc(ggml_backend_reg_t reg, const char * name, T & result) {
    result = reinterpret_cast<T>(ggml_backend_reg_get_proc_address(reg, name));
    return result != nullptr;
}

} // namespace

struct siliang_moe_runtime {
    using cuda_stream_t = ggml_backend_cuda_siliang_stream_t;
    using cuda_event_t = ggml_backend_cuda_siliang_event_t;
    using cuda_status = ggml_backend_cuda_siliang_status;

    using cuda_stream_create_fn = cuda_status (*)(ggml_backend_t, cuda_stream_t *);
    using cuda_stream_destroy_fn = cuda_status (*)(cuda_stream_t);
    using cuda_stream_synchronize_fn = cuda_status (*)(cuda_stream_t);
    using cuda_event_create_fn = cuda_status (*)(ggml_backend_t, cuda_event_t *);
    using cuda_event_destroy_fn = cuda_status (*)(cuda_event_t);
    using cuda_event_synchronize_fn = cuda_status (*)(cuda_event_t);
    using cuda_event_record_fn = cuda_status (*)(cuda_stream_t, cuda_event_t);
    using cuda_main_event_record_fn = cuda_status (*)(ggml_backend_t, cuda_event_t);
    using cuda_stream_wait_event_fn = cuda_status (*)(cuda_stream_t, cuda_event_t);
    using cuda_main_wait_event_fn = cuda_status (*)(ggml_backend_t, cuda_event_t);
    using cuda_h2d_async_fn = cuda_status (*)(cuda_stream_t, ggml_tensor *, const void *, size_t, size_t);
    using cuda_d2h_async_fn = cuda_status (*)(cuda_stream_t, ggml_tensor *, void *, size_t, size_t);
    using cuda_d2d_async_fn = cuda_status (*)(cuda_stream_t, ggml_tensor *, size_t, size_t, size_t);
    using cuda_host_register_fn = cuda_status (*)(ggml_backend_t, void *, size_t);
    using cuda_event_query_fn = cuda_status (*)(cuda_event_t);
    using cuda_host_unregister_fn = cuda_status (*)(ggml_backend_t, void *);
    using cuda_alias_buffer_fn = ggml_backend_buffer_t (*)(ggml_backend_t, const uint64_t *, const size_t *, size_t);
    using cuda_tail_query_fn = cuda_status (*)(ggml_backend_buffer_type_t, void **, size_t *, size_t *, uint64_t *);
    using cuda_vmm_granularity_fn = size_t (*)(ggml_backend_t);
    using cuda_memset_async_fn = cuda_status (*)(cuda_stream_t, ggml_tensor *, size_t, size_t, int);

    using cpu_query_fn = int (*)(ggml_backend_t, ggml_siliangem_cache_info *);
    using cpu_prepare_fn = int (*)(ggml_backend_t, uint32_t, const int32_t *, uint32_t);
    using cpu_prepare_async_fn = int (*)(ggml_backend_t, uint32_t, const int32_t *, uint32_t,
            int32_t *, uint32_t, uint32_t *, uint32_t *, uint32_t *);
    using cpu_wait_fn = int (*)(ggml_backend_t);
    using cpu_wait_expert_fn = int (*)(ggml_backend_t, uint32_t, uint32_t);
    using cpu_copy_part_fn = int (*)(ggml_backend_t, uint32_t, uint32_t, uint32_t, void *, size_t);
    using cpu_location_fn = int (*)(ggml_backend_t, uint32_t, uint32_t);
    using cpu_release_fn = int (*)(ggml_backend_t, uint32_t, uint32_t, uint32_t *);
    using cpu_store_fn = int (*)(ggml_backend_t, uint32_t, uint32_t, uint32_t,
            const void * const *, const size_t *, uint32_t);
    using cpu_arena_span_fn = int (*)(ggml_backend_t, void **, size_t *, uint32_t *);
    using cpu_expert_slot_fn = int (*)(ggml_backend_t, uint32_t, uint32_t, uint32_t *);
    using cpu_lease_part_fn = int (*)(ggml_backend_t, uint32_t, uint32_t, uint32_t,
            const void **, size_t *, uint32_t *);
    using cpu_unlease_fn = int (*)(ggml_backend_t, uint32_t);

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_siliang_expert_cache_params params = {};
    llama_siliang_moe_arena_model_info model_info = {};

    ggml_backend_t cuda = nullptr;
    ggml_backend_t cpu = nullptr;
    cuda_stream_t copy_stream = nullptr;

    cuda_stream_create_fn cuda_stream_create = nullptr;
    cuda_stream_destroy_fn cuda_stream_destroy = nullptr;
    cuda_stream_synchronize_fn cuda_stream_synchronize = nullptr;
    cuda_event_create_fn cuda_event_create = nullptr;
    cuda_event_destroy_fn cuda_event_destroy = nullptr;
    cuda_event_synchronize_fn cuda_event_synchronize = nullptr;
    cuda_event_record_fn cuda_event_record = nullptr;
    cuda_main_event_record_fn cuda_main_event_record = nullptr;
    cuda_stream_wait_event_fn cuda_stream_wait_event = nullptr;
    cuda_main_wait_event_fn cuda_main_wait_event = nullptr;
    cuda_h2d_async_fn cuda_h2d_async = nullptr;
    cuda_d2h_async_fn cuda_d2h_async = nullptr;
    cuda_d2d_async_fn cuda_d2d_async = nullptr;
    cuda_host_register_fn cuda_host_register = nullptr;
    cuda_event_query_fn cuda_event_query = nullptr;
    cuda_host_unregister_fn cuda_host_unregister = nullptr;
    cuda_alias_buffer_fn cuda_alias_buffer = nullptr;
    cuda_tail_query_fn cuda_tail_query = nullptr;
    cuda_vmm_granularity_fn cuda_vmm_granularity = nullptr;
    cuda_memset_async_fn cuda_memset_async = nullptr;

    // Decode K (--expert-cache-l1-k-decode): K slots [0, k_ext) are mapped onto the tail of the CUDA
    // compute buffer. A large prefill ubatch needs that memory for activations; a decode graph plan
    // ends well before it. Every phase change invalidates K, bounded prefill only uses slots from
    // k_ext on, entering decode zeroes the extra slots, and each decode graph plan is checked against
    // the tail start before any extra slot is used.
    uint32_t k_ext = 0;
    bool k_ext_active = false;
    ggml_backend_buffer_type_t k_ext_tail_buft = nullptr;
    void * k_ext_tail_base = nullptr;
    size_t k_ext_tail_head = 0;
    uint64_t k_ext_tail_generation = 0;
    cuda_event_t k_ext_clear_event = nullptr;

    cpu_query_fn cpu_query = nullptr;
    cpu_prepare_fn cpu_prepare = nullptr;
    cpu_prepare_async_fn cpu_prepare_async = nullptr;
    cpu_wait_fn cpu_wait = nullptr;
    // optional: wait for one expert's read so its copy overlaps the reads still in flight
    cpu_wait_expert_fn cpu_wait_expert = nullptr;
    bool l2_reads_pending = false;
    cpu_copy_part_fn cpu_copy_part = nullptr;
    cpu_location_fn cpu_location = nullptr;
    cpu_release_fn cpu_release = nullptr;
    cpu_store_fn cpu_store = nullptr;
    cpu_arena_span_fn cpu_arena_span = nullptr;
    cpu_expert_slot_fn cpu_expert_slot = nullptr;
    cpu_lease_part_fn cpu_lease_part = nullptr;
    cpu_unlease_fn cpu_unlease = nullptr;

    // Leading L2 slots registered with CUDA. Decode and prefill copies of experts resident
    // there read the L2 slot directly; the slot stays leased until the P bank
    // event of the map that issued the copy has completed.
    std::vector<std::pair<uint8_t *, size_t>> pinned_chunks;
    uint32_t pinned_slot_count = 0;
    std::vector<std::vector<uint32_t>> bank_leases;
    std::vector<std::vector<std::pair<int32_t, int32_t>>> bank_deferred_releases;

    // --expert-cache-verify: the previous decode route, checked at the next map
    struct verify_entry {
        int32_t layer = -1;
        int32_t expert = -1;
        int32_t physical = -1;
        char kind = '?'; // H = K hit, A = K admission, R = exchange
        uint64_t route = 0;
    };
    std::vector<verify_entry> verify_pending;
    // copy-ordering probe: the previous map's copies, queried without waiting at the next map
    int32_t probe_prev_layer = -1;
    bool probe_prev_prefill = false;
    bool probe_prev_copied = false;
    staging_copier copier;
    uint64_t verify_route_serial = 0;

    ggml_context_ptr arena_ctx;
    ggml_backend_buffer_ptr arena_buffer;
    std::vector<ggml_backend_buffer_ptr> arena_alias_buffers;
    ggml_backend_buffer_ptr staging_buffer;
    ggml_backend_buffer_ptr demotion_buffer;
    uint8_t * staging_base = nullptr;
    uint8_t * demotion_base = nullptr;
    size_t staging_slot_bytes = 0;
    size_t arena_bytes = 0;

    std::vector<std::array<ggml_tensor *, LLAMA_SILIANG_MOE_ARENA_PART_ROLE_COUNT>> variant_arenas;
    std::vector<llama_siliang_moe_arena_layer_info> schemas;
    std::vector<uint32_t> schema_k_capacity;
    std::vector<uint32_t> schema_r_first;
    std::vector<uint32_t> schema_arena_capacity;
    std::vector<runtime_layer> layers;
    std::vector<int32_t> managed_layers;
    std::vector<uint8_t> managed;

    std::vector<cache_slot> slots;
    std::vector<int32_t> resident;
    std::vector<uint64_t> frequencies;
    std::vector<uint64_t> layer_decode_round;
    std::vector<uint64_t> demotion_start_round;
    std::vector<uint8_t> demotion_reuse_pending;
    std::vector<siliang_moe_prefill::route_bitmap> prefill_bitmaps;
    std::vector<uint8_t> prefill_bitmap_valid;
    std::vector<route_bitmap_metrics> prefill_bitmap_layers;
    std::vector<siliang_moe_prefill::route_bitmap> prefill_sweep_bitmaps;
    size_t prefill_sweep_cursor = 0;
    size_t prefill_sweep_tokens = 0;
    uint64_t prefill_bitmap_epoch = 0;
    uint64_t prefill_bitmap_attempt_serial = 0;
    uint64_t prefill_sweep_attempt = 0;
    uint64_t prefill_sweep_base_unique = 0;
    uint64_t prefill_sweep_base_k_hits = 0;
    uint64_t prefill_sweep_base_k_misses = 0;
    uint64_t prefill_sweep_base_admissions = 0;
    uint64_t prefill_sweep_base_evictions = 0;
    uint64_t prefill_sweep_base_p_waves = 0;
    uint64_t prefill_sweep_base_h2d_ops = 0;
    uint64_t prefill_sweep_base_h2d_bytes = 0;
    uint64_t prefill_sweep_base_prepare_ns = 0;
    uint64_t prefill_sweep_base_copy_ns = 0;
    uint64_t prefill_sweep_base_p_wait_ns = 0;
    std::chrono::steady_clock::time_point prefill_sweep_start {};
    // prefill pass timing: one pass per ubatch, delimited by maps of the first managed layer
    bool pass_open = false;
    uint64_t pass_index = 0;
    size_t pass_tokens = 0;
    std::chrono::steady_clock::time_point pass_start {};
    uint64_t pass_base_prepare_ns = 0;
    uint64_t pass_base_copy_ns = 0;
    uint64_t pass_base_p_wait_ns = 0;
    uint64_t pass_base_misses = 0;
    uint64_t pass_base_unique = 0;
    uint64_t pass_base_k_hits = 0;
    uint64_t pass_base_h2d_bytes = 0;
    uint64_t pass_base_read_wait_ns = 0;
    // decode stretch between two prefills (route stats only)
    bool decode_open = false;
    std::chrono::steady_clock::time_point decode_start {};
    runtime_metrics decode_base {};
    uint64_t clock = 0;

    std::vector<cuda_event_t> layer_ready_events;
    std::vector<uint8_t> layer_event_pending;
    std::vector<cuda_event_t> exchange_release_events;
    std::vector<uint8_t> exchange_bank_used;
    std::vector<cuda_event_t> staging_events;
    std::vector<uint8_t> staging_event_recorded;
    std::vector<cuda_event_t> k_reuse_events;
    cuda_event_t transition_event = nullptr;
    bool transition_event_recorded = false;
    std::vector<pending_k_transition> pending_transitions;
    std::thread transition_worker;
    std::mutex transition_worker_mutex;
    std::condition_variable transition_worker_cv;
    bool transition_worker_stop = false;
    bool transition_worker_job_pending = false;
    bool transition_worker_done = true;
    int32_t transition_worker_error = 0;
    std::string transition_worker_error_stage;
    int32_t transition_worker_error_layer = -1;
    int32_t transition_worker_error_expert = -1;
    uint32_t transition_worker_error_slot = std::numeric_limits<uint32_t>::max();
    uint64_t transition_worker_device_wait_ns = 0;
    uint64_t transition_worker_host_copy_ns = 0;
    size_t next_exchange_bank = 0;
    size_t next_staging_bank = 0;
    size_t active_staging_bank = 0;

    runtime_metrics metrics;
    siliang_moe_hybrid::cost_table hybrid_costs;
    route_stats_metrics route_stats_data;
    size_t route_stats_checkpoint_index = 0;
    std::atomic<int32_t> failure {0};
    std::mutex mutex;
    bool bound = false;
    bool l2_enabled = false;
    bool banked = false;
    bool logged_sync_l2 = false;
    bool logged_first_route = false;
    bool logged_first_prefill = false;
    bool logged_pending_recovery = false;
    bool force_k_reuse_fence = false;

    enum class route_phase : uint8_t {
        none,
        decode,
        prefill,
    };
    route_phase phase = route_phase::none;

    void transition_worker_loop() {
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(transition_worker_mutex);
                transition_worker_cv.wait(lock, [this] {
                    return transition_worker_stop || transition_worker_job_pending;
                });
                if (transition_worker_stop) {
                    return;
                }
                transition_worker_job_pending = false;
            }

            int32_t error = 0;
            const auto device_wait_start = std::chrono::steady_clock::now();
            if (!transition_event ||
                cuda_event_synchronize(transition_event) != GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS) {
                error = SILIANG_RUNTIME_FAILURE_EVENT;
            }
            const auto device_wait_end = std::chrono::steady_clock::now();

            const auto host_copy_start = device_wait_end;
            std::string error_stage;
            int32_t error_layer = -1;
            int32_t error_expert = -1;
            uint32_t error_slot = std::numeric_limits<uint32_t>::max();

            // The CPU cache may still own overlapped reads issued while the preceding
            // graph was executing. The K/L2 swap mutates persistent slot ownership, so
            // it must begin from a quiescent L2 state rather than racing those reads.
            if (error == 0 && (!cpu_wait || !cpu_wait(cpu))) {
                error = SILIANG_RUNTIME_FAILURE_L2;
                error_stage = "quiesce";
            }

            if (error == 0) {
                for (auto & transition : pending_transitions) {
                    uint32_t released_slot = std::numeric_limits<uint32_t>::max();
                    if (!cpu_release(
                            cpu, static_cast<uint32_t>(transition.layer),
                            static_cast<uint32_t>(transition.candidate_expert), &released_slot)) {
                        error = SILIANG_RUNTIME_FAILURE_L2;
                        error_stage = "candidate-release";
                        error_layer = transition.layer;
                        error_expert = transition.candidate_expert;
                        break;
                    }
                    transition.released_l2_slot = released_slot;
                }
            }

            if (error == 0) {
                const auto & source = model->siliang_expert_source;
                for (size_t transition_index = 0; transition_index < pending_transitions.size(); ++transition_index) {
                    auto & transition = pending_transitions[transition_index];
                    const uint32_t victim_layer = static_cast<uint32_t>(transition.victim_key / model_info.expert_count);
                    const uint32_t victim_expert = static_cast<uint32_t>(transition.victim_key % model_info.expert_count);
                    if (victim_layer >= layers.size() || source.n_parts == 0) {
                        error = SILIANG_RUNTIME_FAILURE_L2;
                        break;
                    }
                    const auto & victim_descriptor = layers[victim_layer];
                    std::vector<const void *> parts(source.n_parts, nullptr);
                    std::vector<size_t> part_sizes(source.n_parts, 0);
                    const uint8_t * slot_base = demotion_base + transition_index * staging_slot_bytes;
                    for (int role = 0; role < LLAMA_SILIANG_MOE_ARENA_PART_ROLE_COUNT; ++role) {
                        if (!victim_descriptor.present[role]) {
                            continue;
                        }
                        const auto & part = victim_descriptor.parts[role];
                        if (part.l2_part < 0 || static_cast<uint32_t>(part.l2_part) >= source.n_parts) {
                            continue;
                        }
                        parts[static_cast<size_t>(part.l2_part)] = slot_base + part.staging_offset;
                        part_sizes[static_cast<size_t>(part.l2_part)] = part.bytes;
                    }
                    if (std::any_of(parts.begin(), parts.end(), [](const void * ptr) { return ptr == nullptr; })) {
                        error = SILIANG_RUNTIME_FAILURE_L2;
                        error_stage = "victim-descriptor";
                        error_layer = static_cast<int32_t>(victim_layer);
                        error_expert = static_cast<int32_t>(victim_expert);
                        error_slot = transition.released_l2_slot;
                        break;
                    }
                    const int victim_l2_location = cpu_location
                        ? cpu_location(cpu, victim_layer, victim_expert)
                        : GGML_SILIANGEM_EXPERT_LOCATION_NONE;
                    if (victim_l2_location == GGML_SILIANGEM_EXPERT_LOCATION_RESIDENT) {
                        // With K-prefill disabled, a later CPU prompt pass may legitimately
                        // repopulate L2 for an expert that is still warm in K from the previous
                        // request. The demotion is already represented in L2, so rewriting the
                        // victim into the candidate's released slot would create a duplicate.
                        // Reuse the existing L2 copy and let the normal K commit remove the
                        // temporary cross-tier duplication.
                        continue;
                    }
                    if (!cpu_store(
                            cpu, victim_layer, victim_expert, transition.released_l2_slot,
                            parts.data(), part_sizes.data(), source.n_parts)) {
                        error = SILIANG_RUNTIME_FAILURE_L2;
                        error_stage = "victim-store";
                        error_layer = static_cast<int32_t>(victim_layer);
                        error_expert = static_cast<int32_t>(victim_expert);
                        error_slot = transition.released_l2_slot;
                        break;
                    }
                }
            }
            const auto host_copy_end = std::chrono::steady_clock::now();

            {
                std::lock_guard<std::mutex> lock(transition_worker_mutex);
                transition_worker_error = error;
                transition_worker_error_stage = std::move(error_stage);
                transition_worker_error_layer = error_layer;
                transition_worker_error_expert = error_expert;
                transition_worker_error_slot = error_slot;
                transition_worker_device_wait_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(device_wait_end - device_wait_start).count());
                transition_worker_host_copy_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(host_copy_end - host_copy_start).count());
                transition_worker_done = true;
            }
            transition_worker_cv.notify_all();
        }
    }

    void stop_transition_worker() {
        if (!transition_worker.joinable()) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(transition_worker_mutex);
            transition_worker_stop = true;
        }
        transition_worker_cv.notify_all();
        transition_worker.join();
    }

    ~siliang_moe_runtime() {
        if (copy_stream && cuda_stream_synchronize) {
            (void) cuda_stream_synchronize(copy_stream);
        }
        for (size_t bank = 0; bank < bank_leases.size(); ++bank) {
            return_bank_leases(bank);
        }
        unregister_pinned_l2();
        if (!pending_transitions.empty()) {
            (void) finalize_pending_transitions();
        }
        stop_transition_worker();
        if (bound && ctx) {
            llama_siliang_moe_arena_clear(ctx);
            bound = false;
        }
        destroy_events(layer_ready_events);
        destroy_events(exchange_release_events);
        destroy_events(staging_events);
        destroy_events(k_reuse_events);
        if (transition_event && cuda_event_destroy) {
            (void) cuda_event_destroy(transition_event);
            transition_event = nullptr;
        }
        if (k_ext_clear_event && cuda_event_destroy) {
            (void) cuda_event_destroy(k_ext_clear_event);
            k_ext_clear_event = nullptr;
        }
        if (copy_stream && cuda_stream_destroy) {
            (void) cuda_stream_destroy(copy_stream);
            copy_stream = nullptr;
        }
        finalize_prefill_bitmap_trace();
        print_summary();
    }

    void destroy_events(std::vector<cuda_event_t> & events) {
        if (!cuda_event_destroy) {
            events.clear();
            return;
        }
        for (cuda_event_t event : events) {
            if (event) {
                (void) cuda_event_destroy(event);
            }
        }
        events.clear();
    }

    bool fail(int32_t code, const char * reason) {
        int32_t expected = 0;
        if (failure.compare_exchange_strong(expected, code, std::memory_order_acq_rel)) {
            LLAMA_LOG_ERROR("siliang_moe_runtime: fail-closed code=%d reason=%s\n", code, reason);
        }
        return false;
    }

    static int map_callback(
            void * user_data,
            int32_t layer,
            const int32_t * logical,
            int32_t * physical,
            size_t count) {
        auto * self = static_cast<siliang_moe_runtime *>(user_data);
        return self && self->map(layer, logical, physical, count) ? 1 : 0;
    }

    static int32_t failure_callback(void * user_data) {
        auto * self = static_cast<siliang_moe_runtime *>(user_data);
        return self ? self->failure.load(std::memory_order_acquire) : SILIANG_RUNTIME_FAILURE_INIT;
    }

    static int compute_wait_callback(void * user_data, int32_t layer) {
        auto * self = static_cast<siliang_moe_runtime *>(user_data);
        return self && self->compute_wait(layer) ? 0 : 1;
    }

    static int post_compute_callback(void * user_data, int32_t layer) {
        auto * self = static_cast<siliang_moe_runtime *>(user_data);
        return self && self->post_compute(layer) ? 0 : 1;
    }

    bool initialize() {
        if (!model || !ctx || !params.enabled || params.l1_k == 0 || params.exchange_r == 0 ||
            params.elevator_p == 0 || !llama_siliang_moe_arena_get_model_info(model, &model_info)) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "invalid model or K/R/P configuration");
        }
        if (model_info.layer_count <= 0 || model_info.routed_layer_count <= 0 || model_info.expert_count <= 0 ||
            model_info.top_k <= 0 || model_info.top_k > model_info.expert_count ||
            (params.l1_policy != LLAMA_SILIANG_EXPERT_CACHE_POLICY_LFU &&
             params.l1_policy != LLAMA_SILIANG_EXPERT_CACHE_POLICY_WTINYLFU_W10_SLRU_P80 &&
             params.l1_policy != LLAMA_SILIANG_EXPERT_CACHE_POLICY_CUMULATIVE_LFU_ADMISSION)) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "model geometry or supported L1 policy is invalid (LRU retired)");
        }
        if (params.l1_policy != LLAMA_SILIANG_EXPERT_CACHE_POLICY_CUMULATIVE_LFU_ADMISSION &&
            (!params.admit_k_cold || params.demote_k_hot)) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "SLFU cold-admission/demotion controls used with a non-SLFU policy");
        }
        if ((!params.admit_k_cold || params.demote_k_hot) && params.l2_bytes == 0) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "SLFU cold-admission/demotion controls require L2");
        }
        const uint32_t route_width = static_cast<uint32_t>(model_info.top_k);
        if (params.route_stats) {
            const size_t side = static_cast<size_t>(route_width) + 1;
            route_stats_data.compositions.assign(side * side, 0);
            route_stats_data.execution_compositions.assign(side * side * side, 0);
        }
        const uint32_t prefill_ubatch_cap = llama_n_ubatch(ctx);
        const uint64_t prefill_route_capacity = std::min<uint64_t>(
                static_cast<uint64_t>(prefill_ubatch_cap) * static_cast<uint64_t>(route_width),
                static_cast<uint64_t>(model_info.expert_count));
        if (params.prefill &&
            (model_info.expert_count > static_cast<int32_t>(LLAMA_SILIANG_MOE_PREFILL_MAX_EXPERTS) ||
             prefill_ubatch_cap == 0 || prefill_route_capacity > params.l1_k)) {
            return fail(SILIANG_RUNTIME_FAILURE_PREFILL,
                    "bounded prefill requires at most 512 experts per routed layer and "
                    "min(n_ubatch * top-k, expert-count) <= K");
        }
        const uint64_t double_route_width = 2ULL * route_width;
        if (params.l1_k < route_width || params.exchange_r < double_route_width ||
            params.elevator_p < double_route_width ||
            params.exchange_r % route_width != 0 ||
            params.elevator_p % route_width != 0 ||
            params.l1_k > std::numeric_limits<uint32_t>::max() - params.exchange_r) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "K/R/P geometry is not route aligned");
        }
        k_ext = llama_siliang_decode_tail(ctx, &k_ext_tail_buft);
        if (k_ext != 0 && (!params.prefill || params.demote_k_hot || k_ext_tail_buft == nullptr ||
                           params.l1_k > std::numeric_limits<uint32_t>::max() - k_ext - params.exchange_r)) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT,
                    "decode K requires bounded prefill, no K demotion, and the context's CUDA tail buffer");
        }
        const uint32_t capacity = params.l1_k + k_ext + params.exchange_r;
        if (capacity > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "arena capacity exceeds physical id range");
        }

        cuda = llama_siliang_cuda_backend(ctx);
        cpu = llama_siliang_cpu_backend(ctx);
        if (!cuda || !cpu) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "CUDA and CPU backends are both required");
        }
        if (!load_cuda_procs()) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "CUDA registry bridge is incomplete");
        }
        if (k_ext != 0 && !load_cuda_vmm_procs()) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "decode K requires the CUDA VMM bridge");
        }
        if (cuda_stream_create(cuda, &copy_stream) != GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS || !copy_stream) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "private CUDA copy stream creation failed");
        }

        layers.resize(static_cast<size_t>(model_info.layer_count));
        managed.assign(static_cast<size_t>(model_info.layer_count), 0);
        for (int32_t layer = 0; layer < model_info.layer_count; ++layer) {
            llama_siliang_moe_arena_layer_info info = {};
            if (!llama_siliang_moe_arena_get_layer_info(model, layer, &info)) {
                continue;
            }
            bool host_source = info.expert_count == model_info.expert_count && info.part_count > 0;
            for (size_t index = 0; host_source && index < info.part_count; ++index) {
                const ggml_tensor * source = info.parts[index].source;
                host_source = source && source->data && source->buffer && ggml_backend_buffer_is_host(source->buffer);
            }
            if (!host_source) {
                continue;
            }
            size_t schema_index = schemas.size();
            for (size_t index = 0; index < schemas.size(); ++index) {
                if (same_schema(schemas[index], info)) {
                    schema_index = index;
                    break;
                }
            }
            if (schema_index == schemas.size()) {
                schemas.push_back(info);
            }
            if (!set_layer(info, schema_index)) {
                return fail(SILIANG_RUNTIME_FAILURE_INIT, "invalid routed-expert part descriptor");
            }
            managed[static_cast<size_t>(layer)] = 1;
            managed_layers.push_back(layer);
        }
        if (schemas.empty() || managed_layers.empty()) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "no CPU-backed routed-expert layer is available");
        }
        if (managed_layers.size() != static_cast<size_t>(model_info.routed_layer_count)) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "only a subset of routed-expert layers is CPU-backed");
        }

        if (params.hybrid) {
            for (int32_t layer : managed_layers) {
                const auto & descriptor = layers[static_cast<size_t>(layer)];
                for (int role = LLAMA_SILIANG_MOE_ARENA_GATE_BIAS; role < LLAMA_SILIANG_MOE_ARENA_PART_ROLE_COUNT; ++role) {
                    if (descriptor.present[role]) {
                        return fail(SILIANG_RUNTIME_FAILURE_INIT,
                                "hybrid decode supports routed experts without biases or scales");
                    }
                }
            }
        }
        if (!plan_slot_geometry()) {
            return false;
        }
        if (params.prefill) {
            for (int32_t layer : managed_layers) {
                const auto & descriptor = layers[static_cast<size_t>(layer)];
                if (prefill_route_capacity > descriptor.policy_count - (banked ? 0 : k_ext)) {
                    return fail(SILIANG_RUNTIME_FAILURE_PREFILL,
                            "bounded prefill route union exceeds a layer-local schema-bank K slice; "
                            "reduce ubatch or increase K");
                }
            }
        }
        if (!configure_l2_parts()) {
            return false;
        }
        if (params.hybrid) {
            siliang_moe_hybrid::cost_coefficients cost;
            cost.cpu_l2_us = params.hybrid_cpu_l2_us;
            cost.cpu_miss_us = params.hybrid_cpu_miss_us;
            cost.stage_l2_us = params.hybrid_stage_l2_us;
            cost.stage_pinned_us = params.hybrid_stage_pinned_us;
            cost.stage_miss_us = params.hybrid_stage_miss_us;
            cost.gpu_us = params.hybrid_gpu_us;
            if (!l2_enabled || !cpu_location) {
                return fail(SILIANG_RUNTIME_FAILURE_INIT, "hybrid decode requires L2; CPU experts are served by L2");
            }
            if (!hybrid_costs.build(cost, route_width)) {
                return fail(SILIANG_RUNTIME_FAILURE_INIT, "hybrid decode cost coefficients are invalid");
            }
        }
        if (!allocate_arena() || !allocate_staging() || !create_events()) {
            return false;
        }
        if (params.staging_threads == 0 || params.staging_threads > 16) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "staging thread count must be 1-16");
        }
        copier.start(params.staging_threads);
        bank_leases.assign(staging_events.size(), {});
        bank_deferred_releases.assign(staging_events.size(), {});
        if (params.l2_pinned_bytes != 0 && !register_pinned_l2()) {
            return false;
        }

        slots.resize(static_cast<size_t>(params.l1_k) + k_ext);
        size_t key_count = 0;
        if (!checked_mul(static_cast<size_t>(model_info.layer_count),
                         static_cast<size_t>(model_info.expert_count), key_count)) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "cache key table size overflow");
        }
        resident.assign(key_count, -1);
        frequencies.assign(key_count, 0);
        if (params.route_stats && params.demote_k_hot) {
            layer_decode_round.assign(static_cast<size_t>(model_info.layer_count), 0);
            demotion_start_round.assign(key_count, 0);
            demotion_reuse_pending.assign(key_count, 0);
        }
        prefill_bitmaps.resize(static_cast<size_t>(model_info.layer_count));
        prefill_bitmap_valid.assign(static_cast<size_t>(model_info.layer_count), 0);
        prefill_bitmap_layers.resize(static_cast<size_t>(model_info.layer_count));
        prefill_sweep_bitmaps.resize(static_cast<size_t>(model_info.layer_count));

        std::vector<llama_siliang_moe_arena_part_binding> bindings;
        bindings.reserve(managed_layers.size() * LLAMA_SILIANG_MOE_ARENA_PART_ROLE_COUNT);
        for (int32_t layer : managed_layers) {
            const auto & descriptor = layers[static_cast<size_t>(layer)];
            const uint32_t bank_capacity = schema_arena_capacity[descriptor.schema_index];
            const uint32_t binding_first = banked ? descriptor.physical_first : 0;
            const uint32_t binding_count = descriptor.policy_count;
            const uint32_t exchange_first = schema_r_first[descriptor.schema_index];
            for (int role_index = 0; role_index < LLAMA_SILIANG_MOE_ARENA_PART_ROLE_COUNT; ++role_index) {
                if (!descriptor.present[role_index]) {
                    continue;
                }
                const auto role = static_cast<llama_siliang_moe_arena_part_role>(role_index);
                bindings.push_back({
                    /*.layer               =*/ layer,
                    /*.role                =*/ role,
                    /*.arena               =*/ descriptor.parts[role_index].arena,
                    /*.arena_slot_capacity =*/ bank_capacity,
                    /*.layer_slot_first    =*/ binding_first,
                    /*.layer_slot_count    =*/ binding_count,
                    /*.exchange_slot_first =*/ exchange_first,
                    /*.exchange_slot_count =*/ params.exchange_r,
                });
            }
        }
        if (!llama_siliang_moe_arena_bind(
                ctx, managed_layers.data(), managed_layers.size(), bindings.data(), bindings.size(), capacity,
                map_callback, failure_callback, this)) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "context arena binding was rejected");
        }
        bound = true;
        if (!llama_siliang_moe_arena_set_compute_wait(ctx, compute_wait_callback, this)) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "context compute-wait hook was rejected");
        }
        if (params.demote_k_hot && !llama_siliang_moe_arena_set_post_compute(ctx, post_compute_callback, this)) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "context post-compute hook was rejected");
        }
        if (params.demote_k_hot) {
            transition_worker = std::thread(&siliang_moe_runtime::transition_worker_loop, this);
        }

        LLAMA_LOG_INFO(
                "siliang_moe_runtime: armed %s arena layers=%zu/%d schemas=%zu layout=%s experts=%d top_k=%d "
                "K=%u R=%u P=%u policy=%s source=%s transport=private-stream-staged prefill_cap=%u "
                "prefill_residency=%s arena=%zu MiB pinned=%zu MiB\n",
                params.prefill ? "decode+bounded-prefill" : "decode-only",
                managed_layers.size(), model_info.routed_layer_count, schemas.size(), banked ? "banked-balanced" : "global",
                model_info.expert_count, model_info.top_k,
                params.l1_k, params.exchange_r, params.elevator_p, policy_name(),
                l2_enabled ? "host-l2" : "model-mapping",
                params.prefill ? prefill_ubatch_cap : 1,
                params.prefill ? "K-transient-R-decode-only" : "decode-policy",
                arena_bytes / (1024 * 1024),
                ggml_backend_buffer_get_size(staging_buffer.get()) / (1024 * 1024));
        for (size_t schema_index = 0; schema_index < schemas.size(); ++schema_index) {
            size_t layer_count = 0;
            for (int32_t layer : managed_layers) {
                layer_count += layers[static_cast<size_t>(layer)].schema_index == schema_index;
            }
            LLAMA_LOG_INFO(
                    "siliang_moe_runtime: bank=%zu layers=%zu K_local=%u R_shared=%u physical_capacity=%u\n",
                    schema_index, layer_count, schema_k_capacity[schema_index], params.exchange_r,
                    schema_arena_capacity[schema_index]);
        }
        if (params.hybrid) {
            // one row of the matrix: all bypass candidates L2-resident, no admission
            std::string l2_row;
            std::string miss_row;
            std::string pinned_row;
            for (uint32_t n = 0; n <= route_width; ++n) {
                siliang_moe_hybrid::route_classes l2_route;
                l2_route.l2 = n;
                siliang_moe_hybrid::route_classes miss_route;
                miss_route.miss = n;
                siliang_moe_hybrid::route_classes pinned_route;
                pinned_route.pinned = n;
                l2_row += (n ? "," : "") + std::to_string(hybrid_costs.lookup(l2_route).l2);
                miss_row += (n ? "," : "") + std::to_string(hybrid_costs.lookup(miss_route).miss);
                pinned_row += (n ? "," : "") + std::to_string(hybrid_costs.lookup(pinned_route).pinned);
            }
            LLAMA_LOG_INFO(
                    "siliang_moe_runtime: hybrid decode armed cost_us={cpu_l2=%.1f cpu_miss=%.1f stage_l2=%.1f "
                    "stage_pinned=%.1f stage_miss=%.1f gpu=%.1f} cpu_share_pinned_only=[%s] cpu_share_l2_only=[%s] "
                    "cpu_share_miss_only=[%s]\n",
                    params.hybrid_cpu_l2_us, params.hybrid_cpu_miss_us, params.hybrid_stage_l2_us,
                    params.hybrid_stage_pinned_us, params.hybrid_stage_miss_us, params.hybrid_gpu_us,
                    pinned_row.c_str(), l2_row.c_str(), miss_row.c_str());
        }
        for (int32_t layer : managed_layers) {
            const auto & descriptor = layers[static_cast<size_t>(layer)];
            LLAMA_LOG_DEBUG(
                    "siliang_moe_runtime: layer=%d bank=%zu policy_K=[%u,%u) physical_K=[%u,%u) physical_R=[%u,%u)\n",
                    layer, descriptor.schema_index,
                    descriptor.policy_first, descriptor.policy_first + descriptor.policy_count,
                    descriptor.physical_first, descriptor.physical_first + descriptor.policy_count,
                    schema_r_first[descriptor.schema_index], schema_arena_capacity[descriptor.schema_index]);
        }
        return true;
    }

    bool load_cuda_procs() {
        ggml_backend_dev_t device = ggml_backend_get_device(cuda);
        ggml_backend_reg_t reg = device ? ggml_backend_dev_backend_reg(device) : nullptr;
        return reg &&
            load_proc(reg, "ggml_backend_cuda_siliang_stream_create", cuda_stream_create) &&
            load_proc(reg, "ggml_backend_cuda_siliang_stream_destroy", cuda_stream_destroy) &&
            load_proc(reg, "ggml_backend_cuda_siliang_stream_synchronize", cuda_stream_synchronize) &&
            load_proc(reg, "ggml_backend_cuda_siliang_event_create", cuda_event_create) &&
            load_proc(reg, "ggml_backend_cuda_siliang_event_destroy", cuda_event_destroy) &&
            load_proc(reg, "ggml_backend_cuda_siliang_event_synchronize", cuda_event_synchronize) &&
            load_proc(reg, "ggml_backend_cuda_siliang_event_record", cuda_event_record) &&
            load_proc(reg, "ggml_backend_cuda_siliang_main_stream_event_record", cuda_main_event_record) &&
            load_proc(reg, "ggml_backend_cuda_siliang_stream_wait_event", cuda_stream_wait_event) &&
            load_proc(reg, "ggml_backend_cuda_siliang_main_stream_wait_event", cuda_main_wait_event) &&
            load_proc(reg, "ggml_backend_cuda_siliang_h2d_async", cuda_h2d_async) &&
            load_proc(reg, "ggml_backend_cuda_siliang_d2h_async", cuda_d2h_async) &&
            load_proc(reg, "ggml_backend_cuda_siliang_d2d_async", cuda_d2d_async) &&
            load_proc(reg, "ggml_backend_cuda_siliang_host_register_readonly", cuda_host_register) &&
            load_proc(reg, "ggml_backend_cuda_siliang_event_query", cuda_event_query) &&
            load_proc(reg, "ggml_backend_cuda_siliang_host_unregister", cuda_host_unregister);
    }

    bool load_cuda_vmm_procs() {
        ggml_backend_dev_t device = ggml_backend_get_device(cuda);
        ggml_backend_reg_t reg = device ? ggml_backend_dev_backend_reg(device) : nullptr;
        return reg &&
            load_proc(reg, "ggml_backend_cuda_siliang_alias_buffer", cuda_alias_buffer) &&
            load_proc(reg, "ggml_backend_cuda_siliang_tail_query", cuda_tail_query) &&
            load_proc(reg, "ggml_backend_cuda_siliang_vmm_granularity", cuda_vmm_granularity) &&
            load_proc(reg, "ggml_backend_cuda_siliang_memset_async", cuda_memset_async);
    }

    bool load_cpu_procs() {
        ggml_backend_dev_t device = ggml_backend_get_device(cpu);
        ggml_backend_reg_t reg = device ? ggml_backend_dev_backend_reg(device) : nullptr;
        if (reg) {
            (void) load_proc(reg, "ggml_backend_cpu_siliangem_wait_expert", cpu_wait_expert);
        }
        return reg &&
            load_proc(reg, "ggml_backend_cpu_siliangem_query", cpu_query) &&
            load_proc(reg, "ggml_backend_cpu_siliangem_prepare_experts", cpu_prepare) &&
            load_proc(reg, "ggml_backend_cpu_siliangem_prepare_experts_async", cpu_prepare_async) &&
            load_proc(reg, "ggml_backend_cpu_siliangem_wait_experts", cpu_wait) &&
            load_proc(reg, "ggml_backend_cpu_siliangem_copy_cached_part", cpu_copy_part) &&
            load_proc(reg, "ggml_backend_cpu_siliangem_expert_location", cpu_location) &&
            load_proc(reg, "ggml_backend_cpu_siliangem_release_cached_expert", cpu_release) &&
            load_proc(reg, "ggml_backend_cpu_siliangem_store_cached_expert_at_slot", cpu_store) &&
            load_proc(reg, "ggml_backend_cpu_siliangem_arena_span", cpu_arena_span) &&
            load_proc(reg, "ggml_backend_cpu_siliangem_expert_slot", cpu_expert_slot) &&
            load_proc(reg, "ggml_backend_cpu_siliangem_lease_cached_part", cpu_lease_part) &&
            load_proc(reg, "ggml_backend_cpu_siliangem_unlease_slot", cpu_unlease);
    }

    static bool same_schema(
            const llama_siliang_moe_arena_layer_info & lhs,
            const llama_siliang_moe_arena_layer_info & rhs) {
        if (lhs.expert_count != rhs.expert_count || lhs.part_count != rhs.part_count) {
            return false;
        }
        for (size_t index = 0; index < lhs.part_count; ++index) {
            const auto & a = lhs.parts[index];
            const auto & b = rhs.parts[index];
            if (a.role != b.role || a.type != b.type || a.expert_axis != b.expert_axis ||
                a.bytes_per_expert != b.bytes_per_expert ||
                a.source_expert_stride != b.source_expert_stride) {
                return false;
            }
            for (int dim = 0; dim < 4; ++dim) {
                if (dim != a.expert_axis && a.ne[dim] != b.ne[dim]) {
                    return false;
                }
            }
        }
        return true;
    }

    bool set_layer(const llama_siliang_moe_arena_layer_info & info, size_t schema_index) {
        auto & destination = layers[static_cast<size_t>(info.layer)];
        destination.schema_index = schema_index;
        for (size_t index = 0; index < info.part_count; ++index) {
            const auto & source = info.parts[index];
            const int role = static_cast<int>(source.role);
            if (role < 0 || role >= LLAMA_SILIANG_MOE_ARENA_PART_ROLE_COUNT || destination.present[role]) {
                return false;
            }
            destination.present[role] = 1;
            destination.parts[role].role = source.role;
            destination.parts[role].source = source.source;
            destination.parts[role].bytes = source.bytes_per_expert;
            destination.parts[role].source_stride = source.source_expert_stride;
            ++destination.part_count;
        }
        return destination.part_count == info.part_count;
    }

    bool plan_slot_geometry() {
        if (schemas.empty() || managed_layers.empty()) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "slot geometry has no schema or routed layer");
        }
        banked = schemas.size() > 1;
        schema_k_capacity.assign(schemas.size(), 0);
        schema_r_first.assign(schemas.size(), 0);
        schema_arena_capacity.assign(schemas.size(), 0);

        if (!banked) {
            const uint32_t k_total = params.l1_k + k_ext;
            schema_k_capacity[0] = k_total;
            schema_r_first[0] = k_total;
            schema_arena_capacity[0] = k_total + params.exchange_r;
            for (int32_t layer : managed_layers) {
                auto & descriptor = layers[static_cast<size_t>(layer)];
                descriptor.policy_first = 0;
                descriptor.policy_count = k_total;
                descriptor.physical_first = 0;
            }
            return true;
        }
        if (k_ext != 0) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "decode K requires one expert schema (a global K)");
        }

        size_t minimum = 0;
        if (!checked_mul(managed_layers.size(), static_cast<size_t>(model_info.top_k), minimum) ||
            params.l1_k < minimum) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT,
                    "heterogeneous schema banks require at least one complete K route per routed layer");
        }

        const uint32_t quotient = params.l1_k / static_cast<uint32_t>(managed_layers.size());
        const uint32_t remainder = params.l1_k % static_cast<uint32_t>(managed_layers.size());
        uint32_t policy_cursor = 0;
        for (size_t ordinal = 0; ordinal < managed_layers.size(); ++ordinal) {
            auto & descriptor = layers[static_cast<size_t>(managed_layers[ordinal])];
            const uint32_t count = quotient + static_cast<uint32_t>(ordinal < remainder);
            if (count < static_cast<uint32_t>(model_info.top_k) ||
                policy_cursor > params.l1_k - count ||
                schema_k_capacity[descriptor.schema_index] > std::numeric_limits<uint32_t>::max() - count) {
                return fail(SILIANG_RUNTIME_FAILURE_INIT, "banked K partition overflow");
            }
            descriptor.policy_first = policy_cursor;
            descriptor.policy_count = count;
            policy_cursor += count;
            schema_k_capacity[descriptor.schema_index] += count;
        }
        if (policy_cursor != params.l1_k) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "banked K policy partition is incomplete");
        }

        std::vector<uint32_t> physical_cursor(schemas.size(), 0);
        for (int32_t layer : managed_layers) {
            auto & descriptor = layers[static_cast<size_t>(layer)];
            descriptor.physical_first = physical_cursor[descriptor.schema_index];
            const uint32_t bank_capacity = schema_k_capacity[descriptor.schema_index];
            if (descriptor.policy_count > bank_capacity ||
                physical_cursor[descriptor.schema_index] > bank_capacity - descriptor.policy_count) {
                return fail(SILIANG_RUNTIME_FAILURE_INIT, "banked K physical partition overflow");
            }
            physical_cursor[descriptor.schema_index] += descriptor.policy_count;
        }
        for (size_t schema_index = 0; schema_index < schemas.size(); ++schema_index) {
            if (physical_cursor[schema_index] != schema_k_capacity[schema_index] ||
                schema_k_capacity[schema_index] > std::numeric_limits<uint32_t>::max() - params.exchange_r) {
                return fail(SILIANG_RUNTIME_FAILURE_INIT, "schema-bank physical capacity overflow");
            }
            schema_r_first[schema_index] = schema_k_capacity[schema_index];
            schema_arena_capacity[schema_index] = schema_k_capacity[schema_index] + params.exchange_r;
        }
        return true;
    }

    bool configure_l2_parts() {
        l2_enabled = params.l2_bytes != 0;
        if (!l2_enabled) {
            return true;
        }
        if (!load_cpu_procs()) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "CPU L2 registry bridge is incomplete");
        }
        const auto & source = model->siliang_expert_source;
        if (!source.valid() || source.kind == llama_siliang_expert_source_kind::none ||
            source.n_experts != static_cast<uint32_t>(model_info.expert_count) ||
            source.n_layers != static_cast<uint32_t>(model_info.layer_count)) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "model-owned L2 geometry does not cover the routed model exactly");
        }
        const auto names = split_names(source.part_names);
        if (names.size() != source.n_parts) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "L2 part_names count is invalid");
        }
        std::array<int32_t, LLAMA_SILIANG_MOE_ARENA_PART_ROLE_COUNT> part_by_role;
        part_by_role.fill(-1);
        for (size_t part = 0; part < names.size(); ++part) {
            const int32_t role = source_name_role(names[part]);
            if (role < 0 || part_by_role[static_cast<size_t>(role)] >= 0) {
                return fail(SILIANG_RUNTIME_FAILURE_INIT, "L2 part_names cannot be mapped exactly to routed weights");
            }
            part_by_role[static_cast<size_t>(role)] = static_cast<int32_t>(part);
        }
        for (int32_t layer : managed_layers) {
            auto & descriptor = layers[static_cast<size_t>(layer)];
            for (int role = 0; role < LLAMA_SILIANG_MOE_ARENA_PART_ROLE_COUNT; ++role) {
                if (!descriptor.present[role]) {
                    if (is_weight_role(static_cast<llama_siliang_moe_arena_part_role>(role)) && part_by_role[role] >= 0) {
                        return fail(SILIANG_RUNTIME_FAILURE_INIT, "L2 exposes a weight part absent from the model schema");
                    }
                    continue;
                }
                if (!is_weight_role(static_cast<llama_siliang_moe_arena_part_role>(role))) {
                    continue;
                }
                const int32_t source_part = part_by_role[role];
                if (source_part < 0) {
                    return fail(SILIANG_RUNTIME_FAILURE_INIT, "model weight part is absent from L2 part_names");
                }
                const size_t source_index = static_cast<size_t>(layer) * source.n_parts + static_cast<size_t>(source_part);
                const uint32_t source_bytes = source.kind == llama_siliang_expert_source_kind::expert_major ?
                    source.part_bytes[source_index] : source.stride[source_index];
                if (source_bytes != descriptor.parts[role].bytes) {
                    return fail(SILIANG_RUNTIME_FAILURE_INIT, "L2 part size disagrees with model tensor geometry");
                }
                descriptor.parts[role].l2_part = source_part;
            }
        }

        ggml_siliangem_cache_info info = {};
        info.struct_size = sizeof(info);
        if (!cpu_query(cpu, &info) || !info.configured || !info.ready ||
            info.n_layers != static_cast<uint32_t>(model_info.layer_count) ||
            info.n_experts != static_cast<uint32_t>(model_info.expert_count) ||
            info.capacity_slots < static_cast<uint32_t>(model_info.top_k)) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "CPU L2 is not ready for one complete route");
        }
        const uint64_t prefill_route_capacity = std::min<uint64_t>(
                static_cast<uint64_t>(llama_n_ubatch(ctx)) *
                    static_cast<uint64_t>(model_info.top_k),
                static_cast<uint64_t>(model_info.expert_count));
        if (params.prefill && prefill_route_capacity > info.capacity_slots) {
            return fail(SILIANG_RUNTIME_FAILURE_PREFILL,
                    "CPU L2 cannot hold the bounded prefill route union");
        }
        return true;
    }

    // Decode K: each part's arena is one VMM range [tail block of the compute buffer][new base block].
    // The tensor starts k_ext slots before the base block, so slots [0, k_ext) are the tail block's last
    // bytes and the base K + R slots follow at the same stride.
    bool allocate_extended_arena() {
        void * tail_base = nullptr;
        size_t tail_size = 0;
        size_t tail_head = 0;
        uint64_t generation = 0;
        const size_t granularity = cuda_vmm_granularity(cuda);
        if (variant_arenas.size() != 1 || granularity == 0 ||
            cuda_tail_query(k_ext_tail_buft, &tail_base, &tail_size, &tail_head, &generation) !=
                GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "decode K: the CUDA compute buffer tail is not allocated");
        }
        const uint32_t capacity = schema_arena_capacity[0];
        ggml_backend_buffer_type_t cuda_buft = ggml_backend_get_default_buffer_type(cuda);
        size_t tail_offset = tail_head;
        arena_bytes = 0;
        for (int role = 0; role < LLAMA_SILIANG_MOE_ARENA_PART_ROLE_COUNT; ++role) {
            ggml_tensor * arena = variant_arenas[0][static_cast<size_t>(role)];
            if (!arena) {
                continue;
            }
            const size_t stride = ggml_nbytes(arena) / capacity;
            const size_t ext_bytes = stride * k_ext;
            const size_t alloc_bytes = ggml_backend_buft_get_alloc_size(cuda_buft, arena);
            if (stride * capacity != ggml_nbytes(arena) || ext_bytes % 128 != 0 || alloc_bytes <= ext_bytes) {
                return fail(SILIANG_RUNTIME_FAILURE_INIT, "decode K: arena part geometry is not slot aligned");
            }
            const size_t tail_block = (ext_bytes + granularity - 1) / granularity * granularity;
            const size_t base_block = (alloc_bytes - ext_bytes + granularity - 1) / granularity * granularity;
            if (tail_offset > tail_size || tail_block > tail_size - tail_offset) {
                return fail(SILIANG_RUNTIME_FAILURE_INIT, "decode K: arena parts exceed the compute buffer tail");
            }
            const uint64_t sources[2] = {
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(static_cast<char *>(tail_base) + tail_offset)), 0 };
            const size_t sizes[2] = { tail_block, base_block };
            ggml_backend_buffer_t buffer = cuda_alias_buffer(cuda, sources, sizes, 2);
            if (!buffer) {
                return fail(SILIANG_RUNTIME_FAILURE_INIT, "decode K: CUDA VMM arena mapping failed");
            }
            arena_alias_buffers.emplace_back(buffer);
            ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_buffer_clear(buffer, 0);
            void * address = static_cast<char *>(ggml_backend_buffer_get_base(buffer)) + tail_block - ext_bytes;
            if (ggml_backend_tensor_alloc(buffer, arena, address) != GGML_STATUS_SUCCESS) {
                return fail(SILIANG_RUNTIME_FAILURE_INIT, "decode K: arena tensor placement failed");
            }
            tail_offset += tail_block;
            arena_bytes += base_block;
        }
        if (tail_offset != tail_size) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "decode K: arena parts do not match the compute buffer tail");
        }
        k_ext_tail_base = tail_base;
        k_ext_tail_head = tail_head;
        k_ext_tail_generation = generation;
        k_ext_active = true;
        LLAMA_LOG_INFO("siliang_moe_runtime: decode K armed K_prefill=%u K_decode=%u extra=%u compute_buffer=%.1f MiB "
                       "tail_start=%.1f MiB tail=%.1f MiB base_arena=%.1f MiB\n",
                params.l1_k, params.l1_k + k_ext, k_ext, tail_size / 1048576.0, tail_head / 1048576.0,
                (tail_size - tail_head) / 1048576.0, arena_bytes / 1048576.0);
        return true;
    }

    // Entering decode after prefill: the extra slots hold prefill activations. Only slots admitted in
    // decode are routed, but MMQ may read row padding past a slot into the next one, so zero them all.
    void clear_decode_extension() {
        const uint32_t capacity = schema_arena_capacity[0];
        for (ggml_tensor * arena : variant_arenas[0]) {
            if (!arena) {
                continue;
            }
            const size_t stride = ggml_nbytes(arena) / capacity;
            if (cuda_memset_async(copy_stream, arena, 0, stride * k_ext, 0) != GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS) {
                (void) fail(SILIANG_RUNTIME_FAILURE_EVENT, "decode K extra slot clear failed");
                return;
            }
        }
        if (cuda_event_record(copy_stream, k_ext_clear_event) != GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS ||
            cuda_main_wait_event(cuda, k_ext_clear_event) != GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS) {
            (void) fail(SILIANG_RUNTIME_FAILURE_EVENT, "decode K extra slot clear ordering failed");
            return;
        }
        ++metrics.k_ext_clears;
    }

    // Before the first layer of every decode graph uses an extra slot: that graph's plan must end
    // before the tail, and the compute buffer must still be the one the arena aliases.
    void check_decode_extension() {
        const size_t plan = llama_siliang_sched_plan_bytes(ctx);
        void * base = nullptr;
        size_t size = 0;
        size_t head = 0;
        uint64_t generation = 0;
        const bool live = cuda_tail_query(k_ext_tail_buft, &base, &size, &head, &generation) ==
            GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS;
        if (plan > metrics.k_ext_plan_max) {
            // a new largest decode plan: log it so the tail can be sized against real decode graphs
            LLAMA_LOG_INFO("siliang_moe_runtime: decode K plan %.1f MiB (tail start %.1f MiB)\n",
                    plan / 1048576.0, k_ext_tail_head / 1048576.0);
        }
        metrics.k_ext_plan_max = std::max<uint64_t>(metrics.k_ext_plan_max, plan);
        const char * reason = nullptr;
        if (live && generation != k_ext_tail_generation) {
            // a reallocated compute buffer maps a new head in front of the same tail memory
            LLAMA_LOG_INFO("siliang_moe_runtime: decode K compute buffer reallocated: %.1f MiB, tail start %.1f MiB\n",
                    size / 1048576.0, head / 1048576.0);
            k_ext_tail_generation = generation;
            k_ext_tail_base = base;
            k_ext_tail_head = head;
        }
        if (!live) {
            reason = "the CUDA compute buffer is not allocated";
        } else if (plan == 0 || plan > k_ext_tail_head) {
            reason = "a decode graph plan reaches the compute buffer tail";
        }
        if (reason == nullptr) {
            return;
        }
        k_ext_active = false;
        for (uint32_t slot = 0; slot < k_ext && slot < slots.size(); ++slot) {
            const int64_t key = slots[slot].key;
            if (key >= 0 && static_cast<size_t>(key) < resident.size()) {
                resident[static_cast<size_t>(key)] = -1;
            }
            slots[slot] = {};
        }
        LLAMA_LOG_WARN("siliang_moe_runtime: decode K extra slots disabled: %s (plan %.1f MiB, tail start %.1f MiB); "
                       "decode continues with K=%u\n", reason, plan / 1048576.0, k_ext_tail_head / 1048576.0, params.l1_k);
    }

    bool allocate_arena() {
        size_t tensor_count = 1;
        for (const auto & schema : schemas) {
            if (!checked_add(tensor_count, schema.part_count, tensor_count)) {
                return fail(SILIANG_RUNTIME_FAILURE_INIT, "arena metadata tensor count overflow");
            }
        }
        size_t metadata_bytes = 0;
        if (!checked_mul(ggml_tensor_overhead(), tensor_count, metadata_bytes)) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "arena metadata size overflow");
        }
        ggml_init_params init = {
            /*.mem_size   =*/ metadata_bytes,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        arena_ctx.reset(ggml_init(init));
        if (!arena_ctx) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "arena tensor context allocation failed");
        }
        variant_arenas.resize(schemas.size());
        for (size_t schema_index = 0; schema_index < schemas.size(); ++schema_index) {
            const auto & schema = schemas[schema_index];
            const uint32_t capacity = schema_arena_capacity[schema_index];
            if (capacity == 0) {
                return fail(SILIANG_RUNTIME_FAILURE_INIT, "schema-bank arena capacity is zero");
            }
            for (size_t index = 0; index < schema.part_count; ++index) {
                const auto & part = schema.parts[index];
                int64_t ne[4] = { part.ne[0], part.ne[1], part.ne[2], part.ne[3] };
                ne[part.expert_axis] = capacity;
                for (int dim = part.expert_axis + 1; dim < 4; ++dim) {
                    ne[dim] = 1;
                }
                ggml_tensor * arena = ggml_new_tensor_4d(arena_ctx.get(), part.type, ne[0], ne[1], ne[2], ne[3]);
                if (!arena) {
                    return fail(SILIANG_RUNTIME_FAILURE_INIT, "arena tensor creation failed");
                }
                std::string name = std::string("siliang_moe_arena_v") + std::to_string(schema_index) + "_" + role_name(part.role);
                ggml_set_name(arena, name.c_str());
                variant_arenas[schema_index][static_cast<size_t>(part.role)] = arena;
            }
        }
        for (int32_t layer : managed_layers) {
            auto & descriptor = layers[static_cast<size_t>(layer)];
            if (descriptor.schema_index >= variant_arenas.size()) {
                return fail(SILIANG_RUNTIME_FAILURE_INIT, "layer schema index is invalid");
            }
            for (int role = 0; role < LLAMA_SILIANG_MOE_ARENA_PART_ROLE_COUNT; ++role) {
                if (descriptor.present[role]) {
                    descriptor.parts[role].arena = variant_arenas[descriptor.schema_index][role];
                }
            }
        }
        if (k_ext != 0) {
            return allocate_extended_arena();
        }
        arena_buffer.reset(ggml_backend_alloc_ctx_tensors(arena_ctx.get(), cuda));
        if (!arena_buffer) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "global K+R CUDA arena allocation failed");
        }
        ggml_backend_buffer_set_usage(arena_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        // MMQ kernels may read their documented padding beyond the selected
        // expert payload. Admissions populate only the payload bytes, so make
        // every untouched slot and padding byte deterministic before first use.
        ggml_backend_buffer_clear(arena_buffer.get(), 0);
        arena_bytes = ggml_backend_buffer_get_size(arena_buffer.get());
        return true;
    }

    bool allocate_staging() {
        ggml_backend_dev_t device = ggml_backend_get_device(cuda);
        ggml_backend_buffer_type_t host_buft = device ? ggml_backend_dev_host_buffer_type(device) : nullptr;
        if (!host_buft || !ggml_backend_buft_is_host(host_buft)) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "CUDA pinned host buffer type is unavailable");
        }
        const size_t alignment = std::max<size_t>(64, ggml_backend_buft_get_alignment(host_buft));
        staging_slot_bytes = 0;
        for (size_t schema_index = 0; schema_index < schemas.size(); ++schema_index) {
            const auto & schema = schemas[schema_index];
            size_t cursor = 0;
            for (size_t index = 0; index < schema.part_count; ++index) {
                const auto role = schema.parts[index].role;
                size_t aligned = 0;
                if (!checked_align(cursor, alignment, aligned)) {
                    return fail(SILIANG_RUNTIME_FAILURE_INIT, "pinned staging layout overflow");
                }
                cursor = aligned;
                for (int32_t layer : managed_layers) {
                    auto & descriptor = layers[static_cast<size_t>(layer)];
                    if (descriptor.schema_index == schema_index) {
                        descriptor.parts[static_cast<size_t>(role)].staging_offset = cursor;
                    }
                }
                if (!checked_add(cursor, schema.parts[index].bytes_per_expert, cursor)) {
                    return fail(SILIANG_RUNTIME_FAILURE_INIT, "pinned staging layout overflow");
                }
            }
            size_t schema_slot_bytes = 0;
            if (!checked_align(cursor, alignment, schema_slot_bytes)) {
                return fail(SILIANG_RUNTIME_FAILURE_INIT, "pinned staging layout overflow");
            }
            staging_slot_bytes = std::max(staging_slot_bytes, schema_slot_bytes);
        }
        if (staging_slot_bytes == 0) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "invalid pinned staging slot size");
        }
        size_t staging_bytes = 0;
        if (!checked_mul(staging_slot_bytes, params.elevator_p, staging_bytes)) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "pinned P allocation size overflow");
        }
        staging_buffer.reset(ggml_backend_buft_alloc_buffer(host_buft, staging_bytes));
        if (!staging_buffer) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "pinned P allocation failed");
        }
        staging_base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(staging_buffer.get()));
        if (!staging_base) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "pinned P allocation has no host address");
        }
        if (params.demote_k_hot) {
            size_t demotion_bytes = 0;
            if (!checked_mul(staging_slot_bytes, static_cast<size_t>(model_info.top_k), demotion_bytes)) {
                return fail(SILIANG_RUNTIME_FAILURE_INIT, "demotion staging size overflow");
            }
            demotion_buffer.reset(ggml_backend_buft_alloc_buffer(host_buft, demotion_bytes));
            if (!demotion_buffer) {
                return fail(SILIANG_RUNTIME_FAILURE_INIT, "demotion staging allocation failed");
            }
            demotion_base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(demotion_buffer.get()));
            if (!demotion_base) {
                return fail(SILIANG_RUNTIME_FAILURE_INIT, "demotion staging has no host address");
            }
        }
        return true;
    }

    bool create_events() {
        if (k_ext != 0 && (cuda_event_create(cuda, &k_ext_clear_event) != GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS ||
                           !k_ext_clear_event)) {
            return fail(SILIANG_RUNTIME_FAILURE_EVENT, "decode K clear event creation failed");
        }
        layer_ready_events.assign(static_cast<size_t>(model_info.layer_count), nullptr);
        layer_event_pending.assign(static_cast<size_t>(model_info.layer_count), 0);
        for (int32_t layer : managed_layers) {
            if (cuda_event_create(cuda, &layer_ready_events[static_cast<size_t>(layer)]) !=
                    GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS || !layer_ready_events[static_cast<size_t>(layer)]) {
                return fail(SILIANG_RUNTIME_FAILURE_EVENT, "layer-ready CUDA event creation failed");
            }
        }
        const size_t exchange_banks = params.exchange_r / static_cast<uint32_t>(model_info.top_k);
        exchange_release_events.assign(exchange_banks, nullptr);
        exchange_bank_used.assign(exchange_banks, 0);
        for (cuda_event_t & event : exchange_release_events) {
            if (cuda_event_create(cuda, &event) != GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS || !event) {
                return fail(SILIANG_RUNTIME_FAILURE_EVENT, "R release CUDA event creation failed");
            }
        }
        const size_t staging_banks = params.elevator_p / static_cast<uint32_t>(model_info.top_k);
        staging_events.assign(staging_banks, nullptr);
        staging_event_recorded.assign(staging_banks, 0);
        for (cuda_event_t & event : staging_events) {
            if (cuda_event_create(cuda, &event) != GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS || !event) {
                return fail(SILIANG_RUNTIME_FAILURE_EVENT, "P completion CUDA event creation failed");
            }
        }
        k_reuse_events.assign(staging_banks, nullptr);
        for (cuda_event_t & event : k_reuse_events) {
            if (cuda_event_create(cuda, &event) != GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS || !event) {
                return fail(SILIANG_RUNTIME_FAILURE_EVENT, "K reuse CUDA event creation failed");
            }
        }
        if (params.demote_k_hot &&
            (cuda_event_create(cuda, &transition_event) != GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS || !transition_event)) {
            return fail(SILIANG_RUNTIME_FAILURE_EVENT, "K/L2 transition event creation failed");
        }
        return !exchange_release_events.empty() && !staging_events.empty() &&
            k_reuse_events.size() == staging_events.size();
    }

    int64_t key_for(int32_t layer, int32_t expert) const {
        return static_cast<int64_t>(layer) * model_info.expert_count + expert;
    }

    static size_t demotion_reuse_round_bucket(uint64_t rounds) {
        if (rounds <= 1) return 0;
        if (rounds <= 4) return 1;
        if (rounds <= 8) return 2;
        if (rounds <= 16) return 3;
        return 4;
    }

    void record_demotion_reuse(
            int32_t layer,
            const std::vector<int64_t> & route_keys,
            const l2_prepare_counts & l2) {
        if (!params.route_stats || !params.demote_k_hot || !l2.exact ||
            layer < 0 || static_cast<size_t>(layer) >= layer_decode_round.size()) {
            return;
        }
        const uint64_t current_round = layer_decode_round[static_cast<size_t>(layer)];
        for (int64_t key : route_keys) {
            if (key < 0 || static_cast<size_t>(key) >= demotion_reuse_pending.size() ||
                !demotion_reuse_pending[static_cast<size_t>(key)]) {
                continue;
            }
            const int32_t expert = static_cast<int32_t>(key % model_info.expert_count);
            const bool l2_hit = std::find(l2.hit_experts.begin(), l2.hit_experts.end(), expert) != l2.hit_experts.end();
            const bool cold = std::find(l2.miss_experts.begin(), l2.miss_experts.end(), expert) != l2.miss_experts.end();
            const uint64_t start_round = demotion_start_round[static_cast<size_t>(key)];
            const uint64_t distance = current_round > start_round ? current_round - start_round : 1;
            const size_t bucket = demotion_reuse_round_bucket(distance);
            if (l2_hit) {
                ++metrics.demotion_reuse_l2;
                ++metrics.demotion_reuse_l2_round_buckets[bucket];
            } else if (cold) {
                ++metrics.demotion_reuse_cold;
                ++metrics.demotion_reuse_cold_round_buckets[bucket];
            } else {
                ++metrics.demotion_reuse_unknown;
            }
            demotion_reuse_pending[static_cast<size_t>(key)] = 0;
            if (metrics.demotion_reuse_pending > 0) {
                --metrics.demotion_reuse_pending;
            }
        }
    }

    bool policy_bounds(int32_t layer, uint32_t & first, uint32_t & last, bool prefill_range = false) const {
        if (layer < 0 || static_cast<size_t>(layer) >= layers.size()) {
            return false;
        }
        const auto & descriptor = layers[static_cast<size_t>(layer)];
        const uint32_t k_total = params.l1_k + k_ext;
        first = descriptor.policy_first;
        if (descriptor.policy_count == 0 || first > k_total ||
            descriptor.policy_count > k_total - first) {
            return false;
        }
        last = first + descriptor.policy_count;
        // decode K extra slots hold prefill activations while bounded prefill runs
        if (k_ext != 0 && (prefill_range || !k_ext_active)) {
            first = std::max(first, k_ext);
        }
        return first < last && last <= slots.size();
    }

    int32_t physical_slot_for_policy(int32_t layer, int32_t policy_slot) const {
        uint32_t first = 0;
        uint32_t last = 0;
        if (policy_slot < 0 || !policy_bounds(layer, first, last) ||
            static_cast<uint32_t>(policy_slot) < first || static_cast<uint32_t>(policy_slot) >= last) {
            return -1;
        }
        if (!banked) {
            return policy_slot;
        }
        const auto & descriptor = layers[static_cast<size_t>(layer)];
        const uint64_t physical = static_cast<uint64_t>(descriptor.physical_first) +
            static_cast<uint32_t>(policy_slot) - first;
        return physical < schema_r_first[descriptor.schema_index] &&
               physical <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ?
            static_cast<int32_t>(physical) : -1;
    }

    int32_t physical_slot_for_exchange(int32_t layer, size_t bank, size_t lane) const {
        if (layer < 0 || static_cast<size_t>(layer) >= layers.size() ||
            bank >= exchange_release_events.size() || lane >= static_cast<size_t>(model_info.top_k)) {
            return -1;
        }
        const auto schema_index = layers[static_cast<size_t>(layer)].schema_index;
        const uint64_t physical = static_cast<uint64_t>(schema_r_first[schema_index]) +
            bank * static_cast<size_t>(model_info.top_k) + lane;
        return physical < schema_arena_capacity[schema_index] &&
               physical <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ?
            static_cast<int32_t>(physical) : -1;
    }

    static bool contains_key(const std::vector<int64_t> & keys, int64_t key) {
        return std::find(keys.begin(), keys.end(), key) != keys.end();
    }

    int32_t choose_slot(int32_t layer, const std::vector<int64_t> & protected_keys) {
        uint32_t first = 0;
        uint32_t last = 0;
        if (!policy_bounds(layer, first, last)) {
            return -1;
        }
        int32_t vacant = -1;
        for (uint32_t index = first; index < last; ++index) {
            if (slots[index].key < 0) {
                vacant = static_cast<int32_t>(index);
                break;
            }
        }
        if (params.l1_policy == LLAMA_SILIANG_EXPERT_CACHE_POLICY_WTINYLFU_W10_SLRU_P80) {
            const auto decision = siliang_moe_policy::wtinylfu_choose_slot(
                    slots, frequencies, first, last, static_cast<uint32_t>(model_info.top_k), protected_keys, clock);
            metrics.k_rejections += decision.main_candidate_rejected;
            return decision.slot;
        }
        if (params.l1_policy == LLAMA_SILIANG_EXPERT_CACHE_POLICY_LFU) {
            return siliang_moe_policy::lfu_choose_slot(slots, first, last, protected_keys);
        }
        if (params.l1_policy != LLAMA_SILIANG_EXPERT_CACHE_POLICY_LRU) {
            return -1;
        }
        if (vacant >= 0) {
            return vacant;
        }
        int32_t victim = -1;
        for (uint32_t index = first; index < last; ++index) {
            const auto & value = slots[index];
            if (contains_key(protected_keys, value.key)) {
                continue;
            }
            if (victim < 0) {
                victim = static_cast<int32_t>(index);
                continue;
            }
            const auto & selected = slots[static_cast<size_t>(victim)];
            if (value.last_used < selected.last_used) {
                victim = static_cast<int32_t>(index);
            }
        }
        return victim;
    }

    int32_t choose_hot_or_bypass(
            int32_t layer,
            int64_t candidate_key,
            const std::vector<int64_t> & protected_keys,
            bool & admit) {
        admit = true;
        uint32_t first = 0;
        uint32_t last = 0;
        if (!policy_bounds(layer, first, last)) {
            admit = false;
            return -1;
        }
        if (params.l1_policy == LLAMA_SILIANG_EXPERT_CACHE_POLICY_WTINYLFU_W10_SLRU_P80) {
            const int32_t result = choose_slot(layer, protected_keys);
            if (result < 0) {
                admit = false;
                ++metrics.k_rejections;
            }
            return result;
        }
        for (uint32_t index = first; index < last; ++index) {
            if (slots[index].key < 0) {
                return static_cast<int32_t>(index);
            }
        }
        if (params.l1_policy == LLAMA_SILIANG_EXPERT_CACHE_POLICY_LRU ||
            params.l1_policy == LLAMA_SILIANG_EXPERT_CACHE_POLICY_LFU) {
            return choose_slot(layer, protected_keys);
        }
        if (params.l1_policy == LLAMA_SILIANG_EXPERT_CACHE_POLICY_CUMULATIVE_LFU_ADMISSION) {
            const auto decision = siliang_moe_policy::cumulative_lfu_choose_slot(
                    slots, frequencies, first, last, candidate_key, protected_keys);
            admit = !decision.main_candidate_rejected;
            if (!admit) {
                ++metrics.k_rejections;
            }
            return decision.slot;
        }
        return -1;
    }

    void record_hit(int32_t layer, int32_t slot, const std::vector<int64_t> & protected_keys) {
        if (params.l1_policy == LLAMA_SILIANG_EXPERT_CACHE_POLICY_LFU) {
            siliang_moe_policy::lfu_record_hit(slots[static_cast<size_t>(slot)]);
        }
        if (params.l1_policy != LLAMA_SILIANG_EXPERT_CACHE_POLICY_WTINYLFU_W10_SLRU_P80) {
            return;
        }
        uint32_t first = 0;
        uint32_t last = 0;
        if (!policy_bounds(layer, first, last)) {
            return;
        }
        siliang_moe_policy::wtinylfu_record_hit(
                slots, first, last, static_cast<uint32_t>(model_info.top_k), slot, protected_keys, clock);
    }

    // Registers the leading L2 slots in chunks (one large registration can be
    // refused where several smaller ones are not) and then touches every chunk
    // once, because the driver pins lazily on the first copy.
    bool register_pinned_l2() {
        void * base = nullptr;
        size_t slot_bytes = 0;
        uint32_t slot_count = 0;
        if (!l2_enabled || !cpu_arena_span(cpu, &base, &slot_bytes, &slot_count) || !base || slot_bytes == 0) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "pinned L2 requires a ready L2 arena");
        }
        const uint64_t wanted_slots = std::min<uint64_t>(params.l2_pinned_bytes / slot_bytes, slot_count);
        if (wanted_slots == 0) {
            return fail(SILIANG_RUNTIME_FAILURE_INIT, "pinned L2 size is smaller than one L2 slot");
        }
        const size_t chunk_slots = std::max<size_t>(1, (size_t {1} << 31) / slot_bytes);
        const auto register_start = std::chrono::steady_clock::now();
        for (uint64_t first = 0; first < wanted_slots; first += chunk_slots) {
            const size_t slots_here = static_cast<size_t>(std::min<uint64_t>(chunk_slots, wanted_slots - first));
            auto * chunk = static_cast<uint8_t *>(base) + static_cast<size_t>(first) * slot_bytes;
            const size_t bytes = slots_here * slot_bytes;
            if (cuda_host_register(cuda, chunk, bytes) != GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("siliang_moe_runtime: CUDA refused to register L2 bytes [%zu, %zu) of %zu requested; "
                                "lower --expert-cache-l2-pinned-mib\n",
                        static_cast<size_t>(first) * slot_bytes, static_cast<size_t>(first) * slot_bytes + bytes,
                        static_cast<size_t>(wanted_slots) * slot_bytes);
                unregister_pinned_l2();
                return fail(SILIANG_RUNTIME_FAILURE_INIT, "pinned L2 registration was refused");
            }
            pinned_chunks.emplace_back(chunk, bytes);
        }
        metrics.pin_register_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - register_start).count());

        // warm-up: one small copy per chunk into an exchange slot, which is always rewritten before use
        const auto & descriptor = layers[static_cast<size_t>(managed_layers.front())];
        const runtime_part * probe = nullptr;
        for (int role = 0; role < LLAMA_SILIANG_MOE_ARENA_PART_ROLE_COUNT && !probe; ++role) {
            if (descriptor.present[role]) {
                probe = &descriptor.parts[role];
            }
        }
        const size_t probe_offset = static_cast<size_t>(schema_r_first[descriptor.schema_index]) * probe->bytes;
        const size_t probe_bytes = std::min<size_t>(probe->bytes, 4096);
        const auto warmup_start = std::chrono::steady_clock::now();
        for (const auto & chunk : pinned_chunks) {
            if (cuda_h2d_async(copy_stream, probe->arena, chunk.first, probe_offset, probe_bytes) !=
                    GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS ||
                cuda_stream_synchronize(copy_stream) != GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS) {
                unregister_pinned_l2();
                return fail(SILIANG_RUNTIME_FAILURE_INIT, "pinned L2 warm-up copy failed");
            }
        }
        metrics.pin_warmup_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - warmup_start).count());
        pinned_slot_count = static_cast<uint32_t>(wanted_slots);
        metrics.pinned_slots_armed = pinned_slot_count;
        LLAMA_LOG_INFO("siliang_moe_runtime: pinned L2 slots=%u/%u (%zu MiB) chunks=%zu register=%.1f ms warmup=%.1f ms; "
                       "decode and prefill copies of these slots skip P\n",
                pinned_slot_count, slot_count, static_cast<size_t>(wanted_slots) * slot_bytes / (1024 * 1024),
                pinned_chunks.size(), 1e-6 * metrics.pin_register_ns, 1e-6 * metrics.pin_warmup_ns);
        return true;
    }

    void unregister_pinned_l2() {
        for (const auto & chunk : pinned_chunks) {
            (void) cuda_host_unregister(cuda, chunk.first);
        }
        pinned_chunks.clear();
        pinned_slot_count = 0;
    }

    // Only after every copy recorded under this bank has completed.
    void return_bank_leases(size_t bank) {
        if (bank >= bank_leases.size()) {
            return;
        }
        for (uint32_t slot : bank_leases[bank]) {
            if (cpu_unlease(cpu, slot)) {
                ++metrics.lease_returns;
            }
        }
        bank_leases[bank].clear();
        for (const auto & release : bank_deferred_releases[bank]) {
            uint32_t released_slot = 0;
            if (cpu_release(cpu, static_cast<uint32_t>(release.first), static_cast<uint32_t>(release.second),
                            &released_slot)) {
                ++metrics.deferred_releases;
                ++metrics.l2_releases;
            } else {
                // the expert stays duplicated in L2 until evicted; correctness is unaffected
                ++metrics.deferred_release_failures;
            }
        }
        bank_deferred_releases[bank].clear();
    }

    // Before a phase change: every direct copy must be complete and every lease returned.
    bool drain_direct_copies() {
        bool pending = false;
        for (size_t bank = 0; bank < bank_leases.size(); ++bank) {
            pending = pending || !bank_leases[bank].empty() || !bank_deferred_releases[bank].empty();
        }
        if (!pending) {
            return true;
        }
        if (cuda_stream_synchronize(copy_stream) != GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS) {
            return fail(SILIANG_RUNTIME_FAILURE_STAGING, "direct L2 copy drain failed");
        }
        for (size_t bank = 0; bank < bank_leases.size(); ++bank) {
            return_bank_leases(bank);
        }
        return true;
    }

    // Samples three windows of every part of every slot the previous decode route
    // used and compares them with the model bytes of the expert it expected.
    void verify_previous_route() {
        if (verify_pending.empty()) {
            return;
        }
        if (cuda_stream_synchronize(copy_stream) != GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS) {
            verify_pending.clear();
            return;
        }
        constexpr size_t window = 4096;
        std::vector<uint8_t> device(window);
        for (const auto & entry : verify_pending) {
            const auto & descriptor = layers[static_cast<size_t>(entry.layer)];
            ++metrics.verify_entries;
            for (int role = 0; role < LLAMA_SILIANG_MOE_ARENA_PART_ROLE_COUNT; ++role) {
                if (!descriptor.present[role]) {
                    continue;
                }
                const auto & part = descriptor.parts[role];
                if (!part.source || !part.source->data || !part.source->buffer ||
                    ggml_backend_buffer_is_siliang_managed(part.source->buffer)) {
                    ++metrics.verify_skipped;
                    continue;
                }
                const size_t size = std::min(window, part.bytes);
                const size_t offsets[3] = {0, (part.bytes / 2) & ~size_t {255}, part.bytes - size};
                for (size_t offset : offsets) {
                    const size_t device_offset = static_cast<size_t>(entry.physical) * part.bytes + offset;
                    if (cuda_d2h_async(copy_stream, part.arena, device.data(), device_offset, size) !=
                            GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS ||
                        cuda_stream_synchronize(copy_stream) != GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS) {
                        ++metrics.verify_skipped;
                        continue;
                    }
                    ++metrics.verify_windows;
                    const auto * base = static_cast<const uint8_t *>(part.source->data);
                    const uint8_t * expected = base + static_cast<size_t>(entry.expert) * part.source_stride + offset;
                    if (std::memcmp(device.data(), expected, size) == 0) {
                        continue;
                    }
                    ++metrics.verify_mismatches;
                    int32_t holder = -1;
                    for (int32_t other = 0; other < model_info.expert_count && holder < 0; ++other) {
                        if (std::memcmp(device.data(), base + static_cast<size_t>(other) * part.source_stride + offset,
                                        size) == 0) {
                            holder = other;
                        }
                    }
                    if (metrics.verify_mismatches <= 64) {
                        LLAMA_LOG_WARN("siliang_moe_verify: MISMATCH route=%" PRIu64 " layer=%d expert=%d kind=%c "
                                       "physical=%d role=%s offset=%zu holds_expert=%d\n",
                                entry.route, entry.layer, entry.expert, entry.kind, entry.physical,
                                role_name(part.role), offset, holder);
                    }
                }
            }
        }
        verify_pending.clear();
    }

    // At the start of a map the scheduler has already synchronized the compute stream,
    // and that stream waited for the previous route's copies. A still-pending copy
    // event here means some copy is not covered by a compute dependency.
    void probe_previous_map(int32_t layer, bool prefill) {
        if (probe_prev_layer < 0 || !probe_prev_copied || !cuda_event_query) {
            return;
        }
        ++metrics.probe_checks;
        const cuda_status status = cuda_event_query(layer_ready_events[static_cast<size_t>(probe_prev_layer)]);
        if (status == GGML_BACKEND_CUDA_SILIANG_STATUS_NOT_READY) {
            ++metrics.probe_pending;
            if (metrics.probe_pending <= 40) {
                LLAMA_LOG_WARN("siliang_moe_runtime: copy_order PENDING prev_layer=%d prev_phase=%s "
                               "next_layer=%d next_phase=%s event_pending_flag=%d\n",
                        probe_prev_layer, probe_prev_prefill ? "prefill" : "decode", layer,
                        prefill ? "prefill" : "decode",
                        layer_event_pending[static_cast<size_t>(probe_prev_layer)] ? 1 : 0);
            }
        }
    }

    bool pinned_l2_slot(int32_t layer, int32_t expert) const {
        uint32_t slot = 0;
        return pinned_slot_count != 0 &&
            cpu_expert_slot(cpu, static_cast<uint32_t>(layer), static_cast<uint32_t>(expert), &slot) &&
            slot < pinned_slot_count;
    }

    bool acquire_staging_bank() {
        if (staging_events.empty() || staging_event_recorded.size() != staging_events.size()) {
            return fail(SILIANG_RUNTIME_FAILURE_STAGING, "P event state is invalid");
        }
        const size_t bank = next_staging_bank;
        next_staging_bank = (bank + 1) % staging_events.size();
        if (staging_event_recorded[bank]) {
            const auto wait_start = std::chrono::steady_clock::now();
            const bool waited = cuda_event_synchronize(staging_events[bank]) == GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS;
            metrics.p_bank_wait_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - wait_start).count());
            if (!waited) {
                return fail(SILIANG_RUNTIME_FAILURE_STAGING, "P bank completion wait failed");
            }
            staging_event_recorded[bank] = 0;
            ++metrics.p_bank_reuse_waits;
        }
        return_bank_leases(bank);
        active_staging_bank = bank;
        ++metrics.p_bank_uses;
        return true;
    }

    bool acquire_exchange_bank(size_t & bank) {
        if (exchange_release_events.empty() || exchange_bank_used.size() != exchange_release_events.size()) {
            return fail(SILIANG_RUNTIME_FAILURE_EVENT, "R event state is invalid");
        }
        bank = next_exchange_bank;
        next_exchange_bank = (bank + 1) % exchange_release_events.size();
        if (exchange_bank_used[bank]) {
            if (cuda_main_event_record(cuda, exchange_release_events[bank]) != GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS ||
                cuda_stream_wait_event(copy_stream, exchange_release_events[bank]) != GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS) {
                return fail(SILIANG_RUNTIME_FAILURE_EVENT, "R bank reuse dependency failed");
            }
            ++metrics.r_bank_reuses;
        }
        ++metrics.r_bank_uses;
        return true;
    }

    bool fence_k_reuse(bool & fenced) {
        if (fenced) {
            return true;
        }
        if (active_staging_bank >= k_reuse_events.size() || !k_reuse_events[active_staging_bank]) {
            return fail(SILIANG_RUNTIME_FAILURE_EVENT, "K reuse event state is invalid");
        }
        cuda_event_t event = k_reuse_events[active_staging_bank];
        if (cuda_main_event_record(cuda, event) != GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS ||
            cuda_stream_wait_event(copy_stream, event) != GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS) {
            return fail(SILIANG_RUNTIME_FAILURE_EVENT, "K slot reuse dependency failed");
        }
        fenced = true;
        ++metrics.k_reuse_fences;
        return true;
    }

    // defer_wait: leave the missing experts' reads in flight; the caller then waits for each expert with
    // wait_l2_read before copying it and calls finish_l2_reads at the end
    bool prepare_l2(int32_t layer, const std::vector<int32_t> & needed, l2_prepare_counts * counts = nullptr,
            bool defer_wait = false) {
        l2_prepare_counts local = {};
        if (needed.empty()) {
            if (counts) {
                *counts = local;
            }
            return true;
        }
        if (!l2_enabled) {
            local.misses = static_cast<uint32_t>(needed.size());
            local.miss_experts = needed;
            if (counts) {
                *counts = local;
            }
            return true;
        }
        std::vector<int32_t> order(needed.size(), -1);
        uint32_t hits = 0;
        uint32_t misses = 0;
        uint32_t active = 0;
        const bool async = cpu_prepare_async(
                cpu, static_cast<uint32_t>(layer), needed.data(), static_cast<uint32_t>(needed.size()),
                order.data(), static_cast<uint32_t>(order.size()), &hits, &misses, &active) != 0;
        if (async) {
            if (active != needed.size() || hits + misses != active) {
                return fail(SILIANG_RUNTIME_FAILURE_L2, "L2 async ordering count is invalid");
            }
            local.hit_experts.assign(order.begin(), order.begin() + hits);
            local.miss_experts.assign(order.begin() + hits, order.begin() + hits + misses);
            std::sort(order.begin(), order.end());
            std::vector<int32_t> expected = needed;
            std::sort(expected.begin(), expected.end());
            if (order != expected) {
                return fail(SILIANG_RUNTIME_FAILURE_L2, "L2 async ordering changed the route set");
            }
            ++metrics.l2_async_prepares;
            metrics.l2_hits += hits;
            metrics.l2_misses += misses;
            local.hits = hits;
            local.misses = misses;
            if (misses > 0) {
                if (defer_wait && cpu_wait_expert) {
                    l2_reads_pending = true;
                } else {
                    if (!cpu_wait(cpu)) {
                        return fail(SILIANG_RUNTIME_FAILURE_L2, "L2 deferred read wait failed");
                    }
                    ++metrics.l2_waits;
                }
            }
            if (counts) {
                *counts = local;
            }
            return true;
        }
        if (!logged_sync_l2) {
            LLAMA_LOG_WARN("siliang_moe_runtime: L2 deferred prepare unavailable; using explicit blocking prepare\n");
            logged_sync_l2 = true;
        }
        ++metrics.l2_sync_prepares;

        ggml_siliangem_cache_info before = {};
        ggml_siliangem_cache_info after = {};
        before.struct_size = sizeof(before);
        after.struct_size = sizeof(after);
        const bool have_before = cpu_query && cpu_query(cpu, &before) != 0;
        if (!cpu_prepare(cpu, static_cast<uint32_t>(layer), needed.data(), static_cast<uint32_t>(needed.size()))) {
            return fail(SILIANG_RUNTIME_FAILURE_L2, "L2 blocking prepare failed");
        }
        const bool have_after = have_before && cpu_query && cpu_query(cpu, &after) != 0;
        // The blocking API exposes aggregate deltas only, not which expert ids
        // were hits vs cold misses. Keep totals for legacy metrics but mark the
        // per-expert classification inexact so SLFU never guesses cold state.
        local.exact = false;
        if (have_after && after.hits >= before.hits && after.misses >= before.misses) {
            const uint64_t sync_hits = after.hits - before.hits;
            const uint64_t sync_misses = after.misses - before.misses;
            if (sync_hits + sync_misses == needed.size() &&
                sync_hits <= std::numeric_limits<uint32_t>::max() && sync_misses <= std::numeric_limits<uint32_t>::max()) {
                local.hits = static_cast<uint32_t>(sync_hits);
                local.misses = static_cast<uint32_t>(sync_misses);
                metrics.l2_hits += sync_hits;
                metrics.l2_misses += sync_misses;
            } else {
                local.exact = false;
            }
        } else {
            local.exact = false;
        }
        if (counts) {
            *counts = local;
        }
        return true;
    }

    void record_route_stats(
            size_t count,
            uint64_t l1_hits,
            const l2_prepare_counts & l2,
            uint64_t k_admissions,
            uint64_t r_bypasses) {
        if (!params.route_stats) {
            return;
        }
        auto & stats = route_stats_data;
        ++stats.routes;
        stats.selections += count;
        stats.state_l1 += l1_hits;
        stats.exec_gpu_k_hit += l1_hits;
        stats.exec_gpu_k_admit += k_admissions;
        stats.exec_gpu_r += r_bypasses;

        const uint64_t executed_gpu = l1_hits + k_admissions + r_bypasses;
        if (executed_gpu <= count) {
            stats.exec_cpu += count - executed_gpu;
        } else {
            stats.exec_unknown += count;
        }

        const size_t top_k = static_cast<size_t>(model_info.top_k);
        const size_t side = top_k + 1;
        if (count == top_k && executed_gpu <= count) {
            const size_t exec_cpu = static_cast<size_t>(count - executed_gpu);
            const size_t exec_index =
                (static_cast<size_t>(l1_hits) * side + static_cast<size_t>(k_admissions)) * side +
                static_cast<size_t>(r_bypasses);
            if (exec_index < stats.execution_compositions.size() &&
                static_cast<size_t>(l1_hits + k_admissions + r_bypasses) + exec_cpu == top_k) {
                ++stats.execution_compositions[exec_index];
            } else {
                stats.exec_unknown += count;
            }
        } else if (count != top_k) {
            stats.exec_unknown += count;
        }

        const uint64_t classified = l1_hits + l2.hits + l2.misses;
        if (!l2.exact || classified != count || count != top_k) {
            ++stats.unknown_routes;
            stats.state_unknown += count >= l1_hits ? count - l1_hits : count;
            maybe_print_route_checkpoint();
            return;
        }

        ++stats.exact_routes;
        stats.state_l2 += l2.hits;
        stats.state_uncached += l2.misses;
        const size_t index = static_cast<size_t>(l1_hits) * side + static_cast<size_t>(l2.hits);
        if (index < stats.compositions.size()) {
            ++stats.compositions[index];
        } else {
            ++stats.unknown_routes;
            --stats.exact_routes;
        }
        maybe_print_route_checkpoint();
    }

    void maybe_print_route_checkpoint() {
        if (!params.route_stats || model_info.routed_layer_count <= 0) {
            return;
        }
        while (route_stats_checkpoint_index < SILIANG_ROUTE_STATS_TOKEN_CHECKPOINTS.size()) {
            const uint64_t generated_tokens = SILIANG_ROUTE_STATS_TOKEN_CHECKPOINTS[route_stats_checkpoint_index];
            const uint64_t target_routes = (generated_tokens - 1) * static_cast<uint64_t>(model_info.routed_layer_count);
            if (route_stats_data.routes < target_routes) {
                return;
            }
            const auto & stats = route_stats_data;
            const double denom = stats.selections == 0 ? 1.0 : static_cast<double>(stats.selections);
            uint64_t l2_evictions = 0;
            uint64_t l2_rejections = 0;
            if (l2_enabled && cpu_query) {
                ggml_siliangem_cache_info info = {};
                info.struct_size = sizeof(info);
                if (cpu_query(cpu, &info)) {
                    l2_evictions = info.policy_evictions;
                    l2_rejections = info.policy_rejections;
                }
            }
            std::fprintf(
                    stderr,
                    "siliang_moe_runtime: route_stats checkpoint generated_tokens=%" PRIu64
                    " routes=%" PRIu64 " selections=%" PRIu64
                    " L1=%" PRIu64 "(%.2f%%) L2=%" PRIu64 "(%.2f%%) uncached=%" PRIu64 "(%.2f%%)"
                    " K_hit=%" PRIu64 " K_admit=%" PRIu64 " R=%" PRIu64 " CPU=%" PRIu64
                    " L1_evict=%" PRIu64 " L2_evict=%" PRIu64 " L2_reject=%" PRIu64
                    " demotions=%" PRIu64 " D_reuse_L2=%" PRIu64 " D_reuse_cold=%" PRIu64
                    " D_pending=%" PRIu64 " D_exposed_ms=%.3f unknown=%" PRIu64 "\n",
                    generated_tokens, stats.routes, stats.selections,
                    stats.state_l1, 100.0 * static_cast<double>(stats.state_l1) / denom,
                    stats.state_l2, 100.0 * static_cast<double>(stats.state_l2) / denom,
                    stats.state_uncached, 100.0 * static_cast<double>(stats.state_uncached) / denom,
                    stats.exec_gpu_k_hit, stats.exec_gpu_k_admit, stats.exec_gpu_r, stats.exec_cpu,
                    metrics.k_evictions, l2_evictions, l2_rejections, metrics.l2_demotions,
                    metrics.demotion_reuse_l2, metrics.demotion_reuse_cold, metrics.demotion_reuse_pending,
                    static_cast<double>(metrics.transition_exposed_wait_ns) / 1e6,
                    stats.state_unknown + stats.exec_unknown);
            // counters that otherwise only reach the exit summary, which a killed server never prints
            if (k_ext != 0 || params.verify) {
                std::fprintf(stderr,
                        "siliang_moe_runtime: checkpoint_extra decode_k_active=%d decode_k_admissions=%" PRIu64
                        " decode_k_clears=%" PRIu64 " decode_plan_max_mib=%.1f verify_entries=%" PRIu64
                        " verify_windows=%" PRIu64 " verify_mismatches=%" PRIu64 "\n",
                        k_ext_active ? 1 : 0, metrics.k_ext_admissions, metrics.k_ext_clears,
                        metrics.k_ext_plan_max / 1048576.0, metrics.verify_entries, metrics.verify_windows,
                        metrics.verify_mismatches);
            }
            std::fflush(stderr);
            ++route_stats_checkpoint_index;
        }
    }

    bool wait_l2_read(int32_t layer, int32_t expert) {
        if (!l2_reads_pending) {
            return true;
        }
        const auto start = std::chrono::steady_clock::now();
        const bool ok = cpu_wait_expert(cpu, static_cast<uint32_t>(layer), static_cast<uint32_t>(expert)) != 0;
        metrics.prefill_read_wait_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start).count());
        return ok || fail(SILIANG_RUNTIME_FAILURE_L2, "L2 expert read wait failed");
    }

    bool finish_l2_reads() {
        if (!l2_reads_pending) {
            return true;
        }
        l2_reads_pending = false;
        ++metrics.l2_waits;
        return cpu_wait(cpu) || fail(SILIANG_RUNTIME_FAILURE_L2, "L2 deferred read wait failed");
    }

    bool copy_expert(
            int32_t layer, int32_t expert, int32_t physical, size_t staging_lane, bool release_l2,
            bool direct = false) {
        auto & descriptor = layers[static_cast<size_t>(layer)];
        if (direct && l2_enabled && pinned_l2_slot(layer, expert)) {
            // straight from the registered L2 slot; the lease holds it until this bank's event completes
            for (int role = 0; role < LLAMA_SILIANG_MOE_ARENA_PART_ROLE_COUNT; ++role) {
                if (!descriptor.present[role]) {
                    continue;
                }
                auto & part = descriptor.parts[role];
                const void * source = nullptr;
                size_t source_bytes = 0;
                uint32_t slot = 0;
                if (part.l2_part < 0 ||
                    !cpu_lease_part(cpu, static_cast<uint32_t>(layer), static_cast<uint32_t>(expert),
                                    static_cast<uint32_t>(part.l2_part), &source, &source_bytes, &slot) ) {
                    return fail(SILIANG_RUNTIME_FAILURE_L2, "pinned L2 part lease failed");
                }
                bank_leases[active_staging_bank].push_back(slot);
                if (source_bytes != part.bytes || slot >= pinned_slot_count) {
                    return fail(SILIANG_RUNTIME_FAILURE_L2, "pinned L2 part does not match the arena part");
                }
                const size_t arena_offset = static_cast<size_t>(physical) * part.bytes;
                if (cuda_h2d_async(copy_stream, part.arena, source, arena_offset, part.bytes) !=
                        GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS) {
                    return fail(SILIANG_RUNTIME_FAILURE_H2D, "direct L2 H2D submission failed");
                }
                ++metrics.h2d_ops;
                metrics.h2d_bytes += part.bytes;
                metrics.direct_bytes += part.bytes;
            }
            if (release_l2) {
                bank_deferred_releases[active_staging_bank].emplace_back(layer, expert);
            }
            ++metrics.direct_experts;
            return true;
        }
        if (direct && pinned_slot_count != 0) {
            ++metrics.direct_fallbacks;
        }
        const size_t route_first_slot = active_staging_bank * static_cast<size_t>(model_info.top_k);
        const size_t staging_slot = route_first_slot + staging_lane;
        if (staging_slot >= params.elevator_p) {
            return fail(SILIANG_RUNTIME_FAILURE_STAGING, "P staging lane exceeds configured capacity");
        }
        uint8_t * slot_base = staging_base + staging_slot * staging_slot_bytes;
        if (copier.threads() > 1) {
            return copy_expert_parallel(layer, expert, physical, slot_base, release_l2);
        }
        for (int role = 0; role < LLAMA_SILIANG_MOE_ARENA_PART_ROLE_COUNT; ++role) {
            if (!descriptor.present[role]) {
                continue;
            }
            auto & part = descriptor.parts[role];
            uint8_t * destination = slot_base + part.staging_offset;
            if (l2_enabled && part.l2_part >= 0) {
                if (!cpu_copy_part(
                        cpu, static_cast<uint32_t>(layer), static_cast<uint32_t>(expert),
                        static_cast<uint32_t>(part.l2_part), destination, part.bytes)) {
                    return fail(SILIANG_RUNTIME_FAILURE_L2, "L2 cached part copy failed");
                }
            } else {
                if (ggml_backend_buffer_is_siliang_managed(part.source->buffer)) {
                    return fail(SILIANG_RUNTIME_FAILURE_L2,
                        "managed expert source cannot fall back to model-resident bytes");
                }
                const auto * source = static_cast<const uint8_t *>(part.source->data) +
                    static_cast<size_t>(expert) * part.source_stride;
                std::memcpy(destination, source, part.bytes);
            }
            const size_t arena_offset = static_cast<size_t>(physical) * part.bytes;
            if (cuda_h2d_async(copy_stream, part.arena, destination, arena_offset, part.bytes) !=
                    GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS) {
                return fail(SILIANG_RUNTIME_FAILURE_H2D, "private-stream H2D submission failed");
            }
            metrics.staged_bytes += part.bytes;
            ++metrics.h2d_ops;
            metrics.h2d_bytes += part.bytes;
        }
        if (l2_enabled && release_l2) {
            uint32_t released_slot = 0;
            if (!cpu_release(cpu, static_cast<uint32_t>(layer), static_cast<uint32_t>(expert), &released_slot)) {
                ++metrics.l2_release_failures;
                return fail(SILIANG_RUNTIME_FAILURE_L2, "exclusive L2-to-K release failed");
            }
            ++metrics.l2_releases;
        }
        return true;
    }

    // The L2 slot is leased only while this function copies from it.
    bool copy_expert_parallel(int32_t layer, int32_t expert, int32_t physical, uint8_t * slot_base, bool release_l2) {
        auto & descriptor = layers[static_cast<size_t>(layer)];
        std::vector<staging_copier::segment> segments;
        std::vector<uint32_t> leased;
        std::vector<const runtime_part *> parts;
        const auto unlease_all = [&] {
            for (uint32_t slot : leased) {
                (void) cpu_unlease(cpu, slot);
            }
        };
        for (int role = 0; role < LLAMA_SILIANG_MOE_ARENA_PART_ROLE_COUNT; ++role) {
            if (!descriptor.present[role]) {
                continue;
            }
            const auto & part = descriptor.parts[role];
            staging_copier::segment item;
            item.destination = slot_base + part.staging_offset;
            item.bytes = part.bytes;
            if (l2_enabled && part.l2_part >= 0) {
                const void * source = nullptr;
                size_t source_bytes = 0;
                uint32_t slot = 0;
                if (!cpu_lease_part(cpu, static_cast<uint32_t>(layer), static_cast<uint32_t>(expert),
                                    static_cast<uint32_t>(part.l2_part), &source, &source_bytes, &slot)) {
                    // not in a main L2 slot (SLFU transient scratch): copy it now on this thread
                    if (!cpu_copy_part(cpu, static_cast<uint32_t>(layer), static_cast<uint32_t>(expert),
                                       static_cast<uint32_t>(part.l2_part), item.destination, part.bytes)) {
                        unlease_all();
                        return fail(SILIANG_RUNTIME_FAILURE_L2, "L2 cached part copy failed");
                    }
                    ++metrics.staging_transient_copies;
                    item.bytes = 0; // already in P; still gets its H2D below
                    segments.push_back(item);
                    parts.push_back(&part);
                    continue;
                }
                leased.push_back(slot);
                if (source_bytes != part.bytes) {
                    unlease_all();
                    return fail(SILIANG_RUNTIME_FAILURE_L2, "L2 cached part size does not match the arena part");
                }
                item.source = static_cast<const uint8_t *>(source);
            } else {
                if (ggml_backend_buffer_is_siliang_managed(part.source->buffer)) {
                    unlease_all();
                    return fail(SILIANG_RUNTIME_FAILURE_L2,
                        "managed expert source cannot fall back to model-resident bytes");
                }
                item.source = static_cast<const uint8_t *>(part.source->data) +
                    static_cast<size_t>(expert) * part.source_stride;
            }
            segments.push_back(item);
            parts.push_back(&part);
        }
        const auto copy_start = std::chrono::steady_clock::now();
        copier.copy(segments);
        metrics.staging_copy_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - copy_start).count());
        unlease_all();
        for (size_t index = 0; index < parts.size(); ++index) {
            const auto & part = *parts[index];
            const size_t arena_offset = static_cast<size_t>(physical) * part.bytes;
            if (cuda_h2d_async(copy_stream, part.arena, segments[index].destination, arena_offset, part.bytes) !=
                    GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS) {
                return fail(SILIANG_RUNTIME_FAILURE_H2D, "private-stream H2D submission failed");
            }
            metrics.staged_bytes += part.bytes;
            metrics.staging_copy_bytes += part.bytes;
            ++metrics.h2d_ops;
            metrics.h2d_bytes += part.bytes;
        }
        if (l2_enabled && release_l2) {
            uint32_t released_slot = 0;
            if (!cpu_release(cpu, static_cast<uint32_t>(layer), static_cast<uint32_t>(expert), &released_slot)) {
                ++metrics.l2_release_failures;
                return fail(SILIANG_RUNTIME_FAILURE_L2, "exclusive L2-to-K release failed");
            }
            ++metrics.l2_releases;
        }
        return true;
    }

    bool record_staging_completion() {
        if (active_staging_bank >= staging_events.size() || !staging_events[active_staging_bank] ||
            cuda_event_record(copy_stream, staging_events[active_staging_bank]) !=
                    GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS) {
            return fail(SILIANG_RUNTIME_FAILURE_EVENT, "P wave completion event record failed");
        }
        staging_event_recorded[active_staging_bank] = 1;
        return true;
    }

    void invalidate_k_state() {
        bool occupied = false;
        for (auto & slot : slots) {
            if (slot.key >= 0) {
                if (static_cast<size_t>(slot.key) < resident.size()) {
                    resident[static_cast<size_t>(slot.key)] = -1;
                }
                occupied = true;
            }
            slot = {};
        }
        force_k_reuse_fence = force_k_reuse_fence || occupied;
        ++metrics.phase_invalidations;
    }

    void discard_pending_prefill_sweep(const char * reason) {
        if (prefill_sweep_cursor == 0) {
            return;
        }
        ++metrics.prefill_bitmap_incomplete;
        LLAMA_LOG_DEBUG(
                "siliang_moe_route_sweep_incomplete: v=1 scope=context epoch=%" PRIu64
                " attempt=%" PRIu64 " sweep=%" PRIu64
                " tokens=%zu seen=%zu expected=%zu reason=%s mapped=1\n",
                prefill_bitmap_epoch, prefill_sweep_attempt,
                metrics.prefill_bitmap_sweeps + 1, prefill_sweep_tokens,
                prefill_sweep_cursor, managed_layers.size(), reason);
        std::fill(prefill_sweep_bitmaps.begin(), prefill_sweep_bitmaps.end(), siliang_moe_prefill::route_bitmap {});
        prefill_sweep_cursor = 0;
        prefill_sweep_tokens = 0;
        prefill_sweep_attempt = 0;
    }

    void reset_prefill_bitmaps(const char * reason) {
        discard_pending_prefill_sweep(reason);
        std::fill(prefill_bitmaps.begin(), prefill_bitmaps.end(), siliang_moe_prefill::route_bitmap {});
        std::fill(prefill_bitmap_valid.begin(), prefill_bitmap_valid.end(), 0);
        std::fill(prefill_sweep_bitmaps.begin(), prefill_sweep_bitmaps.end(), siliang_moe_prefill::route_bitmap {});
        prefill_sweep_cursor = 0;
        prefill_sweep_tokens = 0;
        prefill_sweep_attempt = 0;
        ++prefill_bitmap_epoch;
        ++metrics.prefill_bitmap_resets;
        LLAMA_LOG_DEBUG(
                "siliang_moe_route_bitmap_reset: v=1 scope=context epoch=%" PRIu64
                " reason=%s mapped=0\n",
                prefill_bitmap_epoch, reason);
    }

    void finalize_prefill_bitmap_trace() {
        if (params.prefill) {
            discard_pending_prefill_sweep("shutdown");
        }
    }

    bool note_prefill_bitmap(
            int32_t layer,
            size_t token_count,
            const siliang_moe_prefill::route_bitmap & bitmap) {
        if (layer < 0 || static_cast<size_t>(layer) >= prefill_bitmaps.size() ||
            static_cast<size_t>(layer) >= prefill_bitmap_valid.size() ||
            static_cast<size_t>(layer) >= prefill_bitmap_layers.size() ||
            static_cast<size_t>(layer) >= prefill_sweep_bitmaps.size() || managed_layers.empty()) {
            return fail(SILIANG_RUNTIME_FAILURE_PREFILL, "bounded prefill bitmap layer is invalid");
        }
        const bool cursor_valid = prefill_sweep_cursor < managed_layers.size() &&
            layer == managed_layers[prefill_sweep_cursor] &&
            (prefill_sweep_cursor == 0 || token_count == prefill_sweep_tokens);
        if (!cursor_valid) {
            discard_pending_prefill_sweep("order-or-token-mismatch");
            if (layer != managed_layers.front()) {
                return true;
            }
        }
        if (prefill_sweep_cursor == 0) {
            prefill_sweep_tokens = token_count;
            prefill_sweep_attempt = ++prefill_bitmap_attempt_serial;
        }
        prefill_sweep_bitmaps[static_cast<size_t>(layer)] = bitmap;
        ++prefill_sweep_cursor;
        if (prefill_sweep_cursor != managed_layers.size()) {
            return true;
        }

        uint64_t sweep_unique_min = std::numeric_limits<uint64_t>::max();
        uint64_t sweep_unique_max = 0;
        for (int32_t managed_layer : managed_layers) {
            const size_t index = static_cast<size_t>(managed_layer);
            const auto & current = prefill_sweep_bitmaps[index];
            const bool has_previous = prefill_bitmap_valid[index] != 0;
            uint64_t seeded = 0;
            uint64_t needed = siliang_moe_prefill::route_bitmap_count(current);
            sweep_unique_min = std::min(sweep_unique_min, needed);
            sweep_unique_max = std::max(sweep_unique_max, needed);
            uint64_t overlap = 0;
            if (has_previous) {
                seeded = siliang_moe_prefill::route_bitmap_count(prefill_bitmaps[index]);
                overlap = siliang_moe_prefill::route_bitmap_intersection_count(
                        prefill_bitmaps[index], current);
                auto & layer_metrics = prefill_bitmap_layers[index];
                ++layer_metrics.pairs;
                layer_metrics.seeded += seeded;
                layer_metrics.needed += needed;
                layer_metrics.overlap += overlap;
                ++metrics.prefill_bitmap_pairs;
                metrics.prefill_bitmap_seeded += seeded;
                metrics.prefill_bitmap_needed += needed;
                metrics.prefill_bitmap_overlap += overlap;
                metrics.prefill_bitmap_new += needed - overlap;
                metrics.prefill_bitmap_unused += seeded - overlap;
            }
            LLAMA_LOG_DEBUG(
                    "siliang_moe_route_bitmap: v=2 scope=context epoch=%" PRIu64
                    " attempt=%" PRIu64 " sweep=%" PRIu64
                    " layer=%d tokens=%zu unique=%" PRIu64
                    " w0=%016" PRIx64 " w1=%016" PRIx64 " w2=%016" PRIx64 " w3=%016" PRIx64
                    " w4=%016" PRIx64 " w5=%016" PRIx64 " w6=%016" PRIx64 " w7=%016" PRIx64
                    " has_prev=%d overlap=%" PRIu64 " new=%" PRIu64 " unused=%" PRIu64 " mapped=1\n",
                    prefill_bitmap_epoch, prefill_sweep_attempt,
                    metrics.prefill_bitmap_sweeps + 1,
                    managed_layer, prefill_sweep_tokens, needed,
                    current[0], current[1], current[2], current[3], current[4], current[5], current[6], current[7],
                    has_previous ? 1 : 0,
                    overlap, needed - overlap, seeded - overlap);
            prefill_bitmaps[index] = current;
            prefill_bitmap_valid[index] = 1;
        }
        ++metrics.prefill_bitmap_sweeps;
        metrics.prefill_bitmap_sweep_tokens += prefill_sweep_tokens;
        const uint64_t sweep_unique_sum = metrics.prefill_unique - prefill_sweep_base_unique;
        const uint64_t sweep_k_hits = metrics.prefill_k_hits - prefill_sweep_base_k_hits;
        const uint64_t sweep_k_misses = metrics.prefill_k_misses - prefill_sweep_base_k_misses;
        const uint64_t sweep_admissions = metrics.prefill_k_admissions - prefill_sweep_base_admissions;
        const uint64_t sweep_evictions = metrics.prefill_k_evictions - prefill_sweep_base_evictions;
        const uint64_t sweep_p_waves = metrics.prefill_p_waves - prefill_sweep_base_p_waves;
        const uint64_t sweep_h2d_ops = metrics.prefill_h2d_ops - prefill_sweep_base_h2d_ops;
        const uint64_t sweep_h2d_bytes = metrics.prefill_h2d_bytes - prefill_sweep_base_h2d_bytes;
        const double sweep_unique_avg = managed_layers.empty() ? 0.0 :
            static_cast<double>(sweep_unique_sum) / static_cast<double>(managed_layers.size());
        if (params.route_stats) {
            LLAMA_LOG_INFO(
                    "SILIANG_PREFILL_SWEEP tokens=%zu layers=%zu unique_sum=%" PRIu64
                    " unique_avg=%.2f unique_min=%" PRIu64 " unique_max=%" PRIu64
                    " K_hits=%" PRIu64 " K_misses=%" PRIu64
                    " admissions=%" PRIu64 " evictions=%" PRIu64
                    " P_waves=%" PRIu64 " H2D_ops=%" PRIu64 " H2D_bytes=%" PRIu64
                    " wall_ms=%.1f prepare_ms=%.1f copy_ms=%.1f p_wait_ms=%.1f\n",
                    prefill_sweep_tokens, managed_layers.size(), sweep_unique_sum, sweep_unique_avg,
                    sweep_unique_min == std::numeric_limits<uint64_t>::max() ? 0 : sweep_unique_min, sweep_unique_max,
                    sweep_k_hits, sweep_k_misses, sweep_admissions, sweep_evictions,
                    sweep_p_waves, sweep_h2d_ops, sweep_h2d_bytes,
                    1e-6 * static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - prefill_sweep_start).count()),
                    1e-6 * static_cast<double>(metrics.prefill_prepare_ns - prefill_sweep_base_prepare_ns),
                    1e-6 * static_cast<double>(metrics.prefill_copy_ns - prefill_sweep_base_copy_ns),
                    1e-6 * static_cast<double>(metrics.p_bank_wait_ns - prefill_sweep_base_p_wait_ns));
        }
        LLAMA_LOG_DEBUG(
                "siliang_moe_route_sweep: v=1 scope=context epoch=%" PRIu64
                " attempt=%" PRIu64 " sweep=%" PRIu64
                " tokens=%zu layers=%zu records=%zu mapped=1\n",
                prefill_bitmap_epoch, prefill_sweep_attempt,
                metrics.prefill_bitmap_sweeps, prefill_sweep_tokens,
                managed_layers.size(), managed_layers.size());
        prefill_sweep_cursor = 0;
        prefill_sweep_tokens = 0;
        prefill_sweep_attempt = 0;
        return true;
    }

    void close_prefill_pass(const char * reason) {
        if (!pass_open) {
            return;
        }
        pass_open = false;
        if (!params.route_stats) {
            return;
        }
        LLAMA_LOG_INFO("siliang_moe_runtime: prefill_pass index=%" PRIu64 " tokens=%zu wall_ms=%.1f prepare_ms=%.1f "
                       "copy_ms=%.1f read_wait_ms=%.1f p_wait_ms=%.1f unique=%" PRIu64 " K_hits=%" PRIu64
                       " l2_misses=%" PRIu64 " h2d_MiB=%.0f closed_by=%s\n",
                pass_index, pass_tokens,
                1e-6 * static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - pass_start).count()),
                1e-6 * static_cast<double>(metrics.prefill_prepare_ns - pass_base_prepare_ns),
                1e-6 * static_cast<double>(metrics.prefill_copy_ns - pass_base_copy_ns),
                1e-6 * static_cast<double>(metrics.prefill_read_wait_ns - pass_base_read_wait_ns),
                1e-6 * static_cast<double>(metrics.p_bank_wait_ns - pass_base_p_wait_ns),
                metrics.prefill_unique - pass_base_unique, metrics.prefill_k_hits - pass_base_k_hits,
                metrics.l2_misses - pass_base_misses,
                (metrics.prefill_h2d_bytes - pass_base_h2d_bytes) / 1048576.0, reason);
    }

    void open_prefill_pass(size_t tokens) {
        close_prefill_pass("next-pass");
        pass_open = true;
        ++pass_index;
        pass_tokens = tokens;
        pass_start = std::chrono::steady_clock::now();
        pass_base_prepare_ns = metrics.prefill_prepare_ns;
        pass_base_copy_ns = metrics.prefill_copy_ns;
        pass_base_p_wait_ns = metrics.p_bank_wait_ns;
        pass_base_misses = metrics.l2_misses;
        pass_base_unique = metrics.prefill_unique;
        pass_base_k_hits = metrics.prefill_k_hits;
        pass_base_h2d_bytes = metrics.prefill_h2d_bytes;
        pass_base_read_wait_ns = metrics.prefill_read_wait_ns;
    }

    void close_decode_stretch() {
        if (!decode_open) {
            return;
        }
        decode_open = false;
        const auto & b = decode_base;
        const uint64_t choices = metrics.route_choices - b.route_choices;
        LLAMA_LOG_INFO("siliang_moe_runtime: decode_stretch maps=%" PRIu64 " choices=%" PRIu64 " wall_ms=%.1f "
                       "K_hits=%" PRIu64 " cpu_l2=%" PRIu64 " cpu_miss=%" PRIu64 " gpu_admissions=%" PRIu64
                       " l2_hits=%" PRIu64 " l2_misses=%" PRIu64 " h2d_MiB=%.0f host_ms=%.1f\n",
                metrics.map_calls - b.map_calls, choices,
                1e-6 * static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - decode_start).count()),
                metrics.k_hits - b.k_hits, metrics.hybrid_cpu_l2 - b.hybrid_cpu_l2,
                metrics.hybrid_cpu_miss - b.hybrid_cpu_miss, metrics.hybrid_gpu_admissions - b.hybrid_gpu_admissions,
                metrics.l2_hits - b.l2_hits, metrics.l2_misses - b.l2_misses,
                (metrics.h2d_bytes - b.h2d_bytes) / 1048576.0,
                1e-6 * static_cast<double>(metrics.decode_host_ns - b.decode_host_ns));
    }

    void enter_phase(route_phase next) {
        if (phase == next) {
            return;
        }
        if (phase == route_phase::prefill) {
            close_prefill_pass("phase-change");
        }
        if (phase == route_phase::decode) {
            close_decode_stretch();
        }
        if (next == route_phase::decode && params.route_stats) {
            decode_open = true;
            decode_start = std::chrono::steady_clock::now();
            decode_base = metrics;
        }
        (void) drain_direct_copies();
        if (phase != route_phase::none) {
            invalidate_k_state();
        }
        phase = next;
        if (phase == route_phase::prefill) {
            reset_prefill_bitmaps("phase-change");
        }
        if (phase == route_phase::decode && k_ext_active) {
            clear_decode_extension();
        }
    }

    bool map_prefill(
            int32_t layer,
            const int32_t * logical,
            int32_t * physical,
            size_t count) {
        const size_t route_width = static_cast<size_t>(model_info.top_k);
        const size_t token_count = route_width == 0 ? 0 : count / route_width;
        const uint32_t token_cap = llama_n_ubatch(ctx);
        uint32_t policy_first = 0;
        uint32_t policy_last = 0;
        if (!params.prefill || route_width == 0 || count <= route_width || count % route_width != 0 ||
            token_count > token_cap || !policy_bounds(layer, policy_first, policy_last, true)) {
            return fail(SILIANG_RUNTIME_FAILURE_PREFILL, "bounded prefill route geometry is invalid");
        }

        siliang_moe_prefill::route_union route;
        if (!siliang_moe_prefill::build_route_union(
                logical, count, model_info.expert_count,
                static_cast<size_t>(policy_last - policy_first), route)) {
            return fail(SILIANG_RUNTIME_FAILURE_PREFILL,
                    "bounded prefill route union is invalid or exceeds K");
        }
        if (route.union_index_by_choice.size() != count || route.experts.empty() ||
            route.experts.size() != route.occurrences.size()) {
            return fail(SILIANG_RUNTIME_FAILURE_PREFILL, "bounded prefill route union is incomplete");
        }

        std::vector<int64_t> route_keys(route.experts.size(), -1);
        for (size_t index = 0; index < route.experts.size(); ++index) {
            const int64_t key = key_for(layer, route.experts[index]);
            const uint64_t occurrences = route.occurrences[index];
            if (key < 0 || static_cast<size_t>(key) >= frequencies.size() ||
                occurrences > std::numeric_limits<uint64_t>::max() - frequencies[static_cast<size_t>(key)]) {
                return fail(SILIANG_RUNTIME_FAILURE_PREFILL, "bounded prefill frequency counter overflow");
            }
            route_keys[index] = key;
        }

        // Capacity and route validation are complete before the phase transition
        // invalidates any persistent decode mapping.
        enter_phase(route_phase::prefill);
        if (!managed_layers.empty() && layer == managed_layers.front()) {
            open_prefill_pass(count / static_cast<size_t>(model_info.top_k));
        }
        if (prefill_sweep_cursor == 0 && !managed_layers.empty() && layer == managed_layers.front()) {
            prefill_sweep_base_unique = metrics.prefill_unique;
            prefill_sweep_base_k_hits = metrics.prefill_k_hits;
            prefill_sweep_base_k_misses = metrics.prefill_k_misses;
            prefill_sweep_base_admissions = metrics.prefill_k_admissions;
            prefill_sweep_base_evictions = metrics.prefill_k_evictions;
            prefill_sweep_base_p_waves = metrics.prefill_p_waves;
            prefill_sweep_base_h2d_ops = metrics.prefill_h2d_ops;
            prefill_sweep_base_h2d_bytes = metrics.prefill_h2d_bytes;
            prefill_sweep_base_prepare_ns = metrics.prefill_prepare_ns;
            prefill_sweep_base_copy_ns = metrics.prefill_copy_ns;
            prefill_sweep_base_p_wait_ns = metrics.p_bank_wait_ns;
            prefill_sweep_start = std::chrono::steady_clock::now();
        }

        std::vector<int32_t> union_slots(route.experts.size(), -1);
        std::vector<int32_t> needed;
        std::vector<size_t> missing_union_indices;
        needed.reserve(route.experts.size());
        missing_union_indices.reserve(route.experts.size());
        uint64_t route_hits = 0;
        for (size_t index = 0; index < route.experts.size(); ++index) {
            const int32_t slot = resident[static_cast<size_t>(route_keys[index])];
            if (slot >= 0) {
                if (static_cast<uint32_t>(slot) < policy_first || static_cast<uint32_t>(slot) >= policy_last) {
                    return fail(SILIANG_RUNTIME_FAILURE_PREFILL,
                            "bounded prefill K hit is outside the layer policy range");
                }
                union_slots[index] = slot;
                ++route_hits;
            } else {
                needed.push_back(route.experts[index]);
                missing_union_indices.push_back(index);
            }
        }

        std::vector<int32_t> available_slots;
        available_slots.reserve(static_cast<size_t>(policy_last - policy_first));
        for (uint32_t slot = policy_first; slot < policy_last; ++slot) {
            if (!contains_key(route_keys, slots[slot].key)) {
                available_slots.push_back(static_cast<int32_t>(slot));
            }
        }
        std::stable_sort(available_slots.begin(), available_slots.end(), [&](int32_t lhs, int32_t rhs) {
            const auto & a = slots[static_cast<size_t>(lhs)];
            const auto & b = slots[static_cast<size_t>(rhs)];
            if ((a.key < 0) != (b.key < 0)) {
                return a.key < 0;
            }
            if (a.last_used != b.last_used) {
                return a.last_used < b.last_used;
            }
            return lhs < rhs;
        });
        if (available_slots.size() < missing_union_indices.size()) {
            return fail(SILIANG_RUNTIME_FAILURE_PREFILL,
                    "bounded prefill has no stable K placement for the route union");
        }
        for (size_t index = 0; index < missing_union_indices.size(); ++index) {
            union_slots[missing_union_indices[index]] = available_slots[index];
        }

        const auto prepare_start = std::chrono::steady_clock::now();
        l2_prepare_counts prepared = {};
        if (!needed.empty() && !prepare_l2(layer, needed, &prepared, true)) {
            return false;
        }
        // copy order: experts already in L2, then the misses in read-issue order
        std::vector<size_t> copy_order;
        copy_order.reserve(missing_union_indices.size());
        {
            std::vector<uint8_t> is_miss(static_cast<size_t>(model_info.expert_count), 0);
            std::vector<size_t> by_expert(static_cast<size_t>(model_info.expert_count), SIZE_MAX);
            for (int32_t expert : prepared.miss_experts) {
                if (expert >= 0 && expert < model_info.expert_count) {
                    is_miss[static_cast<size_t>(expert)] = 1;
                }
            }
            for (size_t union_index : missing_union_indices) {
                const int32_t expert = route.experts[union_index];
                by_expert[static_cast<size_t>(expert)] = union_index;
                if (!is_miss[static_cast<size_t>(expert)]) {
                    copy_order.push_back(union_index);
                }
            }
            for (int32_t expert : prepared.miss_experts) {
                if (expert >= 0 && expert < model_info.expert_count && by_expert[static_cast<size_t>(expert)] != SIZE_MAX) {
                    copy_order.push_back(by_expert[static_cast<size_t>(expert)]);
                }
            }
            if (copy_order.size() != missing_union_indices.size()) {
                (void) finish_l2_reads();
                return fail(SILIANG_RUNTIME_FAILURE_PREFILL, "bounded prefill copy order lost an expert");
            }
        }
        const auto copy_start = std::chrono::steady_clock::now();
        metrics.prefill_prepare_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(copy_start - prepare_start).count());

        bool copied = false;
        bool k_reuse_fenced = false;
        const uint64_t h2d_ops_before = metrics.h2d_ops;
        const uint64_t h2d_bytes_before = metrics.h2d_bytes;
        size_t missing_cursor = 0;
        while (missing_cursor < missing_union_indices.size()) {
            if (!acquire_staging_bank()) {
                return false;
            }
            ++metrics.prefill_p_waves;
            const size_t wave_count = std::min(route_width, missing_union_indices.size() - missing_cursor);
            bool wave_replaces_resident = force_k_reuse_fence;
            for (size_t lane = 0; lane < wave_count; ++lane) {
                const size_t union_index = copy_order[missing_cursor + lane];
                const int32_t slot = union_slots[union_index];
                if (slot < 0 || static_cast<size_t>(slot) >= slots.size()) {
                    return fail(SILIANG_RUNTIME_FAILURE_PREFILL,
                            "bounded prefill planned an invalid K slot");
                }
                wave_replaces_resident = wave_replaces_resident || slots[static_cast<size_t>(slot)].key >= 0;
            }
            if (wave_replaces_resident && !fence_k_reuse(k_reuse_fenced)) {
                return false;
            }
            if (k_reuse_fenced) {
                force_k_reuse_fence = false;
            }
            for (size_t lane = 0; lane < wave_count; ++lane) {
                const size_t union_index = copy_order[missing_cursor + lane];
                const int32_t physical_slot = physical_slot_for_policy(layer, union_slots[union_index]);
                if (!wait_l2_read(layer, route.experts[union_index])) {
                    return false;
                }
                // an expert in pinned L2 goes straight to K like in decode; the lease and any exclusive
                // release wait for this wave's staging event
                if (physical_slot < 0 || !copy_expert(
                        layer, route.experts[union_index], physical_slot, lane, !params.prefill_l2_retain, true)) {
                    return fail(SILIANG_RUNTIME_FAILURE_PREFILL,
                            "bounded prefill K population failed");
                }
            }
            if (!record_staging_completion()) {
                return false;
            }
            copied = true;
            missing_cursor += wave_count;
        }
        if (!finish_l2_reads()) {
            return false;
        }

        if (copied) {
            metrics.prefill_copy_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - copy_start).count());
            cuda_event_t ready = layer_ready_events[static_cast<size_t>(layer)];
            if (!ready || cuda_event_record(copy_stream, ready) != GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS) {
                return fail(SILIANG_RUNTIME_FAILURE_EVENT,
                        "bounded prefill layer-ready event record failed");
            }
            layer_event_pending[static_cast<size_t>(layer)] = 1;
            ++metrics.ready_events;
            probe_prev_layer = layer;
            probe_prev_prefill = true;
            probe_prev_copied = true;
        }

        uint64_t route_evictions = 0;
        for (size_t index = 0; index < route.experts.size(); ++index) {
            const int64_t key = route_keys[index];
            const int32_t slot = union_slots[index];
            if (slot < 0 || static_cast<size_t>(slot) >= slots.size()) {
                return fail(SILIANG_RUNTIME_FAILURE_PREFILL, "bounded prefill K commit slot is invalid");
            }
            auto & destination = slots[static_cast<size_t>(slot)];
            if (destination.key >= 0 && destination.key != key) {
                if (static_cast<size_t>(destination.key) >= resident.size()) {
                    return fail(SILIANG_RUNTIME_FAILURE_PREFILL,
                            "bounded prefill K victim key is invalid");
                }
                resident[static_cast<size_t>(destination.key)] = -1;
                ++route_evictions;
            }
            frequencies[static_cast<size_t>(key)] += route.occurrences[index];
            destination.key = key;
            destination.last_used = ++clock;
            destination.segment = params.l1_policy == LLAMA_SILIANG_EXPERT_CACHE_POLICY_WTINYLFU_W10_SLRU_P80 ?
                slot_segment::window : slot_segment::none;
            destination.frequency = route.occurrences[index];
            resident[static_cast<size_t>(key)] = slot;
        }
        for (size_t index = 0; index < count; ++index) {
            const size_t union_index = route.union_index_by_choice[index];
            if (union_index >= union_slots.size()) {
                return fail(SILIANG_RUNTIME_FAILURE_PREFILL,
                        "bounded prefill choice-to-union mapping is invalid");
            }
            physical[index] = physical_slot_for_policy(layer, union_slots[union_index]);
            if (physical[index] < 0) {
                return fail(SILIANG_RUNTIME_FAILURE_PREFILL,
                        "bounded prefill K slot has no physical translation");
            }
        }

        const uint64_t route_misses = route.experts.size() - route_hits;
        metrics.route_choices += count;
        metrics.k_hits += route_hits;
        metrics.k_misses += route_misses;
        metrics.k_admissions += route_misses;
        metrics.k_evictions += route_evictions;
        ++metrics.map_calls;
        ++metrics.prefill_maps;
        metrics.prefill_tokens += token_count;
        metrics.prefill_choices += count;
        metrics.prefill_unique += route.experts.size();
        metrics.prefill_unique_max = std::max<uint64_t>(metrics.prefill_unique_max, route.experts.size());
        metrics.prefill_k_hits += route_hits;
        metrics.prefill_k_misses += route_misses;
        metrics.prefill_k_admissions += route_misses;
        metrics.prefill_k_evictions += route_evictions;
        metrics.prefill_h2d_ops += metrics.h2d_ops - h2d_ops_before;
        metrics.prefill_h2d_bytes += metrics.h2d_bytes - h2d_bytes_before;
        if (!note_prefill_bitmap(layer, token_count, route.bitmap)) {
            return false;
        }
        if (!logged_first_prefill) {
            LLAMA_LOG_INFO(
                    "siliang_moe_runtime: serving bounded prefill layer=%d tokens=%zu choices=%zu unique=%zu "
                    "K_hits=%" PRIu64 " K_admissions=%" PRIu64 " P_waves=%zu R_bypass=0 failure=0\n",
                    layer, token_count, count, route.experts.size(), route_hits, route_misses,
                    (missing_union_indices.size() + route_width - 1) / route_width);
            logged_first_prefill = true;
        }
        return true;
    }

    bool finalize_pending_transitions() {
        if (pending_transitions.empty()) {
            transition_event_recorded = false;
            return true;
        }
        if (!transition_event_recorded) {
            metrics.k_transition_cancels += pending_transitions.size();
            pending_transitions.clear();
            return true;
        }

        const auto exposed_wait_start = std::chrono::steady_clock::now();
        int32_t worker_error = 0;
        std::string worker_error_stage;
        int32_t worker_error_layer = -1;
        int32_t worker_error_expert = -1;
        uint32_t worker_error_slot = std::numeric_limits<uint32_t>::max();
        uint64_t device_wait_ns = 0;
        uint64_t host_copy_ns = 0;
        {
            std::unique_lock<std::mutex> lock(transition_worker_mutex);
            transition_worker_cv.wait(lock, [this] { return transition_worker_done; });
            worker_error = transition_worker_error;
            worker_error_stage = transition_worker_error_stage;
            worker_error_layer = transition_worker_error_layer;
            worker_error_expert = transition_worker_error_expert;
            worker_error_slot = transition_worker_error_slot;
            device_wait_ns = transition_worker_device_wait_ns;
            host_copy_ns = transition_worker_host_copy_ns;
        }
        const auto exposed_wait_end = std::chrono::steady_clock::now();
        metrics.transition_exposed_wait_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(exposed_wait_end - exposed_wait_start).count());
        metrics.transition_device_wait_ns += device_wait_ns;
        metrics.transition_host_copy_ns += host_copy_ns;

        if (worker_error != 0) {
            ++metrics.l2_demotion_failures;
            if (worker_error == SILIANG_RUNTIME_FAILURE_EVENT) {
                return fail(SILIANG_RUNTIME_FAILURE_EVENT, "K/L2 transition worker CUDA wait failed");
            }
            LLAMA_LOG_ERROR(
                    "siliang_moe_runtime: K/L2 transition worker L2 swap failed stage=%s layer=%d expert=%d slot=%u\n",
                    worker_error_stage.empty() ? "unknown" : worker_error_stage.c_str(),
                    worker_error_layer, worker_error_expert, worker_error_slot);
            return fail(SILIANG_RUNTIME_FAILURE_L2, "K/L2 transition worker L2 swap failed");
        }

        for (auto & transition : pending_transitions) {
            const size_t policy_slot = static_cast<size_t>(transition.policy_slot);
            if (policy_slot >= slots.size() || slots[policy_slot].key != transition.victim_key) {
                return fail(SILIANG_RUNTIME_FAILURE_ROUTE, "SLFU transition K victim changed before commit");
            }
            resident[static_cast<size_t>(transition.victim_key)] = -1;
            slots[policy_slot].key = transition.candidate_key;
            slots[policy_slot].last_used = ++clock;
            slots[policy_slot].segment = slot_segment::none;
            slots[policy_slot].frequency = 1;
            resident[static_cast<size_t>(transition.candidate_key)] = transition.policy_slot;
            if (params.route_stats && params.demote_k_hot) {
                const size_t victim_index = static_cast<size_t>(transition.victim_key);
                const size_t victim_layer = static_cast<size_t>(transition.victim_key / model_info.expert_count);
                if (victim_index < demotion_reuse_pending.size() && victim_layer < layer_decode_round.size()) {
                    if (!demotion_reuse_pending[victim_index]) {
                        demotion_reuse_pending[victim_index] = 1;
                        ++metrics.demotion_reuse_pending;
                    }
                    demotion_start_round[victim_index] = layer_decode_round[victim_layer];
                }
            }
            ++metrics.l2_releases;
            ++metrics.k_admissions;
            ++metrics.k_evictions;
            ++metrics.l2_demotions;
            ++metrics.k_transition_commits;
        }
        pending_transitions.clear();
        transition_event_recorded = false;
        return true;
    }

    bool post_compute(int32_t layer) {
        std::lock_guard<std::mutex> lock(mutex);
        if (failure.load(std::memory_order_acquire) != 0 || layer < 0 || layer >= model_info.layer_count) {
            return false;
        }
        if (pending_transitions.empty()) {
            return true;
        }
        if (!params.demote_k_hot || !demotion_base || !transition_event || transition_event_recorded) {
            return fail(SILIANG_RUNTIME_FAILURE_EVENT, "SLFU transition post-compute state is invalid");
        }
        if (pending_transitions.size() > static_cast<size_t>(model_info.top_k)) {
            return fail(SILIANG_RUNTIME_FAILURE_ROUTE, "SLFU transition count exceeds route width");
        }
        const auto submit_start = std::chrono::steady_clock::now();

        for (size_t transition_index = 0; transition_index < pending_transitions.size(); ++transition_index) {
            const auto & transition = pending_transitions[transition_index];
            if (transition.layer != layer || transition.victim_key < 0 || transition.candidate_key < 0) {
                return fail(SILIANG_RUNTIME_FAILURE_ROUTE, "SLFU transition layer/key state is invalid");
            }
            const uint32_t victim_layer = static_cast<uint32_t>(transition.victim_key / model_info.expert_count);
            if (victim_layer >= layers.size()) {
                return fail(SILIANG_RUNTIME_FAILURE_ROUTE, "SLFU transition victim layer is invalid");
            }
            const auto & victim_descriptor = layers[victim_layer];
            const auto & candidate_descriptor = layers[static_cast<size_t>(transition.layer)];
            uint8_t * demotion_slot = demotion_base + transition_index * staging_slot_bytes;

            for (int role = 0; role < LLAMA_SILIANG_MOE_ARENA_PART_ROLE_COUNT; ++role) {
                if (!victim_descriptor.present[role]) {
                    continue;
                }
                const auto & part = victim_descriptor.parts[role];
                const size_t source_offset = static_cast<size_t>(transition.k_physical) * part.bytes;
                if (!part.arena || cuda_d2h_async(
                        copy_stream, part.arena, demotion_slot + part.staging_offset,
                        source_offset, part.bytes) != GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS) {
                    return fail(SILIANG_RUNTIME_FAILURE_H2D, "SLFU K victim D2H submission failed");
                }
                ++metrics.demotion_d2h_ops;
                metrics.demotion_d2h_bytes += part.bytes;
            }

            for (int role = 0; role < LLAMA_SILIANG_MOE_ARENA_PART_ROLE_COUNT; ++role) {
                if (!candidate_descriptor.present[role]) {
                    continue;
                }
                const auto & part = candidate_descriptor.parts[role];
                const size_t destination_offset = static_cast<size_t>(transition.k_physical) * part.bytes;
                const size_t source_offset = static_cast<size_t>(transition.candidate_r_physical) * part.bytes;
                if (!part.arena || cuda_d2d_async(
                        copy_stream, part.arena, destination_offset, source_offset, part.bytes) !=
                            GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS) {
                    return fail(SILIANG_RUNTIME_FAILURE_H2D, "SLFU R-to-K D2D submission failed");
                }
                ++metrics.promotion_d2d_ops;
                metrics.promotion_d2d_bytes += part.bytes;
            }
        }
        if (cuda_event_record(copy_stream, transition_event) != GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS) {
            return fail(SILIANG_RUNTIME_FAILURE_EVENT, "SLFU transition completion event record failed");
        }
        transition_event_recorded = true;
        {
            std::lock_guard<std::mutex> worker_lock(transition_worker_mutex);
            if (!transition_worker.joinable() || !transition_worker_done || transition_worker_job_pending) {
                return fail(SILIANG_RUNTIME_FAILURE_EVENT, "SLFU transition worker state is busy or unavailable");
            }
            transition_worker_error = 0;
            transition_worker_error_stage.clear();
            transition_worker_error_layer = -1;
            transition_worker_error_expert = -1;
            transition_worker_error_slot = std::numeric_limits<uint32_t>::max();
            transition_worker_device_wait_ns = 0;
            transition_worker_host_copy_ns = 0;
            transition_worker_done = false;
            transition_worker_job_pending = true;
        }
        transition_worker_cv.notify_one();
        const auto submit_end = std::chrono::steady_clock::now();
        metrics.transition_submit_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(submit_end - submit_start).count());
        return true;
    }

    // Same decision the route loop makes for a non-resident expert, without
    // changing slot state or rejection counters.
    bool predict_k_admission(int32_t layer, int64_t key, bool in_l2, const std::vector<int64_t> & route_keys) {
        if (params.l1_policy == LLAMA_SILIANG_EXPERT_CACHE_POLICY_CUMULATIVE_LFU_ADMISSION &&
            !params.admit_k_cold && !in_l2) {
            return false;
        }
        // the route loop counts this choice before deciding
        ++frequencies[static_cast<size_t>(key)];
        const uint64_t rejections = metrics.k_rejections;
        bool admit = true;
        const int32_t slot = choose_hot_or_bypass(layer, key, route_keys, admit);
        metrics.k_rejections = rejections;
        --frequencies[static_cast<size_t>(key)];
        return admit && slot >= 0;
    }

    // Decode only. Marks the bypass experts that run on CPU from L2. Predicted
    // K admissions always cross to GPU; of the remaining L2-resident and
    // uncached bypass experts the cost table picks how many stay on CPU, and
    // the lowest-ranked ones are chosen. At least one route slot stays on GPU
    // so the masked GPU branch has a valid resident slot to read.
    void plan_hybrid_route(
            int32_t layer,
            const int32_t * logical,
            const std::vector<int64_t> & route_keys,
            std::vector<uint8_t> & cpu_route,
            uint32_t & cpu_l2,
            uint32_t & cpu_miss) {
        const size_t count = route_keys.size();
        cpu_route.assign(count, 0);
        cpu_l2 = 0;
        cpu_miss = 0;
        std::vector<size_t> pinned_candidates;
        std::vector<size_t> l2_candidates;
        std::vector<size_t> miss_candidates;
        std::vector<uint8_t> candidate_in_l2(count, 0);
        uint32_t forced_gpu = 0;
        bool gpu_slot = false;
        for (size_t index = 0; index < count; ++index) {
            const int64_t key = route_keys[index];
            if (resident[static_cast<size_t>(key)] >= 0) {
                gpu_slot = true;
                continue;
            }
            const int location = cpu_location(cpu, static_cast<uint32_t>(layer), static_cast<uint32_t>(logical[index]));
            const bool in_l2 = location == GGML_SILIANGEM_EXPERT_LOCATION_RESIDENT ||
                location == GGML_SILIANGEM_EXPERT_LOCATION_TRANSIENT;
            if (predict_k_admission(layer, key, in_l2, route_keys)) {
                ++forced_gpu;
                gpu_slot = true;
                continue;
            }
            candidate_in_l2[index] = in_l2 ? 1 : 0;
            if (!in_l2) {
                miss_candidates.push_back(index);
            } else if (location == GGML_SILIANGEM_EXPERT_LOCATION_RESIDENT && pinned_l2_slot(layer, logical[index])) {
                pinned_candidates.push_back(index);
            } else {
                l2_candidates.push_back(index);
            }
        }
        siliang_moe_hybrid::route_classes route;
        route.pinned = static_cast<uint32_t>(pinned_candidates.size());
        route.l2 = static_cast<uint32_t>(l2_candidates.size());
        route.miss = static_cast<uint32_t>(miss_candidates.size());
        route.g = forced_gpu;
        const auto share = hybrid_costs.lookup(route);
        for (size_t taken = 0; taken < share.pinned && taken < pinned_candidates.size(); ++taken) {
            cpu_route[pinned_candidates[pinned_candidates.size() - 1 - taken]] = 1;
        }
        for (size_t taken = 0; taken < share.l2 && taken < l2_candidates.size(); ++taken) {
            cpu_route[l2_candidates[l2_candidates.size() - 1 - taken]] = 1;
        }
        for (size_t taken = 0; taken < share.miss && taken < miss_candidates.size(); ++taken) {
            cpu_route[miss_candidates[miss_candidates.size() - 1 - taken]] = 1;
        }
        if (!gpu_slot) {
            for (size_t index = 0; index < count; ++index) {
                if (!cpu_route[index]) {
                    gpu_slot = true;
                    break;
                }
            }
        }
        if (!gpu_slot) {
            // the highest-ranked expert anchors the GPU branch
            cpu_route[0] = 0;
            ++metrics.hybrid_gpu_anchors;
        }
        for (size_t index = 0; index < count; ++index) {
            if (cpu_route[index]) {
                ++(candidate_in_l2[index] ? cpu_l2 : cpu_miss);
            }
        }
        metrics.hybrid_gpu_admissions += forced_gpu;
        metrics.hybrid_gpu_bypasses +=
            pinned_candidates.size() + l2_candidates.size() + miss_candidates.size() - cpu_l2 - cpu_miss;
    }

    bool map(
            int32_t layer,
            const int32_t * logical,
            int32_t * physical,
            size_t count) {
        std::lock_guard<std::mutex> lock(mutex);
        const size_t route_width = static_cast<size_t>(model_info.top_k);
        const bool prefill = route_width > 0 && count > route_width;
        // before verify, which synchronizes the copy stream and would hide pending copies
        probe_previous_map(layer, prefill);
        if (params.verify) {
            verify_previous_route();
        }
        if (!bound || failure.load(std::memory_order_acquire) != 0 || !logical || !physical ||
            layer < 0 || layer >= model_info.layer_count || !managed[static_cast<size_t>(layer)] ||
            count == 0 || route_width == 0 || count % route_width != 0 ||
            (!prefill && count != route_width)) {
            return fail(SILIANG_RUNTIME_FAILURE_ROUTE, "route callback contract failed");
        }
        if (!finalize_pending_transitions()) {
            return false;
        }
        if (layer_event_pending[static_cast<size_t>(layer)]) {
            cuda_event_t ready = layer_ready_events[static_cast<size_t>(layer)];
            if (!ready || cuda_main_wait_event(cuda, ready) != GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS) {
                return fail(SILIANG_RUNTIME_FAILURE_EVENT, "aborted-graph copy dependency recovery failed");
            }
            layer_event_pending[static_cast<size_t>(layer)] = 0;
            ++metrics.compute_waits;
            ++metrics.pending_recoveries;
            if (!logged_pending_recovery) {
                LLAMA_LOG_WARN(
                        "siliang_moe_runtime: recovered a pending layer copy dependency after an interrupted graph\n");
                logged_pending_recovery = true;
            }
        }
        if (prefill) {
            return map_prefill(layer, logical, physical, count);
        }
        if (params.route_stats && params.demote_k_hot) {
            if (static_cast<size_t>(layer) >= layer_decode_round.size() ||
                layer_decode_round[static_cast<size_t>(layer)] == std::numeric_limits<uint64_t>::max()) {
                return fail(SILIANG_RUNTIME_FAILURE_ROUTE, "decode layer-round counter overflow");
            }
            ++layer_decode_round[static_cast<size_t>(layer)];
        }
        std::vector<int64_t> route_keys(count);
        std::vector<int32_t> needed;
        needed.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            if (logical[index] < 0 || logical[index] >= model_info.expert_count) {
                return fail(SILIANG_RUNTIME_FAILURE_ROUTE, "invalid logical route");
            }
            route_keys[index] = key_for(layer, logical[index]);
            for (size_t prior = 0; prior < index; ++prior) {
                if (route_keys[prior] == route_keys[index]) {
                    return fail(SILIANG_RUNTIME_FAILURE_ROUTE, "duplicate logical expert in route");
                }
            }
            if (resident[static_cast<size_t>(route_keys[index])] < 0) {
                needed.push_back(logical[index]);
            }
        }
        enter_phase(route_phase::decode);
        if (k_ext_active && layer == managed_layers.front()) {
            check_decode_extension();
        }
        const auto host_start = std::chrono::steady_clock::now();
        std::vector<uint8_t> cpu_route(count, 0);
        uint32_t hybrid_cpu_l2 = 0;
        uint32_t hybrid_cpu_miss = 0;
        if (params.hybrid) {
            plan_hybrid_route(layer, logical, route_keys, cpu_route, hybrid_cpu_l2, hybrid_cpu_miss);
        }
        needed.clear();
        for (size_t index = 0; index < count; ++index) {
            if (resident[static_cast<size_t>(route_keys[index])] < 0 && !cpu_route[index]) {
                needed.push_back(logical[index]);
            }
        }
        l2_prepare_counts route_l2 = {};
        if (!needed.empty() && (!acquire_staging_bank() || !prepare_l2(layer, needed, &route_l2))) {
            return false;
        }
        metrics.decode_prepare_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - host_start).count());
        metrics.decode_staged_l2 += route_l2.hits;
        metrics.decode_staged_miss += route_l2.misses;
        record_demotion_reuse(layer, route_keys, route_l2);

        bool copied = false;
        bool exchange_used = false;
        size_t exchange_bank = exchange_release_events.size();
        size_t exchange_lane = 0;
        size_t staging_lane = 0;
        uint64_t route_hits = 0;
        uint64_t route_misses = 0;
        uint64_t route_admissions = 0;
        uint64_t route_bypasses = 0;
        bool k_reuse_fenced = false;
        std::vector<int64_t> policy_protected_keys = route_keys;
        std::vector<char> verify_kind(params.verify ? count : 0, 'R');
        for (size_t index = 0; index < count; ++index) {
            const int64_t key = route_keys[index];
            if (frequencies[static_cast<size_t>(key)] == std::numeric_limits<uint64_t>::max()) {
                return fail(SILIANG_RUNTIME_FAILURE_ROUTE, "frequency counter overflow");
            }
            ++frequencies[static_cast<size_t>(key)];
            ++metrics.route_choices;
            int32_t slot = resident[static_cast<size_t>(key)];
            if (slot >= 0) {
                ++metrics.k_hits;
                ++route_hits;
                record_hit(layer, slot, route_keys);
                slots[static_cast<size_t>(slot)].last_used = ++clock;
                physical[index] = physical_slot_for_policy(layer, slot);
                if (physical[index] < 0) {
                    return fail(SILIANG_RUNTIME_FAILURE_ROUTE, "K hit has no valid bank-local translation");
                }
                if (params.verify) {
                    verify_kind[index] = 'H';
                }
                continue;
            }

            ++metrics.k_misses;
            ++route_misses;
            if (cpu_route[index]) {
                // L2 serves it to the CPU branch; K and the P/R path are untouched
                physical[index] = LLAMA_SILIANG_MOE_ARENA_CPU_ROUTE;
                continue;
            }
            const bool was_cold = route_l2.exact &&
                std::find(route_l2.miss_experts.begin(), route_l2.miss_experts.end(), logical[index]) !=
                    route_l2.miss_experts.end();
            const int l2_location = l2_enabled && cpu_location
                ? cpu_location(cpu, static_cast<uint32_t>(layer), static_cast<uint32_t>(logical[index]))
                : GGML_SILIANGEM_EXPERT_LOCATION_NONE;
            const bool l2_persistent = l2_location == GGML_SILIANGEM_EXPERT_LOCATION_RESIDENT;
            const bool l2_transient = l2_location == GGML_SILIANGEM_EXPERT_LOCATION_TRANSIENT;
            if (l2_enabled && route_l2.exact && !l2_persistent && !l2_transient) {
                return fail(SILIANG_RUNTIME_FAILURE_L2, "prepared L2 expert has no resident or transient location");
            }
            bool admit = true;
            if (params.l1_policy == LLAMA_SILIANG_EXPERT_CACHE_POLICY_CUMULATIVE_LFU_ADMISSION &&
                !params.admit_k_cold && was_cold) {
                admit = false;
                slot = -1;
                ++metrics.slfu_cold_bypasses;
            } else {
                slot = choose_hot_or_bypass(layer, key, policy_protected_keys, admit);
            }
            if (!admit) {
                if (exchange_bank == exchange_release_events.size() && !acquire_exchange_bank(exchange_bank)) {
                    return false;
                }
                if (exchange_lane >= static_cast<size_t>(model_info.top_k)) {
                    return fail(SILIANG_RUNTIME_FAILURE_ROUTE, "R bank route capacity exhausted");
                }
                const int32_t exchange_physical = physical_slot_for_exchange(layer, exchange_bank, exchange_lane);
                if (exchange_physical < 0) {
                    return fail(SILIANG_RUNTIME_FAILURE_ROUTE, "R bypass has no valid schema-bank translation");
                }
                if (!copy_expert(layer, logical[index], exchange_physical, staging_lane++, false, true)) {
                    return false;
                }
                physical[index] = exchange_physical;
                ++exchange_lane;
                ++metrics.r_experts;
                ++route_bypasses;
                exchange_used = true;
                copied = true;
                continue;
            }
            if (slot < 0 || static_cast<size_t>(slot) >= slots.size()) {
                return fail(SILIANG_RUNTIME_FAILURE_ROUTE, "K admission could not obtain a physical slot");
            }
            const int32_t admission_physical = physical_slot_for_policy(layer, slot);
            if (admission_physical < 0) {
                return fail(SILIANG_RUNTIME_FAILURE_ROUTE, "K admission has no valid bank-local translation");
            }
            const bool replaces_resident = slots[static_cast<size_t>(slot)].key >= 0;
            const int64_t replaced_key = replaces_resident ? slots[static_cast<size_t>(slot)].key : -1;

            if (params.demote_k_hot && replaces_resident && !l2_persistent) {
                if (exchange_bank == exchange_release_events.size() && !acquire_exchange_bank(exchange_bank)) {
                    return false;
                }
                if (exchange_lane >= static_cast<size_t>(model_info.top_k)) {
                    return fail(SILIANG_RUNTIME_FAILURE_ROUTE, "R bank route capacity exhausted by nonresident L2 K candidate");
                }
                const int32_t exchange_physical = physical_slot_for_exchange(layer, exchange_bank, exchange_lane);
                if (exchange_physical < 0 ||
                    !copy_expert(layer, logical[index], exchange_physical, staging_lane++, false, true)) {
                    return false;
                }
                physical[index] = exchange_physical;
                ++exchange_lane;
                ++metrics.r_experts;
                ++route_bypasses;
                ++metrics.k_transition_cancels;
                exchange_used = true;
                copied = true;
                continue;
            }

            if (params.demote_k_hot && replaces_resident) {
                if (exchange_bank == exchange_release_events.size() && !acquire_exchange_bank(exchange_bank)) {
                    return false;
                }
                if (exchange_lane >= static_cast<size_t>(model_info.top_k)) {
                    return fail(SILIANG_RUNTIME_FAILURE_ROUTE, "R bank route capacity exhausted by deferred K promotion");
                }
                const int32_t exchange_physical = physical_slot_for_exchange(layer, exchange_bank, exchange_lane);
                if (exchange_physical < 0 ||
                    !copy_expert(layer, logical[index], exchange_physical, staging_lane++, false, true)) {
                    return false;
                }
                physical[index] = exchange_physical;
                pending_transitions.push_back({
                    /*.layer                =*/ layer,
                    /*.candidate_expert     =*/ logical[index],
                    /*.candidate_key        =*/ key,
                    /*.candidate_r_physical =*/ exchange_physical,
                    /*.policy_slot          =*/ slot,
                    /*.k_physical           =*/ admission_physical,
                    /*.victim_key           =*/ replaced_key,
                    /*.released_l2_slot     =*/ 0,
                });
                policy_protected_keys.push_back(replaced_key);
                ++exchange_lane;
                ++metrics.r_experts;
                ++route_bypasses;
                ++metrics.k_deferred_promotions;
                exchange_used = true;
                copied = true;
                continue;
            }

            if (((force_k_reuse_fence || replaces_resident) && !fence_k_reuse(k_reuse_fenced)) ||
                // an extra decode K slot is dropped at the next prefill: keeping its L2 copy measured 8.9 vs
                // 7.0 tok/s (uncached 10% vs 16%) against releasing it like a base K admission
                !copy_expert(layer, logical[index], admission_physical, staging_lane++,
                        l2_persistent && admission_physical >= static_cast<int32_t>(k_ext), true)) {
                return false;
            }
            metrics.k_ext_admissions += admission_physical < static_cast<int32_t>(k_ext);
            if (k_reuse_fenced) {
                force_k_reuse_fence = false;
            }
            copied = true;
            if (replaces_resident) {
                resident[static_cast<size_t>(replaced_key)] = -1;
                ++metrics.k_evictions;
            }
            slots[static_cast<size_t>(slot)].key = key;
            slots[static_cast<size_t>(slot)].last_used = ++clock;
            slots[static_cast<size_t>(slot)].segment =
                params.l1_policy == LLAMA_SILIANG_EXPERT_CACHE_POLICY_WTINYLFU_W10_SLRU_P80 ?
                slot_segment::window : slot_segment::none;
            slots[static_cast<size_t>(slot)].frequency = 1;
            resident[static_cast<size_t>(key)] = slot;
            physical[index] = admission_physical;
            if (params.verify) {
                verify_kind[index] = 'A';
            }
            ++metrics.k_admissions;
            ++route_admissions;
        }

        if (exchange_used) {
            exchange_bank_used[exchange_bank] = 1;
        }
        if (copied) {
            cuda_event_t ready = layer_ready_events[static_cast<size_t>(layer)];
            if (!ready || cuda_event_record(copy_stream, ready) != GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS ||
                cuda_event_record(copy_stream, staging_events[active_staging_bank]) !=
                    GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS) {
                return fail(SILIANG_RUNTIME_FAILURE_EVENT, "copy completion event record failed");
            }
            layer_event_pending[static_cast<size_t>(layer)] = 1;
            staging_event_recorded[active_staging_bank] = 1;
            ++metrics.ready_events;
        }
        probe_prev_layer = layer;
        probe_prev_prefill = false;
        probe_prev_copied = copied;
        ++metrics.map_calls;
        if (params.verify) {
            ++verify_route_serial;
            for (size_t index = 0; index < count; ++index) {
                if (physical[index] >= 0) {
                    verify_pending.push_back({layer, logical[index], physical[index], verify_kind[index],
                                              verify_route_serial});
                }
            }
        }
        metrics.hybrid_cpu_l2 += hybrid_cpu_l2;
        metrics.hybrid_cpu_miss += hybrid_cpu_miss;
        metrics.decode_host_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - host_start).count());
        // CPU experts were classified before staging; count them in the residency view too
        route_l2.hits += hybrid_cpu_l2;
        route_l2.misses += hybrid_cpu_miss;
        record_route_stats(count, route_hits, route_l2, route_admissions, route_bypasses);
        if (!logged_first_route) {
            LLAMA_LOG_INFO(
                    "siliang_moe_runtime: serving decode route map=1 layer=%d K_hits=%" PRIu64
                    " K_misses=%" PRIu64 " admissions=%" PRIu64 " R_bypass=%" PRIu64
                    " H2D_ops=%" PRIu64 " failure=0\n",
                    layer, route_hits, route_misses, route_admissions, route_bypasses, metrics.h2d_ops);
            logged_first_route = true;
        }
        return true;
    }

    bool compute_wait(int32_t layer) {
        std::lock_guard<std::mutex> lock(mutex);
        if (failure.load(std::memory_order_acquire) != 0 || layer < 0 || layer >= model_info.layer_count ||
            !managed[static_cast<size_t>(layer)]) {
            return false;
        }
        if (!layer_event_pending[static_cast<size_t>(layer)]) {
            return true;
        }
        cuda_event_t ready = layer_ready_events[static_cast<size_t>(layer)];
        if (!ready || cuda_main_wait_event(cuda, ready) != GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS) {
            return fail(SILIANG_RUNTIME_FAILURE_EVENT, "main-stream ready wait insertion failed");
        }
        layer_event_pending[static_cast<size_t>(layer)] = 0;
        ++metrics.compute_waits;
        if (phase == route_phase::prefill) {
            ++metrics.prefill_compute_waits;
        }
        if (metrics.compute_waits == 1) {
            LLAMA_LOG_INFO(
                    "siliang_moe_runtime: serving %s compute_wait=1 layer=%d failure=0\n",
                    phase == route_phase::prefill ? "bounded prefill" : "decode", layer);
        }
        return true;
    }

    void reset_prefill_trace() {
        std::lock_guard<std::mutex> lock(mutex);
        if (params.prefill) {
            reset_prefill_bitmaps("serving-start");
            metrics.prefill_maps = 0;
            metrics.prefill_tokens = 0;
            metrics.prefill_choices = 0;
            metrics.prefill_unique = 0;
            metrics.prefill_unique_max = 0;
            metrics.prefill_k_hits = 0;
            metrics.prefill_k_misses = 0;
            metrics.prefill_k_admissions = 0;
            metrics.prefill_k_evictions = 0;
            metrics.prefill_p_waves = 0;
            metrics.prefill_h2d_ops = 0;
            metrics.prefill_h2d_bytes = 0;
            metrics.prefill_compute_waits = 0;
            metrics.prefill_bitmap_sweeps = 0;
            metrics.prefill_bitmap_sweep_tokens = 0;
            metrics.prefill_bitmap_pairs = 0;
            metrics.prefill_bitmap_seeded = 0;
            metrics.prefill_bitmap_needed = 0;
            metrics.prefill_bitmap_overlap = 0;
            metrics.prefill_bitmap_new = 0;
            metrics.prefill_bitmap_unused = 0;
            metrics.prefill_bitmap_resets = 0;
            metrics.prefill_bitmap_incomplete = 0;
            std::fill(prefill_bitmap_layers.begin(), prefill_bitmap_layers.end(), route_bitmap_metrics {});
            logged_first_prefill = false;
        }
    }

    const char * policy_name() const {
        switch (params.l1_policy) {
            case LLAMA_SILIANG_EXPERT_CACHE_POLICY_LRU: return "lru";
            case LLAMA_SILIANG_EXPERT_CACHE_POLICY_LFU: return "lfu";
            case LLAMA_SILIANG_EXPERT_CACHE_POLICY_WTINYLFU_W10_SLRU_P80: return "wtinylfu-w10-slru-p80";
            case LLAMA_SILIANG_EXPERT_CACHE_POLICY_CUMULATIVE_LFU_ADMISSION: return "slfu";
        }
        return "invalid";
    }

    void print_route_stats() const {
        if (!params.route_stats || route_stats_data.routes == 0) {
            return;
        }
        const auto & stats = route_stats_data;
        const double denom = stats.selections == 0 ? 1.0 : static_cast<double>(stats.selections);
        const uint64_t gpu_total = stats.exec_gpu_k_hit + stats.exec_gpu_k_admit + stats.exec_gpu_r;
        uint64_t l2_admissions = 0;
        uint64_t l2_evictions = 0;
        uint64_t l2_rejections = 0;
        if (l2_enabled && cpu_query) {
            ggml_siliangem_cache_info info = {};
            info.struct_size = sizeof(info);
            if (cpu_query(cpu, &info)) {
                l2_admissions = info.policy_admissions;
                l2_evictions = info.policy_evictions;
                l2_rejections = info.policy_rejections;
            }
        }
        std::fprintf(
                stderr,
                "siliang_moe_runtime: route_stats routes=%" PRIu64 " exact_routes=%" PRIu64
                " unknown_routes=%" PRIu64 " selections=%" PRIu64
                " residency_L1=%" PRIu64 "(%.2f%%) residency_L2=%" PRIu64 "(%.2f%%)"
                " residency_uncached=%" PRIu64 "(%.2f%%) residency_unknown=%" PRIu64 "(%.2f%%)\n",
                stats.routes, stats.exact_routes, stats.unknown_routes, stats.selections,
                stats.state_l1, 100.0 * static_cast<double>(stats.state_l1) / denom,
                stats.state_l2, 100.0 * static_cast<double>(stats.state_l2) / denom,
                stats.state_uncached, 100.0 * static_cast<double>(stats.state_uncached) / denom,
                stats.state_unknown, 100.0 * static_cast<double>(stats.state_unknown) / denom);
        std::fprintf(
                stderr,
                "siliang_moe_runtime: route_stats execution GPU_K_hit=%" PRIu64
                " GPU_K_admit=%" PRIu64 " GPU_R_transient=%" PRIu64
                " GPU_total=%" PRIu64 " CPU=%" PRIu64 " unknown=%" PRIu64 "\n",
                stats.exec_gpu_k_hit, stats.exec_gpu_k_admit, stats.exec_gpu_r,
                gpu_total, stats.exec_cpu, stats.exec_unknown);
        std::fprintf(
                stderr,
                "siliang_moe_runtime: route_stats l2 admissions=%" PRIu64
                " evictions=%" PRIu64 " rejections=%" PRIu64 "\n",
                l2_admissions, l2_evictions, l2_rejections);
        if (params.l1_policy == LLAMA_SILIANG_EXPERT_CACHE_POLICY_CUMULATIVE_LFU_ADMISSION) {
            std::fprintf(
                    stderr,
                    "siliang_moe_runtime: route_stats slfu admit_k_cold=%s demote_k_hot=%s"
                    " cold_bypass=%" PRIu64 " deferred_promotions=%" PRIu64
                    " transition_commits=%" PRIu64 " transition_cancels=%" PRIu64
                    " demotions=%" PRIu64 " demotion_failures=%" PRIu64
                    " D2H_ops=%" PRIu64 " D2H_bytes=%" PRIu64
                    " D2D_ops=%" PRIu64 " D2D_bytes=%" PRIu64
                    " reuse_L2=%" PRIu64 " reuse_cold=%" PRIu64
                    " reuse_pending=%" PRIu64 " reuse_unknown=%" PRIu64
                    " submit_ms=%.3f device_wait_ms=%.3f host_copy_ms=%.3f exposed_wait_ms=%.3f\n",
                    params.admit_k_cold ? "on" : "off", params.demote_k_hot ? "on" : "off",
                    metrics.slfu_cold_bypasses, metrics.k_deferred_promotions,
                    metrics.k_transition_commits, metrics.k_transition_cancels,
                    metrics.l2_demotions, metrics.l2_demotion_failures,
                    metrics.demotion_d2h_ops, metrics.demotion_d2h_bytes,
                    metrics.promotion_d2d_ops, metrics.promotion_d2d_bytes,
                    metrics.demotion_reuse_l2, metrics.demotion_reuse_cold,
                    metrics.demotion_reuse_pending, metrics.demotion_reuse_unknown,
                    static_cast<double>(metrics.transition_submit_ns) / 1e6,
                    static_cast<double>(metrics.transition_device_wait_ns) / 1e6,
                    static_cast<double>(metrics.transition_host_copy_ns) / 1e6,
                    static_cast<double>(metrics.transition_exposed_wait_ns) / 1e6);
            std::fprintf(
                    stderr,
                    "siliang_moe_runtime: route_stats demotion_reuse_rounds"
                    " L2=[1:%" PRIu64 ",2-4:%" PRIu64 ",5-8:%" PRIu64 ",9-16:%" PRIu64 ",17+:%" PRIu64 "]"
                    " cold=[1:%" PRIu64 ",2-4:%" PRIu64 ",5-8:%" PRIu64 ",9-16:%" PRIu64 ",17+:%" PRIu64 "]\n",
                    metrics.demotion_reuse_l2_round_buckets[0], metrics.demotion_reuse_l2_round_buckets[1],
                    metrics.demotion_reuse_l2_round_buckets[2], metrics.demotion_reuse_l2_round_buckets[3],
                    metrics.demotion_reuse_l2_round_buckets[4], metrics.demotion_reuse_cold_round_buckets[0],
                    metrics.demotion_reuse_cold_round_buckets[1], metrics.demotion_reuse_cold_round_buckets[2],
                    metrics.demotion_reuse_cold_round_buckets[3], metrics.demotion_reuse_cold_round_buckets[4]);
        }

        const size_t top_k = static_cast<size_t>(model_info.top_k);
        const size_t side = top_k + 1;
        for (size_t l1 = 0; l1 <= top_k; ++l1) {
            for (size_t l2 = 0; l2 + l1 <= top_k; ++l2) {
                const size_t index = l1 * side + l2;
                if (index >= stats.compositions.size() || stats.compositions[index] == 0) {
                    continue;
                }
                const size_t uncached = top_k - l1 - l2;
                const double route_share = stats.exact_routes == 0 ? 0.0 :
                    100.0 * static_cast<double>(stats.compositions[index]) /
                        static_cast<double>(stats.exact_routes);
                std::fprintf(
                        stderr,
                        "siliang_moe_runtime: route_stats residency_composition L1=%zu L2=%zu uncached=%zu"
                        " routes=%" PRIu64 " share=%.2f%%\n",
                        l1, l2, uncached, stats.compositions[index], route_share);
            }
        }
        for (size_t k_hit = 0; k_hit <= top_k; ++k_hit) {
            for (size_t k_admit = 0; k_admit + k_hit <= top_k; ++k_admit) {
                for (size_t r = 0; r + k_admit + k_hit <= top_k; ++r) {
                    const size_t index = (k_hit * side + k_admit) * side + r;
                    if (index >= stats.execution_compositions.size() || stats.execution_compositions[index] == 0) {
                        continue;
                    }
                    const size_t cpu = top_k - k_hit - k_admit - r;
                    const double route_share = stats.routes == 0 ? 0.0 :
                        100.0 * static_cast<double>(stats.execution_compositions[index]) /
                            static_cast<double>(stats.routes);
                    std::fprintf(
                            stderr,
                            "siliang_moe_runtime: route_stats execution_composition K_hit=%zu K_admit=%zu"
                            " R=%zu CPU=%zu routes=%" PRIu64 " share=%.2f%%\n",
                            k_hit, k_admit, r, cpu, stats.execution_compositions[index], route_share);
                }
            }
        }
        std::fflush(stderr);
    }

    void print_summary() {
        if (!model || (!bound && metrics.map_calls == 0 && failure.load(std::memory_order_acquire) == 0)) {
            return;
        }
        LLAMA_LOG_INFO(
                "siliang_moe_runtime: summary maps=%" PRIu64 " choices=%" PRIu64
                " K_hits=%" PRIu64 " K_misses=%" PRIu64 " admissions=%" PRIu64
                " evictions=%" PRIu64 " rejections=%" PRIu64 " K_fences=%" PRIu64 " R_experts=%" PRIu64
                " R_banks=%" PRIu64 " R_reuses=%" PRIu64 " P_banks=%" PRIu64
                " P_waits=%" PRIu64 " staged=%" PRIu64 " H2D_ops=%" PRIu64
                " H2D_bytes=%" PRIu64 " ready_events=%" PRIu64 " compute_waits=%" PRIu64
                " pending_recoveries=%" PRIu64
                " L2_async=%" PRIu64 " L2_sync=%" PRIu64 " L2_hits=%" PRIu64
                " L2_misses=%" PRIu64 " L2_waits=%" PRIu64 " L2_releases=%" PRIu64
                " L2_release_failures=%" PRIu64 " phase_invalidations=%" PRIu64
                " prefill_maps=%" PRIu64 " prefill_tokens=%" PRIu64
                " prefill_choices=%" PRIu64 " prefill_unique=%" PRIu64
                " prefill_unique_max=%" PRIu64 " prefill_K_hits=%" PRIu64
                " prefill_K_misses=%" PRIu64 " prefill_admissions=%" PRIu64
                " prefill_evictions=%" PRIu64 " prefill_P_waves=%" PRIu64
                " prefill_H2D_ops=%" PRIu64 " prefill_H2D_bytes=%" PRIu64
                " prefill_compute_waits=%" PRIu64 " failure=%d\n",
                metrics.map_calls, metrics.route_choices, metrics.k_hits, metrics.k_misses,
                metrics.k_admissions, metrics.k_evictions, metrics.k_rejections, metrics.k_reuse_fences,
                metrics.r_experts,
                metrics.r_bank_uses, metrics.r_bank_reuses, metrics.p_bank_uses, metrics.p_bank_reuse_waits,
                metrics.staged_bytes, metrics.h2d_ops, metrics.h2d_bytes, metrics.ready_events,
                metrics.compute_waits, metrics.pending_recoveries,
                metrics.l2_async_prepares, metrics.l2_sync_prepares,
                metrics.l2_hits, metrics.l2_misses, metrics.l2_waits, metrics.l2_releases,
                metrics.l2_release_failures, metrics.phase_invalidations,
                metrics.prefill_maps, metrics.prefill_tokens, metrics.prefill_choices,
                metrics.prefill_unique, metrics.prefill_unique_max, metrics.prefill_k_hits,
                metrics.prefill_k_misses, metrics.prefill_k_admissions, metrics.prefill_k_evictions,
                metrics.prefill_p_waves, metrics.prefill_h2d_ops, metrics.prefill_h2d_bytes,
                metrics.prefill_compute_waits, failure.load(std::memory_order_acquire));
        {
            const uint64_t decode_maps = metrics.map_calls - metrics.prefill_maps;
            const uint64_t staged = metrics.decode_staged_l2 + metrics.decode_staged_miss;
            LLAMA_LOG_INFO(
                    "siliang_moe_runtime: decode_host maps=%" PRIu64 " host_us_per_map=%.1f prepare_us_per_map=%.1f"
                    " staged_l2=%" PRIu64 " staged_miss=%" PRIu64 " host_us_per_staged=%.1f"
                    " hybrid=%d cpu_l2=%" PRIu64 " cpu_miss=%" PRIu64 " gpu_admissions=%" PRIu64
                    " gpu_bypasses=%" PRIu64 " gpu_anchors=%" PRIu64
                    " pinned_slots=%u direct_experts=%" PRIu64 " direct_fallbacks=%" PRIu64
                    " lease_returns=%" PRIu64 " deferred_releases=%" PRIu64 " deferred_release_failures=%" PRIu64
                    " staging_threads=%zu staging_copy_GBps=%.2f\n",
                    decode_maps,
                    decode_maps ? 1e-3 * static_cast<double>(metrics.decode_host_ns) / decode_maps : 0.0,
                    decode_maps ? 1e-3 * static_cast<double>(metrics.decode_prepare_ns) / decode_maps : 0.0,
                    metrics.decode_staged_l2, metrics.decode_staged_miss,
                    staged ? 1e-3 * static_cast<double>(metrics.decode_host_ns) / staged : 0.0,
                    params.hybrid ? 1 : 0, metrics.hybrid_cpu_l2, metrics.hybrid_cpu_miss,
                    metrics.hybrid_gpu_admissions, metrics.hybrid_gpu_bypasses, metrics.hybrid_gpu_anchors,
                    metrics.pinned_slots_armed, metrics.direct_experts, metrics.direct_fallbacks, metrics.lease_returns,
                    metrics.deferred_releases, metrics.deferred_release_failures, copier.threads(),
                    metrics.staging_copy_ns ? static_cast<double>(metrics.staging_copy_bytes) / metrics.staging_copy_ns
                                            : 0.0);
        }
        LLAMA_LOG_INFO("siliang_moe_runtime: copy_order checks=%" PRIu64 " pending=%" PRIu64 "\n",
                metrics.probe_checks, metrics.probe_pending);
        if (k_ext != 0) {
            LLAMA_LOG_INFO("siliang_moe_runtime: decode_k extra=%u active=%d clears=%" PRIu64 " extra_admissions=%" PRIu64
                           " decode_plan_max=%.1f MiB tail_start=%.1f MiB\n",
                    k_ext, k_ext_active ? 1 : 0, metrics.k_ext_clears, metrics.k_ext_admissions,
                    metrics.k_ext_plan_max / 1048576.0, k_ext_tail_head / 1048576.0);
        }
        if (params.verify) {
            LLAMA_LOG_INFO("siliang_moe_verify: entries=%" PRIu64 " windows=%" PRIu64 " mismatches=%" PRIu64
                           " skipped=%" PRIu64 "\n",
                    metrics.verify_entries, metrics.verify_windows, metrics.verify_mismatches, metrics.verify_skipped);
        }
        print_route_stats();
        if (metrics.prefill_bitmap_sweeps != 0 || metrics.prefill_bitmap_incomplete != 0) {
            const double coverage = metrics.prefill_bitmap_needed == 0 ? 0.0 :
                100.0 * static_cast<double>(metrics.prefill_bitmap_overlap) /
                    static_cast<double>(metrics.prefill_bitmap_needed);
            const double precision = metrics.prefill_bitmap_seeded == 0 ? 0.0 :
                100.0 * static_cast<double>(metrics.prefill_bitmap_overlap) /
                    static_cast<double>(metrics.prefill_bitmap_seeded);
            LLAMA_LOG_INFO(
                    "siliang_moe_runtime: prefill_bitmap current_epoch=%" PRIu64
                    " sweeps=%" PRIu64 " sweep_tokens=%" PRIu64
                    " pairs=%" PRIu64
                    " seeded=%" PRIu64 " needed=%" PRIu64 " overlap=%" PRIu64
                    " new=%" PRIu64 " unused=%" PRIu64 " coverage=%.1f%% precision=%.1f%%"
                    " resets=%" PRIu64 " incomplete=%" PRIu64 "\n",
                    prefill_bitmap_epoch, metrics.prefill_bitmap_sweeps,
                    metrics.prefill_bitmap_sweep_tokens,
                    metrics.prefill_bitmap_pairs,
                    metrics.prefill_bitmap_seeded, metrics.prefill_bitmap_needed,
                    metrics.prefill_bitmap_overlap, metrics.prefill_bitmap_new,
                    metrics.prefill_bitmap_unused, coverage, precision, metrics.prefill_bitmap_resets,
                    metrics.prefill_bitmap_incomplete);
            for (size_t layer = 0; layer < prefill_bitmap_layers.size(); ++layer) {
                const auto & layer_metrics = prefill_bitmap_layers[layer];
                if (layer_metrics.pairs == 0) {
                    continue;
                }
                const double layer_coverage = layer_metrics.needed == 0 ? 0.0 :
                    100.0 * static_cast<double>(layer_metrics.overlap) /
                        static_cast<double>(layer_metrics.needed);
                const double layer_precision = layer_metrics.seeded == 0 ? 0.0 :
                    100.0 * static_cast<double>(layer_metrics.overlap) /
                        static_cast<double>(layer_metrics.seeded);
                LLAMA_LOG_DEBUG(
                        "siliang_moe_runtime: prefill_bitmap layer=%zu pairs=%" PRIu64
                        " seeded=%" PRIu64 " needed=%" PRIu64 " overlap=%" PRIu64
                        " coverage=%.1f%% precision=%.1f%%\n",
                        layer, layer_metrics.pairs, layer_metrics.seeded, layer_metrics.needed,
                        layer_metrics.overlap, layer_coverage, layer_precision);
            }
        }
    }
};

siliang_moe_runtime * siliang_moe_runtime_create(
        llama_model * model,
        llama_context * ctx,
        const llama_siliang_expert_cache_params * params) {
    if (!model || !ctx || !params) {
        return nullptr;
    }
    auto * runtime = new (std::nothrow) siliang_moe_runtime;
    if (!runtime) {
        LLAMA_LOG_ERROR("siliang_moe_runtime: allocation failed\n");
        return nullptr;
    }
    runtime->model = model;
    runtime->ctx = ctx;
    runtime->params = *params;
    try {
        if (runtime->initialize()) {
            return runtime;
        }
    } catch (const std::exception & error) {
        LLAMA_LOG_ERROR("siliang_moe_runtime: initialization exception: %s\n", error.what());
    } catch (...) {
        LLAMA_LOG_ERROR("siliang_moe_runtime: initialization raised an unknown exception\n");
    }
    if (runtime) {
        delete runtime;
    }
    return nullptr;
}

void siliang_moe_runtime_free(siliang_moe_runtime * runtime) {
    delete runtime;
}

void siliang_moe_runtime_reset_prefill_trace(siliang_moe_runtime * runtime) {
    if (runtime) {
        runtime->reset_prefill_trace();
    }
}
