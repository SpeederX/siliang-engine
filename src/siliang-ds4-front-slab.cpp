#include "siliang-ds4-front-slab.h"

#include "ggml-cpp.h"
#include "ggml-cuda.h"
#include "llama-context.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstring>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

constexpr int k_ds4_layer_count = 43;
constexpr int k_front_bank_count = 2;

struct front_field {
    const char * name;
    ggml_tensor * llama_layer::* member;
};

const std::array<front_field, 20> k_front_fields = {{
    { "hc_attn_fn",         &llama_layer::hc_attn_fn },
    { "hc_attn_scale",      &llama_layer::hc_attn_scale },
    { "hc_attn_base",       &llama_layer::hc_attn_base },
    { "attn_norm",          &llama_layer::attn_norm },
    { "attn_q_a",           &llama_layer::wq_a },
    { "attn_q_a_norm",      &llama_layer::attn_q_a_norm },
    { "attn_q_b",           &llama_layer::wq_b },
    { "attn_kv",            &llama_layer::wkv },
    { "attn_kv_norm",       &llama_layer::attn_kv_norm },
    { "attn_sinks",         &llama_layer::attn_sinks },
    { "attn_comp_ape",      &llama_layer::attn_comp_ape },
    { "attn_comp_wgate",    &llama_layer::attn_comp_wgate },
    { "attn_comp_wkv",      &llama_layer::attn_comp_wkv },
    { "attn_comp_norm",     &llama_layer::attn_comp_norm },
    { "indexer_attn_q_b",   &llama_layer::indexer_attn_q_b },
    { "indexer_proj",       &llama_layer::indexer_proj },
    { "indexer_comp_ape",   &llama_layer::indexer_comp_ape },
    { "indexer_comp_wgate", &llama_layer::indexer_comp_wgate },
    { "indexer_comp_wkv",   &llama_layer::indexer_comp_wkv },
    { "indexer_comp_norm",  &llama_layer::indexer_comp_norm },
}};

bool checked_add(size_t lhs, size_t rhs, size_t & result) {
    if (lhs > std::numeric_limits<size_t>::max() - rhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool align_up(size_t value, size_t alignment, size_t & result) {
    if (alignment == 0) {
        return false;
    }
    const size_t remainder = value % alignment;
    if (remainder == 0) {
        result = value;
        return true;
    }
    return checked_add(value, alignment - remainder, result);
}

bool parse_marker_layer(const char * name, const char * prefix, int & layer) {
    if (name == nullptr || prefix == nullptr) {
        return false;
    }
    const size_t prefix_length = std::strlen(prefix);
    if (std::strncmp(name, prefix, prefix_length) != 0 || name[prefix_length] != '-') {
        return false;
    }

    const char * cursor = name + prefix_length + 1;
    if (*cursor == '\0') {
        return false;
    }
    int value = 0;
    for (; *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9') {
            return false;
        }
        value = value*10 + (*cursor - '0');
        if (value >= k_ds4_layer_count) {
            return false;
        }
    }
    layer = value;
    return true;
}

int find_marker_layer(const ggml_cgraph * graph, const char * prefix, int64_t & token_count) {
    if (graph == nullptr) {
        return -1;
    }
    ggml_cgraph * readable_graph = const_cast<ggml_cgraph *>(graph);
    int found = -1;
    const int node_count = ggml_graph_n_nodes(readable_graph);
    for (int index = 0; index < node_count; ++index) {
        const ggml_tensor * node = ggml_graph_node(readable_graph, index);
        int layer = -1;
        if (node == nullptr || !parse_marker_layer(node->name, prefix, layer)) {
            continue;
        }
        if (found >= 0 && found != layer) {
            return -2;
        }
        if (found >= 0 && token_count != node->ne[1]) {
            return -2;
        }
        found = layer;
        token_count = node->ne[1];
    }
    return found;
}

} // namespace

struct siliang_ds4_front_slab::impl {
    using stream_create_fn = decltype(&ggml_backend_cuda_siliang_stream_create);
    using stream_destroy_fn = decltype(&ggml_backend_cuda_siliang_stream_destroy);
    using stream_synchronize_fn = decltype(&ggml_backend_cuda_siliang_stream_synchronize);
    using event_create_fn = decltype(&ggml_backend_cuda_siliang_event_create);
    using event_destroy_fn = decltype(&ggml_backend_cuda_siliang_event_destroy);
    using event_synchronize_fn = decltype(&ggml_backend_cuda_siliang_event_synchronize);
    using event_record_fn = decltype(&ggml_backend_cuda_siliang_event_record);
    using main_event_record_fn = decltype(&ggml_backend_cuda_siliang_main_stream_event_record);
    using all_streams_event_record_fn = decltype(&ggml_backend_cuda_siliang_all_streams_event_record);
    using stream_wait_fn = decltype(&ggml_backend_cuda_siliang_stream_wait_event);
    using main_wait_fn = decltype(&ggml_backend_cuda_siliang_main_stream_wait_event);
    using h2d_fn = decltype(&ggml_backend_cuda_siliang_h2d_async);
    using host_register_fn = decltype(&ggml_backend_cuda_siliang_host_register_readonly);
    using host_unregister_fn = decltype(&ggml_backend_cuda_siliang_host_unregister);

    struct cuda_procs {
        stream_create_fn stream_create = nullptr;
        stream_destroy_fn stream_destroy = nullptr;
        stream_synchronize_fn stream_synchronize = nullptr;
        event_create_fn event_create = nullptr;
        event_destroy_fn event_destroy = nullptr;
        event_synchronize_fn event_synchronize = nullptr;
        event_record_fn event_record = nullptr;
        main_event_record_fn main_event_record = nullptr;
        all_streams_event_record_fn all_streams_event_record = nullptr;
        stream_wait_fn stream_wait = nullptr;
        main_wait_fn main_wait = nullptr;
        h2d_fn h2d = nullptr;
        host_register_fn host_register = nullptr;
        host_unregister_fn host_unregister = nullptr;

        bool complete() const {
            return stream_create && stream_destroy && stream_synchronize &&
                event_create && event_destroy && event_synchronize && event_record &&
                main_event_record && all_streams_event_record && stream_wait && main_wait && h2d &&
                host_register && host_unregister;
        }
    };

    struct segment {
        ggml_tensor * tensor = nullptr;
        const ggml_tensor * source = nullptr;
        size_t offset = 0;
        size_t alloc_size = 0;
        size_t payload_size = 0;
        uint64_t source_file_offset = 0;
        bool source_file_managed = false;
    };

    struct layer_plan {
        size_t store_offset = 0;
        size_t span = 0;
        size_t payload = 0;
        std::vector<segment> segments;
    };

    const llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    ggml_backend_t cuda_backend = nullptr;
    cuda_procs cuda = {};

    ggml_context_ptr metadata;
    ggml_backend_buffer_ptr gpu_buffer;
    unsigned char * host_store = nullptr;
    bool host_registered = false;
    ggml_tensor * carrier = nullptr;

    std::array<llama_layer, k_ds4_layer_count> alternate_layers = {};
    std::array<const void *, k_ds4_layer_count> alternate_layer_ptrs = {};
    std::array<layer_plan, k_ds4_layer_count> plans = {};

    ggml_backend_cuda_siliang_stream_t copy_stream = nullptr;
    std::array<ggml_backend_cuda_siliang_event_t, k_front_bank_count> copy_done = {};
    std::array<ggml_backend_cuda_siliang_event_t, k_front_bank_count> use_done = {};

    size_t alignment = 0;
    size_t bank_stride = 0;
    size_t host_store_size = 0;

    std::atomic<int32_t> failure_code { FAILURE_NONE };
    std::atomic<bool> prepared { false };
    std::atomic<bool> bound { false };
    std::atomic<bool> active { false };
    bool context_bound = false;
    std::array<bool, k_front_bank_count> copy_valid = {};
    std::array<bool, k_front_bank_count> use_done_recorded = {};
    std::array<int32_t, k_front_bank_count> bank_resident_layer = {{ -1, -1 }};
    bool graph_mode_known = false;
    bool graph_managed = false;
    bool graph_decode = false;
    int64_t graph_tokens = 0;
    int64_t max_graph_tokens = 1;
    int32_t waited_layer = -1;
    int32_t pending_layer = -1;
    std::atomic<int32_t> resident_layer { -1 };

    std::atomic<uint64_t> tokens { 0 };
    std::atomic<uint64_t> copies { 0 };
    std::atomic<uint64_t> waits { 0 };
    std::atomic<uint64_t> payload_h2d_bytes { 0 };
    std::atomic<uint64_t> wire_h2d_bytes { 0 };
    std::atomic<uint64_t> submission_host_ns { 0 };
    std::atomic<uint64_t> wait_enqueue_host_ns { 0 };
    std::atomic<bool> decode_activity_logged { false };
    std::atomic<bool> prefill_activity_logged { false };

    bool fail(int32_t code) {
        int32_t expected = FAILURE_NONE;
        failure_code.compare_exchange_strong(expected, code, std::memory_order_acq_rel);
        active.store(false, std::memory_order_release);
        return false;
    }

    static bool cuda_ok(ggml_backend_cuda_siliang_status status) {
        return status == GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS;
    }

#if defined(_WIN32)
    static bool read_file_range(HANDLE file, uint64_t offset, void * destination, size_t bytes) {
        if (file == INVALID_HANDLE_VALUE || destination == nullptr || bytes == 0 ||
            offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            return false;
        }
        LARGE_INTEGER position = {};
        position.QuadPart = static_cast<LONGLONG>(offset);
        if (!SetFilePointerEx(file, position, nullptr, FILE_BEGIN)) {
            return false;
        }
        auto * output = static_cast<uint8_t *>(destination);
        size_t completed = 0;
        while (completed < bytes) {
            const DWORD chunk = static_cast<DWORD>(std::min<size_t>(bytes - completed, 64u * 1024u * 1024u));
            DWORD read = 0;
            if (!ReadFile(file, output + completed, chunk, &read, nullptr) || read != chunk) {
                return false;
            }
            completed += read;
        }
        return true;
    }
#endif

    template <typename T>
    static void load_proc(ggml_backend_reg_t reg, const char * name, T & proc) {
        proc = reinterpret_cast<T>(ggml_backend_reg_get_proc_address(reg, name));
    }

    bool load_cuda_bridge(llama_context & target) {
        ggml_backend_sched_t sched = target.get_sched();
        if (sched == nullptr) {
            return fail(FAILURE_CUDA_BRIDGE);
        }

        const int backend_count = ggml_backend_sched_get_n_backends(sched);
        for (int index = 0; index < backend_count; ++index) {
            ggml_backend_t backend = ggml_backend_sched_get_backend(sched, index);
            ggml_backend_dev_t device = backend ? ggml_backend_get_device(backend) : nullptr;
            ggml_backend_reg_t reg = device ? ggml_backend_dev_backend_reg(device) : nullptr;
            if (reg == nullptr) {
                continue;
            }

            cuda_procs candidate = {};
            load_proc(reg, "ggml_backend_cuda_siliang_stream_create", candidate.stream_create);
            load_proc(reg, "ggml_backend_cuda_siliang_stream_destroy", candidate.stream_destroy);
            load_proc(reg, "ggml_backend_cuda_siliang_stream_synchronize", candidate.stream_synchronize);
            load_proc(reg, "ggml_backend_cuda_siliang_event_create", candidate.event_create);
            load_proc(reg, "ggml_backend_cuda_siliang_event_destroy", candidate.event_destroy);
            load_proc(reg, "ggml_backend_cuda_siliang_event_synchronize", candidate.event_synchronize);
            load_proc(reg, "ggml_backend_cuda_siliang_event_record", candidate.event_record);
            load_proc(reg, "ggml_backend_cuda_siliang_main_stream_event_record", candidate.main_event_record);
            load_proc(reg, "ggml_backend_cuda_siliang_all_streams_event_record", candidate.all_streams_event_record);
            load_proc(reg, "ggml_backend_cuda_siliang_stream_wait_event", candidate.stream_wait);
            load_proc(reg, "ggml_backend_cuda_siliang_main_stream_wait_event", candidate.main_wait);
            load_proc(reg, "ggml_backend_cuda_siliang_h2d_async", candidate.h2d);
            load_proc(reg, "ggml_backend_cuda_siliang_host_register_readonly", candidate.host_register);
            load_proc(reg, "ggml_backend_cuda_siliang_host_unregister", candidate.host_unregister);
            if (candidate.complete()) {
                cuda_backend = backend;
                cuda = candidate;
                return true;
            }
        }
        return fail(FAILURE_CUDA_BRIDGE);
    }

    bool build_storage() {
        ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(cuda_backend);
        if (gpu_buft == nullptr) {
            return fail(FAILURE_ALLOCATION);
        }
        alignment = ggml_backend_buft_get_alignment(gpu_buft);
        if (alignment == 0) {
            return fail(FAILURE_ALLOCATION);
        }

        size_t tensor_count = 0;
        for (int layer = 0; layer < k_ds4_layer_count; ++layer) {
            for (const front_field & field : k_front_fields) {
                const ggml_tensor * source = model->layers[layer].*(field.member);
                if (source == nullptr) {
                    continue;
                }
                const bool managed_source = source->buffer != nullptr &&
                    ggml_backend_buffer_is_siliang_managed(source->buffer);
                const bool contiguous = ggml_is_contiguous(source);
                const bool host_buffer = source->buffer != nullptr && ggml_backend_buffer_is_host(source->buffer);
                if (source->buffer == nullptr || !contiguous ||
                    (!managed_source && (source->data == nullptr || !host_buffer))) {
                    LLAMA_LOG_ERROR(
                        "siliang_ds4_front_slab: source placement reject layer=%d field=%s tensor=%s buffer=%s "
                        "managed=%d host=%d contiguous=%d data=%p\n",
                        layer, field.name, source->name,
                        source->buffer ? ggml_backend_buffer_name(source->buffer) : "<null>",
                        managed_source ? 1 : 0, host_buffer ? 1 : 0, contiguous ? 1 : 0, source->data);
                    return fail(FAILURE_FRONT_PLACEMENT);
                }
                if (managed_source) {
                    const auto * receipt = model->siliang_file_source.find(source->name);
                    if (receipt == nullptr || receipt->bytes != ggml_nbytes(source)) {
                        LLAMA_LOG_ERROR(
                            "siliang_ds4_front_slab: source receipt reject layer=%d field=%s tensor=%s receipt=%p "
                            "receipt_bytes=%" PRIu64 " tensor_bytes=%zu\n",
                            layer, field.name, source->name, static_cast<const void *>(receipt),
                            receipt ? receipt->bytes : 0, ggml_nbytes(source));
                        return fail(FAILURE_FRONT_PLACEMENT);
                    }
                }
                ++tensor_count;
            }
        }
        if (tensor_count == 0 || tensor_count > (std::numeric_limits<size_t>::max() - 1024*1024)/ggml_tensor_overhead()) {
            return fail(FAILURE_FRONT_PLACEMENT);
        }

        ggml_init_params init_params = {};
        init_params.mem_size = (tensor_count + 1)*ggml_tensor_overhead() + 1024*1024;
        init_params.no_alloc = true;
        metadata.reset(ggml_init(init_params));
        if (!metadata) {
            return fail(FAILURE_ALLOCATION);
        }

        for (int layer = 0; layer < k_ds4_layer_count; ++layer) {
            alternate_layers[layer] = model->layers[layer];
            layer_plan & plan = plans[layer];
            size_t cursor = 0;
            for (const front_field & field : k_front_fields) {
                const ggml_tensor * source = model->layers[layer].*(field.member);
                if (source == nullptr) {
                    continue;
                }

                size_t offset = 0;
                if (!align_up(cursor, alignment, offset)) {
                    return fail(FAILURE_ALLOCATION);
                }
                ggml_tensor * alias = ggml_dup_tensor(metadata.get(), source);
                if (alias == nullptr) {
                    return fail(FAILURE_ALLOCATION);
                }
                ggml_set_name(alias, source->name);
                if (std::strcmp(alias->name, source->name) != 0) {
                    return fail(FAILURE_FRONT_PLACEMENT);
                }
                const size_t alloc_size = ggml_backend_buft_get_alloc_size(gpu_buft, alias);
                const size_t payload_size = ggml_nbytes(source);
                if (alloc_size == 0 || payload_size == 0 || payload_size > alloc_size ||
                    !checked_add(offset, alloc_size, cursor) ||
                    !checked_add(plan.payload, payload_size, plan.payload)) {
                    return fail(FAILURE_ALLOCATION);
                }
                const bool managed_source = ggml_backend_buffer_is_siliang_managed(source->buffer);
                uint64_t source_file_offset = 0;
                if (managed_source) {
                    const auto * receipt = model->siliang_file_source.find(source->name);
                    if (receipt == nullptr || receipt->bytes != payload_size) {
                        return fail(FAILURE_FRONT_PLACEMENT);
                    }
                    source_file_offset = receipt->offset;
                }
                plan.segments.push_back({
                    /*.tensor              =*/ alias,
                    /*.source              =*/ source,
                    /*.offset              =*/ offset,
                    /*.alloc_size          =*/ alloc_size,
                    /*.payload_size        =*/ payload_size,
                    /*.source_file_offset  =*/ source_file_offset,
                    /*.source_file_managed =*/ managed_source,
                });
                alternate_layers[layer].*(field.member) = alias;
            }
            if (plan.segments.empty() || !align_up(cursor, alignment, plan.span)) {
                return fail(FAILURE_FRONT_PLACEMENT);
            }
            bank_stride = std::max(bank_stride, plan.span);
            alternate_layer_ptrs[layer] = &alternate_layers[layer];
        }
        if (bank_stride == 0 || bank_stride > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
            return fail(FAILURE_ALLOCATION);
        }

        size_t store_cursor = 0;
        for (layer_plan & plan : plans) {
            if (!align_up(store_cursor, alignment, plan.store_offset) ||
                !checked_add(plan.store_offset, plan.span, store_cursor)) {
                return fail(FAILURE_ALLOCATION);
            }
        }
        if (!align_up(store_cursor, alignment, host_store_size)) {
            return fail(FAILURE_ALLOCATION);
        }

#if defined(_WIN32)
        host_store = static_cast<unsigned char *>(VirtualAlloc(
                nullptr, host_store_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
        if (host_store == nullptr) {
            return fail(FAILURE_ALLOCATION);
        }
#else
        return fail(FAILURE_ALLOCATION);
#endif
        std::memset(host_store, 0, host_store_size);
#if defined(_WIN32)
        bool needs_model_file = false;
        for (const layer_plan & plan : plans) {
            for (const segment & item : plan.segments) {
                needs_model_file = needs_model_file || item.source_file_managed;
            }
        }
        HANDLE source_file = INVALID_HANDLE_VALUE;
        if (needs_model_file) {
            if (!model->siliang_file_source.valid()) {
                return fail(FAILURE_FRONT_PLACEMENT);
            }
            source_file = CreateFileA(
                model->siliang_file_source.path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
            if (source_file == INVALID_HANDLE_VALUE) {
                return fail(FAILURE_FRONT_PLACEMENT);
            }
        }
        bool source_ok = true;
        for (const layer_plan & plan : plans) {
            for (const segment & item : plan.segments) {
                void * destination = host_store + plan.store_offset + item.offset;
                if (item.source_file_managed) {
                    if (!read_file_range(source_file, item.source_file_offset, destination, item.payload_size)) {
                        source_ok = false;
                        break;
                    }
                } else {
                    std::memcpy(destination, item.source->data, item.payload_size);
                }
            }
            if (!source_ok) {
                break;
            }
        }
        if (source_file != INVALID_HANDLE_VALUE) {
            CloseHandle(source_file);
        }
        if (!source_ok) {
            return fail(FAILURE_FRONT_PLACEMENT);
        }
        if (needs_model_file) {
            LLAMA_LOG_INFO("siliang_ds4_front_slab: populated FRONT host store from model-owned GGUF receipt (%zu MiB)\n",
                    host_store_size / (1024 * 1024));
        }
#else
        return fail(FAILURE_ALLOCATION);
#endif
        if (!cuda_ok(cuda.host_register(cuda_backend, host_store, host_store_size))) {
            return fail(FAILURE_ALLOCATION);
        }
        host_registered = true;

        size_t gpu_bank_bytes = 0;
        if (!checked_add(bank_stride, bank_stride, gpu_bank_bytes) ||
            gpu_bank_bytes > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
            return fail(FAILURE_ALLOCATION);
        }
        carrier = ggml_new_tensor_1d(metadata.get(), GGML_TYPE_I8, static_cast<int64_t>(gpu_bank_bytes));
        if (carrier == nullptr) {
            return fail(FAILURE_ALLOCATION);
        }
        const size_t gpu_size = ggml_backend_buft_get_alloc_size(gpu_buft, carrier);
        if (gpu_size < gpu_bank_bytes) {
            return fail(FAILURE_ALLOCATION);
        }
        gpu_buffer.reset(ggml_backend_buft_alloc_buffer(gpu_buft, gpu_size));
        if (!gpu_buffer) {
            return fail(FAILURE_ALLOCATION);
        }
        ggml_backend_buffer_set_usage(gpu_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        auto * gpu_base = static_cast<unsigned char *>(ggml_backend_buffer_get_base(gpu_buffer.get()));
        if (gpu_base == nullptr ||
            ggml_backend_tensor_alloc(gpu_buffer.get(), carrier, gpu_base) != GGML_STATUS_SUCCESS) {
            return fail(FAILURE_ALLOCATION);
        }
        for (size_t layer = 0; layer < plans.size(); ++layer) {
            const size_t bank_offset = (layer % k_front_bank_count) * bank_stride;
            for (const segment & item : plans[layer].segments) {
                if (ggml_backend_tensor_alloc(
                        gpu_buffer.get(), item.tensor, gpu_base + bank_offset + item.offset) != GGML_STATUS_SUCCESS) {
                    return fail(FAILURE_ALLOCATION);
                }
            }
        }
        return true;
    }

    bool create_cuda_objects() {
        if (!cuda_ok(cuda.stream_create(cuda_backend, &copy_stream)) || copy_stream == nullptr) {
            return fail(FAILURE_CUDA_BRIDGE);
        }
        for (size_t bank = 0; bank < k_front_bank_count; ++bank) {
            if (!cuda_ok(cuda.event_create(cuda_backend, &copy_done[bank])) || copy_done[bank] == nullptr ||
                !cuda_ok(cuda.event_create(cuda_backend, &use_done[bank])) || use_done[bank] == nullptr) {
                return fail(FAILURE_CUDA_BRIDGE);
            }
        }
        return true;
    }

    void reset_metrics() {
        tokens.store(0, std::memory_order_relaxed);
        copies.store(0, std::memory_order_relaxed);
        waits.store(0, std::memory_order_relaxed);
        payload_h2d_bytes.store(0, std::memory_order_relaxed);
        wire_h2d_bytes.store(0, std::memory_order_relaxed);
        submission_host_ns.store(0, std::memory_order_relaxed);
        wait_enqueue_host_ns.store(0, std::memory_order_relaxed);
        decode_activity_logged.store(false, std::memory_order_relaxed);
        prefill_activity_logged.store(false, std::memory_order_relaxed);
    }

    bool preload_layer_zero() {
        constexpr size_t bank = 0;
        const layer_plan & plan = plans[0];
        if (host_store == nullptr || !cuda_ok(cuda.h2d(
                copy_stream, carrier, host_store + plan.store_offset, bank * bank_stride, plan.span)) ||
            !cuda_ok(cuda.event_record(copy_stream, copy_done[bank])) ||
            !cuda_ok(cuda.event_synchronize(copy_done[bank]))) {
            return fail(FAILURE_CUDA_RUNTIME);
        }
        copy_valid[bank] = true;
        use_done_recorded[bank] = false;
        bank_resident_layer[bank] = 0;
        resident_layer.store(0, std::memory_order_release);
        waited_layer = -1;
        pending_layer = -1;
        return true;
    }

    bool issue_layer(int32_t target_layer) {
        if (target_layer < 0 || target_layer >= k_ds4_layer_count) {
            return fail(FAILURE_MARKER_SEQUENCE);
        }
        const size_t bank = static_cast<size_t>(target_layer % k_front_bank_count);
        if (copy_valid[bank] && bank_resident_layer[bank] == target_layer) {
            pending_layer = -1;
            waited_layer = -1;
            return true;
        }
        const layer_plan & plan = plans[target_layer];
        const auto start = std::chrono::steady_clock::now();

        if (copy_valid[bank] &&
            (!use_done_recorded[bank] || !cuda_ok(cuda.stream_wait(copy_stream, use_done[bank])))) {
            return fail(FAILURE_CUDA_RUNTIME);
        }
        if (host_store == nullptr ||
            !cuda_ok(cuda.h2d(
                copy_stream, carrier, host_store + plan.store_offset, bank * bank_stride, plan.span)) ||
            !cuda_ok(cuda.event_record(copy_stream, copy_done[bank]))) {
            return fail(FAILURE_CUDA_RUNTIME);
        }
        const uint64_t elapsed = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start).count());
        submission_host_ns.fetch_add(elapsed, std::memory_order_relaxed);
        payload_h2d_bytes.fetch_add(plan.payload, std::memory_order_relaxed);
        wire_h2d_bytes.fetch_add(plan.span, std::memory_order_relaxed);
        copies.fetch_add(1, std::memory_order_relaxed);
        copy_valid[bank] = true;
        use_done_recorded[bank] = false;
        bank_resident_layer[bank] = target_layer;
        resident_layer.store(target_layer, std::memory_order_release);
        pending_layer = -1;
        waited_layer = -1;
        return true;
    }

    bool record_layer_use_done(int32_t layer) {
        if (layer < 0 || layer >= k_ds4_layer_count) {
            return fail(FAILURE_MARKER_SEQUENCE);
        }
        const size_t bank = static_cast<size_t>(layer % k_front_bank_count);
        if (!copy_valid[bank] || bank_resident_layer[bank] != layer ||
            !cuda_ok(cuda.all_streams_event_record(cuda_backend, use_done[bank]))) {
            return fail(FAILURE_CUDA_RUNTIME);
        }
        use_done_recorded[bank] = true;
        return true;
    }

    bool issue_pending_layer() {
        if (pending_layer < 0) {
            return true;
        }
        const int32_t target_layer = pending_layer;
        return issue_layer(target_layer);
    }

    bool wait_for_layer(int32_t layer) {
        if (layer < 0 || layer >= k_ds4_layer_count) {
            return fail(FAILURE_MARKER_SEQUENCE);
        }
        const size_t bank = static_cast<size_t>(layer % k_front_bank_count);
        if (!copy_valid[bank] || bank_resident_layer[bank] != layer) {
            return fail(FAILURE_MARKER_SEQUENCE);
        }
        const auto start = std::chrono::steady_clock::now();
        if (!cuda_ok(cuda.main_wait(cuda_backend, copy_done[bank]))) {
            return fail(FAILURE_CUDA_RUNTIME);
        }
        const uint64_t elapsed = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start).count());
        wait_enqueue_host_ns.fetch_add(elapsed, std::memory_order_relaxed);
        const uint64_t wait_count = waits.fetch_add(1, std::memory_order_relaxed) + 1;
        const uint64_t copy_count = copies.load(std::memory_order_relaxed);
        auto & logged = graph_decode ? decode_activity_logged : prefill_activity_logged;
        bool expected = false;
        if (copy_count > 0 && logged.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel)) {
            LLAMA_LOG_INFO(
                    "siliang_ds4_front_slab: serving %s tokens=%" PRId64
                    " copies=%" PRIu64 " waits=%" PRIu64 " H2D_bytes=%" PRIu64 " failure=0\n",
                    graph_decode ? "decode" : "prefill", graph_tokens,
                    copy_count, wait_count, wire_h2d_bytes.load(std::memory_order_relaxed));
        }
        waited_layer = layer;
        return true;
    }

    void observe(int split_index, ggml_backend_t backend, const ggml_cgraph * graph) {
        if (!active.load(std::memory_order_acquire) || failure_code.load(std::memory_order_acquire) != FAILURE_NONE) {
            return;
        }

        if (split_index == 0) {
            graph_mode_known = false;
            graph_managed = false;
            graph_decode = false;
            graph_tokens = 0;
            waited_layer = -1;
        }

        int64_t hc_tokens = -1;
        int64_t out_tokens = -1;
        const int hc_layer = find_marker_layer(graph, "hc_attn_pre", hc_tokens);
        const int out_layer = find_marker_layer(graph, "attn_out", out_tokens);
        if (hc_layer == -2 || out_layer == -2 ||
            (hc_layer >= 0 && out_layer >= 0 &&
             (hc_layer != out_layer || hc_tokens != out_tokens))) {
            fail(FAILURE_MARKER_TOPOLOGY);
            return;
        }

        const int64_t marker_tokens = hc_layer >= 0 ? hc_tokens : out_tokens;
        bool first_managed_marker = false;
        if (!graph_mode_known && (hc_layer >= 0 || out_layer >= 0)) {
            graph_mode_known = true;
            graph_tokens = marker_tokens;
            graph_decode = marker_tokens == 1;
            graph_managed = marker_tokens > 0 && marker_tokens <= max_graph_tokens;
            if (!graph_managed) {
                pending_layer = -1;
                waited_layer = -1;
                return;
            }
            if (hc_layer != 0) {
                fail(FAILURE_MARKER_SEQUENCE);
                return;
            }
            first_managed_marker = true;
        }
        if (!graph_mode_known || !graph_managed) {
            return;
        }
        if ((hc_layer >= 0 && hc_tokens != graph_tokens) ||
            (out_layer >= 0 && out_tokens != graph_tokens)) {
            fail(FAILURE_MARKER_TOPOLOGY);
            return;
        }
        if ((hc_layer >= 0 || out_layer >= 0) && backend != cuda_backend) {
            fail(FAILURE_MARKER_TOPOLOGY);
            return;
        }

        // A pending layer was marked in the preceding attn_out split. Layers
        // alternate between two physical banks, so layer N+2 can be copied into
        // N's bank after the all-stream completion event for N reaches the copy stream.
        if (first_managed_marker) {
            constexpr size_t first_bank = 0;
            if (!copy_valid[first_bank] || bank_resident_layer[first_bank] != 0 || pending_layer >= 0) {
                pending_layer = -1;
                if (!issue_layer(0)) {
                    return;
                }
            } else {
                pending_layer = -1;
                waited_layer = -1;
            }
        } else if (!issue_pending_layer()) {
            return;
        }
        if (hc_layer >= 0 && !wait_for_layer(hc_layer)) {
            return;
        }
        if (out_layer >= 0) {
            if (pending_layer >= 0 || waited_layer != out_layer) {
                fail(FAILURE_MARKER_SEQUENCE);
                return;
            }
            if (!record_layer_use_done(out_layer)) {
                return;
            }
            pending_layer = (out_layer + 1) % k_ds4_layer_count;
            if (out_layer + 1 == k_ds4_layer_count) {
                tokens.fetch_add(static_cast<uint64_t>(graph_tokens), std::memory_order_relaxed);
            }
        }
    }

    void release_binding() {
        active.store(false, std::memory_order_release);
        if (copy_stream && cuda.stream_synchronize) {
            (void) cuda.stream_synchronize(copy_stream);
        }
        if (cuda.event_destroy) {
            for (size_t bank = 0; bank < k_front_bank_count; ++bank) {
                if (copy_done[bank]) {
                    (void) cuda.event_destroy(copy_done[bank]);
                    copy_done[bank] = nullptr;
                }
                if (use_done[bank]) {
                    (void) cuda.event_destroy(use_done[bank]);
                    use_done[bank] = nullptr;
                }
            }
        }
        if (copy_stream && cuda.stream_destroy) {
            (void) cuda.stream_destroy(copy_stream);
        }
        copy_stream = nullptr;
        carrier = nullptr;
        if (host_registered && host_store != nullptr && cuda.host_unregister && cuda_backend) {
            (void) cuda.host_unregister(cuda_backend, host_store);
            host_registered = false;
        }
#if defined(_WIN32)
        if (host_store != nullptr) {
            VirtualFree(host_store, 0, MEM_RELEASE);
            host_store = nullptr;
        }
#endif
        gpu_buffer.reset();
        metadata.reset();
        for (layer_plan & plan : plans) {
            plan = {};
        }
        alternate_layer_ptrs.fill(nullptr);
        cuda = {};
        cuda_backend = nullptr;
        ctx = nullptr;
        alignment = 0;
        bank_stride = 0;
        host_store_size = 0;
        copy_valid.fill(false);
        use_done_recorded.fill(false);
        bank_resident_layer.fill(-1);
        waited_layer = -1;
        pending_layer = -1;
        resident_layer.store(-1, std::memory_order_release);
        graph_mode_known = false;
        graph_managed = false;
        graph_decode = false;
        graph_tokens = 0;
        max_graph_tokens = 1;
        bound.store(false, std::memory_order_release);
    }
};

siliang_ds4_front_slab::siliang_ds4_front_slab() : impl_(new impl) {
}

siliang_ds4_front_slab::~siliang_ds4_front_slab() {
    try {
        deactivate();
    } catch (...) {
    }
    impl_->release_binding();
}

bool siliang_ds4_front_slab::prepare(const llama_model & model) {
    if (impl_->prepared.load(std::memory_order_acquire) ||
        model.arch != LLM_ARCH_DEEPSEEK4 || model.hparams.n_layer() != k_ds4_layer_count ||
        model.hparams.n_layer_nextn != 0 || model.hparams.n_expert != 256 ||
        model.hparams.n_expert_used != 6 || model.layers.size() != k_ds4_layer_count) {
        return impl_->fail(FAILURE_INVALID_MODEL);
    }
    impl_->model = &model;
    impl_->prepared.store(true, std::memory_order_release);
    return true;
}

bool siliang_ds4_front_slab::bind(llama_context & ctx) {
    if (!impl_->prepared.load(std::memory_order_acquire) ||
        impl_->bound.load(std::memory_order_acquire) ||
        impl_->failure_code.load(std::memory_order_acquire) != FAILURE_NONE ||
        &ctx.get_model() != impl_->model || ctx.get_cparams().pipeline_parallel ||
        ctx.n_seq_max() != 1) {
        return impl_->fail(FAILURE_INVALID_STATE);
    }
    impl_->ctx = &ctx;
    impl_->max_graph_tokens = std::max<int64_t>(1, ctx.n_ubatch());
    if (!impl_->load_cuda_bridge(ctx) || !impl_->build_storage() || !impl_->create_cuda_objects()) {
        impl_->release_binding();
        return false;
    }
    impl_->bound.store(true, std::memory_order_release);
    return true;
}

bool siliang_ds4_front_slab::activate() {
    if (!impl_->bound.load(std::memory_order_acquire) || impl_->ctx == nullptr ||
        impl_->active.load(std::memory_order_acquire) || impl_->context_bound ||
        impl_->failure_code.load(std::memory_order_acquire) != FAILURE_NONE) {
        return impl_->fail(FAILURE_INVALID_STATE);
    }
    if (!impl_->preload_layer_zero()) {
        return false;
    }
    if (!llama_siliang_ds4_front_slab_bind(
            impl_->ctx, impl_->alternate_layer_ptrs.data(), impl_->alternate_layer_ptrs.size())) {
        return impl_->fail(FAILURE_CONTEXT_BIND);
    }
    impl_->context_bound = true;
    if (!impl_->ctx->siliang_ds4_front_slab_set_observer(&observer_thunk, this)) {
        llama_siliang_ds4_front_slab_clear(impl_->ctx);
        impl_->context_bound = false;
        return impl_->fail(FAILURE_CONTEXT_BIND);
    }
    impl_->reset_metrics();
    impl_->active.store(true, std::memory_order_release);
    return true;
}

void siliang_ds4_front_slab::deactivate() {
    impl_->active.store(false, std::memory_order_release);
    if (impl_->ctx == nullptr || !impl_->context_bound) {
        return;
    }
    (void) impl_->ctx->siliang_ds4_front_slab_set_observer(nullptr, nullptr);
    llama_siliang_ds4_front_slab_clear(impl_->ctx);
    impl_->context_bound = false;
    if (impl_->copy_stream && impl_->cuda.stream_synchronize) {
        (void) impl_->cuda.stream_synchronize(impl_->copy_stream);
    }
}

ggml_backend_sched_split_observer_callback siliang_ds4_front_slab::observer_callback() const noexcept {
    return &observer_thunk;
}

void * siliang_ds4_front_slab::observer_user_data() noexcept {
    return this;
}

siliang_ds4_front_slab::metrics siliang_ds4_front_slab::snapshot() const noexcept {
    metrics result;
    result.prepared = impl_->prepared.load(std::memory_order_acquire);
    result.bound = impl_->bound.load(std::memory_order_acquire);
    result.active = impl_->active.load(std::memory_order_acquire);
    result.bank_bytes = impl_->bank_stride;
    result.host_store_bytes = impl_->host_store_size;
    result.resident_layer = impl_->resident_layer.load(std::memory_order_acquire);
    result.tokens = impl_->tokens.load(std::memory_order_relaxed);
    result.copies = impl_->copies.load(std::memory_order_relaxed);
    result.waits = impl_->waits.load(std::memory_order_relaxed);
    result.payload_h2d_bytes = impl_->payload_h2d_bytes.load(std::memory_order_relaxed);
    result.wire_h2d_bytes = impl_->wire_h2d_bytes.load(std::memory_order_relaxed);
    result.submission_host_ns = impl_->submission_host_ns.load(std::memory_order_relaxed);
    result.wait_enqueue_host_ns = impl_->wait_enqueue_host_ns.load(std::memory_order_relaxed);
    return result;
}

int32_t siliang_ds4_front_slab::failure() const noexcept {
    return impl_->failure_code.load(std::memory_order_acquire);
}

const char * siliang_ds4_front_slab::failure_message() const noexcept {
    switch (failure()) {
        case FAILURE_NONE:            return "none";
        case FAILURE_INVALID_MODEL:   return "unsupported DeepSeek-V4 model geometry";
        case FAILURE_INVALID_STATE:   return "invalid FRONT slab lifecycle state";
        case FAILURE_CUDA_BRIDGE:     return "CUDA FRONT slab bridge unavailable";
        case FAILURE_FRONT_PLACEMENT: return "FRONT tensors are not fully host-backed and contiguous";
        case FAILURE_ALLOCATION:      return "FRONT slab allocation failed";
        case FAILURE_CONTEXT_BIND:    return "FRONT slab context binding failed";
        case FAILURE_MARKER_TOPOLOGY: return "FRONT slab marker topology mismatch";
        case FAILURE_MARKER_SEQUENCE: return "FRONT slab marker sequence mismatch";
        case FAILURE_CUDA_RUNTIME:    return "FRONT slab CUDA operation failed";
    }
    return "unknown FRONT slab failure";
}

bool siliang_ds4_front_slab::observer_thunk(
        int split_index,
        ggml_backend_t backend,
        const ggml_cgraph * graph,
        void * user_data) {
    if (user_data == nullptr) {
        return false;
    }
    auto * slab = static_cast<siliang_ds4_front_slab *>(user_data);
    slab->impl_->observe(split_index, backend, graph);
    return slab->impl_->failure_code.load(std::memory_order_acquire) == FAILURE_NONE;
}

struct llama_siliang_ds4_front_slab_runtime {
    siliang_ds4_front_slab slab;
};

llama_siliang_ds4_front_slab_runtime * llama_siliang_ds4_front_slab_runtime_create(
        const llama_model * model) {
    if (model == nullptr) {
        return nullptr;
    }
    auto * runtime = new (std::nothrow) llama_siliang_ds4_front_slab_runtime;
    if (runtime == nullptr || !runtime->slab.prepare(*model)) {
        delete runtime;
        return nullptr;
    }
    return runtime;
}

int llama_siliang_ds4_front_slab_runtime_bind(
        llama_siliang_ds4_front_slab_runtime * runtime,
        llama_context * ctx) {
    return runtime && ctx && runtime->slab.bind(*ctx) ? 1 : 0;
}

int llama_siliang_ds4_front_slab_runtime_activate(
        llama_siliang_ds4_front_slab_runtime * runtime) {
    return runtime && runtime->slab.activate() ? 1 : 0;
}

void llama_siliang_ds4_front_slab_runtime_deactivate(
        llama_siliang_ds4_front_slab_runtime * runtime) {
    if (runtime) {
        runtime->slab.deactivate();
    }
}

void llama_siliang_ds4_front_slab_runtime_free(
        llama_siliang_ds4_front_slab_runtime * runtime) {
    delete runtime;
}

int32_t llama_siliang_ds4_front_slab_runtime_failure(
        const llama_siliang_ds4_front_slab_runtime * runtime) {
    return runtime ? runtime->slab.failure() : siliang_ds4_front_slab::FAILURE_INVALID_STATE;
}

const char * llama_siliang_ds4_front_slab_runtime_failure_message(
        const llama_siliang_ds4_front_slab_runtime * runtime) {
    return runtime ? runtime->slab.failure_message() : "FRONT slab runtime is null";
}

int llama_siliang_ds4_front_slab_runtime_metrics(
        const llama_siliang_ds4_front_slab_runtime * runtime,
        llama_siliang_ds4_front_slab_metrics * metrics) {
    if (runtime == nullptr || metrics == nullptr) {
        return 0;
    }
    const siliang_ds4_front_slab::metrics source = runtime->slab.snapshot();
    metrics->prepared = source.prepared ? 1 : 0;
    metrics->bound = source.bound ? 1 : 0;
    metrics->active = source.active ? 1 : 0;
    metrics->bank_bytes = source.bank_bytes;
    metrics->host_store_bytes = source.host_store_bytes;
    metrics->resident_layer = source.resident_layer;
    metrics->tokens = source.tokens;
    metrics->copies = source.copies;
    metrics->waits = source.waits;
    metrics->payload_h2d_bytes = source.payload_h2d_bytes;
    metrics->wire_h2d_bytes = source.wire_h2d_bytes;
    metrics->submission_host_ns = source.submission_host_ns;
    metrics->wait_enqueue_host_ns = source.wait_enqueue_host_ns;
    return 1;
}
