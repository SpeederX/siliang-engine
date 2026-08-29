#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;
using clock_type = std::chrono::steady_clock;

namespace {

constexpr int64_t k_embd = 4096;
constexpr int64_t k_ff = 2048;
constexpr size_t k_shared_part_bytes = 8912896;
constexpr size_t k_slot_bytes = 7096320;
constexpr size_t k_max_experts = 6;
constexpr size_t k_gate_bytes = 2162688;
constexpr size_t k_up_bytes = 2162688;
constexpr size_t k_down_bytes = 2752512;
constexpr size_t k_gate_offset = 0;
constexpr size_t k_up_offset = k_gate_bytes;
constexpr size_t k_down_offset = k_gate_bytes + k_up_bytes;

struct BackendDeleter { void operator()(ggml_backend_t p) const { if (p) ggml_backend_free(p); } };
struct BufferDeleter { void operator()(ggml_backend_buffer_t p) const { if (p) ggml_backend_buffer_free(p); } };
struct ContextDeleter { void operator()(ggml_context * p) const { if (p) ggml_free(p); } };
struct AllocDeleter { void operator()(ggml_gallocr_t p) const { if (p) ggml_gallocr_free(p); } };
using BackendPtr = std::unique_ptr<ggml_backend, BackendDeleter>;
using BufferPtr = std::unique_ptr<ggml_backend_buffer, BufferDeleter>;
using ContextPtr = std::unique_ptr<ggml_context, ContextDeleter>;
using AllocPtr = std::unique_ptr<ggml_gallocr, AllocDeleter>;

std::vector<uint8_t> read_bytes(const fs::path & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot read " + path.string());
    file.seekg(0, std::ios::end);
    const auto end = file.tellg();
    file.seekg(0);
    if (end < 0) throw std::runtime_error("cannot stat " + path.string());
    std::vector<uint8_t> out(static_cast<size_t>(end));
    if (!out.empty() && !file.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(out.size()))) {
        throw std::runtime_error("short read " + path.string());
    }
    return out;
}

ContextPtr make_context(size_t tensors = 128) {
    ggml_init_params params = {};
    params.mem_size = tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(256, false) + 16384;
    params.no_alloc = true;
    auto * ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("ggml_init failed");
    return ContextPtr(ctx);
}

double ms(clock_type::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() & 1 ? values[middle] : 0.5 * (values[middle - 1] + values[middle]);
}

struct SharedWeights {
    ContextPtr ctx;
    BufferPtr buffer;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * down = nullptr;

    static SharedWeights load(const fs::path & package, ggml_backend_buffer_type_t buft) {
        auto gate_bytes = read_bytes(package / "m03-shared-gate.bin");
        auto up_bytes = read_bytes(package / "m03-shared-up.bin");
        auto down_bytes = read_bytes(package / "m03-shared-down.bin");
        if (gate_bytes.size() != k_shared_part_bytes || up_bytes.size() != k_shared_part_bytes ||
            down_bytes.size() != k_shared_part_bytes) {
            throw std::runtime_error("shared expert package geometry mismatch");
        }

        SharedWeights result;
        result.ctx = make_context(16);
        result.buffer.reset(ggml_backend_buft_alloc_buffer(buft, 3 * k_shared_part_bytes));
        if (!result.buffer) throw std::runtime_error("shared weight buffer allocation failed");
        ggml_backend_buffer_set_usage(result.buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        auto * base = static_cast<char *>(ggml_backend_buffer_get_base(result.buffer.get()));
        if (!base) throw std::runtime_error("shared weight buffer has no base");

        result.gate = ggml_new_tensor_2d(result.ctx.get(), GGML_TYPE_Q8_0, k_embd, k_ff);
        result.up = ggml_new_tensor_2d(result.ctx.get(), GGML_TYPE_Q8_0, k_embd, k_ff);
        result.down = ggml_new_tensor_2d(result.ctx.get(), GGML_TYPE_Q8_0, k_ff, k_embd);
        if (ggml_backend_tensor_alloc(result.buffer.get(), result.gate, base) != GGML_STATUS_SUCCESS ||
            ggml_backend_tensor_alloc(result.buffer.get(), result.up, base + k_shared_part_bytes) != GGML_STATUS_SUCCESS ||
            ggml_backend_tensor_alloc(result.buffer.get(), result.down, base + 2 * k_shared_part_bytes) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("shared weight tensor bind failed");
        }
        // Match the frozen M03 reference helper exactly for CPU-host weights:
        // the quantized tensor payloads already use ggml's physical layout.
        if (ggml_backend_buft_is_host(buft)) {
            std::memcpy(base, gate_bytes.data(), gate_bytes.size());
            std::memcpy(base + k_shared_part_bytes, up_bytes.data(), up_bytes.size());
            std::memcpy(base + 2 * k_shared_part_bytes, down_bytes.data(), down_bytes.size());
        } else {
            ggml_backend_tensor_set(result.gate, gate_bytes.data(), 0, gate_bytes.size());
            ggml_backend_tensor_set(result.up, up_bytes.data(), 0, up_bytes.size());
            ggml_backend_tensor_set(result.down, down_bytes.data(), 0, down_bytes.size());
        }
        return result;
    }
};

struct SharedGraph {
    ContextPtr ctx;
    AllocPtr alloc;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;

    static SharedGraph build(ggml_backend_t backend, SharedWeights & weights, float clamp) {
        SharedGraph result;
        result.ctx = make_context(128);
        result.input = ggml_new_tensor_2d(result.ctx.get(), GGML_TYPE_F32, k_embd, 1);
        ggml_set_input(result.input);
        auto * up = ggml_mul_mat(result.ctx.get(), weights.up, result.input);
        auto * gate = ggml_mul_mat(result.ctx.get(), weights.gate, result.input);
        if (clamp > 1e-6f) {
            up = ggml_clamp(result.ctx.get(), up, -clamp, clamp);
            gate = ggml_clamp(result.ctx.get(), gate, -INFINITY, clamp);
        }
        auto * act = ggml_swiglu_split(result.ctx.get(), gate, up);
        result.output = ggml_mul_mat(result.ctx.get(), weights.down, act);
        ggml_set_output(result.output);
        result.graph = ggml_new_graph_custom(result.ctx.get(), 128, false);
        ggml_build_forward_expand(result.graph, result.output);
        result.alloc.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
        if (!result.alloc || !ggml_gallocr_alloc_graph(result.alloc.get(), result.graph)) {
            throw std::runtime_error("shared graph allocation failed");
        }
        return result;
    }
};

struct DeviceExpertArena {
    ContextPtr ctx;
    BufferPtr buffer;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * down = nullptr;

    static DeviceExpertArena create(ggml_backend_t cuda) {
        DeviceExpertArena result;
        result.ctx = make_context(16);
        result.gate = ggml_new_tensor_1d(result.ctx.get(), GGML_TYPE_I8, k_max_experts * k_gate_bytes);
        result.up = ggml_new_tensor_1d(result.ctx.get(), GGML_TYPE_I8, k_max_experts * k_up_bytes);
        result.down = ggml_new_tensor_1d(result.ctx.get(), GGML_TYPE_I8, k_max_experts * k_down_bytes);
        result.buffer.reset(ggml_backend_alloc_ctx_tensors(result.ctx.get(), cuda));
        if (!result.buffer) throw std::runtime_error("device expert arena allocation failed");
        ggml_backend_buffer_set_usage(result.buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        return result;
    }
};

void require_cuda(ggml_backend_cuda_siliang_status status, const char * operation) {
    if (status != GGML_BACKEND_CUDA_SILIANG_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with Siliang CUDA status " + std::to_string(status));
    }
}

struct CopyEngine {
    ggml_backend_cuda_siliang_stream_t stream = nullptr;
    ggml_backend_cuda_siliang_event_t done = nullptr;

    explicit CopyEngine(ggml_backend_t cuda) {
        require_cuda(ggml_backend_cuda_siliang_stream_create(cuda, &stream), "copy stream create");
        require_cuda(ggml_backend_cuda_siliang_event_create(cuda, &done), "copy event create");
    }
    ~CopyEngine() {
        if (stream) (void) ggml_backend_cuda_siliang_stream_synchronize(stream);
        if (done) (void) ggml_backend_cuda_siliang_event_destroy(done);
        if (stream) (void) ggml_backend_cuda_siliang_stream_destroy(stream);
    }

    void synchronize() {
        require_cuda(ggml_backend_cuda_siliang_stream_synchronize(stream), "copy stream synchronize");
    }

    void submit(DeviceExpertArena & dst, const uint8_t * src, int experts) {
        if (experts < 0 || experts > static_cast<int>(k_max_experts)) throw std::runtime_error("invalid expert count");
        for (int expert = 0; expert < experts; ++expert) {
            const size_t source_slot = static_cast<size_t>(expert) * k_slot_bytes;
            require_cuda(ggml_backend_cuda_siliang_h2d_async(
                    stream, dst.gate, src + source_slot + k_gate_offset,
                    static_cast<size_t>(expert) * k_gate_bytes, k_gate_bytes), "gate H2D");
            require_cuda(ggml_backend_cuda_siliang_h2d_async(
                    stream, dst.up, src + source_slot + k_up_offset,
                    static_cast<size_t>(expert) * k_up_bytes, k_up_bytes), "up H2D");
            require_cuda(ggml_backend_cuda_siliang_h2d_async(
                    stream, dst.down, src + source_slot + k_down_offset,
                    static_cast<size_t>(expert) * k_down_bytes, k_down_bytes), "down H2D");
        }
        require_cuda(ggml_backend_cuda_siliang_event_record(stream, done), "copy completion record");
    }

    void wait_done() {
        require_cuda(ggml_backend_cuda_siliang_event_synchronize(done), "copy completion wait");
    }
};

class CpuSharedWorker {
public:
    CpuSharedWorker(ggml_backend_t cpu, SharedGraph & graph, int affinity_cpu)
        : cpu_(cpu), graph_(graph), affinity_cpu_(affinity_cpu), thread_(&CpuSharedWorker::loop, this) {}

    ~CpuSharedWorker() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    uint64_t submit() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pending_) throw std::runtime_error("CPU shared worker already has a pending job");
        ++generation_;
        pending_ = true;
        done_ = false;
        cv_.notify_all();
        return generation_;
    }

    std::pair<clock_type::time_point, clock_type::time_point> wait(uint64_t generation) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&]() { return stop_ || (done_ && completed_generation_ == generation); });
        if (stop_) throw std::runtime_error("CPU shared worker stopped");
        return { start_, end_ };
    }

private:
    void loop() {
        if (affinity_cpu_ >= 0 && affinity_cpu_ < 64) {
            const DWORD_PTR mask = static_cast<DWORD_PTR>(1ull << affinity_cpu_);
            if (SetThreadAffinityMask(GetCurrentThread(), mask) == 0) {
                std::fprintf(stderr, "warning: failed to pin shared worker to logical CPU %d\n", affinity_cpu_);
            }
        }
        for (;;) {
            uint64_t generation = 0;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&]() { return stop_ || pending_; });
                if (stop_) return;
                generation = generation_;
                pending_ = false;
            }
            const auto start = clock_type::now();
            const ggml_status status = ggml_backend_graph_compute(cpu_, graph_.graph);
            const auto end = clock_type::now();
            if (status != GGML_STATUS_SUCCESS) {
                std::lock_guard<std::mutex> lock(mutex_);
                stop_ = true;
                cv_.notify_all();
                return;
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                start_ = start;
                end_ = end;
                completed_generation_ = generation;
                done_ = true;
            }
            cv_.notify_all();
        }
    }

    ggml_backend_t cpu_ = nullptr;
    SharedGraph & graph_;
    int affinity_cpu_ = -1;
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
    bool pending_ = false;
    bool done_ = false;
    uint64_t generation_ = 0;
    uint64_t completed_generation_ = 0;
    clock_type::time_point start_ = {};
    clock_type::time_point end_ = {};
};

struct Options {
    fs::path package;
    fs::path output;
    int repeats = 21;
    int warmups = 8;
    int cpu_threads = 1;
    int worker_cpu = -1;
};

Options parse_options(int argc, char ** argv) {
    Options result;
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::puts("ds4-shared-cpu-overlap-assay --package DIR --output-file FILE [--repeats N] [--warmups N] [--cpu-threads N] [--worker-cpu N]");
        std::exit(0);
    }
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::runtime_error("missing argument value");
        const std::string key = argv[i];
        const std::string value = argv[i + 1];
        if (key == "--package") result.package = fs::absolute(value);
        else if (key == "--output-file") result.output = fs::absolute(value);
        else if (key == "--repeats") result.repeats = std::stoi(value);
        else if (key == "--warmups") result.warmups = std::stoi(value);
        else if (key == "--cpu-threads") result.cpu_threads = std::stoi(value);
        else if (key == "--worker-cpu") result.worker_cpu = std::stoi(value);
        else throw std::runtime_error("unknown argument " + key);
    }
    if (result.package.empty() || result.output.empty() || result.repeats < 5 || result.warmups < 0 ||
        result.cpu_threads < 1 || result.cpu_threads > 12 || fs::exists(result.output)) {
        throw std::runtime_error("invalid assay request");
    }
    return result;
}

float max_abs_diff(const std::vector<uint8_t> & reference_bytes, ggml_tensor * output) {
    if (reference_bytes.size() != static_cast<size_t>(k_embd) * sizeof(float)) {
        throw std::runtime_error("shared reference output geometry mismatch");
    }
    std::array<float, k_embd> actual = {};
    ggml_backend_tensor_get(output, actual.data(), 0, sizeof(actual));
    const auto * reference = reinterpret_cast<const float *>(reference_bytes.data());
    float result = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) result = std::max(result, std::fabs(actual[i] - reference[i]));
    return result;
}

float max_abs_between(ggml_tensor * lhs, ggml_tensor * rhs) {
    std::array<float, k_embd> a = {};
    std::array<float, k_embd> b = {};
    ggml_backend_tensor_get(lhs, a.data(), 0, sizeof(a));
    ggml_backend_tensor_get(rhs, b.data(), 0, sizeof(b));
    float result = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) result = std::max(result, std::fabs(a[i] - b[i]));
    return result;
}

void verify_expert_copy(DeviceExpertArena & arena, const std::vector<uint8_t> & source) {
    std::vector<uint8_t> gate(k_max_experts * k_gate_bytes);
    std::vector<uint8_t> up(k_max_experts * k_up_bytes);
    std::vector<uint8_t> down(k_max_experts * k_down_bytes);
    ggml_backend_tensor_get(arena.gate, gate.data(), 0, gate.size());
    ggml_backend_tensor_get(arena.up, up.data(), 0, up.size());
    ggml_backend_tensor_get(arena.down, down.data(), 0, down.size());
    for (size_t expert = 0; expert < k_max_experts; ++expert) {
        const auto * slot = source.data() + expert * k_slot_bytes;
        if (std::memcmp(gate.data() + expert * k_gate_bytes, slot + k_gate_offset, k_gate_bytes) != 0 ||
            std::memcmp(up.data() + expert * k_up_bytes, slot + k_up_offset, k_up_bytes) != 0 ||
            std::memcmp(down.data() + expert * k_down_bytes, slot + k_down_offset, k_down_bytes) != 0) {
            throw std::runtime_error("expert H2D verification mismatch");
        }
    }
}

} // namespace

int main(int argc, char ** argv) try {
    const Options options = parse_options(argc, argv);
    ggml_backend_load_all();

    BackendPtr cuda(ggml_backend_cuda_init(0));
    BackendPtr cpu(ggml_backend_cpu_init());
    if (!cuda || !cpu) throw std::runtime_error("backend initialization failed");
    ggml_backend_cpu_set_n_threads(cpu.get(), options.cpu_threads);

    const auto manifest_bytes = read_bytes(options.package / "m03-work-unit-package.json");
    const json manifest = json::parse(manifest_bytes.begin(), manifest_bytes.end());
    const float shared_clamp = manifest.at("shared_expert").at("clamp").get<float>();
    const auto activation = read_bytes(options.package / "m03-ffn-input-f32le.bin");
    const auto expert_bytes = read_bytes(options.package / "m03-selected-experts.bin");
    const auto shared_reference = read_bytes(options.package / "m03-shared-out-f32le.bin");
    if (activation.size() != static_cast<size_t>(k_embd) * sizeof(float) ||
        expert_bytes.size() != k_max_experts * k_slot_bytes) {
        throw std::runtime_error("M03 package geometry mismatch");
    }

    auto cpu_weights = SharedWeights::load(options.package, ggml_backend_cpu_buffer_type());
    auto cpu_graph = SharedGraph::build(cpu.get(), cpu_weights, shared_clamp);
    ggml_backend_tensor_set(cpu_graph.input, activation.data(), 0, activation.size());

    auto gpu_weights = SharedWeights::load(options.package, ggml_backend_get_default_buffer_type(cuda.get()));
    auto gpu_graph = SharedGraph::build(cuda.get(), gpu_weights, shared_clamp);
    ggml_backend_tensor_set(gpu_graph.input, activation.data(), 0, activation.size());
    if (ggml_backend_graph_compute(cuda.get(), gpu_graph.graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("current CUDA shared reference compute failed");
    }
    ggml_backend_synchronize(cuda.get());

    DeviceExpertArena device_arena = DeviceExpertArena::create(cuda.get());
    ggml_backend_buffer_type_t pinned_type = ggml_backend_cuda_host_buffer_type();
    if (!pinned_type) throw std::runtime_error("CUDA pinned host buffer type unavailable");
    BufferPtr pinned(ggml_backend_buft_alloc_buffer(pinned_type, expert_bytes.size()));
    if (!pinned) throw std::runtime_error("pinned source allocation failed");
    auto * pinned_base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(pinned.get()));
    if (!pinned_base) throw std::runtime_error("pinned source has no base");
    std::memcpy(pinned_base, expert_bytes.data(), expert_bytes.size());

    CopyEngine copy(cuda.get());
    copy.submit(device_arena, pinned_base, 6);
    copy.wait_done();
    verify_expert_copy(device_arena, expert_bytes);

    CpuSharedWorker worker(cpu.get(), cpu_graph, options.worker_cpu);
    for (int i = 0; i < options.warmups; ++i) {
        const uint64_t generation = worker.submit();
        (void) worker.wait(generation);
    }
    const uint64_t correctness_generation = worker.submit();
    (void) worker.wait(correctness_generation);
    const float shared_max_abs = max_abs_diff(shared_reference, cpu_graph.output);
    const float shared_cuda_frozen_max_abs = max_abs_diff(shared_reference, gpu_graph.output);
    const float shared_current_cpu_cuda_max_abs = max_abs_between(cpu_graph.output, gpu_graph.output);

    json cells = json::array();
    for (int experts = 0; experts <= 6; ++experts) {
        std::vector<double> h2d_only;
        std::vector<double> shared_only;
        std::vector<double> concurrent_h2d_completion;
        std::vector<double> concurrent_shared;
        std::vector<double> concurrent_tail;
        std::vector<double> concurrent_critical;
        std::vector<double> enqueue_wall;

        for (int rep = 0; rep < options.repeats; ++rep) {
            if (experts > 0) {
                copy.synchronize();
                const auto start = clock_type::now();
                copy.submit(device_arena, pinned_base, experts);
                copy.wait_done();
                h2d_only.push_back(ms(clock_type::now() - start));
            } else {
                h2d_only.push_back(0.0);
            }

            const auto shared_start = clock_type::now();
            const uint64_t shared_generation = worker.submit();
            const auto shared_interval = worker.wait(shared_generation);
            (void) shared_start;
            shared_only.push_back(ms(shared_interval.second - shared_interval.first));

            copy.synchronize();
            const auto wall_start = clock_type::now();
            auto enqueue_end = wall_start;
            if (experts > 0) {
                copy.submit(device_arena, pinned_base, experts);
                enqueue_end = clock_type::now();
            }
            const uint64_t generation = worker.submit();

            clock_type::time_point copy_done = wall_start;
            if (experts > 0) {
                copy.wait_done();
                copy_done = clock_type::now();
            }
            const auto cpu_interval = worker.wait(generation);
            const auto wall_end = std::max(copy_done, cpu_interval.second);

            enqueue_wall.push_back(ms(enqueue_end - wall_start));
            concurrent_h2d_completion.push_back(ms(copy_done - wall_start));
            concurrent_shared.push_back(ms(cpu_interval.second - cpu_interval.first));
            concurrent_tail.push_back(copy_done > cpu_interval.second ? ms(copy_done - cpu_interval.second) : 0.0);
            concurrent_critical.push_back(ms(wall_end - wall_start));
        }

        const double h2d = median(h2d_only);
        const double cpu_only = median(shared_only);
        const double h2d_concurrent = median(concurrent_h2d_completion);
        const double cpu_concurrent = median(concurrent_shared);
        cells.push_back({
            {"experts_moved", experts},
            {"h2d_bytes", static_cast<uint64_t>(experts) * k_slot_bytes},
            {"h2d_only_wall_ms_median", h2d},
            {"shared_cpu_only_ms_median", cpu_only},
            {"copy_enqueue_cpu_ms_median", median(enqueue_wall)},
            {"concurrent_h2d_completion_ms_median", h2d_concurrent},
            {"concurrent_shared_cpu_ms_median", cpu_concurrent},
            {"concurrent_tail_after_shared_ms_median", median(concurrent_tail)},
            {"concurrent_critical_wall_ms_median", median(concurrent_critical)},
            {"h2d_slowdown_ratio", h2d > 0.0 ? h2d_concurrent / h2d : 1.0},
            {"shared_slowdown_ratio", cpu_only > 0.0 ? cpu_concurrent / cpu_only : 1.0},
        });
    }

    json result = {
        {"schema", "siliang-v013-ds4-shared-cpu-overlap-v1"},
        {"complete", true},
        {"scope", "isolated-equal-work-shared-cpu-vs-production-style-expert-h2d"},
        {"base_checkpoint", "738c1804a88e9f99742714d7fbc354ecf7e0b279"},
        {"repeats", options.repeats},
        {"warmups", options.warmups},
        {"cpu_threads", options.cpu_threads},
        {"worker_logical_cpu", options.worker_cpu},
        {"source", "cudaMallocHost-pinned-hot-prepositioned"},
        {"copy_style", "production-siliang-private-stream-3-h2d-ops-per-expert"},
        {"expert_slot_bytes", k_slot_bytes},
        {"shared_expert_payload_bytes", 3 * k_shared_part_bytes},
        {"correctness", {
            {"shared_cpu_vs_frozen_max_abs", shared_max_abs},
            {"shared_cuda_vs_frozen_max_abs", shared_cuda_frozen_max_abs},
            {"shared_current_cpu_vs_cuda_max_abs", shared_current_cpu_cuda_max_abs},
            {"expert_h2d_exact_bytes", true}
        }},
        {"cells", cells},
    };

    std::ofstream output(options.output, std::ios::binary);
    if (!output) throw std::runtime_error("cannot create output file");
    output << result.dump(2) << "\n";
    std::fprintf(stderr, "ds4-shared-cpu-overlap-assay: complete: %s\n", options.output.string().c_str());
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "ds4-shared-cpu-overlap-assay: fatal: %s\n", error.what());
    return 2;
}
