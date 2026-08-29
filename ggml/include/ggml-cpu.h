#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    // the compute plan that needs to be prepared for ggml_graph_compute()
    // since https://github.com/ggml-org/ggml/issues/287
    struct ggml_cplan {
        size_t    work_size; // size of work buffer, calculated by `ggml_graph_plan()`
        uint8_t * work_data; // work buffer, to be allocated by caller before calling to `ggml_graph_compute()`

        int n_threads;
        struct ggml_threadpool * threadpool;

        // abort ggml_graph_compute when true
        ggml_abort_callback abort_callback;
        void *              abort_callback_data;

        // use only reference implementations
        bool use_ref;

        // Opaque per-backend Siliang expert-cache state. The CPU backend sets
        // this on plans it creates; direct ggml_graph_plan() callers leave it
        // NULL and retain the stock, cache-disabled behavior.
        struct ggml_siliangem_cache_state * siliangem_cache;
    };

    // numa strategies
    enum ggml_numa_strategy {
        GGML_NUMA_STRATEGY_DISABLED   = 0,
        GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
        GGML_NUMA_STRATEGY_ISOLATE    = 2,
        GGML_NUMA_STRATEGY_NUMACTL    = 3,
        GGML_NUMA_STRATEGY_MIRROR     = 4,
        GGML_NUMA_STRATEGY_COUNT
    };

    GGML_BACKEND_API void    ggml_numa_init(enum ggml_numa_strategy numa); // call once for better performance on NUMA systems
    GGML_BACKEND_API bool    ggml_is_numa(void); // true if init detected that system has >1 NUMA node

    GGML_BACKEND_API struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);
    GGML_BACKEND_API struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);

    GGML_BACKEND_API struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
    GGML_BACKEND_API struct ggml_tensor * ggml_set_f32 (struct ggml_tensor * tensor, float value);

    GGML_BACKEND_API int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
    GGML_BACKEND_API void    ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);

    GGML_BACKEND_API int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    GGML_BACKEND_API void    ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);

    GGML_BACKEND_API float   ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
    GGML_BACKEND_API void    ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);

    GGML_BACKEND_API float   ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    GGML_BACKEND_API void    ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);

    GGML_BACKEND_API struct ggml_threadpool *      ggml_threadpool_new           (struct ggml_threadpool_params  * params);
    GGML_BACKEND_API void                          ggml_threadpool_free          (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API int                           ggml_threadpool_get_n_threads (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API void                          ggml_threadpool_pause         (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API void                          ggml_threadpool_resume        (struct ggml_threadpool * threadpool);

    // ggml_graph_plan() has to be called before ggml_graph_compute()
    // when plan.work_size > 0, caller must allocate memory for plan.work_data
    GGML_BACKEND_API struct ggml_cplan ggml_graph_plan(
                  const struct ggml_cgraph * cgraph,
                                       int   n_threads, /* = GGML_DEFAULT_N_THREADS */
                    struct ggml_threadpool * threadpool /* = NULL */ );
    GGML_BACKEND_API enum ggml_status  ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);

    // same as ggml_graph_compute() but the work data is allocated as a part of the context
    // note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
    GGML_BACKEND_API enum ggml_status  ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);

    //
    // system info
    //

    // x86
    GGML_BACKEND_API int ggml_cpu_has_sse3       (void);
    GGML_BACKEND_API int ggml_cpu_has_ssse3      (void);
    // Siliang Engine managed host expert cache. All state belongs to one CPU
    // backend. Configuration is explicit and default-disabled; no operator
    // environment variables are read by this subsystem.
    enum ggml_siliangem_cache_policy {
        GGML_SILIANGEM_CACHE_POLICY_LRU = 0,
        GGML_SILIANGEM_CACHE_POLICY_LFU = 1,
        GGML_SILIANGEM_CACHE_POLICY_WTINYLFU_W10_SLRU_P80 = 2,
        GGML_SILIANGEM_CACHE_POLICY_SLFU = 3,
    };

    enum ggml_siliangem_expert_location {
        GGML_SILIANGEM_EXPERT_LOCATION_NONE = 0,
        GGML_SILIANGEM_EXPERT_LOCATION_RESIDENT = 1,
        GGML_SILIANGEM_EXPERT_LOCATION_TRANSIENT = 2,
    };

    enum ggml_siliangem_source_kind {
        GGML_SILIANGEM_SOURCE_NONE = 0,
        GGML_SILIANGEM_SOURCE_LEGACY_SLAB = 1,
        GGML_SILIANGEM_SOURCE_EXPERT_MAJOR = 2,
        GGML_SILIANGEM_SOURCE_SCATTERED = 3,
    };

    struct ggml_siliangem_cache_config {
        size_t struct_size;
        uint32_t enabled;
        uint32_t capacity_mib;
        enum ggml_siliangem_cache_policy policy;
        uint32_t deferred_io;
        uint32_t verbose;
        uint32_t memory_report;
        uint32_t mmap_prefetch;
    };

    // The configure call deep-copies this descriptor and every referenced
    // array. For EXPERT_MAJOR, base/stride have n_layers elements and
    // part_offset/part_bytes have n_layers*n_parts elements. For SCATTERED,
    // base/stride have n_layers*n_parts elements. LEGACY_SLAB uses only path.
    struct ggml_siliangem_source_desc {
        size_t struct_size;
        enum ggml_siliangem_source_kind kind;
        const char * path;
        uint32_t n_layers;
        uint32_t n_experts;
        uint32_t n_parts;
        const uint64_t * base;
        const uint32_t * stride;
        const uint32_t * part_offset;
        const uint32_t * part_bytes;
        const char * part_names;
    };

    struct ggml_siliangem_cache_info {
        size_t struct_size;
        uint32_t configured;
        uint32_t ready;
        enum ggml_siliangem_cache_policy policy;
        uint32_t capacity_slots;
        uint32_t occupied_slots;
        uint32_t pending_reads;
        uint32_t n_layers;
        uint32_t n_experts;
        uint32_t expert_bytes;
        uint32_t policy_window_slots;
        uint32_t policy_protected_slots;
        uint64_t hits;
        uint64_t misses;
        uint64_t bytes_read;
        uint64_t wait_calls;
        uint64_t wait_ns;
        uint64_t policy_admissions;
        uint64_t policy_evictions;
        uint64_t policy_rejections;
    };

    // Configure/reset only while the backend is idle. Graph plans retain the
    // backend-owned state object; cache state is never stored in a threadpool.
    GGML_BACKEND_API int ggml_backend_cpu_siliangem_configure(
            ggml_backend_t backend_cpu,
            const struct ggml_siliangem_cache_config * config,
            const struct ggml_siliangem_source_desc * source);
    GGML_BACKEND_API void ggml_backend_cpu_siliangem_reset(ggml_backend_t backend_cpu);
    GGML_BACKEND_API int ggml_backend_cpu_siliangem_query(
            ggml_backend_t backend_cpu, struct ggml_siliangem_cache_info * info);

    // K/P support. prepare_async returns expert ids ordered as resident hits
    // first, followed by submitted misses. wait_experts must complete before
    // copying a miss into a bounded pinned staging buffer.
    GGML_BACKEND_API int ggml_backend_cpu_siliangem_prepare_experts(
            ggml_backend_t backend_cpu, uint32_t layer,
            const int32_t * experts, uint32_t expert_count);
    GGML_BACKEND_API int ggml_backend_cpu_siliangem_prepare_experts_async(
            ggml_backend_t backend_cpu, uint32_t layer,
            const int32_t * experts, uint32_t expert_count,
            int32_t * order, uint32_t order_capacity,
            uint32_t * n_hits, uint32_t * n_misses, uint32_t * n_active);
    GGML_BACKEND_API int ggml_backend_cpu_siliangem_wait_experts(ggml_backend_t backend_cpu);
    GGML_BACKEND_API int ggml_backend_cpu_siliangem_copy_cached_part(
            ggml_backend_t backend_cpu, uint32_t layer, uint32_t expert, uint32_t part,
            void * destination, size_t destination_size);
    GGML_BACKEND_API int ggml_backend_cpu_siliangem_expert_location(
            ggml_backend_t backend_cpu, uint32_t layer, uint32_t expert);
    GGML_BACKEND_API int ggml_backend_cpu_siliangem_release_cached_expert(
            ggml_backend_t backend_cpu, uint32_t layer, uint32_t expert,
            uint32_t * released_slot);
    GGML_BACKEND_API int ggml_backend_cpu_siliangem_store_cached_expert_at_slot(
            ggml_backend_t backend_cpu, uint32_t layer, uint32_t expert, uint32_t slot,
            const void * const * parts, const size_t * part_sizes, uint32_t part_count);
    GGML_BACKEND_API int ggml_backend_cpu_siliangem_cache_occupancy(
            ggml_backend_t backend_cpu, uint32_t * capacity_slots, uint32_t * occupied_slots);

    GGML_BACKEND_API int ggml_cpu_has_avx        (void);
    GGML_BACKEND_API int ggml_cpu_has_avx_vnni   (void);
    GGML_BACKEND_API int ggml_cpu_has_avx2       (void);
    GGML_BACKEND_API int ggml_cpu_has_bmi2       (void);
    GGML_BACKEND_API int ggml_cpu_has_f16c       (void);
    GGML_BACKEND_API int ggml_cpu_has_fma        (void);
    GGML_BACKEND_API int ggml_cpu_has_avx512     (void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_vbmi(void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_vnni(void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_bf16(void);
    GGML_BACKEND_API int ggml_cpu_has_amx_int8   (void);
    // ARM
    GGML_BACKEND_API int ggml_cpu_has_neon       (void);
    GGML_BACKEND_API int ggml_cpu_has_arm_fma    (void);
    GGML_BACKEND_API int ggml_cpu_has_fp16_va    (void);
    GGML_BACKEND_API int ggml_cpu_has_dotprod    (void);
    GGML_BACKEND_API int ggml_cpu_has_matmul_int8(void);
    GGML_BACKEND_API int ggml_cpu_has_sve        (void);
    GGML_BACKEND_API int ggml_cpu_get_sve_cnt    (void);  // sve vector length in bytes
    GGML_BACKEND_API int ggml_cpu_has_sme        (void);
    GGML_BACKEND_API int ggml_cpu_has_sme2       (void);
    // other
    GGML_BACKEND_API int ggml_cpu_has_riscv_v    (void);
    GGML_BACKEND_API int ggml_cpu_get_rvv_vlen   (void);  // risc-v vector length in bytes
    GGML_BACKEND_API int ggml_cpu_has_vsx        (void);
    GGML_BACKEND_API int ggml_cpu_has_vxe        (void);
    GGML_BACKEND_API int ggml_cpu_has_wasm_simd  (void);
    GGML_BACKEND_API int ggml_cpu_has_llamafile  (void);

    // Internal types and functions exposed for tests and benchmarks

    typedef void (*ggml_vec_dot_t)  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT x, size_t bx,
                                       const void * GGML_RESTRICT y, size_t by, int nrc);

    struct ggml_type_traits_cpu {
        ggml_from_float_t        from_float;
        ggml_vec_dot_t           vec_dot;
        enum ggml_type           vec_dot_type;
        int64_t                  nrows; // number of rows to process simultaneously
    };

    GGML_BACKEND_API const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type);

    GGML_BACKEND_API void ggml_cpu_init(void);

    //
    // CPU backend
    //

    GGML_BACKEND_API ggml_backend_t ggml_backend_cpu_init(void);

    GGML_BACKEND_API bool ggml_backend_is_cpu                (ggml_backend_t backend);
    GGML_BACKEND_API void ggml_backend_cpu_set_n_threads     (ggml_backend_t backend_cpu, int n_threads);
    GGML_BACKEND_API void ggml_backend_cpu_set_threadpool    (ggml_backend_t backend_cpu, ggml_threadpool_t threadpool);
    GGML_BACKEND_API void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data);

    GGML_BACKEND_API void ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, bool use_ref);

    GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cpu_reg(void);

    GGML_BACKEND_API void ggml_cpu_fp32_to_fp32(const float *,       float *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp32_to_i32 (const float *,     int32_t *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp32_to_fp16(const float *, ggml_fp16_t *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp16_to_fp32(const ggml_fp16_t *, float *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp32_to_bf16(const float *, ggml_bf16_t *, int64_t);
    GGML_BACKEND_API void ggml_cpu_bf16_to_fp32(const ggml_bf16_t *, float *, int64_t);

#ifdef __cplusplus
}
#endif
