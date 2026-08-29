#pragma once

#include "llama.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

#define LLAMA_MAX_SEQ 256

constexpr uint32_t LLAMA_SILIANG_MOE_PREFILL_MAX_EXPERTS = 256;

struct llama_siliang_moe_arena_state;

struct llama_siliang_moe_arena_map_call {
    llama_siliang_moe_arena_state * state = nullptr;
    int32_t layer = -1;
    uint64_t generation = 0;
};

struct llama_siliang_moe_arena_wait_call {
    llama_siliang_moe_arena_state * state = nullptr;
    int32_t layer = -1;
    uint64_t generation = 0;
};

struct llama_siliang_moe_arena_state {
    uint64_t generation = 0;
    std::vector<std::array<ggml_tensor *, LLAMA_SILIANG_MOE_ARENA_PART_ROLE_COUNT>> parts_by_layer;
    std::vector<uint8_t> managed_layers;
    std::vector<uint32_t> physical_slot_first_by_layer;
    std::vector<uint32_t> physical_slot_count_by_layer;
    std::vector<uint32_t> exchange_slot_first_by_layer;
    std::vector<uint32_t> exchange_slot_count_by_layer;
    uint32_t capacity = 0;
    int32_t expert_count = 0;
    int32_t top_k = 0;
    uint32_t prefill_ubatch_cap = 1;
    bool prefill_enabled = false;
    llama_siliang_moe_arena_slot_mapper mapper = nullptr;
    llama_siliang_moe_arena_failure_query failure_query = nullptr;
    llama_siliang_moe_arena_compute_wait_hook compute_wait_hook = nullptr;
    void * user_data = nullptr;
    void * compute_wait_user_data = nullptr;
    std::atomic<int32_t> failure_code {0};
    std::atomic<uint64_t> map_calls {0};
    std::atomic<bool> contract_failure_logged {false};
    std::vector<llama_siliang_moe_arena_map_call> map_calls_by_layer;
    std::vector<llama_siliang_moe_arena_wait_call> wait_calls_by_layer;
};

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    uint32_t n_outputs_max;   // max outputs supported by the context
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    int32_t  nextn_layer_offset = 0;

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool fused_lid;          // use fused lightning indexer
    bool auto_flid;
    bool fused_dsv4_hc_pre;
    bool fused_dsv4_hc_comb;
    bool fused_dsv4_hc_post;
    bool auto_fhc;
    bool no_perf;
    bool warmup;             // TODO: remove [TAG_LLAMA_GRAPH_NO_WARMUP]
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;

    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer

    enum llama_context_type ctx_type;
    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    llama_siliang_expert_cache_params expert_cache;

    llama_siliang_moe_arena_state * siliang_moe_arena_state;
    uint64_t siliang_moe_arena_generation;
    bool siliang_moe_arena_enabled;

    std::array<const void *, 43> siliang_ds4_front_slab_layers;
    uint64_t siliang_ds4_front_slab_generation;
    bool siliang_ds4_front_slab_enabled;

    llama_context * ctx_other;
};
