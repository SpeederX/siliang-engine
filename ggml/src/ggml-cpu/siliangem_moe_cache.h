// Siliang Engine: managed out-of-core MoE expert cache.
//
// Replaces where routed-expert bytes come from, and nothing else. llama.cpp's
// graph, kernels, attention, router and tokenizer are untouched; the only hook
// is the pointer `ggml_compute_forward_mul_mat_id` uses to locate expert
// `cur_a`. Everything below is behind that one substitution, and falls back to
// the mmap pointer whenever it cannot serve a request.
//
// Why it exists: mmap page faults can serialize expert reads at low queue depth.
// Explicit unbuffered, overlapped reads can expose storage parallelism directly;
// a PrefetchVirtualMemory hint is not equivalent to queued asynchronous I/O.
//
// So this owns the read path: unbuffered (no page cache), overlapped (real
// queue depth), against an expert-major layout where one expert's projections
// are contiguous instead of scattered across separate tensor regions.
//
// Cache granularity is the WHOLE expert, not the individual matrix. The gate
// matmul faults in all three parts as one contiguous read; the up and down
// matmuls that follow are then guaranteed hits. That is the entire reason the
// repack is expert-major.
//
// Configuration and source geometry are supplied explicitly through the
// backend-scoped API in ggml-cpu.h. The cache remains disabled unless that API
// successfully configures a particular CPU backend.
#pragma once

#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>      /* GetPerformanceInfo - the file-cache size, which is
                         * the margin that actually protects the arena */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SILIANGEM_MAGIC        0x424D5845u   /* "EXMB" */
#define SILIANGEM_HEADER_BYTES 4096u
/* Upper bound on ACTIVE experts in one mul_mat_id call.
 *
 * A bound based only on n_expert_used is valid for a small decode step, but
 * during prefill the active set is the union over the whole ubatch and can be
 * much larger. Active sets larger than this storage bound are handled by the
 * explicit capacity check in siliangem_prepare_async(), which declines rather than
 * overruns. */
#define SILIANGEM_MAX_BATCH    256
#define SILIANGEM_EMPTY        0xFFFFFFFFu

/* Routed experts per token, with headroom so a larger top-k does not silently
 * truncate the reuse probe. */
#define SILIANGEM_TOPK_MAX     8

#define SILIANGEM_POLICY_LRU                         0
#define SILIANGEM_POLICY_LFU                         1
#define SILIANGEM_POLICY_WTINYLFU_W10_SLRU_P80      2
#define SILIANGEM_POLICY_SLFU                         3
#define SILIANGEM_BYPASS                              0xFFFFFFFEu
#define SILIANGEM_SEGMENT_NONE                       0
#define SILIANGEM_SEGMENT_WINDOW                     1
#define SILIANGEM_SEGMENT_PROBATION                  2
#define SILIANGEM_SEGMENT_PROTECTED                  3

typedef struct {
    uint32_t key;          /* layer<<16 | expert, or SILIANGEM_EMPTY */
    uint64_t stamp;        /* LRU clock */
    uint64_t frequency;    /* LFU count, with LRU stamp as the tie break */
    uint32_t leases;       /* reserved for bounded asynchronous consumers */
    uint8_t  segment;      /* W-TinyLFU window / SLRU main segment */
} siliangem_slot;

typedef struct {
    int      ready;        /* 1 armed, 0 disabled (fall back to mmap) */
    HANDLE   file;
    uint32_t n_layers, n_experts;
    uint32_t part_bytes[3];      /* gate, up, down - slab order */
    uint32_t part_off[3];        /* byte offset of each part inside an expert */
    uint32_t expert_bytes;

    /* ---- expert-major GGUF source ----------------------------------------
     * The slab format assumed ONE geometry for the whole model: a global
     * expert stride and one part layout. An expert-major GGUF does not work
     * that way -- Q4_K_M varies the down-projection type per layer, so both
     * the stride and the part sizes differ layer to layer.
     *
     * When the model loader publishes a source (ggml_siliangem_set_expert_
     * source), these per-layer tables replace the slab header and the file
     * read is the MODEL ITSELF. No sidecar, no duplicated expert bytes.
     *
     * em_base[L]      absolute file offset of layer L's packed region
     * em_stride[L]    bytes between consecutive experts in that region
     * em_poff[L*np+p] byte offset of part p inside one expert
     * em_pbytes[..]   logical size of part p for one expert
     */
    int       em;                /* 1 when reading an expert-major GGUF */
    int       em_scattered;      /* 1 when the geometry came from a STOCK GGUF:
                                  * same slot layout, but nb[2] is a PART stride
                                  * rather than the whole expert */
    int       em_nparts;
    uint64_t *em_base;
    uint32_t *em_stride;
    uint32_t *em_poff;
    uint32_t *em_pbytes;

    uint8_t *arena;              /* nslots * expert_bytes, sector aligned */
    siliangem_slot *slots;
    uint32_t nslots;

    /* SLFU rejected candidates still need bytes for the current request. This
     * is bounded scratch, never persistent cache residency. It follows the
     * backend request bound (SILIANGEM_MAX_BATCH), not the old top-k telemetry cap. */
    uint8_t *transient_arena;
    uint32_t transient_capacity;
    uint32_t transient_count;
    uint32_t transient_keys[SILIANGEM_MAX_BATCH];

    uint32_t *table;             /* open-addressed key -> slot, capacity mask+1 */
    uint32_t  mask;

    uint64_t clock, hits, misses, bytes_read;
    uint64_t expert_requests, expert_hits;
    uint64_t last_report;        /* lookups at the last periodic stats emit */
    int      placement_policy;
    uint32_t policy_window_slots;
    uint32_t policy_protected_slots;
    uint64_t policy_admissions;
    uint64_t policy_evictions;
    uint64_t policy_main_candidate_rejections;
    int      verbose;
    int      mem_report;         /* emit the siliangem[mem] residency line */

    /* In-flight reads, deferred so cached experts can be computed while the
     * misses are still landing. Only ever touched from ith==0, between
     * barriers, so no locking is needed. */
    OVERLAPPED pend_ov[SILIANGEM_MAX_BATCH];
    HANDLE     pend_ev[SILIANGEM_MAX_BATCH];
    uint32_t   pend_slot[SILIANGEM_MAX_BATCH];
    uint32_t   pend_transient[SILIANGEM_MAX_BATCH];
    int        n_pending;

    /* Distribution of cne1 -- how many tokens routed to each expert.
     * llama.cpp takes a fast path when cne1 == 1:
     *     chunk_size = 16; if (nr0 == 1 || nr1 == 1) chunk_size = 64;
     * As an ubatch grows, more tokens can route to the same expert and move
     * work out of that path. Record the histogram instead of assuming a
     * model-independent transition point. */
    uint64_t cne1_hist[8];   /* 1, 2, 3, 4, 5-8, 9-16, 17-32, 33+ */

    /* Deferred wait on/off, supplied by the context-owned typed config.
     *
     * Exists so the deferred wait can be A/B'd within ONE binary. Comparing
     * two builds would confound it with every other difference between them,
     * and comparing cache-on against cache-off measures the whole arena, not
     * this change. */
    int defer;

    /* Time attribution for the read path. Separate wait time from submission
     * time so storage transfer, request overhead, and synchronization are not
     * conflated. Interpret the counters only with target-system measurements.
     *
     * wait_ns  time blocked in siliangem_wait/siliangem_prepare waiting on I/O
     * fetch_ns time issuing reads (submission cost, not completion)
     * QPC because it is monotonic and cheap; ith==0 only, so no atomics. */
    int64_t  qpc_freq;
    int64_t  t0;
    uint64_t wait_ns;
    uint64_t fetch_ns;
    uint64_t wait_calls;

    /* Sub-split of fetch_ns. Several mechanisms can contribute to submission
     * cost, so record them separately before optimizing:
     *   evict_ns  victim selection scans all slots and siliangem_table_rebuild() rebuilds
     *             the whole table on every eviction -- both O(nslots), fixable
     *             with an intrusive LRU list and a table that updates in place
     *   event_ns  CreateEventA per read is a kernel object creation per miss;
     *             fixable by allocating the events once and calling ResetEvent
     *   read_ns   ReadFile itself. Unbuffered overlapped reads must probe and
     *             lock destination pages at submission. Large pages are a
     *             distinct, privileged experiment rather than an assumption. */
    uint64_t evict_ns;
    uint64_t event_ns;
    uint64_t read_ns;

    /* A slow ReadFile submission has two possible explanations with different
     * remedies:
     *   page locking      -- the kernel probes and locks destination pages;
     *                        large pages would require SeLockMemoryPrivilege.
     *   sync completion   -- the I/O finishes inside ReadFile, which returns
     *                        TRUE instead of ERROR_IO_PENDING. Then it is real
     *                        transfer time, not overhead, and large pages do
     *                        nothing; the fix is making reads actually async.
     * n_sync counts ReadFile returning TRUE, which settles it outright. */
    uint64_t n_sync;
    uint64_t n_pend;

    /* ---- arm C: is cross-layer expert prefetch viable at all? -------------
     * It requires knowing layer L+1's experts while layer L computes, but L+1's
     * router consumes L's output, so exact IDs do not exist yet. Predicting
     * L+1's set from L's only works if consecutive layers route similarly.
     *
     * This measures that directly during one-token decode: how many of layer
     * L+1's active experts also appeared in layer L's. Batched prefill unions
     * are excluded because they do not represent a decode top-k route.
     *
     * A wrong prefetch can displace a needed read when the transfer path is
     * saturated. Compare observed overlap with the chance baseline derived from
     * the active-expert count before treating prefetch as viable.
     *
     * Recorded per layer transition; counts how many of the current layer's
     * active experts were in the previous layer's active set. */
    uint32_t prev_set[SILIANGEM_MAX_BATCH];
    int      prev_n;
    int      prev_layer;
    uint64_t xlayer_hist[8];   /* overlap count 0..6, index 7 = 7+ */
    uint64_t xlayer_obs;
    uint64_t xlayer_sum;
    uint64_t xlayer_chance_sum; /* sum of previous_count * current_count */

    /* ---- file-adjacent co-activation ------------------------------------
     * A bandwidth probe cannot tell whether extra bytes beyond one expert are
     * useful or wasted, so it cannot establish file-adjacent locality.
     *
     * This can. Experts sit in the file ordered by id, so reading one extra
     * extent past expert e delivers expert e+1. The question is simply how
     * often e+1 is also active in the same step.
     *
     * DECODE ONLY, deliberately. Prefill lights up most of the population, so
     * adjacency there would be near-certain and would say nothing about
     * whether read-ahead is worth its bytes.
     *
     * Compared against chance at report time: with n of n_experts active, a
     * given expert's successor is active with probability (n-1)/(n_experts-1).
     * Materially above that means adjacent experts co-activate; at chance a
     * wider read would mostly buy unused bytes. */
    uint64_t adj_obs;          /* decode (layer, batch) observations */
    uint64_t adj_active;       /* total active experts across them */
    uint64_t adj_hits;         /* those whose id+1 was also active */

    /* ---- expert access frequency, per (layer, expert) --------------------
     * Frequency-aware placement or an online learner is useful only when the
     * access distribution is skewed. If experts are equally likely there is
     * nothing stable to learn, and pure LRU is not being compared against a
     * meaningful frequency tier.
     *
     * Reported as coverage: what share of all accesses the top-N hottest
     * experts of each layer account for. Compare that coverage with the uniform
     * N/n_experts baseline before drawing a placement conclusion. */
    uint32_t *freq;            /* n_layers * n_experts counters, POOLED */
    uint64_t  freq_total;

    /* ---- the same counters, SPLIT BY PHASE -------------------------------
     * Pooling prefill and decode together makes several questions
     * unanswerable:
     *
     *  - whether prefill's routing predicts decode's, which is the whole
     *    premise of deriving a pinned tier from the prompt for free;
     *  - which experts a VRAM-resident tier should hold;
     *  - separating prefill and decode access statistics.
     *
     * They need not be the same distribution: a larger prefill can touch a much
     * wider expert population than a one-token decode. Mixing their counts can
     * therefore describe neither phase accurately.
     *
     * PHASE TEST: decode is the case where every active expert received exactly
     * one token, so total == n_active. Needs no knowledge of top_k. A two-token
     * prefill whose tokens route disjointly would be misfiled; rare, harmless,
     * and it fails toward calling prefill "decode" rather than the reverse. */
    uint32_t *freq_pf;         /* prefill-only counters */
    uint32_t *freq_dec;        /* decode-only counters  */
    uint64_t  freq_pf_total;
    uint64_t  freq_dec_total;

    /* ---- token-to-token expert reuse within decode -----------------------
     * Cross-request batching cannot establish how adjacent tokens in the same
     * context route. Measure that reuse directly before using it to justify a
     * speculative-decoding or prefetch policy.
     *
     * Per layer, hold the previous decode step's expert set and count how many
     * of this step's experts appear in it. High overlap means k drafted tokens
     * cost barely more I/O than one, and speculative decoding pays on both the
     * barrier and the bandwidth axis. Low overlap means it inherits the same
     * thrashing as cross-request batching and a drafter is wasted work. */
    uint16_t *dec_prev;        /* n_layers * SILIANGEM_TOPK_MAX previous expert ids */
    uint8_t  *dec_prev_n;      /* n_layers, count valid in dec_prev          */
    uint64_t  reuse_hist[9];   /* overlap 0..8 experts with the prior token  */
    uint64_t  reuse_obs;
    uint64_t  reuse_sum;
    uint64_t  reuse_denom;     /* total experts examined, for a share        */
} siliangem_cache;

/* Source geometry is copied into backend-owned state before the first
 * mul_mat_id and remains separate from the live cache allocation. */
typedef struct {
    int      set;
    char     path[1024];
    int      n_layers, n_experts, n_parts;
    uint64_t base[512];
    uint32_t stride[512];
    uint32_t poff[512*4];
    uint32_t pbytes[512*4];
} siliangem_em_source;

/* Scattered geometry from a STOCK GGUF, published the same way. An expert's
 * gate/up/down are three separate tensors, so there are three bases per layer
 * and three per-part strides.
 *
 * This is what lets the cache run on a stock model. Without it siliangem_init requires
 * a compatible slab or expert-major metadata and otherwise declines to plain
 * mmap.
 *
 * A stock file's tensor directory already carries the required offsets and
 * shapes, and llama_model_loader already parses it. Publishing that geometry
 * directly avoids a separate offset sidecar that could drift from the model. */
typedef struct {
    int      set;
    char     path[1024];
    int      n_layers, n_experts;
    uint32_t stride[512 * 3];  /* bytes between experts, per (layer, part) */
    uint64_t base[512 * 3];    /* file offsets,           per (layer, part) */
} siliangem_scattered_source;

/* Monotonic nanoseconds. ith==0 only, so no synchronisation needed.
 * Declared here because siliangem_report() below uses it before its definition. */
static int64_t siliangem_now_ns(void);

/* ---- scattered-source mode (attribution arm) ------------------------------
 * The cache bundles two changes and reports them as one number:
 *   LAYOUT     contiguous slab   vs  scattered projection reads
 *   RESIDENCY  committed arena   vs  file-backed mmap
 * This mode keeps the arena and the unbuffered path but reads the ORIGINAL
 * GGUF offsets, giving a third arm:
 *   OFF -> GGUF-offset  isolates residency
 *   GGUF-offset -> ON   isolates layout
 * Which decides whether this is a disk-streaming trick or a general one.
 *
 * Armed by the backend-scoped configure API with geometry read from the
 * model's own tensor directory. No environment variables or sidecar are used.
 *
 * GGUF tensor offsets are 32-byte aligned, not sector aligned, so unbuffered
 * reads must start on a rounded-down 4096 boundary and land in a bounce buffer;
 * the wanted bytes are then copied into the slot. Expert-major destinations are
 * already aligned, so that arm needs no bounce and no copy -- an inherent
 * difference this attribution arm cannot remove. */
#define SILIANGEM_GGUF_MAX_READS 24            /* 8 experts x 3 parts in flight */

typedef struct {
    int       enabled;
    HANDLE    file;
    uint32_t  n_layers, n_experts;
    /* Bytes per expert, PER (layer, part). One global stride is unsafe when a
     * projection's quantization varies by layer: a wrong stride can return a
     * neighbouring expert's bytes without crashing. The legacy slab format
     * carries global strides, so that path replicates them across layers. */
    uint32_t *stride;                   /* n_layers*3 */
    uint64_t *base;                     /* n_layers*3 tensor base offsets */
    uint8_t  *bounce;                   /* SILIANGEM_GGUF_MAX_READS * bounce_slot */
    size_t    bounce_slot;
} siliangem_src;

static void siliangem_src_finish(void);  /* shared tail: open file, size bounce, arm  */
static void siliangem_scat_init(void);   /* loader-published stock-GGUF arm           */
static void siliangem_wait(void);

/* The model's expert part names, in the order the file packs them. Some models
 * fuse gate and up. Gemma 4, for example, stores
 *
 *     blk.N.ffn_gate_up_exps.weight   (fused)
 *     blk.N.ffn_down_exps.weight
 *     blk.N.ffn_down_exps.scale       (per-expert, F32, stays resident)
 *
 * A fixed gate/up/down name set would not match the fused tensor and would make
 * siliangem_ptr decline the cache path. tools/make_expert_major_gguf.py tries
 * ("gate","up","down") then ("gate_up","down") and writes the winner into
 * siliangem.part_names, and llama_model_loader already parses that key and
 * synthesises blk.%d.ffn_%s_exps.weight from it.
 *
 * Default is the three-part layout, which is what the SCATTERED path uses -
 * it reads a stock GGUF whose gate/up/down are separate tensors, so it has no
 * part_names key to consult. */
#define SILIANGEM_MAX_PARTS 4

struct ggml_siliangem_cache_state {
    siliangem_cache cache;
    siliangem_em_source expert_major;
    siliangem_scattered_source scattered;
    siliangem_src scattered_io;

    uint64_t source_calls;
    uint64_t source_ok;
    uint64_t source_expected_bytes;
    uint32_t source_layers_seen;
    uint8_t source_layer_seen[512];
    uint32_t source_layers_substituted;
    uint8_t source_layer_substituted[512];

    char part_name[SILIANGEM_MAX_PARTS][24];
    int n_part_name;

    int configured;
    int init_attempted;
    int enabled;
    uint32_t capacity_mib;
    int policy;
    int deferred_io;
    int verbose;
    int memory_report;
    int mmap_prefetch;
    enum ggml_siliangem_source_kind source_kind;
    char source_path[1024];

    int order[SILIANGEM_MAX_BATCH];
    int n_hits;
    int n_active;
    int reordered;
};

#if defined(_MSC_VER)
#define SILIANGEM_THREAD_LOCAL __declspec(thread)
#else
#define SILIANGEM_THREAD_LOCAL _Thread_local
#endif

// Non-owning selector only. Ownership and configuration live in the CPU
// backend context; each compute worker binds the state carried by its cplan.
static SILIANGEM_THREAD_LOCAL struct ggml_siliangem_cache_state * siliangem_tls_state = NULL;

static void siliangem_bind_state(struct ggml_siliangem_cache_state * state) {
    siliangem_tls_state = state;
}

#define g_siliangem              (siliangem_tls_state->cache)
#define g_em                      (siliangem_tls_state->expert_major)
#define g_scat                    (siliangem_tls_state->scattered)
#define g_src                     (siliangem_tls_state->scattered_io)
#define g_src_calls               (siliangem_tls_state->source_calls)
#define g_src_ok                  (siliangem_tls_state->source_ok)
#define g_src_expected_bytes      (siliangem_tls_state->source_expected_bytes)
#define g_src_layers_seen         (siliangem_tls_state->source_layers_seen)
#define g_src_layer_seen          (siliangem_tls_state->source_layer_seen)
#define g_src_layers_substituted  (siliangem_tls_state->source_layers_substituted)
#define g_src_layer_substituted   (siliangem_tls_state->source_layer_substituted)
#define g_part_name               (siliangem_tls_state->part_name)
#define g_n_part_name             (siliangem_tls_state->n_part_name)

/* Where expert `a` of `layer` lives in the source file, and how many bytes it
 * occupies. The slab uses one global geometry; an expert-major GGUF has a
 * per-layer base and stride because quantisation varies by layer. */
static uint64_t siliangem_expert_off(int layer, int a) {
    if (g_siliangem.em) {
        return g_siliangem.em_base[layer] + (uint64_t) a * (uint64_t) g_siliangem.em_stride[layer];
    }
    return (uint64_t) SILIANGEM_HEADER_BYTES +
           ((uint64_t) layer * g_siliangem.n_experts + (uint64_t) a) * g_siliangem.expert_bytes;
}

static uint32_t siliangem_expert_len(int layer) {
    return g_siliangem.em ? g_siliangem.em_stride[layer] : g_siliangem.expert_bytes;
}

/* Comma-separated, as it appears in siliangem.part_names. Ignored if empty, so
 * a file without the key keeps the gate/up/down default. */
static void siliangem_set_part_names(const char *csv) {
    if (!csv || !csv[0]) return;
    int n = 0;
    const char *b = csv;
    while (*b && n < SILIANGEM_MAX_PARTS) {
        const char *e = strchr(b, ',');
        size_t len = e ? (size_t)(e - b) : strlen(b);
        if (len > 0 && len < sizeof(g_part_name[0])) {
            memcpy(g_part_name[n], b, len);
            g_part_name[n][len] = '\0';
            n++;
        }
        if (!e) break;
        b = e + 1;
    }
    if (n > 0) g_n_part_name = n;
}

/* "blk.<N>.ffn_<part>_exps.weight" -> layer, part index. 0 if not an expert.
 *
 * Matches the whole remainder, not a prefix, so "gate" cannot swallow
 * "gate_up_exps.weight" even if both names were somehow present. */
static int siliangem_parse_name(const char *name, int *layer, int *part) {
    if (strncmp(name, "blk.", 4) != 0) return 0;
    const char *p = name + 4;
    char *end;
    long l = strtol(p, &end, 10);
    if (end == p || *end != '.') return 0;
    p = end + 1;
    if (strncmp(p, "ffn_", 4) != 0) return 0;
    p += 4;
    for (int i = 0; i < g_n_part_name; i++) {
        const size_t ln = strlen(g_part_name[i]);
        if (strncmp(p, g_part_name[i], ln) == 0 &&
            strcmp(p + ln, "_exps.weight") == 0) {
            *part  = i;
            *layer = (int) l;
            return 1;
        }
    }
    return 0;
}

static uint32_t siliangem_hash(uint32_t k) {           /* fibonacci mix */
    k *= 2654435761u;
    return k ^ (k >> 16);
}

/* Emit cache statistics.
 *
 * Called periodically as well as at exit. Exit-only reporting silently loses
 * the numbers whenever the process is terminated rather than returning --
 * which is exactly what happens to llama-server, torn down with a forced kill
 * (TerminateProcess skips atexit). Periodic reporting also shows whether reuse
 * DEGRADES as context grows, which a single final figure cannot.
 *
 * `lookups` counts one entry per (expert, projection). `expert_requests`
 * counts only the first packed part, so periodic reports remain exact even
 * when emitted between projection calls. */
/* Physical residency, which is what actually bounds this cache.
 *
 * The arena is VirtualAlloc'd PRIVATE memory, so its pages are backed by the
 * PAGEFILE, not by the model file. Windows cannot discard them the way it
 * discards mmap'd pages -- it has no way to know the contents also exist in
 * the GGUF -- so reclaiming one requires WRITING it out. That gives the arena
 * a cliff where mmap has a slope, and the cliff is invisible from inside the
 * process unless we look.
 *
 * Two counters matter and neither is "free memory":
 *   avail_mb  physical pages available without evicting anything
 *   cache_mb  the evictable file-backed page cache -- the real margin. It sits
 *             between the arena and the wall and disappears FIRST, so it falls
 *             to near zero while free memory still looks survivable.
 * Near a paging cliff, available memory and file-cache margin can collapse
 * while pagefile traffic rises; the thresholds are workload-specific.
 */
typedef struct { uint64_t avail_mb; uint64_t cache_mb; uint64_t pf_mb; } siliangem_mem;

static void siliangem_mem_status(siliangem_mem *out) {
    out->avail_mb = 0; out->cache_mb = 0; out->pf_mb = 0;

    MEMORYSTATUSEX ms; ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        out->avail_mb = (uint64_t)(ms.ullAvailPhys / (1024ull * 1024ull));
        /* Commit charge in use; the pagefile portion of it is what hurts. */
        out->pf_mb = (uint64_t)((ms.ullTotalPageFile - ms.ullAvailPageFile)
                                / (1024ull * 1024ull));
    }
    /* SystemFileCacheInformation via the documented performance struct. */
    PERFORMANCE_INFORMATION pi; pi.cb = sizeof(pi);
    if (GetPerformanceInfo(&pi, sizeof(pi))) {
        out->cache_mb = (uint64_t)((uint64_t) pi.SystemCache * pi.PageSize
                                   / (1024ull * 1024ull));
    }
}

static void siliangem_report(const char *tag) {
    uint64_t tot = g_siliangem.hits + g_siliangem.misses;
    if (!tot) return;
    const double expert_hit = g_siliangem.expert_requests
            ? 100.0 * (double) g_siliangem.expert_hits / (double) g_siliangem.expert_requests
            : 0.0;
    fprintf(stderr,
            "siliangem[%s]: %llu lookups, %llu hits (%.1f%%), %llu misses, "
            "expert-request hit %.1f%% (%llu/%llu), %.2f GiB read from slab\n",
            tag,
            (unsigned long long) tot, (unsigned long long) g_siliangem.hits,
            100.0 * (double) g_siliangem.hits / (double) tot,
            (unsigned long long) g_siliangem.misses,
            expert_hit,
            (unsigned long long) g_siliangem.expert_hits,
            (unsigned long long) g_siliangem.expert_requests,
            (double) g_siliangem.bytes_read / (1024.0 * 1024.0 * 1024.0));

    if (g_siliangem.mem_report) {
        /* Residency, every report. Throughput numbers taken while the file
         * cache is near zero are measuring the pagefile, not this cache.
         *
         * GetPerformanceInfo is not free - it takes a system-wide lock and
         * walks the process/thread counts - but this runs once per report,
         * once per SILIANGEM_REPORT_EVERY lookups rather than on every lookup.
         * siliangem_mem_status has exactly one call site: this one. */
        siliangem_mem mem; siliangem_mem_status(&mem);
        fprintf(stderr,
                "siliangem[mem]: available %llu MiB | file cache %llu MiB "
                "(the real margin) | commit charged %llu MiB%s\n",
                (unsigned long long) mem.avail_mb,
                (unsigned long long) mem.cache_mb,
                (unsigned long long) mem.pf_mb,
                (mem.avail_mb < 400 || mem.cache_mb < 64)
                    ? "  <-- PRESSURE, results here are suspect" : "");
    }

    if (g_siliangem.t0) {
        double wall_s  = (double)(siliangem_now_ns() - g_siliangem.t0) / 1e9;
        double wait_s  = (double) g_siliangem.wait_ns  / 1e9;
        double fetch_s = (double) g_siliangem.fetch_ns / 1e9;
        double gib     = (double) g_siliangem.bytes_read / (1024.0*1024.0*1024.0);
        fprintf(stderr,
                "siliangem[time]: wall %.1fs | blocked on I/O %.1fs (%.1f%%) | "
                "issuing %.2fs | %llu waits | effective %.2f GiB/s while "
                "blocked, %.2f GiB/s overall\n",
                wall_s, wait_s, wall_s > 0 ? 100.0*wait_s/wall_s : 0.0,
                fetch_s, (unsigned long long) g_siliangem.wait_calls,
                wait_s  > 0 ? gib / wait_s  : 0.0,
                wall_s  > 0 ? gib / wall_s  : 0.0);

        double m = g_siliangem.misses ? (double) g_siliangem.misses : 1.0;
        fprintf(stderr,
                "siliangem[issue]: per miss -- evict+rehash %.0f us | "
                "CreateEvent %.0f us | ReadFile submit %.0f us  "
                "(%.1f%% / %.1f%% / %.1f%% of wall)\n",
                (double) g_siliangem.evict_ns / m / 1e3,
                (double) g_siliangem.event_ns / m / 1e3,
                (double) g_siliangem.read_ns  / m / 1e3,
                wall_s > 0 ? 100.0 * (double) g_siliangem.evict_ns / 1e9 / wall_s : 0.0,
                wall_s > 0 ? 100.0 * (double) g_siliangem.event_ns / 1e9 / wall_s : 0.0,
                wall_s > 0 ? 100.0 * (double) g_siliangem.read_ns  / 1e9 / wall_s : 0.0);

        uint64_t rtot = g_siliangem.n_sync + g_siliangem.n_pend;
        fprintf(stderr,
                "siliangem[submit]: %llu sync (%.1f%%), %llu pending (%.1f%%) "
                "-- sync means ReadFile did the transfer inline, so that time "
                "is I/O and not overhead\n",
                (unsigned long long) g_siliangem.n_sync,
                rtot ? 100.0*(double)g_siliangem.n_sync/(double)rtot : 0.0,
                (unsigned long long) g_siliangem.n_pend,
                rtot ? 100.0*(double)g_siliangem.n_pend/(double)rtot : 0.0);
    }

    if (g_siliangem.xlayer_obs) {
        double o = (double) g_siliangem.xlayer_obs;
        double mean = (double) g_siliangem.xlayer_sum / o;
        double chance = g_siliangem.n_experts
                ? (double) g_siliangem.xlayer_chance_sum / o / (double) g_siliangem.n_experts
                : 0.0;
        fprintf(stderr,
                "siliangem[xlayer]: DECODE mean overlap %.2f experts | "
                "0=%.1f%% 1=%.1f%% 2=%.1f%% 3=%.1f%% 4=%.1f%% 5=%.1f%% 6=%.1f%% 7+=%.1f%% "
                "| chance would be %.2f -- prefetching L+1 from L is viable "
                "only well above chance, since the bus is saturated and a wrong "
                "prefetch displaces a needed read\n",
                mean,
                100.0*(double)g_siliangem.xlayer_hist[0]/o,
                100.0*(double)g_siliangem.xlayer_hist[1]/o,
                100.0*(double)g_siliangem.xlayer_hist[2]/o,
                100.0*(double)g_siliangem.xlayer_hist[3]/o,
                100.0*(double)g_siliangem.xlayer_hist[4]/o,
                100.0*(double)g_siliangem.xlayer_hist[5]/o,
                100.0*(double)g_siliangem.xlayer_hist[6]/o,
                100.0*(double)g_siliangem.xlayer_hist[7]/o,
                chance);
    }

    /* Coverage of the top-N hottest experts per layer, averaged over layers.
     * Compared against N/n_experts, which is what uniform routing would give. */
    if (g_siliangem.freq && g_siliangem.freq_total) {
        const int Ns[4] = { 8, 16, 40, 80 };
        double cov[4] = {0,0,0,0};
        uint32_t *tmp = (uint32_t *) malloc(g_siliangem.n_experts * sizeof(uint32_t));
        if (tmp) {
            for (uint32_t L = 0; L < g_siliangem.n_layers; L++) {
                uint32_t *row = g_siliangem.freq + (size_t) L * g_siliangem.n_experts;
                uint64_t tot = 0;
                for (uint32_t e = 0; e < g_siliangem.n_experts; e++) { tmp[e] = row[e]; tot += row[e]; }
                if (!tot) continue;
                /* Partial selection sort; bounded by n_experts and report scope. */
                for (int k = 0; k < 80 && (uint32_t) k < g_siliangem.n_experts; k++) {
                    int mx = k;
                    for (uint32_t j = k + 1; j < g_siliangem.n_experts; j++)
                        if (tmp[j] > tmp[mx]) mx = (int) j;
                    uint32_t t = tmp[k]; tmp[k] = tmp[mx]; tmp[mx] = t;
                }
                for (int c = 0; c < 4; c++) {
                    uint64_t s = 0;
                    for (int k = 0; k < Ns[c] && (uint32_t) k < g_siliangem.n_experts; k++) s += tmp[k];
                    cov[c] += (double) s / (double) tot;
                }
            }
            free(tmp);
            double L = (double) g_siliangem.n_layers;
            fprintf(stderr,
                    "siliangem[freq]: top-N expert coverage (avg over layers) -- "
                    "top8 %.1f%% (uniform %.1f%%) | top16 %.1f%% (%.1f%%) | "
                    "top40 %.1f%% (%.1f%%) | top80 %.1f%% (%.1f%%)  "
                    "[skew above uniform is what a learned table or a pinned "
                    "frequency tier could exploit]\n",
                    100.0*cov[0]/L, 100.0*8.0 /(double)g_siliangem.n_experts,
                    100.0*cov[1]/L, 100.0*16.0/(double)g_siliangem.n_experts,
                    100.0*cov[2]/L, 100.0*40.0/(double)g_siliangem.n_experts,
                    100.0*cov[3]/L, 100.0*80.0/(double)g_siliangem.n_experts);
        }
    }

    /* ---- phase split: does prefill's routing predict decode's? ------------
     * The pooled coverage above can average substantially different routing
     * distributions. Report prefill and decode separately, plus the overlap
     * that decides whether a pinned tier can be derived from the prompt. */
    if (g_siliangem.freq_pf && g_siliangem.freq_dec && g_siliangem.freq_dec_total) {
        double ov_sum = 0.0; uint32_t ov_layers = 0;
        uint64_t dec_distinct = 0, pf_distinct = 0;

        for (uint32_t L = 0; L < g_siliangem.n_layers; L++) {
            const uint32_t *pf  = g_siliangem.freq_pf  + (size_t) L * g_siliangem.n_experts;
            const uint32_t *dec = g_siliangem.freq_dec + (size_t) L * g_siliangem.n_experts;

            /* Rank prefill's experts, take the top-N, and ask what share of
             * DECODE's accesses those N would have covered. N = 40 to match the
             * coverage buckets above. */
            const uint32_t N = g_siliangem.n_experts < 40 ? g_siliangem.n_experts : 40;
            uint64_t dec_tot = 0, dec_in_top = 0;
            for (uint32_t e = 0; e < g_siliangem.n_experts; e++) {
                dec_tot += dec[e];
                if (dec[e]) dec_distinct++;
                if (pf[e])  pf_distinct++;
            }
            if (!dec_tot) continue;

            /* Selection of prefill's top-N by threshold: find the Nth largest
             * prefill count, then sum decode accesses at or above it. Avoids a
             * second sort buffer and is exact enough for a coverage figure. */
            uint32_t thresh = 0;
            {
                uint32_t *pcopy = (uint32_t *) malloc(g_siliangem.n_experts * sizeof(uint32_t));
                if (!pcopy) continue;
                for (uint32_t e = 0; e < g_siliangem.n_experts; e++) pcopy[e] = pf[e];
                for (uint32_t k = 0; k < N; k++) {
                    uint32_t mx = k;
                    for (uint32_t j = k + 1; j < g_siliangem.n_experts; j++)
                        if (pcopy[j] > pcopy[mx]) mx = j;
                    uint32_t t = pcopy[k]; pcopy[k] = pcopy[mx]; pcopy[mx] = t;
                }
                thresh = pcopy[N - 1];
                free(pcopy);
            }
            for (uint32_t e = 0; e < g_siliangem.n_experts; e++) {
                if (pf[e] >= thresh && pf[e] > 0) dec_in_top += dec[e];
            }
            ov_sum += (double) dec_in_top / (double) dec_tot;
            ov_layers++;
        }

        if (ov_layers) {
            fprintf(stderr,
                    "siliangem[phase]: prefill %llu accesses over %llu distinct "
                    "(layer,expert); decode %llu over %llu. Prefill's top-40 per "
                    "layer covers %.1f%% of DECODE accesses (uniform would be "
                    "%.1f%%) [above uniform = a pinned tier can be derived from "
                    "the prompt for free; at uniform, prefill says nothing about "
                    "decode and the tier must be learned from decode itself]\n",
                    (unsigned long long) g_siliangem.freq_pf_total,
                    (unsigned long long) pf_distinct,
                    (unsigned long long) g_siliangem.freq_dec_total,
                    (unsigned long long) dec_distinct,
                    100.0 * ov_sum / (double) ov_layers,
                    100.0 * 40.0 / (double) g_siliangem.n_experts);
        }
    }

    /* ---- token-to-token expert reuse: sizes speculative decoding ---------- */
    if (g_src.enabled) {
        fprintf(stderr, "siliangem[scat]: fetch calls %llu, ok %llu (%.1f%%), "
                "layers served %u/%u, layers substituted %u/%u, "
                "expected bytes %.2f GiB vs counted %.2f GiB\n",
                (unsigned long long) g_src_calls,
                (unsigned long long) g_src_ok,
                g_src_calls ? 100.0 * (double) g_src_ok / (double) g_src_calls : 0.0,
                g_src_layers_seen, g_src.n_layers,
                g_src_layers_substituted, g_src.n_layers,
                (double) g_src_expected_bytes / (1024.0*1024.0*1024.0),
                (double) g_siliangem.bytes_read / (1024.0*1024.0*1024.0));
    }

    if (g_siliangem.reuse_obs && g_siliangem.reuse_denom) {
        double mean = (double) g_siliangem.reuse_sum / (double) g_siliangem.reuse_obs;
        double share = 100.0 * (double) g_siliangem.reuse_sum / (double) g_siliangem.reuse_denom;
        fprintf(stderr,
                "siliangem[reuse]: consecutive DECODE tokens share %.2f of their "
                "experts per layer on average (%.1f%%). Distribution 0..6: ",
                mean, share);
        for (int i = 0; i <= 6; i++) {
            fprintf(stderr, "%.0f%% ",
                    100.0 * (double) g_siliangem.reuse_hist[i] / (double) g_siliangem.reuse_obs);
        }
        fprintf(stderr,
                " [share bounds how much EXTRA population a drafted token drags "
                "in. Do not read a high share as 'speculation pays': the outcome "
                "also depends on total bytes fetched against arena capacity, so "
                "it requires an end-to-end target-system measurement.]\n");
    }

    if (g_siliangem.adj_obs && g_siliangem.adj_active) {
        const double mean_n   = (double) g_siliangem.adj_active / (double) g_siliangem.adj_obs;
        const double observed = 100.0 * (double) g_siliangem.adj_hits / (double) g_siliangem.adj_active;
        const double chance   = g_siliangem.n_experts > 1
                              ? 100.0 * (mean_n - 1.0) / (double) (g_siliangem.n_experts - 1)
                              : 0.0;
        fprintf(stderr,
                "siliangem[adjacent]: expert id+1 also active in %.1f%% of DECODE "
                "activations (chance %.1f%% at %.1f active of %u) -- ratio %.2fx "
                "[experts sit in the file ordered by id, so this is exactly the "
                "hit rate a double-width read would get for its second extent. "
                "Near 1.00x means read-ahead buys bytes nobody wants; well above "
                "means adjacent experts co-activate and reordering the file by "
                "co-activation is worth evaluating. A bandwidth probe alone "
                "cannot tell a used byte from a wasted one.]\n",
                observed, chance, mean_n, g_siliangem.n_experts,
                chance > 0.0 ? observed / chance : 0.0);
    }

    uint64_t ch = 0;
    for (int i = 0; i < 8; i++) ch += g_siliangem.cne1_hist[i];
    if (ch) {
        fprintf(stderr,
                "siliangem[cne1]: 1=%.1f%% 2=%.1f%% 3=%.1f%% 4=%.1f%% "
                "5-8=%.1f%% 9-16=%.1f%% 17-32=%.1f%% 33+=%.1f%%  "
                "(cne1==1 takes llama.cpp's chunk_size=64 fast path)\n",
                100.0*(double)g_siliangem.cne1_hist[0]/(double)ch,
                100.0*(double)g_siliangem.cne1_hist[1]/(double)ch,
                100.0*(double)g_siliangem.cne1_hist[2]/(double)ch,
                100.0*(double)g_siliangem.cne1_hist[3]/(double)ch,
                100.0*(double)g_siliangem.cne1_hist[4]/(double)ch,
                100.0*(double)g_siliangem.cne1_hist[5]/(double)ch,
                100.0*(double)g_siliangem.cne1_hist[6]/(double)ch,
                100.0*(double)g_siliangem.cne1_hist[7]/(double)ch);
    }
}

static int64_t siliangem_now_ns(void) {
    LARGE_INTEGER c;
    if (!g_siliangem.qpc_freq) {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        g_siliangem.qpc_freq = f.QuadPart ? f.QuadPart : 1;
    }
    QueryPerformanceCounter(&c);
    return (int64_t) ((double) c.QuadPart * 1e9 / (double) g_siliangem.qpc_freq);
}

/* Arm C: record how much a decode layer's active set overlaps layer L-1's.
 *
 * Called once per layer, from the first projection only (gate), so each layer
 * transition is counted once rather than three times. ith==0 only. */
static void siliangem_note_xlayer(int layer, const int64_t *counts, int n_as,
                           int64_t n_tokens) {
    uint32_t cur[SILIANGEM_MAX_BATCH];
    int n = 0;
    int64_t total = 0;
    for (int a = 0; a < n_as && n < SILIANGEM_MAX_BATCH; a++) {
        if (counts[a] != 0) { cur[n++] = (uint32_t) a; total += counts[a]; }
    }

    /* Decode means the batch carries exactly one token. Read ids->ne[1] from
     * the caller: inferring the phase from active-expert counts can misclassify
     * short prefills whose routes do not collide. */
    const int is_decode = (n > 0 && n_tokens == 1);

    if (g_siliangem.freq && (uint32_t) layer < g_siliangem.n_layers) {
        uint32_t *pooled = g_siliangem.freq     + (size_t) layer * g_siliangem.n_experts;
        uint32_t *phase  = is_decode ? g_siliangem.freq_dec : g_siliangem.freq_pf;
        for (int i = 0; i < n; i++) {
            if (cur[i] < g_siliangem.n_experts) {
                pooled[cur[i]]++;
                g_siliangem.freq_total++;
                if (phase) {
                    phase[(size_t) layer * g_siliangem.n_experts + cur[i]]++;
                    if (is_decode) g_siliangem.freq_dec_total++; else g_siliangem.freq_pf_total++;
                }
            }
        }
    }

    /* File-adjacent co-activation: would a double-width read pay? See the
     * note on adj_obs. n is 6 here, so the O(n^2) scan is 36 compares. */
    if (is_decode) {
        int adj = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (cur[j] == cur[i] + 1) { adj++; break; }
            }
        }
        g_siliangem.adj_obs++;
        g_siliangem.adj_active += (uint64_t) n;
        g_siliangem.adj_hits   += (uint64_t) adj;
    }

    /* Token-to-token reuse, decode only. Prefill is excluded deliberately: it
     * touches nearly the whole expert population, so "overlap with the previous
     * step" would be near-total and would say nothing about drafting. */
    if (is_decode && g_siliangem.dec_prev && g_siliangem.dec_prev_n &&
        (uint32_t) layer < g_siliangem.n_layers) {
        uint16_t *prev  = g_siliangem.dec_prev + (size_t) layer * SILIANGEM_TOPK_MAX;
        int       prevn = g_siliangem.dec_prev_n[layer];
        if (prevn > 0) {
            int hit = 0;
            for (int i = 0; i < n && i < SILIANGEM_TOPK_MAX; i++) {
                for (int j = 0; j < prevn; j++) {
                    if ((uint16_t) cur[i] == prev[j]) { hit++; break; }
                }
            }
            g_siliangem.reuse_hist[hit < 8 ? hit : 8]++;
            g_siliangem.reuse_obs++;
            g_siliangem.reuse_sum   += (uint64_t) hit;
            g_siliangem.reuse_denom += (uint64_t) (n < SILIANGEM_TOPK_MAX ? n : SILIANGEM_TOPK_MAX);
        }
        int keep = n < SILIANGEM_TOPK_MAX ? n : SILIANGEM_TOPK_MAX;
        for (int i = 0; i < keep; i++) prev[i] = (uint16_t) cur[i];
        g_siliangem.dec_prev_n[layer] = (uint8_t) keep;
    }

    if (!is_decode) {
        g_siliangem.prev_n = 0;
        g_siliangem.prev_layer = -1;
        return;
    }

    /* Only consecutive decode layers within the same forward pass are comparable. */
    if (g_siliangem.prev_n > 0 && layer == g_siliangem.prev_layer + 1) {
        int hit = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < g_siliangem.prev_n; j++) {
                if (cur[i] == g_siliangem.prev_set[j]) { hit++; break; }
            }
        }
        g_siliangem.xlayer_hist[hit < 7 ? hit : 7]++;
        g_siliangem.xlayer_obs++;
        g_siliangem.xlayer_sum += (uint64_t) hit;
        g_siliangem.xlayer_chance_sum += (uint64_t) n * (uint64_t) g_siliangem.prev_n;
    }
    for (int i = 0; i < n; i++) g_siliangem.prev_set[i] = cur[i];
    g_siliangem.prev_n     = n;
    g_siliangem.prev_layer = layer;
}

/* Entry point for the cross-layer probe, called unconditionally from the
 * mul_mat_id hook so the measurement exists in every configuration -- including
 * mmap and scattered-source, where siliangem_prepare_async declines early.
 *
 * Filters to the gate projection so each layer transition counts once rather
 * than three times. */
static void siliangem_observe_layer(const char *name, const int64_t *counts, int n_as,
                             int64_t n_tokens) {
    int layer, part;
    if (!siliangem_parse_name(name, &layer, &part)) return;
    if (part != 0) return;                   /* gate only */
    siliangem_note_xlayer(layer, counts, n_as, n_tokens);
}

/* Record one expert's token count. Called from ith==0 only. */
static void siliangem_note_cne1(long long cne1) {
    if (cne1 <= 0) return;
    int b = cne1 == 1 ? 0 : cne1 == 2 ? 1 : cne1 == 3 ? 2 : cne1 == 4 ? 3
          : cne1 <= 8 ? 4 : cne1 <= 16 ? 5 : cne1 <= 32 ? 6 : 7;
    g_siliangem.cne1_hist[b]++;
}

/* Report every N lookups so the numbers survive any shutdown path. The
 * threshold is low enough to emit during short controlled runs while keeping
 * reporting off the per-lookup path. */
#define SILIANGEM_REPORT_EVERY 5000

static void siliangem_maybe_report(void) {
    if (!g_siliangem.verbose) return;
    uint64_t tot = g_siliangem.hits + g_siliangem.misses;
    if (tot - g_siliangem.last_report >= SILIANGEM_REPORT_EVERY) {
        g_siliangem.last_report = tot;
        siliangem_report("periodic");
    }
}

static void siliangem_transient_reset(void);

static void siliangem_shutdown(void) {
    if (g_siliangem.verbose && (g_siliangem.hits + g_siliangem.misses)) {
        siliangem_report("final");
    }
    for (int i = 0; i < g_siliangem.n_pending; ++i) {
        DWORD ignored = 0;
        if (g_siliangem.file && g_siliangem.file != INVALID_HANDLE_VALUE) {
            GetOverlappedResult(g_siliangem.file, &g_siliangem.pend_ov[i], &ignored, TRUE);
        }
        if (g_siliangem.pend_ev[i]) CloseHandle(g_siliangem.pend_ev[i]);
    }
    if (g_siliangem.file && g_siliangem.file != INVALID_HANDLE_VALUE) CloseHandle(g_siliangem.file);
    if (g_siliangem.arena) VirtualFree(g_siliangem.arena, 0, MEM_RELEASE);
    if (g_siliangem.transient_arena) VirtualFree(g_siliangem.transient_arena, 0, MEM_RELEASE);
    if (g_src.bounce) VirtualFree(g_src.bounce, 0, MEM_RELEASE);
    if (g_src.file && g_src.file != INVALID_HANDLE_VALUE) CloseHandle(g_src.file);
    free(g_src.stride);
    free(g_src.base);
    memset(&g_src, 0, sizeof(g_src));
    free(g_siliangem.freq);
    free(g_siliangem.freq_pf);
    free(g_siliangem.freq_dec);
    free(g_siliangem.dec_prev);
    free(g_siliangem.dec_prev_n);
    free(g_siliangem.slots);
    free(g_siliangem.table);
    memset(&g_siliangem, 0, sizeof(g_siliangem));
}

static void siliangem_init(void) {
    g_siliangem.ready = 0;                              /* disabled unless we finish */
    if (!siliangem_tls_state->configured || !siliangem_tls_state->enabled ||
        siliangem_tls_state->capacity_mib == 0) {
        return;
    }

    const uint64_t budget = siliangem_tls_state->capacity_mib;
    const char * path = siliangem_tls_state->source_path;

    /* A STOCK GGUF published scattered geometry. Derive the arena's SLOT layout
     * from it and reuse the expert-major plumbing wholesale: a slot holds
     * gate|up|down contiguously either way, with per-layer part offsets and
     * sizes, so siliangem_ptr needs no new case. Only the FETCH differs -- three reads
     * at three offsets instead of one -- and g_src owns that.
     *
     * Lowest priority. A legacy slab or an expert-major source both win, so
     * nothing that worked before changes behaviour. */
    if (!g_em.set && g_scat.set) {
        snprintf(g_em.path, sizeof(g_em.path), "%s", g_scat.path);
        g_em.n_layers  = g_scat.n_layers;
        g_em.n_experts = g_scat.n_experts;
        g_em.n_parts   = 3;
        for (int L = 0; L < g_scat.n_layers; L++) {
            uint32_t tot = 0;
            for (int p = 0; p < 3; p++) {
                g_em.poff[L*3 + p]   = tot;
                g_em.pbytes[L*3 + p] = g_scat.stride[L*3 + p];
                tot += g_scat.stride[L*3 + p];
            }
            g_em.stride[L] = tot;      /* bytes per expert in the slot */
            g_em.base[L]   = 0;        /* unused: fetches go through g_src */
        }
        g_em.set = 1;
        g_siliangem.em_scattered = 1;         /* nb[2] will be a PART stride, not the
                                        * whole expert -- see siliangem_ptr */
    }

    /* An expert-major model publishes its own source, which supersedes the
     * slab because the experts live in the model file itself. */
    if (g_em.set) path = g_em.path;
    if (!path || !path[0]) return;
    g_siliangem.verbose = siliangem_tls_state->verbose;
    g_siliangem.mem_report = siliangem_tls_state->memory_report;
    g_siliangem.defer = siliangem_tls_state->deferred_io;
    g_siliangem.placement_policy = siliangem_tls_state->policy;

    /* Unbuffered so the page cache is bypassed entirely (we manage residency),
     * overlapped so several misses can be in flight at once. */
    g_siliangem.file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING,
                            FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, NULL);
    if (g_siliangem.file == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "siliangem: cannot open %s (err %lu) - using mmap\n",
                path, GetLastError());
        return;
    }

    /* ---- expert-major GGUF: geometry comes from the loader, not a header --- */
    if (g_em.set) {
        g_siliangem.em         = 1;
        g_siliangem.em_nparts  = g_em.n_parts;
        g_siliangem.em_base    = g_em.base;
        g_siliangem.em_stride  = g_em.stride;
        g_siliangem.em_poff    = g_em.poff;
        g_siliangem.em_pbytes  = g_em.pbytes;
        g_siliangem.n_layers   = (uint32_t) g_em.n_layers;
        g_siliangem.n_experts  = (uint32_t) g_em.n_experts;

        /* Slots are fixed size, so they must fit the LARGEST layer's expert.
         * Layers with a smaller stride simply leave the tail unused. */
        uint32_t mx = 0;
        for (int i = 0; i < g_em.n_layers; i++) {
            if (g_em.stride[i] > mx) mx = g_em.stride[i];
        }
        g_siliangem.expert_bytes = mx;

        /* Unbuffered I/O needs sector-multiple offsets AND lengths. The writer
         * pads every stride to lcm(type sizes, 512) and starts each packed
         * region on a 512 boundary, so this should hold; refuse rather than
         * fall into misaligned reads if a file was produced differently. */
        for (int i = 0; i < g_em.n_layers; i++) {
            if ((g_em.stride[i] & 511u) || (g_em.base[i] & 511ull)) {
                fprintf(stderr, "siliangem: layer %d stride/base not sector aligned "
                        "(%u / %llu) - using mmap\n", i, g_em.stride[i],
                        (unsigned long long) g_em.base[i]);
                CloseHandle(g_siliangem.file); g_siliangem.file = NULL;
                return;
            }
        }
        goto have_geometry;
    }

    /* Header must be read through a buffered handle: unbuffered reads must be
     * sector-multiples, and we want an exact 36-byte struct. */
    uint32_t h[9] = {0};
    {
        HANDLE hb = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        DWORD got = 0;
        if (hb == INVALID_HANDLE_VALUE || !ReadFile(hb, h, sizeof(h), &got, NULL)
            || got != sizeof(h)) {
            fprintf(stderr, "siliangem: cannot read slab header - using mmap\n");
            if (hb != INVALID_HANDLE_VALUE) CloseHandle(hb);
            CloseHandle(g_siliangem.file); g_siliangem.file = NULL;
            return;
        }
        CloseHandle(hb);
    }
    if (h[0] != SILIANGEM_MAGIC || h[1] != 1u || h[2] != SILIANGEM_HEADER_BYTES) {
        fprintf(stderr, "siliangem: bad slab header (magic %08x v%u) - using mmap\n",
                h[0], h[1]);
        CloseHandle(g_siliangem.file); g_siliangem.file = NULL;
        return;
    }
    g_siliangem.n_layers  = h[3];
    g_siliangem.n_experts = h[4];
    g_siliangem.part_bytes[0] = h[5];
    g_siliangem.part_bytes[1] = h[6];
    g_siliangem.part_bytes[2] = h[7];
    g_siliangem.part_off[0] = 0;
    g_siliangem.part_off[1] = h[5];
    g_siliangem.part_off[2] = h[5] + h[6];
    g_siliangem.expert_bytes = h[5] + h[6] + h[7];

    /* Unbuffered I/O requires sector-multiple offsets and lengths. The repack
     * geometry satisfies this (7,077,888 == 1728 * 4096); refuse rather than
     * silently fall into misaligned reads if a future layout does not. */
    if (g_siliangem.expert_bytes == 0 || (g_siliangem.expert_bytes & 4095u)) {
        fprintf(stderr, "siliangem: expert stride %u not sector aligned - using mmap\n",
                g_siliangem.expert_bytes);
        CloseHandle(g_siliangem.file); g_siliangem.file = NULL;
        return;
    }

have_geometry:
    (void) 0;
    /* The typed capacity_mib value is authoritative. There is deliberately no clamp here.
     *
     * Two tempting clamps are deliberately not used:
     *
     *   ullAvailPhys minus a fixed reserve. By the time siliangem_init runs, llama.cpp
     *     has mmap'd the model and the file cache may have absorbed free memory,
     *     so ullAvailPhys can be pessimistically low. Silently cutting the
     *     requested arena can reduce throughput; page cache is evictable.
     *
     *   ullTotalPhys minus a fixed reserve. A fixed reserve cannot derive a
     *     generally safe limit and would silently resize an arena the operator
     *     requested by name.
     *
     * The real ceiling is physical RAM minus everything else that must stay
     * resident (GPU SHARED memory, KV at n_ctx_slot x n_slots, the non-expert
     * weights). It cannot be computed here because the GPU allocations happen
     * after this point. Sizing the arena is a measured decision. What catches
     * the cliff at runtime is the
     * siliangem[mem] line, which reports the file-cache margin that actually
     * protects us and collapses first. */

    uint64_t arena_bytes = budget * 1024ull * 1024ull;
    const uint64_t nslots = arena_bytes / g_siliangem.expert_bytes;
    if (nslots == 0) {
        fprintf(stderr, "siliangem: cache budget too small - using mmap\n");
        CloseHandle(g_siliangem.file); g_siliangem.file = NULL;
        return;
    }
    if (nslots > UINT32_MAX || nslots > (uint64_t) ((SIZE_T) -1) / g_siliangem.expert_bytes) {
        fprintf(stderr, "siliangem: cache budget too large - using mmap\n");
        CloseHandle(g_siliangem.file); g_siliangem.file = NULL;
        return;
    }
    g_siliangem.nslots = (uint32_t) nslots;

    /* VirtualAlloc is page aligned, which satisfies the unbuffered requirement
     * that the destination buffer be sector aligned. */
    g_siliangem.arena = (uint8_t *) VirtualAlloc(
        NULL, (SIZE_T) g_siliangem.nslots * g_siliangem.expert_bytes,
        MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!g_siliangem.arena) {
        fprintf(stderr, "siliangem: cannot commit %llu MiB - using mmap\n",
                (unsigned long long) budget);
        CloseHandle(g_siliangem.file); g_siliangem.file = NULL;
        return;
    }

    g_siliangem.mask = 1;
    while (g_siliangem.mask < g_siliangem.nslots * 2u) g_siliangem.mask <<= 1;
    g_siliangem.mask -= 1;

    g_siliangem.slots = (siliangem_slot *)  malloc(sizeof(siliangem_slot) * g_siliangem.nslots);
    g_siliangem.table = (uint32_t *) malloc(sizeof(uint32_t) * (g_siliangem.mask + 1));
    if (!g_siliangem.slots || !g_siliangem.table) {
        fprintf(stderr, "siliangem: metadata alloc failed - using mmap\n");
        siliangem_shutdown();
        return;
    }
    for (uint32_t i = 0; i < g_siliangem.nslots; i++) {
        g_siliangem.slots[i].key = SILIANGEM_EMPTY;
        g_siliangem.slots[i].stamp = 0;
        g_siliangem.slots[i].frequency = 0;
        g_siliangem.slots[i].leases = 0;
        g_siliangem.slots[i].segment = SILIANGEM_SEGMENT_NONE;
    }
    for (uint32_t i = 0; i <= g_siliangem.mask; i++)   g_siliangem.table[i] = SILIANGEM_EMPTY;
    siliangem_transient_reset();

    if (g_siliangem.em_scattered) {
        siliangem_scat_init();
        if (!g_src.enabled) {
            fprintf(stderr, "siliangem: stock scattered source initialization failed - using mmap\n");
            siliangem_shutdown();
            return;
        }
    }

    g_siliangem.ready = 1;
    g_siliangem.t0 = siliangem_now_ns();
    if (g_siliangem.placement_policy == SILIANGEM_POLICY_WTINYLFU_W10_SLRU_P80) {
        g_siliangem.policy_window_slots = (g_siliangem.nslots + 9u) / 10u;
        if (g_siliangem.policy_window_slots == 0) g_siliangem.policy_window_slots = 1;
        const uint32_t main_slots = g_siliangem.nslots - g_siliangem.policy_window_slots;
        g_siliangem.policy_protected_slots = (main_slots * 80u) / 100u;
    }
    /* All four are optional: every consumer checks for NULL, so a failed
     * allocation disables that probe rather than the cache. */
    {
        const size_t ncell = (size_t) g_siliangem.n_layers * g_siliangem.n_experts;
        g_siliangem.freq       = (uint32_t *) calloc(ncell, sizeof(uint32_t));
        g_siliangem.freq_pf    = (uint32_t *) calloc(ncell, sizeof(uint32_t));
        g_siliangem.freq_dec   = (uint32_t *) calloc(ncell, sizeof(uint32_t));
        g_siliangem.dec_prev   = (uint16_t *) calloc((size_t) g_siliangem.n_layers * SILIANGEM_TOPK_MAX,
                                              sizeof(uint16_t));
        g_siliangem.dec_prev_n = (uint8_t  *) calloc((size_t) g_siliangem.n_layers, sizeof(uint8_t));
    }
    fprintf(stderr,
            "siliangem: source %ux%u experts, %.3f MiB each, cache %u slots "
            "(%.2f GiB), policy %d, unbuffered+overlapped, deferred-wait %s\n",
            g_siliangem.n_layers, g_siliangem.n_experts, g_siliangem.expert_bytes / 1048576.0,
            g_siliangem.nslots,
            (double) g_siliangem.nslots * g_siliangem.expert_bytes / (1024.0*1024.0*1024.0),
            g_siliangem.placement_policy,
            g_siliangem.defer ? "ON" : "OFF");
}

/* The GGUF is the source of truth for stock-model tensor geometry. A separate
 * offset sidecar would duplicate information, could drift from the model, and
 * could not independently verify that the pairing remained valid.
 * llama_model_loader publishes the geometry straight from the tensor
 * directory via the typed source descriptor, so siliangem_scat_init()
 * below fills g_src from the GGUF itself. Same struct, same fetch path, no
 * sidecar, nothing to keep in sync. */

/* Shared tail for both scattered sources: open the file unbuffered, size the
 * bounce buffer to the LARGEST stride over every (layer, part), and arm.
 * Assumes g_src.{n_layers,n_experts,stride,base} are already filled. */
static void siliangem_src_finish(void) {
    uint32_t maxs = 0;
    for (size_t i = 0; i < (size_t) g_src.n_layers * 3; i++) {
        if (g_src.stride[i] > maxs) maxs = g_src.stride[i];
    }
    if (maxs == 0) {
        fprintf(stderr, "siliangem: zero expert stride - scattered source declined\n");
        free(g_src.stride); g_src.stride = NULL;
        free(g_src.base);   g_src.base   = NULL;
        if (g_src.file && g_src.file != INVALID_HANDLE_VALUE) CloseHandle(g_src.file);
        g_src.file = NULL;
        return;
    }
    g_src.bounce_slot = ((size_t) maxs + 2 * 4096 + 4095) & ~(size_t) 4095;
    g_src.bounce = (uint8_t *) VirtualAlloc(
        NULL, g_src.bounce_slot * SILIANGEM_GGUF_MAX_READS,
        MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!g_src.bounce) {
        fprintf(stderr, "siliangem: bounce alloc failed\n");
        CloseHandle(g_src.file); g_src.file = NULL;
        free(g_src.stride); g_src.stride = NULL;
        free(g_src.base);   g_src.base   = NULL;
        return;
    }
    g_src.enabled = 1;
    fprintf(stderr, "siliangem: SCATTERED-SOURCE mode - reading original GGUF "
                    "offsets (%u layers x %u experts, max stride %u B, "
                    "bounce %.1f MiB)\n",
            g_src.n_layers, g_src.n_experts, maxs,
            g_src.bounce_slot * SILIANGEM_GGUF_MAX_READS / 1048576.0);
}

/* Loader-published scattered source: a STOCK GGUF, no sidecar, no repack.
 * Called from siliangem_init whenever the loader published geometry, which it does
 * for any model whose tensor directory contains ffn_*_exps. This is now the
 * ONLY way a scattered source is armed. */
static void siliangem_scat_init(void) {
    if (!g_scat.set) return;

    const size_t n = (size_t) g_scat.n_layers * 3;
    g_src.stride = (uint32_t *) malloc(n * sizeof(uint32_t));
    g_src.base   = (uint64_t *) malloc(n * sizeof(uint64_t));
    if (!g_src.stride || !g_src.base) {
        free(g_src.stride); g_src.stride = NULL;
        free(g_src.base);   g_src.base   = NULL;
        return;
    }
    g_src.n_layers  = (uint32_t) g_scat.n_layers;
    g_src.n_experts = (uint32_t) g_scat.n_experts;
    for (size_t i = 0; i < n; i++) {
        g_src.stride[i] = g_scat.stride[i];
        g_src.base[i]   = g_scat.base[i];
    }
    g_src.file = CreateFileA(g_scat.path, GENERIC_READ, FILE_SHARE_READ, NULL,
                             OPEN_EXISTING,
                             FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, NULL);
    if (g_src.file == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "siliangem: cannot open %s unbuffered (err %lu)\n",
                g_scat.path, GetLastError());
        g_src.file = NULL;
        free(g_src.stride); g_src.stride = NULL;
        free(g_src.base);   g_src.base   = NULL;
        return;
    }
    siliangem_src_finish();
}

/* Fetch one expert as 3 scattered unbuffered reads into `dst`. 0 on failure. */
static int siliangem_src_fetch(int layer, int expert, uint8_t *dst) {
    g_src_calls++;
    OVERLAPPED ov[3]; HANDLE ev[3];
    uint64_t lo[3]; uint32_t off_in[3], span[3];
    uint32_t part_off = 0;
    int issued = 0;
    const uint32_t *st = g_src.stride + (size_t) layer * 3;   /* per-layer */
    for (int p = 0; p < 3; p++) {
        uint64_t o = g_src.base[(size_t) layer * 3 + p]
                   + (uint64_t) expert * st[p];
        lo[p]     = o & ~(uint64_t) 4095;
        off_in[p] = (uint32_t) (o - lo[p]);
        span[p]   = (off_in[p] + st[p] + 4095) & ~(uint32_t) 4095;
        memset(&ov[p], 0, sizeof(OVERLAPPED));
        ov[p].Offset     = (DWORD) (lo[p] & 0xFFFFFFFFull);
        ov[p].OffsetHigh = (DWORD) (lo[p] >> 32);
        ev[p] = CreateEventA(NULL, TRUE, FALSE, NULL);
        ov[p].hEvent = ev[p];
        if (!ReadFile(g_src.file, g_src.bounce + (size_t) p * g_src.bounce_slot,
                      span[p], NULL, &ov[p])
            && GetLastError() != ERROR_IO_PENDING) {
            /* Diagnostic belongs HERE as well as in the wait loop: an issue-time
             * failure breaks out before the wait loop, so wait-only logging can
             * hide the failure mode. */
            static int issue_complained = 0;
            if (issue_complained < 8) {
                issue_complained++;
                fprintf(stderr, "siliangem: scattered ISSUE failed L%d e%d p%d - "
                        "err %lu (off %llu, span %u, buf+%llu, stride %u)\n",
                        layer, expert, p, GetLastError(),
                        (unsigned long long) lo[p], span[p],
                        (unsigned long long) ((size_t) p * g_src.bounce_slot),
                        st[p]);
            }
            CloseHandle(ev[p]);
            break;
        }
        issued++;
    }
    int ok = (issued == 3);
    for (int p = 0; p < issued; p++) {
        DWORD got = 0;
        /* st[p], not g_src.stride[p]: stride is now a per-(layer,part) array,
         * so the bare index would always read LAYER 0's value. Identical for a
         * uniformly quantised model and wrong for Q4_K_M. */
        BOOL  gr  = GetOverlappedResult(g_src.file, &ov[p], &got, TRUE);
        if (gr && got >= off_in[p] + st[p]) {
            g_siliangem.bytes_read += st[p];
        } else {
            /* Rate-limited: a failing fetch poisons the slot, so this can fire
             * once per miss and would otherwise bury the log. */
            static int complained = 0;
            if (complained < 8) {
                complained++;
                fprintf(stderr, "siliangem: scattered read failed L%d e%d p%d - "
                        "err %lu, got %lu, wanted >= %u (off_in %u, span %u, "
                        "base %llu, stride %u)\n",
                        layer, expert, p, gr ? 0UL : GetLastError(),
                        (unsigned long) got, off_in[p] + st[p],
                        off_in[p], span[p],
                        (unsigned long long) g_src.base[(size_t) layer * 3 + p],
                        st[p]);
            }
            ok = 0;
        }
        CloseHandle(ev[p]);
    }
    if (!ok) return 0;
    g_src_ok++;
    for (int p = 0; p < 3; p++) {
        memcpy(dst + part_off,
               g_src.bounce + (size_t) p * g_src.bounce_slot + off_in[p],
               st[p]);
        part_off += st[p];
    }
    g_src_expected_bytes += part_off;
    if (!g_src_layer_seen[layer]) {
        g_src_layer_seen[layer] = 1;
        g_src_layers_seen++;
    }
    return 1;
}

static void siliangem_ensure_init(void) {
    if (!siliangem_tls_state) return;
    if (!siliangem_tls_state->init_attempted) {
        siliangem_tls_state->init_attempted = 1;
        siliangem_init();
    }
}

/* slot index for key, or SILIANGEM_EMPTY */
static uint32_t siliangem_lookup(uint32_t key) {
    uint32_t i = siliangem_hash(key) & g_siliangem.mask;
    for (;;) {
        uint32_t s = g_siliangem.table[i];
        if (s == SILIANGEM_EMPTY) return SILIANGEM_EMPTY;
        if (g_siliangem.slots[s].key == key) return s;
        i = (i + 1) & g_siliangem.mask;
    }
}

static void siliangem_table_insert(uint32_t key, uint32_t slot) {
    uint32_t i = siliangem_hash(key) & g_siliangem.mask;
    while (g_siliangem.table[i] != SILIANGEM_EMPTY) i = (i + 1) & g_siliangem.mask;
    g_siliangem.table[i] = slot;
}

/* Rebuild the probe table; cheap enough at eviction rates we care about and
 * avoids tombstone bookkeeping in the hot path. */
static void siliangem_table_rebuild(void) {
    for (uint32_t i = 0; i <= g_siliangem.mask; i++) g_siliangem.table[i] = SILIANGEM_EMPTY;
    for (uint32_t s = 0; s < g_siliangem.nslots; s++) {
        if (g_siliangem.slots[s].key != SILIANGEM_EMPTY) siliangem_table_insert(g_siliangem.slots[s].key, s);
    }
}

static void siliangem_transient_reset(void) {
    g_siliangem.transient_count = 0;
    for (uint32_t i = 0; i < SILIANGEM_MAX_BATCH; ++i) g_siliangem.transient_keys[i] = SILIANGEM_EMPTY;
}

static int siliangem_transient_ensure(uint32_t capacity) {
    if (g_siliangem.placement_policy != SILIANGEM_POLICY_SLFU) return 1;
    if (capacity == 0) return 1;
    if (capacity > SILIANGEM_MAX_BATCH || g_siliangem.expert_bytes == 0 || g_siliangem.n_pending != 0) return 0;
    if (g_siliangem.transient_arena && g_siliangem.transient_capacity >= capacity) return 1;
    const SIZE_T bytes = (SIZE_T) capacity * g_siliangem.expert_bytes;
    uint8_t * replacement = (uint8_t *) VirtualAlloc(NULL, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!replacement) return 0;
    if (g_siliangem.transient_arena) VirtualFree(g_siliangem.transient_arena, 0, MEM_RELEASE);
    g_siliangem.transient_arena = replacement;
    g_siliangem.transient_capacity = capacity;
    siliangem_transient_reset();
    return 1;
}

static uint32_t siliangem_transient_find(uint32_t key) {
    for (uint32_t i = 0; i < g_siliangem.transient_count; ++i) {
        if (g_siliangem.transient_keys[i] == key) return i;
    }
    return SILIANGEM_EMPTY;
}

static int siliangem_policy_key_is_active(
        uint32_t key, int layer, const int64_t * counts, int n_as) {
    if ((int) (key >> 16) != layer) return 0;
    const uint32_t expert = key & 0xffffu;
    return expert < (uint32_t) n_as && counts[expert] != 0;
}

static uint64_t siliangem_policy_frequency(uint32_t key) {
    const uint32_t layer = key >> 16;
    const uint32_t expert = key & 0xffffu;
    if (!g_siliangem.freq || layer >= g_siliangem.n_layers || expert >= g_siliangem.n_experts) return 0;
    return g_siliangem.freq[(size_t) layer * g_siliangem.n_experts + expert];
}

static uint32_t siliangem_policy_oldest_segment(
        uint8_t segment, int layer, const int64_t * counts, int n_as) {
    uint32_t best = SILIANGEM_EMPTY;
    uint64_t oldest = ~0ull;
    for (uint32_t i = 0; i < g_siliangem.nslots; ++i) {
        const siliangem_slot * slot = &g_siliangem.slots[i];
        if (slot->key == SILIANGEM_EMPTY || slot->leases != 0 || slot->segment != segment ||
            siliangem_policy_key_is_active(slot->key, layer, counts, n_as)) continue;
        if (slot->stamp < oldest) { oldest = slot->stamp; best = i; }
    }
    return best;
}

static uint32_t siliangem_policy_segment_count(uint8_t segment) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < g_siliangem.nslots; ++i) {
        count += g_siliangem.slots[i].key != SILIANGEM_EMPTY &&
                 g_siliangem.slots[i].segment == segment;
    }
    return count;
}

static void siliangem_wtinylfu_normalize(
        int layer, const int64_t * counts, int n_as) {
    uint32_t window_count = siliangem_policy_segment_count(SILIANGEM_SEGMENT_WINDOW);
    while (window_count > g_siliangem.policy_window_slots) {
        const uint32_t demote = siliangem_policy_oldest_segment(
                SILIANGEM_SEGMENT_WINDOW, layer, counts, n_as);
        if (demote == SILIANGEM_EMPTY) break;
        g_siliangem.slots[demote].segment = SILIANGEM_SEGMENT_PROBATION;
        g_siliangem.slots[demote].stamp = ++g_siliangem.clock;
        --window_count;
    }

    uint32_t protected_count = siliangem_policy_segment_count(SILIANGEM_SEGMENT_PROTECTED);
    while (protected_count > g_siliangem.policy_protected_slots) {
        const uint32_t demote = siliangem_policy_oldest_segment(
                SILIANGEM_SEGMENT_PROTECTED, layer, counts, n_as);
        if (demote == SILIANGEM_EMPTY) break;
        g_siliangem.slots[demote].segment = SILIANGEM_SEGMENT_PROBATION;
        g_siliangem.slots[demote].stamp = ++g_siliangem.clock;
        --protected_count;
    }
}

static void siliangem_policy_record_hit(
        uint32_t slot, int layer, const int64_t * counts, int n_as) {
    ++g_siliangem.slots[slot].frequency;
    if (g_siliangem.placement_policy != SILIANGEM_POLICY_WTINYLFU_W10_SLRU_P80) return;
    if (g_siliangem.slots[slot].segment == SILIANGEM_SEGMENT_PROBATION) {
        g_siliangem.slots[slot].segment = SILIANGEM_SEGMENT_PROTECTED;
    }
    siliangem_wtinylfu_normalize(layer, counts, n_as);
}

static uint32_t siliangem_policy_choose_basic_slot(
        int layer, const int64_t * counts, int n_as) {
    uint32_t best = SILIANGEM_EMPTY;
    uint64_t best_frequency = UINT64_MAX;
    uint64_t oldest = UINT64_MAX;
    for (uint32_t i = 0; i < g_siliangem.nslots; ++i) {
        const siliangem_slot * slot = &g_siliangem.slots[i];
        if (slot->leases != 0 || siliangem_policy_key_is_active(slot->key, layer, counts, n_as)) continue;
        if (slot->key == SILIANGEM_EMPTY) return i;
        const uint64_t frequency = g_siliangem.placement_policy == SILIANGEM_POLICY_LFU
                ? slot->frequency : 0;
        if (best == SILIANGEM_EMPTY || frequency < best_frequency ||
            (frequency == best_frequency && slot->stamp < oldest)) {
            best = i;
            best_frequency = frequency;
            oldest = slot->stamp;
        }
    }
    return best;
}

static uint32_t siliangem_policy_choose_slot(
        uint32_t candidate_key, int layer, const int64_t * counts, int n_as) {
    if (g_siliangem.placement_policy == SILIANGEM_POLICY_SLFU) {
        uint32_t vacant = SILIANGEM_EMPTY;
        uint32_t victim = SILIANGEM_EMPTY;
        uint64_t victim_frequency = UINT64_MAX;
        uint64_t oldest = UINT64_MAX;
        for (uint32_t i = 0; i < g_siliangem.nslots; ++i) {
            const siliangem_slot * slot = &g_siliangem.slots[i];
            if (slot->leases != 0 || siliangem_policy_key_is_active(slot->key, layer, counts, n_as)) continue;
            if (slot->key == SILIANGEM_EMPTY) { vacant = i; break; }
            const uint64_t frequency = siliangem_policy_frequency(slot->key);
            if (victim == SILIANGEM_EMPTY || frequency < victim_frequency ||
                (frequency == victim_frequency && slot->stamp < oldest)) {
                victim = i;
                victim_frequency = frequency;
                oldest = slot->stamp;
            }
        }
        if (vacant != SILIANGEM_EMPTY) return vacant;
        if (victim == SILIANGEM_EMPTY) {
            // Every resident slot is active in the current request. Serving the
            // candidate from transient request scratch is correct; evicting an
            // active resident would not be.
            ++g_siliangem.policy_main_candidate_rejections;
            return SILIANGEM_BYPASS;
        }
        if (siliangem_policy_frequency(candidate_key) > victim_frequency) return victim;
        ++g_siliangem.policy_main_candidate_rejections;
        return SILIANGEM_BYPASS;
    }
    if (g_siliangem.placement_policy != SILIANGEM_POLICY_WTINYLFU_W10_SLRU_P80) {
        return siliangem_policy_choose_basic_slot(layer, counts, n_as);
    }

    siliangem_wtinylfu_normalize(layer, counts, n_as);

    uint32_t vacant = SILIANGEM_EMPTY;
    for (uint32_t i = 0; i < g_siliangem.nslots; ++i) {
        if (g_siliangem.slots[i].key == SILIANGEM_EMPTY && g_siliangem.slots[i].leases == 0) {
            vacant = i;
            break;
        }
    }
    const uint32_t window_count = siliangem_policy_segment_count(SILIANGEM_SEGMENT_WINDOW);
    if (vacant != SILIANGEM_EMPTY && window_count < g_siliangem.policy_window_slots) return vacant;

    uint32_t candidate = siliangem_policy_oldest_segment(
            SILIANGEM_SEGMENT_WINDOW, layer, counts, n_as);
    if (candidate == SILIANGEM_EMPTY) {
        if (vacant != SILIANGEM_EMPTY) return vacant;
        candidate = siliangem_policy_oldest_segment(
                SILIANGEM_SEGMENT_PROBATION, layer, counts, n_as);
        if (candidate == SILIANGEM_EMPTY) candidate = siliangem_policy_oldest_segment(
                SILIANGEM_SEGMENT_PROTECTED, layer, counts, n_as);
        return candidate;
    }

    if (vacant != SILIANGEM_EMPTY) {
        g_siliangem.slots[candidate].segment = SILIANGEM_SEGMENT_PROBATION;
        g_siliangem.slots[candidate].stamp = ++g_siliangem.clock;
        return vacant;
    }

    uint32_t victim = siliangem_policy_oldest_segment(
            SILIANGEM_SEGMENT_PROBATION, layer, counts, n_as);
    if (victim == SILIANGEM_EMPTY) victim = siliangem_policy_oldest_segment(
            SILIANGEM_SEGMENT_PROTECTED, layer, counts, n_as);
    if (victim == SILIANGEM_EMPTY) return candidate;

    if (siliangem_policy_frequency(g_siliangem.slots[candidate].key) >
        siliangem_policy_frequency(g_siliangem.slots[victim].key)) {
        g_siliangem.slots[candidate].segment = SILIANGEM_SEGMENT_PROBATION;
        g_siliangem.slots[candidate].stamp = ++g_siliangem.clock;
        return victim;
    }
    ++g_siliangem.policy_main_candidate_rejections;
    return candidate;
}

/* Ensure every expert selected this call is resident. Misses are issued as one
 * batch of overlapped unbuffered reads, so they queue on the device together
 * rather than serializing. Called on ith == 0 before the barrier. */
static int siliangem_prepare(const char *name, const int64_t *counts, int n_as) {
    siliangem_ensure_init();
    if (!g_siliangem.ready) return 0;

    int layer, part;
    if (!siliangem_parse_name(name, &layer, &part)) return 0;
    if ((uint32_t) layer >= g_siliangem.n_layers) return 0;
    if (part == 0 && g_siliangem.n_pending == 0) siliangem_transient_reset();

    uint32_t active = 0;
    uint32_t missing = 0;
    for (int a = 0; a < n_as; ++a) {
        if (counts[a] == 0) continue;
        ++active;
        const uint32_t key = ((uint32_t) layer << 16) | (uint32_t) a;
        missing += siliangem_lookup(key) == SILIANGEM_EMPTY;
    }
    if (active > SILIANGEM_MAX_BATCH) return 0;
    if (g_siliangem.placement_policy != SILIANGEM_POLICY_SLFU && active > g_siliangem.nslots) return 0;
    if (g_siliangem.placement_policy == SILIANGEM_POLICY_SLFU && !siliangem_transient_ensure(missing)) return 0;

    OVERLAPPED ov[SILIANGEM_MAX_BATCH];
    HANDLE     ev[SILIANGEM_MAX_BATCH];
    uint32_t   slot_of[SILIANGEM_MAX_BATCH];
    uint32_t   transient_of[SILIANGEM_MAX_BATCH];
    int        nmiss = 0;

    for (int a = 0; a < n_as && nmiss < SILIANGEM_MAX_BATCH; a++) {
        if (counts[a] == 0) continue;
        if ((uint32_t) a >= g_siliangem.n_experts) continue;
        if (part == 0) g_siliangem.expert_requests++;
        uint32_t key = ((uint32_t) layer << 16) | (uint32_t) a;
        uint32_t s = siliangem_lookup(key);
        const uint32_t transient = s == SILIANGEM_EMPTY ? siliangem_transient_find(key) : SILIANGEM_EMPTY;
        if (s != SILIANGEM_EMPTY) {                       /* persistent hit */
            g_siliangem.slots[s].stamp = ++g_siliangem.clock;
            siliangem_policy_record_hit(s, layer, counts, n_as);
            g_siliangem.hits++;
            if (part == 0) g_siliangem.expert_hits++;
            continue;
        }
        if (transient != SILIANGEM_EMPTY) {               /* current-request bypass already fetched */
            continue;
        }
        g_siliangem.misses++;
        s = siliangem_policy_choose_slot(key, layer, counts, n_as);
        if (s == SILIANGEM_EMPTY) return 0;
        uint32_t transient_index = SILIANGEM_EMPTY;
        uint8_t * fetch_destination = NULL;
        if (s == SILIANGEM_BYPASS) {
            if (!g_siliangem.transient_arena || g_siliangem.transient_count >= g_siliangem.transient_capacity) return 0;
            transient_index = g_siliangem.transient_count++;
            g_siliangem.transient_keys[transient_index] = key;
            fetch_destination = g_siliangem.transient_arena + (size_t) transient_index * g_siliangem.expert_bytes;
        } else {
            int was_occupied = (g_siliangem.slots[s].key != SILIANGEM_EMPTY);
            g_siliangem.slots[s].key   = key;
            g_siliangem.slots[s].stamp = ++g_siliangem.clock;
            g_siliangem.slots[s].frequency = 1;
            g_siliangem.slots[s].segment = g_siliangem.placement_policy == SILIANGEM_POLICY_WTINYLFU_W10_SLRU_P80
                    ? SILIANGEM_SEGMENT_WINDOW : SILIANGEM_SEGMENT_NONE;
            ++g_siliangem.policy_admissions;
            if (was_occupied) ++g_siliangem.policy_evictions;
            if (was_occupied) siliangem_table_rebuild(); else siliangem_table_insert(key, s);
            fetch_destination = g_siliangem.arena + (size_t) s * g_siliangem.expert_bytes;
        }

        /* Scattered-source arm: same arena and unbuffered path, but bytes come
         * from the stock GGUF as three projection reads instead of one packed
         * read. The reads within one expert are concurrent, while different
         * misses are processed per expert. Measure the layout effect on the
         * target model and storage device. */
        if (g_src.enabled) {
            if (!siliangem_src_fetch(layer, a, fetch_destination)) {
                if (transient_index != SILIANGEM_EMPTY) {
                    g_siliangem.transient_keys[transient_index] = SILIANGEM_EMPTY;
                } else {
                    g_siliangem.slots[s].key = SILIANGEM_EMPTY;
                    siliangem_table_rebuild();
                }
            }
            continue;
        }

        uint64_t off = siliangem_expert_off(layer, a);
        memset(&ov[nmiss], 0, sizeof(OVERLAPPED));
        ov[nmiss].Offset     = (DWORD) (off & 0xFFFFFFFFull);
        ov[nmiss].OffsetHigh = (DWORD) (off >> 32);
        ev[nmiss] = CreateEventA(NULL, TRUE, FALSE, NULL);
        ov[nmiss].hEvent = ev[nmiss];
        slot_of[nmiss] = s;
        transient_of[nmiss] = transient_index;

        if (!ReadFile(g_siliangem.file, fetch_destination,
                      siliangem_expert_len(layer), NULL, &ov[nmiss])
            && GetLastError() != ERROR_IO_PENDING) {
            /* Poison the destination so a failed read is never mistaken for data. */
            if (transient_index != SILIANGEM_EMPTY) {
                g_siliangem.transient_keys[transient_index] = SILIANGEM_EMPTY;
            } else {
                g_siliangem.slots[s].key = SILIANGEM_EMPTY;
                siliangem_table_rebuild();
            }
            CloseHandle(ev[nmiss]);
            continue;
        }
        nmiss++;
    }

    /* Same accounting as siliangem_wait(), so DEFER=0 and DEFER=1 are comparable. */
    if (nmiss) {
        int64_t t_start = siliangem_now_ns();
        g_siliangem.wait_calls++;
        for (int i = 0; i < nmiss; i++) {
            DWORD got = 0;
            if (GetOverlappedResult(g_siliangem.file, &ov[i], &got, TRUE)) {
                g_siliangem.bytes_read += got;
            } else {
                if (transient_of[i] != SILIANGEM_EMPTY) {
                    g_siliangem.transient_keys[transient_of[i]] = SILIANGEM_EMPTY;
                } else {
                    g_siliangem.slots[slot_of[i]].key = SILIANGEM_EMPTY;
                    siliangem_table_rebuild();
                }
            }
            CloseHandle(ev[i]);
        }
        g_siliangem.wait_ns += (uint64_t)(siliangem_now_ns() - t_start);
    }
    siliangem_maybe_report();
    return 1;
}

/* ---- deferred-wait path --------------------------------------------------
 * siliangem_prepare() issues reads for missing experts and then BLOCKS on all of them
 * before compute begins. When some experts are already resident, their compute
 * can instead overlap reads that are still in flight.
 *
 * siliangem_prepare_async() issues the reads and returns immediately, handing back an
 * ordering with hits first. The caller computes the hits, then calls siliangem_wait()
 * before touching the misses. No prediction is involved: the router has already
 * run, so this is exact -- it simply stops discarding overlap we already have.
 *
 * Returns 1 if it took ownership of ordering, 0 to fall back (no arena, not an
 * expert tensor, or the scattered-source arm, which fetches synchronously).
 *   order[]   active expert indices, hits first then misses
 *   *n_hits   how many leading entries are already resident
 *   *n_active total entries written to order[] */
static int siliangem_prepare_async(const char *name, const int64_t *counts, int n_as,
                            int *order, int *n_hits, int *n_active) {
    siliangem_ensure_init();
    *n_hits = 0; *n_active = 0;
    if (!g_siliangem.ready) return 0;
    if (!g_siliangem.defer)  return 0;       /* A/B switch: fall back to blocking */
    if (g_src.enabled) return 0;      /* scattered arm stays synchronous */
    if (g_siliangem.n_pending != 0) return 0;

    int layer, part;
    if (!siliangem_parse_name(name, &layer, &part)) return 0;
    if ((uint32_t) layer >= g_siliangem.n_layers) return 0;

    int miss_idx[SILIANGEM_MAX_BATCH], nmiss = 0;

    /* Capacity check BEFORE writing anything.
     *
     * order[] is sized SILIANGEM_MAX_BATCH and the caller's buffer is too, so an
     * active set larger than that has to be declined, not truncated: dropping
     * experts would silently lose their contribution. Declining falls back to
     * the natural order and the blocking path, which is slower but correct. */
    int n_act = 0;
    for (int a = 0; a < n_as; a++) {
        if (counts[a] != 0 && (uint32_t) a < g_siliangem.n_experts) n_act++;
    }
    if (n_act > SILIANGEM_MAX_BATCH) return 0;
    if (g_siliangem.placement_policy != SILIANGEM_POLICY_SLFU && (uint32_t) n_act > g_siliangem.nslots) return 0;
    if (g_siliangem.placement_policy == SILIANGEM_POLICY_SLFU && part == 0) siliangem_transient_reset();

    /* pass 1: hits and already-fetched transient bypasses go straight into
     * order[]; only not-yet-available experts are collected as misses. */
    for (int a = 0; a < n_as; a++) {
        if (counts[a] == 0) continue;
        if ((uint32_t) a >= g_siliangem.n_experts) continue;
        if (part == 0) g_siliangem.expert_requests++;
        uint32_t key = ((uint32_t) layer << 16) | (uint32_t) a;
        uint32_t s = siliangem_lookup(key);
        const uint32_t transient = s == SILIANGEM_EMPTY ? siliangem_transient_find(key) : SILIANGEM_EMPTY;
        if (s != SILIANGEM_EMPTY) {
            g_siliangem.slots[s].stamp = ++g_siliangem.clock;
            siliangem_policy_record_hit(s, layer, counts, n_as);
            g_siliangem.hits++;
            if (part == 0) g_siliangem.expert_hits++;
            order[(*n_hits)++] = a;
        } else if (transient != SILIANGEM_EMPTY) {
            order[(*n_hits)++] = a;
        } else if (nmiss < SILIANGEM_MAX_BATCH) {
            miss_idx[nmiss++] = a;
        }
    }
    *n_active = *n_hits;
    if (g_siliangem.placement_policy == SILIANGEM_POLICY_SLFU && !siliangem_transient_ensure((uint32_t) nmiss)) return 0;

    /* pass 2: issue the misses, do NOT wait */
    int64_t t_issue = nmiss ? siliangem_now_ns() : 0;
    for (int i = 0; i < nmiss; i++) {
        int a = miss_idx[i];
        g_siliangem.misses++;
        uint32_t key = ((uint32_t) layer << 16) | (uint32_t) a;
        int64_t t_ev = siliangem_now_ns();
        uint32_t s = siliangem_policy_choose_slot(key, layer, counts, n_as);
        if (s == SILIANGEM_EMPTY) {
            siliangem_wait();
            return 0;
        }
        uint32_t transient_index = SILIANGEM_EMPTY;
        uint8_t * fetch_destination = NULL;
        if (s == SILIANGEM_BYPASS) {
            if (!g_siliangem.transient_arena || g_siliangem.transient_count >= g_siliangem.transient_capacity) {
                siliangem_wait();
                return 0;
            }
            transient_index = g_siliangem.transient_count++;
            g_siliangem.transient_keys[transient_index] = key;
            fetch_destination = g_siliangem.transient_arena + (size_t) transient_index * g_siliangem.expert_bytes;
        } else {
            int was_occupied = (g_siliangem.slots[s].key != SILIANGEM_EMPTY);
            g_siliangem.slots[s].key = key;
            g_siliangem.slots[s].stamp = ++g_siliangem.clock;
            g_siliangem.slots[s].frequency = 1;
            g_siliangem.slots[s].segment = g_siliangem.placement_policy == SILIANGEM_POLICY_WTINYLFU_W10_SLRU_P80
                    ? SILIANGEM_SEGMENT_WINDOW : SILIANGEM_SEGMENT_NONE;
            ++g_siliangem.policy_admissions;
            if (was_occupied) ++g_siliangem.policy_evictions;
            if (was_occupied) siliangem_table_rebuild(); else siliangem_table_insert(key, s);
            fetch_destination = g_siliangem.arena + (size_t) s * g_siliangem.expert_bytes;
        }
        int64_t t_after_evict = siliangem_now_ns();
        g_siliangem.evict_ns += (uint64_t)(t_after_evict - t_ev);

        uint64_t off = siliangem_expert_off(layer, a);
        int p = g_siliangem.n_pending;
        memset(&g_siliangem.pend_ov[p], 0, sizeof(OVERLAPPED));
        g_siliangem.pend_ov[p].Offset     = (DWORD) (off & 0xFFFFFFFFull);
        g_siliangem.pend_ov[p].OffsetHigh = (DWORD) (off >> 32);
        g_siliangem.pend_ev[p] = CreateEventA(NULL, TRUE, FALSE, NULL);
        g_siliangem.pend_ov[p].hEvent = g_siliangem.pend_ev[p];
        g_siliangem.pend_slot[p] = s;
        g_siliangem.pend_transient[p] = transient_index;
        int64_t t_after_event = siliangem_now_ns();
        g_siliangem.event_ns += (uint64_t)(t_after_event - t_after_evict);

        BOOL rf_ok = ReadFile(g_siliangem.file, fetch_destination,
                              siliangem_expert_len(layer), NULL, &g_siliangem.pend_ov[p]);
        DWORD rf_err = rf_ok ? 0 : GetLastError();
        g_siliangem.read_ns += (uint64_t)(siliangem_now_ns() - t_after_event);
        if (rf_ok) g_siliangem.n_sync++; else if (rf_err == ERROR_IO_PENDING) g_siliangem.n_pend++;

        if (!rf_ok && rf_err != ERROR_IO_PENDING) {
            if (transient_index != SILIANGEM_EMPTY) {
                g_siliangem.transient_keys[transient_index] = SILIANGEM_EMPTY;
            } else {
                g_siliangem.slots[s].key = SILIANGEM_EMPTY;
                siliangem_table_rebuild();
            }
            CloseHandle(g_siliangem.pend_ev[p]);
            /* Still emit it. Dropping it from order[] would skip the expert
             * entirely and silently lose its contribution to those tokens --
             * turning an I/O error into wrong output instead of a slow one.
             * With the key poisoned, siliangem_ptr() returns NULL and the caller
             * falls back to the mmap'd weights, which is what the blocking
             * path does too. */
            order[(*n_active)++] = a;
            continue;
        }
        g_siliangem.n_pending++;
        order[(*n_active)++] = a;            /* placed AFTER all hits */
    }
    if (nmiss) g_siliangem.fetch_ns += (uint64_t)(siliangem_now_ns() - t_issue);
    siliangem_maybe_report();
    return 1;
}

/* Block until every read issued by siliangem_prepare_async has landed. */
static void siliangem_wait(void) {
    if (g_siliangem.n_pending == 0) return;
    int64_t t_start = siliangem_now_ns();
    g_siliangem.wait_calls++;
    for (int i = 0; i < g_siliangem.n_pending; i++) {
        DWORD got = 0;
        if (GetOverlappedResult(g_siliangem.file, &g_siliangem.pend_ov[i], &got, TRUE)) {
            g_siliangem.bytes_read += got;
        } else if (g_siliangem.pend_transient[i] != SILIANGEM_EMPTY) {
            g_siliangem.transient_keys[g_siliangem.pend_transient[i]] = SILIANGEM_EMPTY;
        } else {
            g_siliangem.slots[g_siliangem.pend_slot[i]].key = SILIANGEM_EMPTY;
            siliangem_table_rebuild();
        }
        CloseHandle(g_siliangem.pend_ev[i]);
    }
    g_siliangem.n_pending = 0;
    g_siliangem.wait_ns += (uint64_t)(siliangem_now_ns() - t_start);
}

/* Pointer to expert `a`'s slice of `name`. NULL permits ordinary model-backing
 * fallback, but managed-only callers treat NULL as a correctness failure.
 * Read-only and called from every worker thread after the barrier. */
static const char *siliangem_ptr(const char *name, int a, size_t stride, int ith, int managed_only) {
    if (!g_siliangem.ready) return NULL;
    int layer, part;
    if (!siliangem_parse_name(name, &layer, &part)) return NULL;
    if ((uint32_t) layer >= g_siliangem.n_layers || (uint32_t) a >= g_siliangem.n_experts) return NULL;
    /* The tensor's own stride must match the source layout, otherwise the two
     * disagree and the mmap pointer is the safe answer.
     *   slab: nb02 is the part's own per-expert size
     *   expert-major GGUF: the loader set nb02 to the whole expert stride */
    uint32_t poff;
    if (g_siliangem.em) {
        if (part >= g_siliangem.em_nparts) return NULL;
        /* What nb[2] means depends on where the geometry came from:
         *
         *   expert-major mmap  loader SET nb[2] to the whole expert stride
         *   managed no-mmap    proxy keeps logical per-PART contiguous stride
         *   stock GGUF         nb[2] is the tensor's own per-PART stride
         *
         * Comparing a stock tensor's per-part stride with a packed whole-expert
         * stride fails, and siliangem_ptr returning NULL is silent: every expert can
         * fall back to mmap while the arena still pays to fetch. */
        const uint32_t want = (managed_only || g_siliangem.em_scattered)
                            ? g_siliangem.em_pbytes[layer * g_siliangem.em_nparts + part]
                            : g_siliangem.em_stride[layer];
        if (stride != (size_t) want) return NULL;
        poff = g_siliangem.em_poff[layer * g_siliangem.em_nparts + part];
    } else {
        if (stride != (size_t) g_siliangem.part_bytes[part]) return NULL;
        poff = g_siliangem.part_off[part];
    }
    const uint32_t key = ((uint32_t) layer << 16) | (uint32_t) a;
    const uint32_t s = siliangem_lookup(key);
    const uint32_t transient = s == SILIANGEM_EMPTY ? siliangem_transient_find(key) : SILIANGEM_EMPTY;
    if (s == SILIANGEM_EMPTY && transient == SILIANGEM_EMPTY) return NULL;
    const char *cached = s != SILIANGEM_EMPTY
        ? (const char *) (g_siliangem.arena + (size_t) s * g_siliangem.expert_bytes + poff)
        : (const char *) (g_siliangem.transient_arena + (size_t) transient * g_siliangem.expert_bytes + poff);
    if (g_src.enabled && ith == 0 && !g_src_layer_substituted[layer]) {
        g_src_layer_substituted[layer] = 1;
        g_src_layers_substituted++;
    }
    return cached;
}

#endif /* _WIN32 */
