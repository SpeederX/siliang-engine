#pragma once

#include "llama.h"

#include "llama-impl.h"
#include "llama-arch.h"
#include "llama-hparams.h"
#include "llama-mmap.h"

#include "ggml-cpp.h"

#include <cstddef>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>

using llama_buf_map = std::unordered_map<uint32_t, ggml_backend_buffer_t>;

// lists of buffer types used for each layer
using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;

enum llama_fver {
    GGUF_FILE_VERSION_V1 = 1,
    GGUF_FILE_VERSION_V2 = 2,
    GGUF_FILE_VERSION_V3 = 3,
};

const char * llama_file_version_name(llama_fver version);

struct llama_model_loader {
    // Holds information on a model weight
    struct llama_tensor_weight {
        uint16_t  idx; // source file index
        size_t   offs; // tensor data offset in the original file

        ggml_tensor * tensor;

        // Direct construction, for entries that are SYNTHESISED rather than
        // read from the tensor directory. The Siliang expert-major layout
        // stores each layer's experts as one opaque packed tensor; the loader
        // derives gate/up/down entries pointing into it, which have no
        // directory entry of their own for gguf_find_tensor to locate.
        llama_tensor_weight(uint16_t idx, size_t offs, ggml_tensor * tensor)
            : idx(idx), offs(offs), tensor(tensor) {}

        llama_tensor_weight(const llama_file * file, uint16_t idx, const struct gguf_context * gguf_ctx, ggml_tensor * tensor) : idx(idx), tensor(tensor) {
            const int tensor_idx = gguf_find_tensor(gguf_ctx,  ggml_get_name(tensor));
            if (tensor_idx < 0) {
                throw std::runtime_error(format("tensor '%s' not found in the model", ggml_get_name(tensor)));
            }

            offs = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);
            if (offs + ggml_nbytes(tensor) < offs || offs + ggml_nbytes(tensor) > file->size()) {
                throw std::runtime_error(format("tensor '%s' data is not within the file bounds, model is corrupted or incomplete", ggml_get_name(tensor)));
            }
        }
    };

    // custom comparator to sort weights more nicely by layer
    struct weight_name_comparer {
        bool operator()(const std::string & a, const std::string & b) const {
            int a_layer = -1;
            int b_layer = -1;
            sscanf(a.c_str(), "blk.%d.", &a_layer);
            sscanf(b.c_str(), "blk.%d.", &b_layer);
            if (a_layer != b_layer) {
                return a_layer < b_layer;
            }
            return a < b;
        }
    };

    static const int TENSOR_NOT_REQUIRED    = 1 << 0;
    static const int TENSOR_DUPLICATED      = 1 << 1;
    static const int TENSOR_SKIP            = 1 << 2;
    static const int TENSOR_SKIP_IF_VIRTUAL = 1 << 3;
    static const int TENSOR_ALLOW_RESHAPE   = 1 << 4;

    int n_kv      = 0;
    int n_tensors = 0;
    int n_created = 0;

    uint64_t n_elements = 0;
    size_t   n_bytes    = 0;

    bool use_mmap = false;
    bool use_direct_io = false;
    bool check_tensors;
    bool no_alloc;
    bool load_mtp;

    llama_files files;
    llama_ftype ftype;
    llama_fver  fver;

    llama_mmaps mappings;

    // ---- Siliang expert-major layout ------------------------------------
    // An expert-major GGUF stores each layer's experts interleaved:
    //   [e0: gate|up|down|pad][e1: ...]
    // so one expert is one contiguous read instead of scattered projection
    // reads. gate/up/down remain real tensor-directory entries with their true
    // names, shapes and types, given overlapping offsets into that region.
    //
    // Everything in llama.cpp then works unchanged EXCEPT nb[2], which is
    // derived assuming contiguity. This map supplies the true expert stride
    // per layer; create_tensor() writes it into nb[2].
    bool expert_major = false;
    std::map<int, uint64_t> expert_stride;   // layer -> bytes between experts

    // ---- expert-major on a DEVICE buffer ---------------------------------
    // The interleaving exists to make a DISK read contiguous, and only the
    // out-of-core cache reads from disk. A device-resident expert layer is
    // copied once at load and never touched by the cache again, so it has no
    // reason to be interleaved - and staying interleaved is expensive:
    // ggml_nbytes() measures an address SPAN, part + (ne2-1)*nb2, so each view
    // spans the packed region rather than only its logical bytes. The allocator
    // can then reserve and copy redundant device memory, reducing the number of
    // expert layers that fit.
    //
    // So for a device buffer create_tensor() leaves nb[] contiguous - the
    // tensor is then perfectly ordinary, and CUDA's nb[2]/type_size division
    // stops being load-bearing - and load_all_data() gathers the experts one
    // at a time out of the strided file region. Host tensors are untouched:
    // they still point straight into the mapping with the true stride, which
    // is what the cache requires.
    std::set<const struct ggml_tensor *> em_deinterleaved;

    // Sum over em_deinterleaved of (address span - true size). init_mappings()
    // subtracts it from size_data, which is summed over the strided metas.
    size_t em_span_excess = 0;

    std::map<std::string, llama_tensor_weight, weight_name_comparer> weights_map;
    std::unordered_map<std::string, llama_model_kv_override> kv_overrides;
    const llama_model_tensor_buft_override * tensor_buft_overrides;

    gguf_context_ptr metadata_ptr;
    struct gguf_context * metadata; // either metadata_ptr.get() or externally set
    llama_model_set_tensor_data_t set_tensor_data;
    void * set_tensor_data_ud;
    std::vector<ggml_context_ptr> contexts;

    std::string arch_name;
    LLM_KV      llm_kv    = LLM_KV(LLM_ARCH_UNKNOWN);

    size_t size_done = 0;
    size_t size_data = 0;
    std::vector<std::pair<size_t, size_t>> mmaps_used;

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };

    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // track tensors that had to be moved for debugging:
    size_t n_tensors_moved = 0;
    std::string first_tensor_moved_name;
    std::string first_tensor_moved_type_name;
    ggml_backend_buffer_type_t first_moved_from_buft = nullptr;
    ggml_backend_buffer_type_t first_moved_to_buft = nullptr;

    llama_model_loader(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        const std::string & fname,
        std::vector<std::string> & splits, // optional, only need if the split does not follow naming scheme
        FILE * file,
        llama_load_mode load_mode,
        bool check_tensors,
        bool no_alloc,
        bool load_mtp,
        const llama_model_kv_override * param_overrides_p,
        const llama_model_tensor_buft_override * param_tensor_buft_overrides_p);

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(const std::string & key, T & result, bool required = true);

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(enum llm_kv kid, T & result, bool required = true);

    template<typename T>
    bool get_arr(const std::string & key, std::vector<T> & result, bool required = true);

    template<typename T, size_t N_MAX>
    bool get_arr(const std::string & key, std::array<T, N_MAX> & result, bool required = true);

    template<typename T>
    bool get_arr(enum llm_kv kid, T & result, bool required = true);

    template<typename T>
    bool get_key(const std::string & key, T & result, bool required = true);

    template<typename T>
    bool get_key(enum llm_kv kid, T & result, bool required = true);

    template<typename T, size_t N_MAX>
    bool get_key_or_arr(const std::string & key, std::array<T, N_MAX> & result, uint32_t n, bool required = true);

    template<typename T>
    bool get_key_or_arr(enum llm_kv kid, T & result, uint32_t n, bool required = true);

    bool get_key_or_arr(enum llm_kv kid, uint32_t & result, bool required = true);

    std::string get_arch_name() const;

    enum llm_arch get_arch() const;

    const llama_tensor_weight * get_weight(const char * name) const;

    const llama_tensor_weight & require_weight(const char * name) const;

    struct ggml_tensor * get_tensor_meta(const char * name) const;

    struct ggml_tensor * require_tensor_meta(const std::string & name) const;

    const struct ggml_tensor * check_tensor_dims(
            const std::string & name,
            const std::vector<int64_t> & ne,
            bool required,
            bool allow_reshape) const;

    struct ggml_tensor * create_tensor(
        const llama_hparams & hparams, const buft_list_t * buft_list_cpu, const buft_list_t * buft_list_input, const buft_list_t * buft_list_output,
        const buft_list_t * buft_list_layer, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags);

    void done_getting_tensors(bool partial = false) const;

    void init_mappings(bool prefetch = true, llama_mlocks * mlock_mmaps = nullptr);

    void get_mapping_range(size_t * first, size_t * last, void ** addr, int idx, ggml_context * ctx) const;

    // for backwards compatibility, does not support ggml-backend
    void load_data_for(struct ggml_tensor * cur) const;

    // Returns false if cancelled by progress_callback
    bool load_all_data(
            struct ggml_context * ctx,
            llama_buf_map & bufs,
            llama_mlocks * lmlocks,
            llama_progress_callback progress_callback,
            void * progress_callback_user_data);

    std::string ftype_name() const;

    void print_info() const;
};
