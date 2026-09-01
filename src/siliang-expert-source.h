#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>


// Non-owning receipt for tensor bytes that remain owned by the model file.
// Managed residency layers may use these verified offsets to materialize their
// own transient/resident copies, but do not become semantic/source owners.
struct llama_siliang_file_tensor_ref {
    std::string name;
    uint64_t offset = 0;
    uint64_t bytes = 0;
};

struct llama_siliang_file_source_receipt {
    std::string path;
    std::vector<llama_siliang_file_tensor_ref> tensors;

    const llama_siliang_file_tensor_ref * find(const char * name) const {
        if (name == nullptr) {
            return nullptr;
        }
        for (const auto & tensor : tensors) {
            if (tensor.name == name) {
                return &tensor;
            }
        }
        return nullptr;
    }

    bool valid() const {
        return !path.empty() && !tensors.empty();
    }
};

enum class llama_siliang_expert_source_kind : uint8_t {
    none = 0,
    legacy_slab,
    expert_major,
    scattered,
};

// Immutable geometry copied from the loader into the model. A context uses it
// to configure its own CPU expert cache without process-global source setters.
struct llama_siliang_expert_source {
    llama_siliang_expert_source_kind kind = llama_siliang_expert_source_kind::none;
    std::string path;
    uint32_t n_layers = 0;
    uint32_t n_experts = 0;
    uint32_t n_parts = 0;
    std::vector<uint64_t> base;
    std::vector<uint32_t> stride;
    std::vector<uint32_t> part_offset;
    std::vector<uint32_t> part_bytes;
    std::string part_names;

    bool valid() const {
        if (kind == llama_siliang_expert_source_kind::expert_major) {
            return !path.empty() && n_layers > 0 && n_experts > 0 && n_parts > 0 &&
                   base.size() == n_layers && stride.size() == n_layers &&
                   part_offset.size() == static_cast<size_t>(n_layers) * n_parts &&
                   part_bytes.size() == static_cast<size_t>(n_layers) * n_parts &&
                   !part_names.empty();
        }
        if (kind == llama_siliang_expert_source_kind::scattered) {
            return !path.empty() && n_layers > 0 && n_experts > 0 && n_parts > 0 &&
                   base.size() == static_cast<size_t>(n_layers) * n_parts &&
                   stride.size() == static_cast<size_t>(n_layers) * n_parts &&
                   !part_names.empty();
        }
        return kind == llama_siliang_expert_source_kind::none;
    }
};
