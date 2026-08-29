#pragma once

#include "llama.h"
#include "llama-cparams.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace siliang_moe_prefill {

constexpr size_t route_bitmap_word_bits = 64;
constexpr size_t route_bitmap_word_count = 4;
constexpr size_t route_bitmap_expert_capacity = route_bitmap_word_bits * route_bitmap_word_count;
static_assert(route_bitmap_expert_capacity == LLAMA_SILIANG_MOE_PREFILL_MAX_EXPERTS);
using route_bitmap = std::array<uint64_t, route_bitmap_word_count>;

inline uint32_t route_bitmap_count(const route_bitmap & bitmap) {
    uint32_t count = 0;
    for (uint64_t word : bitmap) {
        while (word != 0) {
            word &= word - 1;
            ++count;
        }
    }
    return count;
}

inline uint32_t route_bitmap_intersection_count(
        const route_bitmap & lhs,
        const route_bitmap & rhs) {
    route_bitmap intersection = {};
    for (size_t index = 0; index < intersection.size(); ++index) {
        intersection[index] = lhs[index] & rhs[index];
    }
    return route_bitmap_count(intersection);
}

struct route_union {
    std::vector<int32_t> experts;
    std::vector<uint32_t> occurrences;
    std::vector<uint32_t> union_index_by_choice;
    route_bitmap bitmap = {};
};

inline bool build_route_union(
        const int32_t * logical,
        size_t count,
        int32_t expert_count,
        size_t capacity,
        route_union & result) {
    result = {};
    if (!logical || count == 0 || expert_count <= 0 ||
        expert_count > static_cast<int32_t>(route_bitmap_expert_capacity) || capacity == 0) {
        return false;
    }

    std::vector<int32_t> union_index_by_expert(static_cast<size_t>(expert_count), -1);
    result.experts.reserve(std::min(count, capacity));
    result.occurrences.reserve(std::min(count, capacity));
    result.union_index_by_choice.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        const int32_t expert = logical[index];
        if (expert < 0 || expert >= expert_count) {
            result = {};
            return false;
        }
        result.bitmap[static_cast<size_t>(expert) / route_bitmap_word_bits] |=
            uint64_t {1} << (static_cast<size_t>(expert) % route_bitmap_word_bits);
        int32_t union_index = union_index_by_expert[static_cast<size_t>(expert)];
        if (union_index < 0) {
            if (result.experts.size() >= capacity) {
                result = {};
                return false;
            }
            union_index = static_cast<int32_t>(result.experts.size());
            union_index_by_expert[static_cast<size_t>(expert)] = union_index;
            result.experts.push_back(expert);
            result.occurrences.push_back(0);
        }
        auto & occurrences = result.occurrences[static_cast<size_t>(union_index)];
        if (occurrences == UINT32_MAX) {
            result = {};
            return false;
        }
        ++occurrences;
        result.union_index_by_choice.push_back(static_cast<uint32_t>(union_index));
    }
    return true;
}

} // namespace siliang_moe_prefill

namespace siliang_moe_policy {

enum class slot_segment : uint8_t {
    none,
    window,
    probation,
    protected_main,
};

struct cache_slot {
    int64_t key = -1;
    uint64_t last_used = 0;
    slot_segment segment = slot_segment::none;
    uint64_t frequency = 0;
};

struct admission_decision {
    int32_t slot = -1;
    bool main_candidate_rejected = false;
};

inline bool contains_key(const std::vector<int64_t> & keys, int64_t key) {
    return std::find(keys.begin(), keys.end(), key) != keys.end();
}

inline bool cumulative_lfu_should_admit(
        const std::vector<uint64_t> & frequencies,
        int64_t candidate_key,
        int64_t victim_key) {
    return candidate_key >= 0 && victim_key >= 0 &&
        static_cast<size_t>(candidate_key) < frequencies.size() &&
        static_cast<size_t>(victim_key) < frequencies.size() &&
        frequencies[static_cast<size_t>(candidate_key)] > frequencies[static_cast<size_t>(victim_key)];
}

inline admission_decision cumulative_lfu_choose_slot(
        const std::vector<cache_slot> & slots,
        const std::vector<uint64_t> & frequencies,
        uint32_t first,
        uint32_t last,
        int64_t candidate_key,
        const std::vector<int64_t> & protected_keys) {
    if (first >= last || last > slots.size()) {
        return {-1, true};
    }

    int32_t victim = -1;
    for (uint32_t index = first; index < last; ++index) {
        const auto & slot = slots[index];
        if (slot.key < 0) {
            return {static_cast<int32_t>(index), false};
        }
        if (contains_key(protected_keys, slot.key)) {
            continue;
        }
        if (static_cast<size_t>(slot.key) >= frequencies.size()) {
            return {-1, true};
        }
        if (victim < 0) {
            victim = static_cast<int32_t>(index);
            continue;
        }
        const auto & selected = slots[static_cast<size_t>(victim)];
        const uint64_t slot_frequency = frequencies[static_cast<size_t>(slot.key)];
        const uint64_t victim_frequency = frequencies[static_cast<size_t>(selected.key)];
        if (slot_frequency < victim_frequency ||
            (slot_frequency == victim_frequency && slot.last_used < selected.last_used)) {
            victim = static_cast<int32_t>(index);
        }
    }

    if (victim >= 0 && cumulative_lfu_should_admit(
            frequencies, candidate_key, slots[static_cast<size_t>(victim)].key)) {
        return {victim, false};
    }
    return {-1, true};
}

inline int32_t lfu_choose_slot(
        const std::vector<cache_slot> & slots,
        uint32_t first,
        uint32_t last,
        const std::vector<int64_t> & protected_keys) {
    int32_t result = -1;
    for (uint32_t index = first; index < last && index < slots.size(); ++index) {
        const auto & slot = slots[index];
        if (slot.key < 0) {
            return static_cast<int32_t>(index);
        }
        if (contains_key(protected_keys, slot.key)) {
            continue;
        }
        if (result < 0 || slot.frequency < slots[static_cast<size_t>(result)].frequency ||
            (slot.frequency == slots[static_cast<size_t>(result)].frequency &&
             slot.last_used < slots[static_cast<size_t>(result)].last_used)) {
            result = static_cast<int32_t>(index);
        }
    }
    return result;
}

inline void lfu_record_hit(cache_slot & slot) {
    if (slot.frequency != std::numeric_limits<uint64_t>::max()) {
        ++slot.frequency;
    }
}

inline uint32_t wtinylfu_window_limit(uint32_t capacity, uint32_t route_width) {
    const uint32_t tenth = capacity / 10 + (capacity % 10 != 0);
    return std::min(capacity, std::max(route_width, tenth));
}

inline int32_t oldest_slot(
        const std::vector<cache_slot> & slots,
        uint32_t first,
        uint32_t last,
        slot_segment segment,
        const std::vector<int64_t> & protected_keys) {
    int32_t result = -1;
    for (uint32_t index = first; index < last; ++index) {
        const auto & slot = slots[index];
        if (slot.key < 0 || slot.segment != segment || contains_key(protected_keys, slot.key)) {
            continue;
        }
        if (result < 0 || slot.last_used < slots[static_cast<size_t>(result)].last_used) {
            result = static_cast<int32_t>(index);
        }
    }
    return result;
}

inline admission_decision wtinylfu_choose_slot(
        std::vector<cache_slot> & slots,
        const std::vector<uint64_t> & frequencies,
        uint32_t first,
        uint32_t last,
        uint32_t route_width,
        const std::vector<int64_t> & protected_keys,
        uint64_t & clock) {
    if (route_width == 0 || first >= last || last > slots.size()) {
        return {};
    }

    int32_t vacant = -1;
    uint32_t window_count = 0;
    for (uint32_t index = first; index < last; ++index) {
        vacant = vacant < 0 && slots[index].key < 0 ? static_cast<int32_t>(index) : vacant;
        window_count += slots[index].key >= 0 && slots[index].segment == slot_segment::window;
    }

    const uint32_t window_limit = wtinylfu_window_limit(last - first, route_width);
    if (window_count < window_limit) {
        if (vacant >= 0) {
            return {vacant, false};
        }
        for (slot_segment segment : {slot_segment::probation, slot_segment::protected_main}) {
            const int32_t victim = oldest_slot(slots, first, last, segment, protected_keys);
            if (victim >= 0) {
                return {victim, false};
            }
        }
        return {};
    }

    const int32_t candidate = oldest_slot(
            slots, first, last, slot_segment::window, protected_keys);
    if (candidate < 0) {
        return {};
    }
    if (vacant >= 0) {
        slots[static_cast<size_t>(candidate)].segment = slot_segment::probation;
        slots[static_cast<size_t>(candidate)].last_used = ++clock;
        return {vacant, false};
    }

    int32_t victim = oldest_slot(slots, first, last, slot_segment::probation, protected_keys);
    if (victim < 0) {
        victim = oldest_slot(slots, first, last, slot_segment::protected_main, protected_keys);
    }
    if (victim < 0) {
        return {candidate, false};
    }

    const int64_t candidate_key = slots[static_cast<size_t>(candidate)].key;
    const int64_t victim_key = slots[static_cast<size_t>(victim)].key;
    if (candidate_key < 0 || victim_key < 0 ||
        static_cast<size_t>(candidate_key) >= frequencies.size() ||
        static_cast<size_t>(victim_key) >= frequencies.size()) {
        return {};
    }
    if (frequencies[static_cast<size_t>(candidate_key)] > frequencies[static_cast<size_t>(victim_key)]) {
        slots[static_cast<size_t>(candidate)].segment = slot_segment::probation;
        slots[static_cast<size_t>(candidate)].last_used = ++clock;
        return {victim, false};
    }
    return {candidate, true};
}

inline bool wtinylfu_record_hit(
        std::vector<cache_slot> & slots,
        uint32_t first,
        uint32_t last,
        uint32_t route_width,
        int32_t slot,
        const std::vector<int64_t> & protected_keys,
        uint64_t & clock) {
    if (route_width == 0 || first >= last || last > slots.size() || slot < 0 ||
        static_cast<uint32_t>(slot) < first || static_cast<uint32_t>(slot) >= last ||
        slots[static_cast<size_t>(slot)].segment != slot_segment::probation) {
        return false;
    }

    const uint32_t window_limit = wtinylfu_window_limit(last - first, route_width);
    const uint32_t main_capacity = last - first - window_limit;
    const uint32_t protected_limit = main_capacity * 80 / 100;
    if (protected_limit == 0) {
        return false;
    }

    uint32_t protected_count = 0;
    for (uint32_t index = first; index < last; ++index) {
        protected_count += slots[index].segment == slot_segment::protected_main;
    }
    while (protected_count >= protected_limit) {
        const int32_t demote = oldest_slot(
                slots, first, last, slot_segment::protected_main, protected_keys);
        if (demote < 0) {
            return false;
        }
        slots[static_cast<size_t>(demote)].segment = slot_segment::probation;
        slots[static_cast<size_t>(demote)].last_used = ++clock;
        --protected_count;
    }
    slots[static_cast<size_t>(slot)].segment = slot_segment::protected_main;
    return true;
}

} // namespace siliang_moe_policy

struct siliang_moe_runtime;

LLAMA_API struct siliang_moe_runtime * siliang_moe_runtime_create(
        struct llama_model * model,
        struct llama_context * ctx,
        const struct llama_siliang_expert_cache_params * params);

LLAMA_API void siliang_moe_runtime_free(struct siliang_moe_runtime * runtime);

LLAMA_API void siliang_moe_runtime_reset_prefill_trace(struct siliang_moe_runtime * runtime);
