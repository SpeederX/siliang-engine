#include "../src/siliang-moe-runtime.h"

#include <cstdint>
#include <type_traits>
#include <vector>

#undef NDEBUG
#include <cassert>

using expected_slot_mapper = int (*)(void *, int32_t, const int32_t *, int32_t *, size_t);
static_assert(std::is_same<llama_siliang_moe_arena_slot_mapper, expected_slot_mapper>::value, "slot mapper must be IDs-only");

static void test_batch_union_preserves_first_seen_order_and_choice_mapping() {
    const std::vector<int32_t> logical = {
        0, 1, 2, 3, 4, 5,
        0, 2, 4, 6, 8, 10,
        10, 8, 6, 4, 2, 0,
    };
    siliang_moe_prefill::route_union route;

    assert(siliang_moe_prefill::build_route_union(
            logical.data(), logical.size(), 256, 216, route));
    assert((route.experts == std::vector<int32_t> {0, 1, 2, 3, 4, 5, 6, 8, 10}));
    assert((route.occurrences == std::vector<uint32_t> {3, 1, 3, 1, 3, 1, 2, 2, 2}));
    assert(route.union_index_by_choice.size() == logical.size());
    for (size_t index = 0; index < logical.size(); ++index) {
        assert(route.experts[route.union_index_by_choice[index]] == logical[index]);
    }
    assert(siliang_moe_prefill::route_bitmap_count(route.bitmap) == route.experts.size());
}

static void test_union_capacity_is_fail_closed() {
    const std::vector<int32_t> logical = {0, 1, 2, 3, 4, 5, 6};
    siliang_moe_prefill::route_union route;

    assert(siliang_moe_prefill::build_route_union(
            logical.data(), logical.size(), 256, logical.size(), route));
    assert(route.experts.size() == logical.size());
    assert(!siliang_moe_prefill::build_route_union(
            logical.data(), logical.size(), 256, logical.size() - 1, route));
    assert(route.experts.empty());
    assert(route.occurrences.empty());
    assert(route.union_index_by_choice.empty());
    assert(siliang_moe_prefill::route_bitmap_count(route.bitmap) == 0);
}

static void test_large_microbatch_is_bounded_by_expert_count() {
    std::vector<int32_t> logical(512 * 6);
    for (size_t index = 0; index < logical.size(); ++index) {
        logical[index] = static_cast<int32_t>(index % 256);
    }
    siliang_moe_prefill::route_union route;

    assert(siliang_moe_prefill::build_route_union(
            logical.data(), logical.size(), 256, 256, route));
    assert(route.experts.size() == 256);
    assert(route.union_index_by_choice.size() == logical.size());
    assert(siliang_moe_prefill::route_bitmap_count(route.bitmap) == 256);
    assert(!siliang_moe_prefill::build_route_union(
            logical.data(), logical.size(), 256, 255, route));
}

static void test_invalid_route_is_fail_closed() {
    siliang_moe_prefill::route_union route;
    const std::vector<int32_t> negative = {0, -1};
    const std::vector<int32_t> too_large = {0, 256};

    assert(!siliang_moe_prefill::build_route_union(
            negative.data(), negative.size(), 256, 2, route));
    assert(!siliang_moe_prefill::build_route_union(
            too_large.data(), too_large.size(), 256, 2, route));
    assert(!siliang_moe_prefill::build_route_union(
            nullptr, 2, 256, 2, route));
    assert(!siliang_moe_prefill::build_route_union(
            too_large.data(), 0, 256, 2, route));
    assert(!siliang_moe_prefill::build_route_union(
            too_large.data(), too_large.size(), 257, 2, route));
}

static void test_bitmap_word_boundaries_and_intersection() {
    const std::vector<int32_t> first = {0, 63, 64, 127, 128, 191, 192, 255, 255};
    const std::vector<int32_t> second = {0, 64, 128, 192};
    siliang_moe_prefill::route_union first_route;
    siliang_moe_prefill::route_union second_route;

    assert(siliang_moe_prefill::build_route_union(
            first.data(), first.size(), 256, 256, first_route));
    assert(siliang_moe_prefill::build_route_union(
            second.data(), second.size(), 256, 256, second_route));
    assert(siliang_moe_prefill::route_bitmap_count(first_route.bitmap) == 8);
    assert(first_route.occurrences.back() == 2);
    assert(siliang_moe_prefill::route_bitmap_intersection_count(
            first_route.bitmap, second_route.bitmap) == 4);

    const std::vector<int32_t> replacement = {1};
    assert(siliang_moe_prefill::build_route_union(
            replacement.data(), replacement.size(), 256, 256, first_route));
    assert(siliang_moe_prefill::route_bitmap_count(first_route.bitmap) == 1);
    assert(first_route.bitmap[0] == (uint64_t {1} << 1));
}

int main() {
    test_batch_union_preserves_first_seen_order_and_choice_mapping();
    test_union_capacity_is_fail_closed();
    test_large_microbatch_is_bounded_by_expert_count();
    test_invalid_route_is_fail_closed();
    test_bitmap_word_boundaries_and_intersection();
    return 0;
}
