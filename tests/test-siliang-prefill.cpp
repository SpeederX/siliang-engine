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
    // expert_count above the 512-expert bitmap capacity
    assert(!siliang_moe_prefill::build_route_union(
            too_large.data(), too_large.size(), 513, 2, route));
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

static siliang_moe_hybrid::cost_coefficients hybrid_cost(
        double cpu_l2, double cpu_miss, double stage_l2, double stage_pinned, double stage_miss, double gpu) {
    siliang_moe_hybrid::cost_coefficients cost;
    cost.cpu_l2_us = cpu_l2;
    cost.cpu_miss_us = cpu_miss;
    cost.stage_l2_us = stage_l2;
    cost.stage_pinned_us = stage_pinned;
    cost.stage_miss_us = stage_miss;
    cost.gpu_us = gpu;
    return cost;
}

static siliang_moe_hybrid::route_classes hybrid_route(uint32_t pinned, uint32_t l2, uint32_t miss, uint32_t g) {
    siliang_moe_hybrid::route_classes route;
    route.pinned = pinned;
    route.l2 = l2;
    route.miss = miss;
    route.g = g;
    return route;
}

static void test_hybrid_cost_table_keeps_gpu_when_staging_is_cheap() {
    siliang_moe_hybrid::cost_table table;
    assert(table.build(hybrid_cost(900, 2000, 50, 20, 400, 50), 10));
    for (uint32_t l = 0; l <= 10; ++l) {
        for (uint32_t m = 0; l + m <= 10; ++m) {
            const auto share = table.lookup(hybrid_route(0, l, m, 0));
            assert(share.pinned == 0 && share.l2 == 0 && share.miss == 0);
        }
    }
}

static void test_hybrid_cost_table_moves_work_to_cpu_when_staging_dominates() {
    siliang_moe_hybrid::cost_table table;
    assert(table.build(hybrid_cost(100, 300, 200, 200, 900, 100), 10));
    const auto l2_only = table.lookup(hybrid_route(0, 4, 0, 0));
    assert(l2_only.l2 == 4 && l2_only.miss == 0);
    const auto mixed = table.lookup(hybrid_route(0, 2, 3, 1));
    assert(mixed.l2 == 2 && mixed.miss == 3);
}

static void test_hybrid_cost_table_balances_the_overlap() {
    // x=1 of four L2 hits: 300 staged + max(300 cpu, 450 gpu) = 750, the minimum
    siliang_moe_hybrid::cost_table table;
    const auto cost = hybrid_cost(300, 900, 100, 100, 700, 150);
    assert(table.build(cost, 10));
    siliang_moe_hybrid::cpu_share one;
    one.l2 = 1;
    assert(siliang_moe_hybrid::predicted_us(cost, hybrid_route(0, 4, 0, 0), one) == 750.0);
    const auto share = table.lookup(hybrid_route(0, 4, 0, 0));
    assert(share.l2 == 1 && share.miss == 0);
}

static void test_hybrid_cost_table_sends_pinned_hits_to_gpu_and_the_rest_to_cpu() {
    // measured-style costs: CPU 350, P staging 400, pinned submit 40, miss staging 900, H2D + GPU 170.
    // Three pinned hits, one other L2 hit, one miss: pinned to GPU = 120 + max(350 + 650, 510) = 1120.
    siliang_moe_hybrid::cost_table table;
    const auto cost = hybrid_cost(350, 650, 400, 40, 900, 170);
    assert(table.build(cost, 10));
    const auto share = table.lookup(hybrid_route(3, 1, 1, 0));
    assert(share.pinned == 0 && share.l2 == 1 && share.miss == 1);
    siliang_moe_hybrid::cpu_share chosen;
    chosen.l2 = 1;
    chosen.miss = 1;
    assert(siliang_moe_hybrid::predicted_us(cost, hybrid_route(3, 1, 1, 0), chosen) == 1120.0);
}

static void test_hybrid_cost_table_ties_and_bounds_keep_gpu() {
    siliang_moe_hybrid::cost_table table;
    // one L2 hit: 100 staged + 1 gpu == 101 cpu, so the tie stays on GPU
    assert(table.build(hybrid_cost(101, 900, 100, 100, 700, 1), 6));
    assert(table.lookup(hybrid_route(0, 1, 0, 0)).l2 == 0);
    // outside pinned + l2 + miss + g <= top_k
    assert(table.lookup(hybrid_route(0, 7, 0, 0)).l2 == 0);
    const auto outside = table.lookup(hybrid_route(1, 3, 3, 0));
    assert(outside.pinned == 0 && outside.l2 == 0 && outside.miss == 0);
    assert(!table.build(hybrid_cost(0, 900, 100, 100, 700, 1), 6));
    assert(!table.ready());
    assert(!table.build(hybrid_cost(100, 900, 100, 100, 700, 1), 0));
}

int main() {
    test_batch_union_preserves_first_seen_order_and_choice_mapping();
    test_union_capacity_is_fail_closed();
    test_large_microbatch_is_bounded_by_expert_count();
    test_invalid_route_is_fail_closed();
    test_bitmap_word_boundaries_and_intersection();
    test_hybrid_cost_table_keeps_gpu_when_staging_is_cheap();
    test_hybrid_cost_table_moves_work_to_cpu_when_staging_dominates();
    test_hybrid_cost_table_balances_the_overlap();
    test_hybrid_cost_table_sends_pinned_hits_to_gpu_and_the_rest_to_cpu();
    test_hybrid_cost_table_ties_and_bounds_keep_gpu();
    return 0;
}
