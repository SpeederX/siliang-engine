#include "arg.h"
#include "common.h"
#include "download.h"
#include "llama.h"
#include "../src/siliang-moe-runtime.h"

#include <string>
#include <vector>
#include <sstream>
#include <unordered_set>
#include <limits>

#undef NDEBUG
#include <cassert>

static void test_wtinylfu_policy() {
    using siliang_moe_policy::cache_slot;
    using siliang_moe_policy::slot_segment;

    std::vector<cache_slot> slots(10);
    std::vector<uint64_t> frequencies(16, 1);
    uint64_t clock = 1;

    slots[0] = {0, clock, slot_segment::window};
    auto decision = siliang_moe_policy::wtinylfu_choose_slot(
            slots, frequencies, 0, slots.size(), 1, {}, clock);
    assert(decision.slot == 1);
    assert(!decision.main_candidate_rejected);
    assert(slots[0].segment == slot_segment::probation);

    slots[1] = {1, ++clock, slot_segment::window};
    assert(siliang_moe_policy::wtinylfu_record_hit(
            slots, 0, slots.size(), 1, 0, {}, clock));
    assert(slots[0].segment == slot_segment::protected_main);

    std::vector<cache_slot> small(2);
    small[0] = {0, 1, slot_segment::probation};
    uint64_t small_clock = 1;
    assert(!siliang_moe_policy::wtinylfu_record_hit(
            small, 0, small.size(), 1, 0, {0}, small_clock));
    assert(small[0].segment == slot_segment::probation);

    for (size_t index = 0; index < slots.size(); ++index) {
        slots[index] = {
            static_cast<int64_t>(index),
            static_cast<uint64_t>(index + 1),
            index == 0 ? slot_segment::window :
            index == 1 ? slot_segment::probation : slot_segment::protected_main,
        };
    }
    frequencies[0] = 5;
    frequencies[1] = 1;
    decision = siliang_moe_policy::wtinylfu_choose_slot(
            slots, frequencies, 0, slots.size(), 1, {}, clock);
    assert(decision.slot == 1);
    assert(!decision.main_candidate_rejected);
    assert(slots[0].segment == slot_segment::probation);

    slots[0].segment = slot_segment::window;
    slots[0].last_used = 1;
    frequencies[0] = 1;
    frequencies[1] = 5;
    decision = siliang_moe_policy::wtinylfu_choose_slot(
            slots, frequencies, 0, slots.size(), 1, {}, clock);
    assert(decision.slot == 0);
    assert(decision.main_candidate_rejected);

    slots[0] = {0, 10, slot_segment::window};
    slots[1] = {1, 1, slot_segment::probation};
    slots[2] = {2, 2, slot_segment::probation};
    frequencies[0] = 10;
    frequencies[1] = 1;
    frequencies[2] = 1;
    decision = siliang_moe_policy::wtinylfu_choose_slot(
            slots, frequencies, 0, slots.size(), 1, {1}, clock);
    assert(decision.slot == 2);

    for (size_t index = 0; index < slots.size(); ++index) {
        slots[index] = {
            static_cast<int64_t>(index),
            static_cast<uint64_t>(index + 1),
            index < 7 ? slot_segment::protected_main :
            index == 7 ? slot_segment::probation : slot_segment::window,
        };
    }
    const std::vector<int64_t> active_protected = {0, 1, 2, 3, 4, 5, 6};
    assert(!siliang_moe_policy::wtinylfu_record_hit(
            slots, 0, slots.size(), 1, 7, active_protected, clock));
    assert(slots[7].segment == slot_segment::probation);
    assert(siliang_moe_policy::wtinylfu_record_hit(
            slots, 0, slots.size(), 1, 7, {}, clock));
    assert(slots[0].segment == slot_segment::probation);
    assert(slots[7].segment == slot_segment::protected_main);
}

static void test_lfu_policy_split() {
    using siliang_moe_policy::cache_slot;
    using siliang_moe_policy::slot_segment;

    static_assert(COMMON_EXPERT_CACHE_POLICY_LRU == 0);
    static_assert(COMMON_EXPERT_CACHE_POLICY_LFU == 1);
    static_assert(COMMON_EXPERT_CACHE_POLICY_WTINYLFU_W10_SLRU_P80 == 2);
    static_assert(COMMON_EXPERT_CACHE_POLICY_CUMULATIVE_LFU_ADMISSION == 3);
    static_assert(LLAMA_SILIANG_EXPERT_CACHE_POLICY_LRU == 0);
    static_assert(LLAMA_SILIANG_EXPERT_CACHE_POLICY_LFU == 1);
    static_assert(LLAMA_SILIANG_EXPERT_CACHE_POLICY_WTINYLFU_W10_SLRU_P80 == 2);
    static_assert(LLAMA_SILIANG_EXPERT_CACHE_POLICY_CUMULATIVE_LFU_ADMISSION == 3);

    std::vector<cache_slot> slots(3);
    slots[0] = {0, 5, slot_segment::none, 4};
    slots[1] = {1, 2, slot_segment::none, 1};
    slots[2] = {2, 1, slot_segment::none, 1};
    assert(siliang_moe_policy::lfu_choose_slot(slots, 0, slots.size(), {}) == 2);
    assert(siliang_moe_policy::lfu_choose_slot(slots, 0, slots.size(), {2}) == 1);
    assert(siliang_moe_policy::lfu_choose_slot(slots, 0, slots.size(), {0, 1, 2}) == -1);

    siliang_moe_policy::lfu_record_hit(slots[1]);
    assert(slots[1].frequency == 2);
    slots[0].frequency = std::numeric_limits<uint64_t>::max();
    siliang_moe_policy::lfu_record_hit(slots[0]);
    assert(slots[0].frequency == std::numeric_limits<uint64_t>::max());
    slots[2] = {};
    assert(slots[2].frequency == 0);
    assert(siliang_moe_policy::lfu_choose_slot(slots, 0, slots.size(), {}) == 2);

    const std::vector<uint64_t> frequencies = {3, 3, 4};

    assert(!siliang_moe_policy::cumulative_lfu_should_admit(frequencies, 0, 1));
    assert(siliang_moe_policy::cumulative_lfu_should_admit(frequencies, 2, 0));
    assert(!siliang_moe_policy::cumulative_lfu_should_admit(frequencies, 0, 2));
    assert(!siliang_moe_policy::cumulative_lfu_should_admit(frequencies, 3, 0));

    slots[0] = {0, 5, slot_segment::none, 1};
    slots[1] = {1, 2, slot_segment::none, 100};
    slots[2] = {};
    const std::vector<uint64_t> cumulative = {8, 1, 4, 2, 9, 1};
    auto decision = siliang_moe_policy::cumulative_lfu_choose_slot(
            slots, cumulative, 0, slots.size(), 3, {});
    assert(decision.slot == 2);
    assert(!decision.main_candidate_rejected);

    slots[2] = {2, 1, slot_segment::none, 50};
    assert(siliang_moe_policy::lfu_choose_slot(slots, 0, slots.size(), {}) == 0);
    decision = siliang_moe_policy::cumulative_lfu_choose_slot(
            slots, cumulative, 0, slots.size(), 3, {});
    assert(decision.slot == 1);
    assert(!decision.main_candidate_rejected);

    decision = siliang_moe_policy::cumulative_lfu_choose_slot(
            slots, cumulative, 0, slots.size(), 5, {});
    assert(decision.slot == -1);
    assert(decision.main_candidate_rejected);

    decision = siliang_moe_policy::cumulative_lfu_choose_slot(
            slots, cumulative, 0, slots.size(), 4, {1});
    assert(decision.slot == 2);
    assert(!decision.main_candidate_rejected);

    decision = siliang_moe_policy::cumulative_lfu_choose_slot(
            slots, cumulative, 0, slots.size(), 4, {0, 1, 2});
    assert(decision.slot == -1);
    assert(decision.main_candidate_rejected);
}

static void test(void) {
    test_wtinylfu_policy();
    test_lfu_policy_split();

    common_params params;

    printf("test-arg-parser: make sure there is no duplicated arguments in any examples\n\n");
    for (int ex = 0; ex < LLAMA_EXAMPLE_COUNT; ex++) {
        try {
            auto ctx_arg = common_params_parser_init(params, (enum llama_example)ex);
            common_params_add_preset_options(ctx_arg.options);
            std::unordered_set<std::string> seen_args;
            std::unordered_set<std::string> seen_env_vars;
            for (const auto & opt : ctx_arg.options) {
                // check for args duplications
                for (const auto & arg : opt.get_args()) {
                    if (seen_args.find(arg) == seen_args.end()) {
                        seen_args.insert(arg);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same argument: %s", arg.c_str());
                        exit(1);
                    }
                }
                // check for env var duplications
                for (const auto & env : opt.get_env()) {
                    if (seen_env_vars.find(env) == seen_env_vars.end()) {
                        seen_env_vars.insert(env);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same env var: %s", env.c_str());
                        exit(1);
                    }
                }

                // exclude spec args from this check
                // ref: https://github.com/ggml-org/llama.cpp/pull/22397
                const bool skip = opt.is_spec;

                // ensure shorter argument precedes longer argument
                if (!skip && opt.args.size() > 1) {
                    const std::string first(opt.args.front());
                    const std::string last(opt.args.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }

                // same check for negated arguments
                if (opt.args_neg.size() > 1) {
                    const std::string first(opt.args_neg.front());
                    const std::string last(opt.args_neg.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter negated argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }
            }
        } catch (std::exception & e) {
            printf("%s\n", e.what());
            assert(false);
        }
    }

    auto list_str_to_char = [](std::vector<std::string> & argv) -> std::vector<char *> {
        std::vector<char *> res;
        for (auto & arg : argv) {
            res.push_back(const_cast<char *>(arg.data()));
        }
        return res;
    };

    std::vector<std::string> argv;

    printf("test-arg-parser: test expert cache option contract\n\n");

    {
        common_params expert_defaults;
        assert(expert_defaults.expert_cache.enabled == false);
        assert(expert_defaults.expert_cache.l2_mib == 0);
        assert(expert_defaults.expert_cache.l2_policy == COMMON_EXPERT_CACHE_POLICY_LRU);
        assert(expert_defaults.expert_cache.l1_k == 0);
        assert(expert_defaults.expert_cache.exchange_r == 0);
        assert(expert_defaults.expert_cache.elevator_p == 0);
        assert(expert_defaults.expert_cache.l1_policy == COMMON_EXPERT_CACHE_POLICY_CUMULATIVE_LFU_ADMISSION);
        assert(expert_defaults.expert_cache.roll == COMMON_EXPERT_CACHE_ROLL_OFF);
        assert(expert_defaults.expert_cache.memory_report == true);
        assert(expert_defaults.expert_cache.route_stats == false);
        assert(expert_defaults.expert_cache.admit_k_cold == true);
        assert(expert_defaults.expert_cache.demote_k_hot == false);
        assert(expert_defaults.expert_cache.deferred_wait == true);
        assert(expert_defaults.expert_cache.tier_configured == false);
        const auto llama_defaults = llama_context_default_params();
        assert(llama_defaults.expert_cache.l1_policy == LLAMA_SILIANG_EXPERT_CACHE_POLICY_CUMULATIVE_LFU_ADMISSION);
        assert(llama_defaults.expert_cache.route_stats == false);
        assert(llama_defaults.expert_cache.admit_k_cold == true);
        assert(llama_defaults.expert_cache.demote_k_hot == false);
        assert(llama_defaults.expert_cache.deferred_wait == true);

        auto find_option = [](common_params_context & ctx, const std::string & arg) -> const common_arg * {
            for (const common_arg & opt : ctx.options) {
                for (const std::string & candidate : opt.get_args()) {
                    if (candidate == arg) {
                        return &opt;
                    }
                }
            }
            return nullptr;
        };

        const char * expert_args[] = {
            "--expert-cache",
            "--no-expert-cache",
            "--expert-cache-l2-mib",
            "--expert-cache-l2-policy",
            "--expert-cache-l1-k",
            "--expert-cache-exchange-r",
            "--expert-cache-elevator-p",
            "--expert-cache-l1-policy",
            "--admit-k-cold",
            "--demote-k-hot",
            "--expert-cache-roll",
            "--expert-cache-prefill",
            "--no-expert-cache-prefill",
            "--expert-cache-memory-report",
            "--no-expert-cache-memory-report",
            "--expert-cache-route-stats",
            "--no-expert-cache-route-stats",
            "--expert-cache-deferred-wait",
            "--no-expert-cache-deferred-wait",
        };

        common_params cli_params;
        common_params server_params;
        auto cli_ctx = common_params_parser_init(cli_params, LLAMA_EXAMPLE_CLI);
        auto server_ctx = common_params_parser_init(server_params, LLAMA_EXAMPLE_SERVER);
        for (const char * arg : expert_args) {
            const common_arg * cli_opt = find_option(cli_ctx, arg);
            const common_arg * server_opt = find_option(server_ctx, arg);
            assert(cli_opt != nullptr);
            assert(server_opt != nullptr);
            assert(cli_opt->env == nullptr);
            assert(server_opt->env == nullptr);
        }

        const common_arg * l2_policy_opt = find_option(cli_ctx, "--expert-cache-l2-policy");
        assert(l2_policy_opt != nullptr);
        assert(std::string(l2_policy_opt->value_hint).find("wtinylfu-w10-slru-p80") != std::string::npos);
        assert(std::string(l2_policy_opt->value_hint).find("cumulative-lfu") == std::string::npos);
        assert(l2_policy_opt->help.find("W-TinyLFU W10/SLRU-P80") != std::string::npos);
        const common_arg * l1_k_opt = find_option(cli_ctx, "--expert-cache-l1-k");
        assert(l1_k_opt != nullptr);
        assert(l1_k_opt->help.find("incompatible with LoRA adapters") != std::string::npos);
        const common_arg * l1_policy_opt = find_option(cli_ctx, "--expert-cache-l1-policy");
        assert(l1_policy_opt != nullptr);
        assert(std::string(l1_policy_opt->value_hint).find("slfu") != std::string::npos);
        assert(std::string(l1_policy_opt->value_hint).find("{lfu,slfu") != std::string::npos);
        assert(l1_policy_opt->help.find("SLFU") != std::string::npos);
        assert(l1_policy_opt->help.find("LRU is retired") != std::string::npos);
    }

    auto parse_expert_args = [&](std::vector<std::string> args, common_params & expert_params) {
        auto arg_ptrs = list_str_to_char(args);
        return common_params_parse(
            static_cast<int>(arg_ptrs.size()), arg_ptrs.data(), expert_params, LLAMA_EXAMPLE_SERVER);
    };

    {
        common_params expert_params;
        assert(parse_expert_args({
            "binary_name",
            "--expert-cache",
            "--expert-cache-l2-mib", "8192",
            "--expert-cache-l2-policy", "wtinylfu-w10-slru-p80",
            "--expert-cache-l1-k", "64",
            "--expert-cache-exchange-r", "16",
            "--expert-cache-elevator-p", "8",
            "--expert-cache-l1-policy", "slfu",
            "--admit-k-cold", "off",
            "--demote-k-hot", "on",
            "--expert-cache-roll", "deepseek4",
            "--expert-cache-prefill",
            "--ubatch-size", "8",
            "--no-expert-cache-memory-report",
            "--expert-cache-route-stats",
            "--no-expert-cache-deferred-wait",
            "--parallel", "1",
        }, expert_params));
        assert(expert_params.expert_cache.enabled == true);
        assert(expert_params.expert_cache.l2_mib == 8192);
        assert(expert_params.expert_cache.l2_policy == COMMON_EXPERT_CACHE_POLICY_WTINYLFU_W10_SLRU_P80);
        assert(expert_params.expert_cache.l1_k == 64);
        assert(expert_params.expert_cache.exchange_r == 16);
        assert(expert_params.expert_cache.elevator_p == 8);
        assert(expert_params.expert_cache.l1_policy == COMMON_EXPERT_CACHE_POLICY_CUMULATIVE_LFU_ADMISSION);
        assert(expert_params.expert_cache.roll == COMMON_EXPERT_CACHE_ROLL_DEEPSEEK4);
        assert(expert_params.expert_cache.prefill == true);
        assert(expert_params.n_ubatch == 8);
        assert(expert_params.expert_cache.memory_report == false);
        assert(expert_params.expert_cache.route_stats == true);
        assert(expert_params.expert_cache.admit_k_cold == false);
        assert(expert_params.expert_cache.demote_k_hot == true);
        assert(expert_params.expert_cache.deferred_wait == false);
        const auto context_params = common_context_params_to_llama(expert_params);
        assert(context_params.expert_cache.l1_policy == LLAMA_SILIANG_EXPERT_CACHE_POLICY_CUMULATIVE_LFU_ADMISSION);
        assert(context_params.expert_cache.route_stats == true);
        assert(context_params.expert_cache.admit_k_cold == false);
        assert(context_params.expert_cache.demote_k_hot == true);
    }
    {
        common_params expert_params;
        assert(parse_expert_args({
            "binary_name",
            "--expert-cache",
            "--expert-cache-l1-k", "64",
            "--expert-cache-exchange-r", "16",
            "--expert-cache-elevator-p", "16",
            "--expert-cache-l1-policy", "lfu",
            "--parallel", "1",
        }, expert_params));
        assert(expert_params.expert_cache.l1_policy == COMMON_EXPERT_CACHE_POLICY_LFU);
        const auto context_params = common_context_params_to_llama(expert_params);
        assert(context_params.expert_cache.l1_policy == LLAMA_SILIANG_EXPERT_CACHE_POLICY_LFU);
    }

    {
        common_params expert_params;
        assert(false == parse_expert_args({
            "binary_name", "--expert-cache-l2-mib", "1",
        }, expert_params));
    }
    {
        common_params expert_params;
        assert(false == parse_expert_args({
            "binary_name", "--expert-cache-prefill",
        }, expert_params));
    }
    {
        common_params expert_params;
        assert(false == parse_expert_args({
            "binary_name", "--expert-cache", "--expert-cache-l1-k", "64",
            "--expert-cache-exchange-r", "16", "--expert-cache-elevator-p", "16",
            "--expert-cache-prefill", "--ubatch-size", "1", "--parallel", "1",
        }, expert_params));
    }
    {
        common_params expert_params;
        assert(false == parse_expert_args({
            "binary_name", "--expert-cache",
        }, expert_params));
    }
    {
        common_params expert_params;
        assert(false == parse_expert_args({
            "binary_name", "--expert-cache", "--expert-cache-l1-k", "1",
            "--expert-cache-exchange-r", "1",
        }, expert_params));
    }
    {
        common_params expert_params;
        assert(false == parse_expert_args({
            "binary_name", "--expert-cache", "--expert-cache-l2-mib", "1",
            "--expert-cache-exchange-r", "1",
        }, expert_params));
    }
    {
        common_params expert_params;
        assert(false == parse_expert_args({
            "binary_name", "--expert-cache", "--expert-cache-l1-k", "4294967295",
            "--expert-cache-exchange-r", "1", "--expert-cache-elevator-p", "1",
        }, expert_params));
    }
    {
        common_params expert_params;
        assert(false == parse_expert_args({
            "binary_name", "--expert-cache", "--expert-cache-l2-mib", "-1",
        }, expert_params));
    }
    {
        common_params expert_params;
        assert(false == parse_expert_args({
            "binary_name", "--expert-cache", "--expert-cache-l2-mib", "17592186044416",
        }, expert_params));
    }
    {
        common_params expert_params;
        assert(false == parse_expert_args({
            "binary_name", "--expert-cache", "--expert-cache-l2-mib", "1",
            "--expert-cache-l2-policy", "fifo",
        }, expert_params));
    }
    {
        common_params expert_params;
        assert(false == parse_expert_args({
            "binary_name", "--expert-cache", "--expert-cache-l2-mib", "1",
            "--expert-cache-l2-policy", "cumulative-lfu",
        }, expert_params));
    }
    {
        common_params expert_params;
        assert(false == parse_expert_args({
            "binary_name", "--expert-cache",
            "--expert-cache-l1-k", "64",
            "--expert-cache-exchange-r", "16",
            "--expert-cache-elevator-p", "16",
            "--parallel", "1",
            "--spec-type", "ngram-simple",
        }, expert_params));
    }
    {
        common_params expert_params;
        assert(false == parse_expert_args({
            "binary_name", "--expert-cache",
            "--expert-cache-l1-k", "64",
            "--expert-cache-exchange-r", "16",
            "--expert-cache-elevator-p", "16",
        }, expert_params));
    }
    {
        common_params expert_params;
        assert(false == parse_expert_args({
            "binary_name", "--expert-cache",
            "--expert-cache-l1-k", "64",
            "--expert-cache-exchange-r", "16",
            "--expert-cache-elevator-p", "16",
            "--parallel", "4",
        }, expert_params));
    }
    {
        common_params expert_params;
        assert(false == parse_expert_args({
            "binary_name", "--expert-cache",
            "--expert-cache-l1-k", "64",
            "--expert-cache-exchange-r", "16",
            "--expert-cache-elevator-p", "16",
            "--parallel", "1",
            "--lora", "adapter.gguf",
        }, expert_params));
    }
    {
        common_params expert_params;
        assert(parse_expert_args({
            "binary_name", "--expert-cache", "--expert-cache-l2-mib", "1",
            "--lora", "adapter.gguf",
        }, expert_params));
    }

    printf("test-arg-parser: test invalid usage\n\n");

    // missing value
    argv = {"binary_name", "-m"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (int)
    argv = {"binary_name", "-ngl", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (enum)
    argv = {"binary_name", "-sm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    {
        common_params penalty_params;

        argv = {"binary_name", "--repeat-penalty", "0"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "nan"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "inf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "-inf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        const char * penalty_options[] = {"--frequency-penalty", "--presence-penalty"};
        const char * nonfinite_values[] = {"nan", "inf", "-inf"};
        for (const char * option : penalty_options) {
            for (const char * value : nonfinite_values) {
                argv = {"binary_name", option, value};
                assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));
            }
        }
    }

    // non-existence arg in specific example (--draft cannot be used outside llama-speculative)
    argv = {"binary_name", "--draft", "123"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_EMBEDDING));

    argv = {"binary_name", "-lm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    printf("test-arg-parser: test valid usage\n\n");

    argv = {"binary_name", "-m", "model_file.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "model_file.gguf");

    argv = {"binary_name", "-t", "1234"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.cpuparams.n_threads == 1234);

    argv = {"binary_name", "--verbose"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.verbosity > 1);

    argv = {"binary_name", "-m", "abc.gguf", "--predict", "6789", "--batch-size", "9090"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "abc.gguf");
    assert(params.n_predict == 6789);
    assert(params.n_batch == 9090);

    // --draft cannot be used outside llama-speculative
    argv = {"binary_name", "--spec-draft-n-max", "123"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.draft.n_max == 123);

    argv = {"binary_name", "-lm", "none"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_NONE);

    argv = {"binary_name", "-lm", "mmap"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP);

    argv = {"binary_name", "-lm", "mlock"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MLOCK);

    argv = {"binary_name", "-lm", "mmap+mlock"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);

    argv = {"binary_name", "-lm", "dio"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);

    // multi-value args (CSV)
    argv = {"binary_name", "--lora", "file1.gguf,\"file2,2.gguf\",\"file3\"\"3\"\".gguf\",file4\".gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.lora_adapters.size() == 4);
    assert(params.lora_adapters[0].path == "file1.gguf");
    assert(params.lora_adapters[1].path == "file2,2.gguf");
    assert(params.lora_adapters[2].path == "file3\"3\".gguf");
    assert(params.lora_adapters[3].path == "file4\".gguf");

// skip this part on windows, because setenv is not supported
#ifdef _WIN32
    printf("test-arg-parser: skip on windows build\n");
#else
    printf("test-arg-parser: test environment variables (valid + invalid usages)\n\n");

    setenv("LLAMA_ARG_THREADS", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "blah.gguf");
    assert(params.cpuparams.n_threads == 1010);

    setenv("LLAMA_ARG_LOAD_MODE", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_LOAD_MODE", "mmap", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP);

    setenv("LLAMA_ARG_LOAD_MODE", "mlock", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MLOCK);

    setenv("LLAMA_ARG_LOAD_MODE", "mmap+mlock", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);

    setenv("LLAMA_ARG_LOAD_MODE", "dio", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);

    printf("test-arg-parser: test negated environment variables\n\n");

    setenv("LLAMA_ARG_LOAD_MODE", "none", true);
    setenv("LLAMA_ARG_NO_PERF", "1", true); // legacy format
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_NONE);
    assert(params.no_perf == true);

    printf("test-arg-parser: test environment variables being overwritten\n\n");

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name", "-m", "overwritten.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "overwritten.gguf");
    assert(params.cpuparams.n_threads == 1010);
#endif // _WIN32

    printf("test-arg-parser: test download functions\n\n");
    const char * GOOD_URL = "http://ggml.ai/";
    const char * BAD_URL  = "http://ggml.ai/404";

    {
        printf("test-arg-parser: test good URL\n\n");
        auto res = common_remote_get_content(GOOD_URL, {});
        assert(res.first == 200);
        assert(res.second.size() > 0);
        std::string str(res.second.data(), res.second.size());
        assert(str.find("llama.cpp") != std::string::npos);
    }

    {
        printf("test-arg-parser: test bad URL\n\n");
        auto res = common_remote_get_content(BAD_URL, {});
        assert(res.first == 404);
    }

    {
        printf("test-arg-parser: test max size error\n");
        common_remote_params params;
        params.max_size = 1;
        try {
            common_remote_get_content(GOOD_URL, params);
            assert(false && "it should throw an error");
        } catch (std::exception & e) {
            printf("  expected error: %s\n\n", e.what());
        }
    }

    printf("test-arg-parser: all tests OK\n\n");
}

int main(void) {
    try {
        test();
    } catch (std::exception & e) {
        fprintf(stderr, "test-arg-parser: exception: %s\n", e.what());
        return 1;
    }
    return 0;
}
