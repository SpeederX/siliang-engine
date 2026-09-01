from __future__ import annotations

from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CACHE_SOURCE = (REPOSITORY_ROOT / "ggml/src/ggml-cpu/siliangem_moe_cache.h").read_text(encoding="utf-8")
CPU_HEADER = (REPOSITORY_ROOT / "ggml/include/ggml-cpu.h").read_text(encoding="utf-8")
COMMON_HEADER = (REPOSITORY_ROOT / "common/common.h").read_text(encoding="utf-8")
COMMON_SOURCE = (REPOSITORY_ROOT / "common/common.cpp").read_text(encoding="utf-8")
ARG_SOURCE = (REPOSITORY_ROOT / "common/arg.cpp").read_text(encoding="utf-8")
README = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
CONFIGURATION = (REPOSITORY_ROOT / "docs/CONFIGURATION.md").read_text(encoding="utf-8")
LLAMA_HEADER = (REPOSITORY_ROOT / "include/llama.h").read_text(encoding="utf-8")
LLAMA_CPARAMS = (REPOSITORY_ROOT / "src/llama-cparams.h").read_text(encoding="utf-8")
LLAMA_GRAPH = (REPOSITORY_ROOT / "src/llama-graph.cpp").read_text(encoding="utf-8")
LLAMA_CONTEXT = (REPOSITORY_ROOT / "src/llama-context.cpp").read_text(encoding="utf-8")
MOE_RUNTIME = (REPOSITORY_ROOT / "src/siliang-moe-runtime.cpp").read_text(encoding="utf-8")
BACKEND_HEADER = (REPOSITORY_ROOT / "ggml/include/ggml-backend.h").read_text(encoding="utf-8")
BACKEND_SOURCE = (REPOSITORY_ROOT / "ggml/src/ggml-backend.cpp").read_text(encoding="utf-8")
CPU_SOURCE = (REPOSITORY_ROOT / "ggml/src/ggml-cpu/ggml-cpu.c").read_text(encoding="utf-8")
MODEL_LOADER = (REPOSITORY_ROOT / "src/llama-model-loader.cpp").read_text(encoding="utf-8")
MODEL_SOURCE = (REPOSITORY_ROOT / "src/llama-model.cpp").read_text(encoding="utf-8")
CLI_HELP_DOC = (REPOSITORY_ROOT / "tools/cli/README.md").read_text(encoding="utf-8")
SERVER_HELP_DOC = (REPOSITORY_ROOT / "tools/server/README.md").read_text(encoding="utf-8")
SERVER_SOURCE = (REPOSITORY_ROOT / "tools/server/server-context.cpp").read_text(encoding="utf-8")


def function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening : index + 1]
    raise AssertionError(f"unterminated function: {signature}")


class ArenaOptInContractTests(unittest.TestCase):
    def test_cache_is_context_configured_and_disabled_by_default(self) -> None:
        self.assertIn("bool enabled = false;", COMMON_HEADER)
        self.assertIn("uint64_t l2_mib = 0;", COMMON_HEADER)

        body = function_body(CACHE_SOURCE, "static void siliangem_init(void)")
        self.assertIn("!siliangem_tls_state->configured", body)
        self.assertIn("!siliangem_tls_state->enabled", body)
        self.assertIn("siliangem_tls_state->capacity_mib == 0", body)
        self.assertNotIn("getenv(", body)

    def test_ds4_front_double_bank_reuse_is_event_ordered_without_full_cuda_fence(self) -> None:
        front = (REPOSITORY_ROOT / "src/siliang-ds4-front-slab.cpp").read_text(encoding="utf-8")
        cuda_header = (REPOSITORY_ROOT / "ggml/include/ggml-cuda.h").read_text(encoding="utf-8")
        issue = function_body(front, "bool issue_layer(int32_t target_layer)")
        use_done = function_body(front, "bool record_layer_use_done(int32_t layer)")
        self.assertIn("constexpr int k_front_bank_count = 2;", front)
        self.assertIn("bank * bank_stride", issue)
        self.assertIn("cuda.stream_wait(copy_stream, use_done[bank])", issue)
        self.assertIn("cuda.all_streams_event_record(cuda_backend, use_done[bank])", use_done)
        self.assertIn("ggml_backend_cuda_siliang_all_streams_event_record", cuda_header)
        self.assertNotIn("ggml_backend_synchronize(cuda_backend);", issue)

    def test_no_mmap_managed_experts_have_no_model_backing_fallback(self) -> None:
        reserve = function_body(
            BACKEND_SOURCE,
            "static ggml_backend_buffer_t ggml_backend_cpu_siliang_managed_buffer_type_alloc_buffer("
        )
        self.assertIn("ggml_backend_cpu_siliang_managed_buffer_type", BACKEND_HEADER)
        self.assertIn("ggml_backend_buffer_is_siliang_managed", BACKEND_HEADER)
        self.assertIn("MEM_RESERVE", reserve)
        self.assertIn("PAGE_NOACCESS", reserve)
        self.assertNotIn("MEM_COMMIT", reserve)

        self.assertIn("managed_expert_source", COMMON_SOURCE)
        self.assertIn("params.load_mode == LLAMA_LOAD_MODE_NONE", COMMON_SOURCE)
        self.assertIn("params.expert_cache.l2_mib > 0", COMMON_SOURCE)
        self.assertIn("managed_override_patterns", COMMON_SOURCE)
        self.assertIn("routed-expert sidecar CPU", COMMON_SOURCE)
        self.assertIn("CPU_SILIANG_MANAGED", BACKEND_SOURCE)

        self.assertIn("expert_major_managed_only", MODEL_LOADER)
        self.assertIn("em_managed_only", MODEL_LOADER)
        self.assertIn("routed expert tensor bytes are not materialized", MODEL_LOADER)
        self.assertIn("managed expert tensor lost its proxy buffer", MODEL_LOADER)

        self.assertIn("const int managed_only = ggml_backend_buffer_is_siliang_managed", CPU_SOURCE)
        self.assertIn("refusing model-backing fallback", CPU_SOURCE)
        self.assertIn("if (src0_cur == NULL)", CPU_SOURCE)
        self.assertIn("managed_only || g_siliangem.em_scattered", CACHE_SOURCE)
        self.assertIn("managed expert source cannot fall back to model-resident bytes", MOE_RUNTIME)
        self.assertIn("proxy reserves only the logical projection size", MODEL_LOADER)
        self.assertIn("proxy VA reserve", MODEL_SOURCE)
        self.assertIn("committed = 0 MiB", MODEL_SOURCE)

    def test_public_cpu_api_uses_typed_configuration(self) -> None:
        self.assertIn("struct ggml_siliangem_cache_config", CPU_HEADER)
        self.assertIn("uint32_t capacity_mib;", CPU_HEADER)
        self.assertIn("uint32_t enabled;", CPU_HEADER)
        self.assertIn("ggml_backend_cpu_siliangem_configure", CPU_HEADER)
        self.assertIn("no operator\n    // environment variables are read", CPU_HEADER)

    def test_cli_and_server_share_the_typed_flags(self) -> None:
        for option in (
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
            "--expert-cache-route-stats",
            "--expert-cache-deferred-wait",
        ):
            self.assertIn(option, ARG_SOURCE)
            self.assertIn(option, CLI_HELP_DOC)
            self.assertIn(option, SERVER_HELP_DOC)
        self.assertIn("LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER", ARG_SOURCE)
        retired_prefix = "SILIANG" + "EM_"
        self.assertNotIn(f'set_env("{retired_prefix}', ARG_SOURCE)

    def test_public_docs_describe_the_same_opt_in_contract(self) -> None:
        self.assertIn("The expert cache is opt-in", README)
        self.assertIn("`--expert-cache`", README)
        self.assertIn("The expert cache is opt-in", CONFIGURATION)
        self.assertIn("`--no-expert-cache`", CONFIGURATION)
        retired_helper = "siliang-" + "env.ps1"
        self.assertNotIn(retired_helper, README)
        self.assertNotIn(retired_helper, CONFIGURATION)

    def test_banked_mapper_is_limited_to_own_k_slice_and_shared_r_tail(self) -> None:
        self.assertIn("uint32_t exchange_slot_first;", LLAMA_HEADER)
        self.assertIn("uint32_t exchange_slot_count;", LLAMA_HEADER)
        self.assertIn("exchange_slot_first_by_layer", LLAMA_CPARAMS)
        self.assertIn("exchange_slot_count_by_layer", LLAMA_CPARAMS)
        self.assertIn("const bool in_persistent", LLAMA_GRAPH)
        self.assertIn("const bool in_exchange", LLAMA_GRAPH)
        self.assertIn("(!in_persistent && !in_exchange)", LLAMA_GRAPH)
        self.assertIn("/*.layer_slot_count    =*/ binding_count", MOE_RUNTIME)
        self.assertIn("/*.exchange_slot_first =*/ exchange_first", MOE_RUNTIME)
        self.assertIn("/*.exchange_slot_count =*/ params.exchange_r", MOE_RUNTIME)

    def test_generic_l1_runtime_is_model_structure_derived(self) -> None:
        model_info_body = function_body(
            LLAMA_CONTEXT, "static bool siliang_moe_arena_get_model_info_impl("
        )
        bind_body = function_body(LLAMA_CONTEXT, "bool llama_context::siliang_moe_arena_bind(")
        self.assertNotIn("LLM_ARCH_", MOE_RUNTIME)
        self.assertNotIn("LLM_ARCH_", model_info_body)
        self.assertNotIn("LLM_ARCH_", bind_body)
        self.assertIn("same_schema", MOE_RUNTIME)
        self.assertIn("schema_k_capacity", MOE_RUNTIME)
        self.assertIn("physical_slot_for_policy", MOE_RUNTIME)

    def test_k_l2_transition_worker_quiesces_l2_before_slot_swap(self) -> None:
        worker = function_body(MOE_RUNTIME, "void transition_worker_loop()")
        wait_call = "if (error == 0 && (!cpu_wait || !cpu_wait(cpu)))"
        release_call = "if (!cpu_release("
        store_call = "if (!cpu_store("
        self.assertIn(wait_call, worker)
        self.assertIn(release_call, worker)
        self.assertIn(store_call, worker)
        self.assertLess(worker.index(wait_call), worker.index(release_call))
        self.assertLess(worker.index(wait_call), worker.index(store_call))
        self.assertIn("const int victim_l2_location = cpu_location", worker)
        self.assertIn(
            "victim_l2_location == GGML_SILIANGEM_EXPERT_LOCATION_RESIDENT", worker
        )
        location_check = "victim_l2_location == GGML_SILIANGEM_EXPERT_LOCATION_RESIDENT"
        self.assertLess(worker.index("const int victim_l2_location = cpu_location"), worker.index(store_call))
        duplicate_guard = worker[worker.index(location_check):worker.index(store_call)]
        self.assertIn("Reuse the existing L2 copy", duplicate_guard)
        self.assertIn("continue;", duplicate_guard)
        self.assertIn('error_stage = "candidate-release";', worker)
        self.assertIn('error_stage = "victim-store";', worker)
        self.assertIn("K/L2 transition worker L2 swap failed stage=%s", MOE_RUNTIME)

    def test_l1_arena_rejects_lora_at_bind_and_dynamic_application(self) -> None:
        bind_body = function_body(LLAMA_CONTEXT, "bool llama_context::siliang_moe_arena_bind(")
        self.assertIn("!loras || !loras->empty()", bind_body)

        apply_body = function_body(LLAMA_CONTEXT, "int32_t llama_set_adapters_lora(")
        self.assertIn("has_nonzero_adapter", apply_body)
        self.assertIn("!ctx->siliang_moe_arena_lora_compatible()", apply_body)
        self.assertIn("return -1;", apply_body)

        compatibility_body = function_body(
            LLAMA_CONTEXT, "bool llama_context::siliang_moe_arena_lora_compatible() const"
        )
        self.assertIn("cparams.expert_cache.l1_k == 0", compatibility_body)
        self.assertIn("!cparams.siliang_moe_arena_enabled", compatibility_body)

        common_apply_body = function_body(COMMON_SOURCE, "bool common_set_adapter_lora(")
        self.assertIn("return llama_set_adapters_lora", common_apply_body)
        server_update_body = function_body(SERVER_SOURCE, "void update_slots()")
        self.assertIn("if (!common_set_adapter_lora", server_update_body)
        self.assertIn("abort_all_slots(reason)", server_update_body)


if __name__ == "__main__":
    unittest.main()
