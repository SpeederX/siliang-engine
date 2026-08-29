from __future__ import annotations

from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LLAMA_HEADER = (REPOSITORY_ROOT / "include/llama.h").read_text(encoding="utf-8")
LLAMA_CPARAMS = (REPOSITORY_ROOT / "src/llama-cparams.h").read_text(encoding="utf-8")
LLAMA_CONTEXT = (REPOSITORY_ROOT / "src/llama-context.cpp").read_text(encoding="utf-8")
LLAMA_GRAPH = (REPOSITORY_ROOT / "src/llama-graph.cpp").read_text(encoding="utf-8")
MOE_RUNTIME = (REPOSITORY_ROOT / "src/siliang-moe-runtime.cpp").read_text(encoding="utf-8")
COMMON_SOURCE = (REPOSITORY_ROOT / "common/common.cpp").read_text(encoding="utf-8")
SERVER_SOURCE = (REPOSITORY_ROOT / "tools/server/server-context.cpp").read_text(encoding="utf-8")


class PrefillArenaContractTests(unittest.TestCase):
    def test_prefill_is_explicit_and_bounded_by_the_real_union_limit(self) -> None:
        self.assertIn("bool prefill;", LLAMA_HEADER)
        self.assertIn("bool prefill_enabled = false;", LLAMA_CPARAMS)
        self.assertIn("prefill_ubatch_cap", LLAMA_CPARAMS)
        self.assertIn("min(n_ubatch * top-k, expert-count) <= L1 K", LLAMA_CONTEXT)
        self.assertIn("LLAMA_SILIANG_MOE_PREFILL_MAX_EXPERTS", LLAMA_CONTEXT)
        self.assertIn("LLAMA_SILIANG_MOE_PREFILL_MAX_EXPERTS", MOE_RUNTIME)
        self.assertIn("prefill_route_capacity > descriptor.policy_count", MOE_RUNTIME)
        self.assertIn("layer-local schema-bank K slice", MOE_RUNTIME)
        self.assertNotIn("schemas.size() != 1", MOE_RUNTIME)
        self.assertNotIn("model->arch != LLM_ARCH_DEEPSEEK4", MOE_RUNTIME)

    def test_graph_accepts_only_exact_bounded_route_shapes(self) -> None:
        self.assertIn("logical_ids->ne[0] == state->top_k", LLAMA_GRAPH)
        self.assertIn("ggml_cont(ctx0, selected_experts)", LLAMA_GRAPH)
        self.assertIn("ggml_map_custom1(", LLAMA_GRAPH)
        self.assertIn("ctx0, logical_dispatch_experts, &moe_arena->map_calls_by_layer[il]", LLAMA_GRAPH)
        self.assertNotIn("natural_weights", LLAMA_GRAPH)
        self.assertNotIn("ggml_map_custom2(", LLAMA_GRAPH)
        self.assertIn("experts = ggml_mul(ctx0, experts, weights)", LLAMA_GRAPH)
        self.assertIn("n_tokens == n_tokens_at_build", LLAMA_GRAPH)
        self.assertIn("n_tokens <= static_cast<int64_t>(state->prefill_ubatch_cap)", LLAMA_GRAPH)
        self.assertIn("(prefill && !in_persistent)", LLAMA_GRAPH)
        self.assertIn("physical_by_logical", LLAMA_GRAPH)
        self.assertIn("logical_by_persistent", LLAMA_GRAPH)
        self.assertIn("logical_by_exchange", LLAMA_GRAPH)

    def test_prefill_is_k_transient_exclusive_p_waved_and_r_free(self) -> None:
        self.assertIn("build_route_union", MOE_RUNTIME)
        self.assertIn("enter_phase(route_phase::prefill)", MOE_RUNTIME)
        self.assertIn("record_staging_completion()", MOE_RUNTIME)
        self.assertIn("copy_expert(\n                        layer, route.experts[union_index], physical_slot, lane, true)", MOE_RUNTIME)
        self.assertIn('"exclusive L2-to-K release failed"', MOE_RUNTIME)
        self.assertIn("++metrics.l2_release_failures", MOE_RUNTIME)
        self.assertIn("++metrics.l2_releases", MOE_RUNTIME)
        self.assertIn("R_bypass=0", MOE_RUNTIME)
        self.assertIn("phase_invalidations", MOE_RUNTIME)

    def test_prefill_telemetry_is_phase_specific(self) -> None:
        self.assertIn("serving bounded prefill", MOE_RUNTIME)
        self.assertIn("prefill_unique_max=", MOE_RUNTIME)
        self.assertIn("prefill_P_waves=", MOE_RUNTIME)
        self.assertIn("prefill_H2D_bytes=", MOE_RUNTIME)
        self.assertIn("prefill_bitmap current_epoch=", MOE_RUNTIME)
        self.assertIn("coverage=%.1f%% precision=%.1f%%", MOE_RUNTIME)
        self.assertIn('reset_prefill_bitmaps("serving-start")', MOE_RUNTIME)
        self.assertIn("siliang_moe_route_bitmap_reset: v=1 scope=context epoch=", MOE_RUNTIME)
        self.assertIn("reason=%s mapped=0", MOE_RUNTIME)
        self.assertIn("prefill_bitmap_incomplete", MOE_RUNTIME)
        self.assertIn('discard_pending_prefill_sweep("shutdown")', MOE_RUNTIME)
        self.assertIn("siliang_moe_route_bitmap: v=1 scope=context epoch=", MOE_RUNTIME)
        self.assertIn("siliang_moe_route_sweep: v=1 scope=context epoch=", MOE_RUNTIME)
        self.assertIn("prefill_sweep_attempt = ++prefill_bitmap_attempt_serial", MOE_RUNTIME)
        self.assertIn("metrics.prefill_maps = 0", MOE_RUNTIME)
        self.assertIn("res->reset_expert_cache_prefill_trace();", COMMON_SOURCE)
        self.assertGreater(
            COMMON_SOURCE.index("res->reset_expert_cache_prefill_trace();"),
            COMMON_SOURCE.index("if (params.warmup)"),
        )
        self.assertIn("llama_init->reset_expert_cache_prefill_trace();", SERVER_SOURCE)


if __name__ == "__main__":
    unittest.main()
