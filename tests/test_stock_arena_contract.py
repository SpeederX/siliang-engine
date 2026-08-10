from __future__ import annotations

from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CPU_SOURCE = (REPOSITORY_ROOT / "ggml/src/ggml-cpu/ggml-cpu.c").read_text(encoding="utf-8")
CACHE_SOURCE = (REPOSITORY_ROOT / "ggml/src/ggml-cpu/siliangem_moe_cache.h").read_text(encoding="utf-8")
LOADER_SOURCE = (REPOSITORY_ROOT / "src/llama-model-loader.cpp").read_text(encoding="utf-8")


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


class StockArenaSourceContractTests(unittest.TestCase):
    def test_loader_publishes_every_layer_part_stride(self) -> None:
        self.assertIn("std::vector<uint32_t> sc_stride;", LOADER_SOURCE)
        self.assertIn("sc_stride.push_back((uint32_t) stride);", LOADER_SOURCE)
        self.assertIn("sc_stride.size() == (size_t) sc_layers * 3", LOADER_SOURCE)
        self.assertIn("sc_base.data(), sc_stride.data()", LOADER_SOURCE)
        self.assertIn("if (max_expert_bytes <= 0xFFFFFFFFull)", LOADER_SOURCE)
        self.assertNotIn("uint32_t sc_stride[3]", LOADER_SOURCE)

    def test_setter_validates_and_copies_the_flattened_geometry(self) -> None:
        body = function_body(CPU_SOURCE, "void ggml_siliangem_set_scattered_source(")
        validation = "const uint32_t part_bytes = stride[L * 3 + p];"
        copy_stride = "g_scat.stride[i] = stride[i];"
        copy_base = "g_scat.base[i]   = base[i];"

        self.assertIn("for (int L = 0; L < n_layers; L++)", body)
        self.assertIn(validation, body)
        self.assertIn("if (part_bytes == 0) return;", body)
        self.assertIn("if (expert_bytes > UINT32_MAX) return;", body)
        self.assertIn("for (int i = 0; i < n_layers * 3; i++)", body)
        self.assertIn(copy_stride, body)
        self.assertIn(copy_base, body)
        self.assertLess(body.index(validation), body.index(copy_stride))
        self.assertLess(body.index(copy_stride), body.index("g_scat.set = 1;"))

    def test_cache_indexes_stock_geometry_by_layer_and_part(self) -> None:
        self.assertIn("g_scat.stride[L*3 + p]", CACHE_SOURCE)
        self.assertIn("g_src.stride + (size_t) layer * 3", CACHE_SOURCE)
        self.assertIn("g_src.base[(size_t) layer * 3 + p]", CACHE_SOURCE)

    def test_stock_initialization_fails_closed_before_ready(self) -> None:
        body = function_body(CACHE_SOURCE, "static void siliangem_init(void)")
        stock_branch = body.index("if (g_siliangem.em_scattered)")
        initialize = body.index("siliangem_scat_init();", stock_branch)
        require_enabled = body.index("if (!g_src.enabled)", initialize)
        cleanup = body.index("siliangem_shutdown();", require_enabled)
        ready = body.index("g_siliangem.ready = 1;", cleanup)

        self.assertEqual(body.count("siliangem_scat_init();"), 1)
        self.assertLess(stock_branch, initialize)
        self.assertLess(initialize, require_enabled)
        self.assertLess(require_enabled, cleanup)
        self.assertLess(cleanup, ready)
        self.assertIn("stock scattered source initialization failed - using mmap", body)

    def test_split_gguf_never_publishes_a_single_file_arena_source(self) -> None:
        self.assertIn("bool em_source_is_monolithic = files.size() == 1;", LOADER_SOURCE)
        self.assertIn("em_source_is_monolithic = em_source_is_monolithic && fidx == 0;", LOADER_SOURCE)
        self.assertIn("if (!fname.empty() && em_source_is_monolithic)", LOADER_SOURCE)
        self.assertIn("expert-major model file as an arena source - using mmap", LOADER_SOURCE)
        self.assertIn("!fname.empty() && files.size() != 1", LOADER_SOURCE)
        self.assertIn("stock scattered model files as an arena source - using mmap", LOADER_SOURCE)
        self.assertIn("!fname.empty() && files.size() == 1", LOADER_SOURCE)
        self.assertIn("it->second.tensor == nullptr || it->second.idx != 0", LOADER_SOURCE)
        self.assertEqual(LOADER_SOURCE.count('LLAMA_LOG_WARN("%s: siliangem: split GGUF'), 2)

    def test_scattered_telemetry_reports_complete_source_evidence(self) -> None:
        self.assertIn("layers served %u/%u", CACHE_SOURCE)
        self.assertIn("layers substituted %u/%u", CACHE_SOURCE)
        self.assertIn("g_src_expected_bytes += part_off;", CACHE_SOURCE)
        self.assertIn("if (!g_src_layer_seen[layer])", CACHE_SOURCE)
        self.assertIn("g_src_layers_seen++;", CACHE_SOURCE)
        self.assertIn("if (g_src.enabled && ith == 0 && !g_src_layer_substituted[layer])", CACHE_SOURCE)
        self.assertIn("g_src_layers_substituted++;", CACHE_SOURCE)
        self.assertIn("return cached;", CACHE_SOURCE)
        self.assertIn("siliangem_ptr(src0->name, cur_a, (size_t) nb02, ith)", CPU_SOURCE)


if __name__ == "__main__":
    unittest.main()
