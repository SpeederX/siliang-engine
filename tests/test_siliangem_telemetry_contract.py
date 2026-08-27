from __future__ import annotations

from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CACHE_SOURCE = (REPOSITORY_ROOT / "ggml/src/ggml-cpu/siliangem_moe_cache.h").read_text(encoding="utf-8")


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


class SiliangemTelemetryContractTests(unittest.TestCase):
    def test_expert_hit_rate_uses_explicit_request_counters(self) -> None:
        report = function_body(CACHE_SOURCE, "static void siliangem_report(")
        blocking = function_body(CACHE_SOURCE, "static int siliangem_prepare(")
        deferred = function_body(CACHE_SOURCE, "static int siliangem_prepare_async(")

        self.assertIn("expert_requests, expert_hits", CACHE_SOURCE)
        self.assertIn("g_siliangem.expert_hits / (double) g_siliangem.expert_requests", report)
        self.assertIn("expert-request hit", report)
        self.assertNotIn("tot / 3", report)
        self.assertNotIn("sel - g_siliangem.misses", report)
        for body in (blocking, deferred):
            self.assertIn("if (part == 0) g_siliangem.expert_requests++;", body)
            self.assertIn("if (part == 0) g_siliangem.expert_hits++;", body)

    def test_cross_layer_overlap_is_decode_only_and_reports_overflow_bucket(self) -> None:
        observe = function_body(CACHE_SOURCE, "static void siliangem_note_xlayer(")
        report = function_body(CACHE_SOURCE, "static void siliangem_report(")

        reset = observe.index("if (!is_decode)")
        sample = observe.index("g_siliangem.xlayer_hist")
        self.assertLess(reset, sample)
        self.assertIn("g_siliangem.prev_n = 0;", observe[reset:sample])
        self.assertIn("return;", observe[reset:sample])
        self.assertIn("xlayer_chance_sum += (uint64_t) n * (uint64_t) g_siliangem.prev_n", observe)
        self.assertIn("siliangem[xlayer]: DECODE mean overlap", report)
        self.assertIn("7+=%.1f%%", report)
        self.assertIn("g_siliangem.xlayer_hist[7]", report)
        self.assertNotIn("36.0 /", report)


if __name__ == "__main__":
    unittest.main()
