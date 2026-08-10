from __future__ import annotations

from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CACHE_SOURCE = (REPOSITORY_ROOT / "ggml/src/ggml-cpu/siliangem_moe_cache.h").read_text(encoding="utf-8")
README = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
CONFIGURATION = (REPOSITORY_ROOT / "docs/CONFIGURATION.md").read_text(encoding="utf-8")


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
    def test_budget_parser_accepts_only_positive_decimal_mib(self) -> None:
        body = function_body(CACHE_SOURCE, "static int siliangem_parse_cache_mib(")

        self.assertIn("if (!text || !text[0] || !value_out) return 0;", body)
        self.assertIn("if (*p < '0' || *p > '9') return 0;", body)
        self.assertIn("if (value > (UINT64_MAX - digit) / 10u) return 0;", body)
        self.assertIn("if (value == 0 || value > UINT64_MAX / (1024ull * 1024ull)) return 0;", body)
        self.assertIn("*value_out = value;", body)

    def test_absent_or_invalid_budget_returns_before_source_or_arena_work(self) -> None:
        body = function_body(CACHE_SOURCE, "static void siliangem_init(void)")
        disable = body.index('getenv("SILIANGEM_DISABLE")')
        cache_env = body.index('getenv("SILIANGEM_CACHE_MIB")')
        absent = body.index("SILIANGEM_CACHE_MIB is not set - arena is opt-in; using mmap")
        parse = body.index("siliangem_parse_cache_mib(mib, &budget)")
        invalid = body.index("invalid SILIANGEM_CACHE_MIB='%s'")
        open_source = body.index("CreateFileA(path")
        allocate = body.index("g_siliangem.arena = (uint8_t *) VirtualAlloc(")

        self.assertLess(disable, cache_env)
        self.assertLess(cache_env, absent)
        self.assertLess(absent, parse)
        self.assertLess(parse, invalid)
        self.assertLess(invalid, open_source)
        self.assertLess(open_source, allocate)

    def test_no_implicit_default_or_partial_numeric_parse_remains(self) -> None:
        body = function_body(CACHE_SOURCE, "static void siliangem_init(void)")

        self.assertNotIn("uint64_t budget = 8192ull", body)
        self.assertNotIn("atoll(", body)
        self.assertIn("const uint64_t nslots = arena_bytes / g_siliangem.expert_bytes;", body)
        self.assertIn("if (nslots > UINT32_MAX", body)

    def test_public_docs_describe_the_same_opt_in_contract(self) -> None:
        self.assertIn("The arena is opt-in", README)
        self.assertIn("without an explicit positive `SILIANGEM_CACHE_MIB`", README)
        self.assertIn("The arena is opt-in", CONFIGURATION)
        self.assertIn("there is no implicit arena size", CONFIGURATION)
        self.assertIn("`SILIANGEM_DISABLE` takes precedence", CONFIGURATION)


if __name__ == "__main__":
    unittest.main()
