from __future__ import annotations

import json
from pathlib import Path
import re
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SILIANG_AUTHORED_FILES = (
    ".gitattributes",
    ".github/ISSUE_TEMPLATE/bug_report.md",
    ".github/workflows/ci.yaml",
    ".github/workflows/release.yml",
    ".gitignore",
    "README.md",
    "SILIANG_AGENTS.md",
    "THIRD_PARTY_NOTICES.md",
    "assets/siliang-engine.png",
    "docs/CONFIGURATION.md",
    "docs/PERFORMANCE.md",
    "docs/PROVENANCE.md",
    "docs/REPOSITORY_LAYOUT.md",
    "docs/releases/v0.1.3.md",
    "docs/source-manifest.json",
    "licenses/Apache-2.0.txt",
    "licenses/LLVM-exception.txt",
    "licenses/SILIANG-ENGINE-MIT.txt",
    "patches/siliang-engine.patch",
    "scripts/README.md",
    "scripts/build.ps1",
    "scripts/check_expert_major.py",
    "scripts/preflight_repack.py",
    "scripts/repack-model.ps1",
    "scripts/runtime-gate.ps1",
    "scripts/test.ps1",
    "scripts/verify-snapshot.ps1",
    "tests/test_arena_opt_in_contract.py",
    "tests/test_converter.py",
    "tests/test_hygiene.py",
    "tests/test_prefill_arena_contract.py",
    "tests/test_repack_wrapper.py",
    "tests/test_release_packaging.py",
    "tests/test_runtime_gate.py",
    "tests/test_siliangem_telemetry_contract.py",
    "tests/test_stock_arena_contract.py",
    "tools/README.md",
    "tools/gguf_reader.py",
    "tools/make_expert_major_gguf.py",
)
ENGINE_DELTA_FILES = (
    "common/arg.cpp",
    "common/common.cpp",
    "common/common.h",
    "common/speculative.cpp",
    "ggml/include/ggml-backend.h",
    "ggml/include/ggml-cpu.h",
    "ggml/include/ggml-cuda.h",
    "ggml/src/ggml-backend.cpp",
    "ggml/src/ggml-cpu/ggml-cpu-impl.h",
    "ggml/src/ggml-cpu/ggml-cpu.c",
    "ggml/src/ggml-cpu/ggml-cpu.cpp",
    "ggml/src/ggml-cpu/siliangem_moe_cache.h",
    "ggml/src/ggml-cuda/ggml-cuda.cu",
    "include/llama.h",
    "src/CMakeLists.txt",
    "src/llama-context.cpp",
    "src/llama-context.h",
    "src/llama-cparams.h",
    "src/llama-graph.cpp",
    "src/llama-graph.h",
    "src/llama-model-loader.cpp",
    "src/llama-model-loader.h",
    "src/llama-model.h",
    "src/llama.cpp",
    "src/models/deepseek4.cpp",
    "src/siliang-ds4-front-slab.cpp",
    "src/siliang-ds4-front-slab.h",
    "src/siliang-expert-source.h",
    "src/siliang-moe-runtime.cpp",
    "src/siliang-moe-runtime.h",
    "tests/CMakeLists.txt",
    "tests/test-arg-parser.cpp",
    "tests/test-siliang-prefill.cpp",
    "tools/server/server-context.cpp",
)


class PublicTreeHygieneTests(unittest.TestCase):
    def test_engine_delta_provenance_uses_the_same_path_boundary(self) -> None:
        manifest = json.loads((REPOSITORY_ROOT / "docs/source-manifest.json").read_text(encoding="utf-8"))
        manifest_paths = {
            entry["path"] for entry in manifest["source"]["engineDelta"]["files"]
        }
        self.assertEqual(manifest_paths, set(ENGINE_DELTA_FILES))
        self.assertEqual(len(manifest_paths), len(ENGINE_DELTA_FILES))

        patch_text = (REPOSITORY_ROOT / "patches/siliang-engine.patch").read_text(encoding="utf-8")
        patch_paths = set(re.findall(r"^diff --git a/(.+) b/\1$", patch_text, re.MULTILINE))
        self.assertEqual(patch_paths, set(ENGINE_DELTA_FILES))

        provenance = (REPOSITORY_ROOT / "docs/PROVENANCE.md").read_text(encoding="utf-8")
        verifier = (REPOSITORY_ROOT / "scripts/verify-snapshot.ps1").read_text(encoding="utf-8")
        self.assertIn(f"across {len(ENGINE_DELTA_FILES)} paths", provenance)
        self.assertIn(f"exact {len(ENGINE_DELTA_FILES)}-path boundary", provenance)
        self.assertIn("$expectedFiles.Count", verifier)

    def test_no_machine_bound_paths_or_obvious_secrets(self) -> None:
        patterns = {
            "Windows user-profile path": re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\r\n]+", re.IGNORECASE),
            "Unix home path": re.compile(r"/(?:home|Users)/[^/\s]+/"),
            "private-key header": re.compile("-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
            "AWS access-key shape": re.compile(r"AKIA[0-9A-Z]{16}"),
            "GitHub token shape": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,255}"),
        }
        delta_only_patterns = {
            "missing private-history document": re.compile(r"\b(?:PLAN|STATUS)\.md\b", re.IGNORECASE),
            "private launch script": re.compile(r"\brun_server\.bat\b", re.IGNORECASE),
            "private ticket identifier": re.compile(
                r"\b(?:T-\d{3,}|(?:RS|GX|GZ2?|TS|TA|ST|SP|A|E)-20\d{2}-\d{2}-\d{2}(?:-[A-Za-z0-9-]+)?)\b"
            ),
            "private job path": re.compile(r"(?:\.claude|\.codex)[\\/]jobs[\\/]", re.IGNORECASE),
        }
        failures: list[str] = []

        for relative_text in SILIANG_AUTHORED_FILES:
            path = REPOSITORY_ROOT / relative_text
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for label, pattern in patterns.items():
                if pattern.search(text):
                    failures.append(f"{relative_text}: {label}")

        for relative_text in ENGINE_DELTA_FILES:
            path = REPOSITORY_ROOT / relative_text
            self.assertTrue(path.is_file(), f"required hygiene target is missing: {relative_text}")
            text = path.read_text(encoding="utf-8")
            for label, pattern in {**patterns, **delta_only_patterns}.items():
                if pattern.search(text):
                    failures.append(f"{relative_text}: {label}")

        self.assertEqual(failures, [], "public-tree hygiene failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
