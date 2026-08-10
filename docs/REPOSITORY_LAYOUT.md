# Repository layout

The public repository keeps end-user guidance at the top level and places
technical material beside the code or workflow it describes.

| Path | Purpose |
| --- | --- |
| `README.md` | Product overview, performance summary, build, and quickstart. |
| Root source tree (`ggml/`, `src/`, `include/`, and related paths) | Pinned `llama.cpp` upstream base plus the Siliang engine delta. |
| `tools/` | Expert-major GGUF conversion code and model-preparation guide. |
| `scripts/` | Build, verification, test, repack, and runtime-gate entry points. |
| `tests/` | Upstream tests plus model-free tests for Siliang tooling and source contracts. |
| `docs/` | Configuration, provenance, and repository documentation. |
| `patches/` | Canonical engine patch relative to the pinned upstream source. |
| `licenses/` | Siliang additions license and license texts required by bundled components. |
| `.github/` | Continuous integration and contribution templates. |
| `AGENTS.md` | Upstream contributor and coding-agent rules, with a pointer to the Siliang policy. |
| `SILIANG_AGENTS.md` | Siliang evidence, safety, compatibility, and public-hygiene rules. |
| `LICENSE` | Upstream `llama.cpp` MIT license. |
| `licenses/SILIANG-ENGINE-MIT.txt` | MIT terms for the Siliang additions. |
| `THIRD_PARTY_NOTICES.md` | Notices for upstream and other bundled components. |

Generated models, builds, caches, logs, and machine-specific benchmark packets
do not belong in the public repository.
