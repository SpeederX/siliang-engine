# Source provenance

Siliang Engine is maintained directly on a fork of llama.cpp. The upstream
source tree, the Siliang engine delta, and the product documentation therefore
share one repository; there is no vendored `llama.cpp/` directory or submodule.

## Repository identity

| Field | Value |
| --- | --- |
| Siliang fork | [`SpeederX/siliang-engine`](https://github.com/SpeederX/siliang-engine) |
| Official upstream | [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) |
| Pinned upstream base | `e85e15cf6d810cd1268498c2e5b657bb3ece47bc` |
| Upstream tag at base | `b11188` |
| Base root tree | `cd9cd8e6821229813a9b5a35673e4d1e2f876d56` |

The Siliang branch must descend from that exact upstream commit. Locally,
`origin` points to the Siliang fork and `upstream` points to the official
llama.cpp repository. A CI checkout may omit the convenience `upstream` remote;
the verifier still validates the recorded base object and its ancestry.

## Canonical patch

[`patches/siliang-engine.patch`](../patches/siliang-engine.patch) is the
canonical engine-only delta from the pinned upstream base.

| Identity | Value |
| --- | --- |
| SHA-256 | `4CA4C3881011682906F83AB83F618045186EB3F012A7507A9577AE5F3BA8B518` |
| Git blob | `1aca5140a0862cb482dc57287d9840fbb022b2be` |

The patch uses full Git object IDs and LF line endings. It contains exactly
**11,230 insertions and 38 deletions** across 35 paths.

## Engine delta

The v0.1.6 engine boundary covers the shared typed argument and context
configuration, CPU/CUDA backend transfer hooks, model-owned expert sources,
the generic K/R/P runtime, the DeepSeek4 FRONT slab, graph and server error
propagation, the C++ argument-parser contract test, and the focused bounded-
prefill/bitmap test. Product documentation,
release workflows, PowerShell helpers, and Python product-contract tests are
maintained in the fork but are deliberately excluded from the canonical engine
patch.

| Path | Base Git blob | Siliang Git blob |
| --- | --- | --- |
| `common/arg.cpp` | `63e342776d549b6abaf9c4aef5543a729e6dfd97` | `1f685113f8228efa57445c8e16531dc55dd256ca` |
| `common/common.cpp` | `364688ef466dab454e278e4e7c8f85864e39f432` | `7e5ff09b17a5e7f0016d9b91c6edba556e751583` |
| `common/common.h` | `9194d3dcb650a37c10dade1f10b7f1430936db9c` | `9102beb9fe390a49365d2e9016d1d0649fbf44ea` |
| `common/speculative.cpp` | `6fdfa4dc33f0514462f8e8d40cdf093a0029a73e` | `2eb9ce9c722052ef40fdc986e4687ac1354473aa` |
| `ggml/include/ggml-backend.h` | `cc3f8cd36e3549c4f3ed3d6ad5f6b577af4b837b` | `7f5cf0f4f39d27998cbcf9c2ac1dc62fd52dc3dc` |
| `ggml/include/ggml-cpu.h` | `dc6453c6eaa16667f720f987659ad42d03a403a2` | `8f1634660f907e8e2d52ad8c20c122e576544184` |
| `ggml/include/ggml-cuda.h` | `1cd81eeaebcdf4abcd46c87ba1a9a46e275aa12b` | `6c4776a86af28a6485d8cfb8b4242b2080e7bd16` |
| `ggml/src/ggml-backend.cpp` | `20bf965017e31338f579c1d8d1a64b0ab56c78ee` | `fa80801b1ed78088f5b04d426a40931e1df87161` |
| `ggml/src/ggml-cpu/ggml-cpu-impl.h` | `5dd9ec8e628acad32537e87eccfdeb01c1c1dc46` | `699e60a2680883f6872dad7cf26ac72c3eb587e1` |
| `ggml/src/ggml-cpu/ggml-cpu.c` | `8bb0ff7bc3366be957dd20aba3fbe0f50dd249d7` | `0be59cf484cc81ba3da5c5b0d6c066d1a593487f` |
| `ggml/src/ggml-cpu/ggml-cpu.cpp` | `1df0f2bb926894eba96beef691ff2bf520ab70b3` | `1dc3a427112d12dd268388fbfc519dd56ccba6d1` |
| `ggml/src/ggml-cpu/siliangem_moe_cache.h` | absent | `fe88c90c6583eeb1a67f018e04f53c57123bc8ea` |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `e76ff3128fddd7bf248e16b8469d27c2f441b8e1` | `011f69f075910448498ed2fbe0c6d2f7e5b346d0` |
| `include/llama.h` | `1805ed0559f92a818dfe951b012608ee2be3115c` | `22c232bb3d86d746747514d04ff213400846c68f` |
| `src/CMakeLists.txt` | `afdaddc79de81bc03dadee2707e67d2af4b814c0` | `4ff52fe171c6f3f525997cb3156d095792821b14` |
| `src/llama-context.cpp` | `8675f6087336069c43dffff0435002bb4358c6f5` | `6c2f288db48faabb9c1d33664b68229036bd4440` |
| `src/llama-context.h` | `b403b099b76fefea6f3d8a4959bb12287b633876` | `5a147f5459407f34c7a37804ef8c2481ea0e9f71` |
| `src/llama-cparams.h` | `b592de18c79470243ece446cb7db1ca00b0b803b` | `d9bfa8ca9c167e1fdbc56c09299924e607e8054f` |
| `src/llama-graph.cpp` | `0b3bab612375b27b357971287857826dcc8b66e8` | `158cf8b74993536ea05ebd7a37b2c26fa73a51d1` |
| `src/llama-graph.h` | `3daa425bc07bdb2ee9b618124bfa7dfcebc6094f` | `e03a1b59a38ed55d5556a3e457b93d21e9a3b531` |
| `src/llama-model-loader.cpp` | `43c396f15af3475e405da8dd73e6a8010ce3d7a5` | `9707a9ada9b7b21cf2fa402554a0df394ab29ad6` |
| `src/llama-model-loader.h` | `9e51d0ce750503788d2236b4892b8fa3e409b478` | `da89930317474ed8a6ad0f103056d9fd09d49786` |
| `src/llama-model.cpp` | `ab5e744b5d1076c2ad9c4700051ae3bdb9edb024` | `371fa9c9750400746f0d02633168194e42ee02a9` |
| `src/llama-model.h` | `a0f9f11423e36699996d1d5f8e56514a0d19b7e8` | `dbc5b6102da3242d25e859cd1c163e3e7cc47a36` |
| `src/llama.cpp` | `ad8e443882ad66e8a68c028f970a0fffd03b92d4` | `42440f6ebedebdd3c8da8eb4651ebf0618772cb9` |
| `src/models/deepseek4.cpp` | `6bf9d34449426504cea436531118eb9b13941f20` | `bf61c024c1cf005a85c9f7e3c8f21c0abb2514b3` |
| `src/siliang-ds4-front-slab.cpp` | absent | `05636ef8d52cbde133dff107f53bd8c35e05c6da` |
| `src/siliang-ds4-front-slab.h` | absent | `a3c1052be580ed7ceecbd3d64d01350e439a5a74` |
| `src/siliang-expert-source.h` | absent | `72b1d213be6b1d40f687f040e8913e648aeaefd3` |
| `src/siliang-moe-runtime.cpp` | absent | `33132c85bcc7630c6761fbe18995047dbe05a048` |
| `src/siliang-moe-runtime.h` | absent | `5bf98bd7fdaa26a64c66f8c4231ea133d6136cf6` |
| `tests/CMakeLists.txt` | `9b3a4fcc4bbf6afc77cc8554fa14412722b887b4` | `8244dcbb99f2fb414f49a9a4aaf2e6f29bc5906f` |
| `tests/test-arg-parser.cpp` | `e0907631abd8a89e5b6dbadf10d28b65ed483b9a` | `ea164bfe12351845c9bad0e1e3953daafc834017` |
| `tests/test-siliang-prefill.cpp` | absent | `7d740fe1e709cf8a7985ff418124a732c3f281c8` |
| `tools/server/server-context.cpp` | `e95fb63ab3cca4352f627738edb077ce8413b31f` | `2f4325a704f82a07174369d1f3e861cfc67deb8a` |

[`source-manifest.json`](source-manifest.json) is the machine-readable record of
these identities. The rest of the llama.cpp tree is inherited through Git from
the pinned base rather than duplicated in a generated 3,321-file manifest.

## Verification

After the Siliang changes are committed, run the strict gate from the repository
root:

```powershell
.\scripts\verify-snapshot.ps1
```

Before the first commit, or while intentionally preparing a provenance update,
use authoring mode:

```powershell
.\scripts\verify-snapshot.ps1 -Authoring
```

Both modes verify the repository identity, upstream base and ancestry, canonical
patch identities and line counts, exact 35-path boundary, base and final Git
blobs, and an isolated application of the patch to the pinned base. Authoring
mode additionally accepts each engine path in its valid pre-commit index state:
the index and `HEAD` may still contain the upstream blob while the worktree must
already contain the final Siliang blob. When an already committed engine path is
being updated, the verifier can also admit its explicitly recorded previous
Siliang blob during authoring. It never stages or commits files.

Strict mode requires the engine paths and provenance artifacts to agree across
`HEAD`, the index, and the worktree. CI checks out full history so the pinned
base object and ancestry remain independently verifiable.

For a separate reconstruction check:

```powershell
git clone https://github.com/ggml-org/llama.cpp.git "<fresh-llama-checkout>"
git -C "<fresh-llama-checkout>" checkout e85e15cf6d810cd1268498c2e5b657bb3ece47bc
git -C "<fresh-llama-checkout>" apply --check "<siliang-root>\patches\siliang-engine.patch"
git -C "<fresh-llama-checkout>" apply "<siliang-root>\patches\siliang-engine.patch"
```

Build outputs, generated binaries, models, caches, and raw runtime logs are not
part of the source manifest.

## CI boundary

Siliang intentionally carries two workflows. `.github/workflows/ci.yaml`
validates the Windows release path plus representative Linux CPU and macOS Metal
builds, and packages Windows CPU/CUDA artifacts for version tags.
`.github/workflows/release.yml` is the tag-triggered and manually dispatchable
publisher for an existing `v*` tag; it rebuilds and verifies the release packages
before creating a draft prerelease. These compatibility jobs prove that the Siliang delta does
not silently narrow the upstream build surface; they do not claim that the
Windows-only expert arena is implemented on those platforms. Other upstream
llama.cpp workflows are not carried into the fork because their scheduled,
self-hosted, and upstream-project automation is outside Siliang releases.
