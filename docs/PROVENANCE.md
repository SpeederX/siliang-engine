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
| SHA-256 | `E17AF8BD6A38BCC4CFEF09A934EC719B4050A8815D76285325D6B0359B6DCD68` |
| Git blob | `a5f5c67474591f22250cfc1d3ca1fa62c4f4f020` |

The patch uses full Git object IDs and LF line endings. It contains exactly
**15,119 insertions and 266 deletions** across 50 paths.

## Engine delta

The v0.1.8 engine boundary covers the shared typed argument and context
configuration, CPU/CUDA backend transfer hooks, the ggml plan-size query, the
CUDA decode tail buffer and deterministic top-k, model-owned expert sources,
the generic K/R/P runtime, the DeepSeek4 FRONT slab, the batch splitter, KV and
recurrent memory replay for layer-major prefill and in-batch checkpoints, the
qwen4exp layer-range graph, graph and server error propagation, the C++ argument-parser contract test, and the focused bounded-
prefill/bitmap test. Product documentation,
release workflows, PowerShell helpers, and Python product-contract tests are
maintained in the fork but are deliberately excluded from the canonical engine
patch.

| Path | Base Git blob | Siliang Git blob |
| --- | --- | --- |
| `common/arg.cpp` | `63e342776d549b6abaf9c4aef5543a729e6dfd97` | `99fba78ea8e1c06c42bd2aa8d4c2f48becfc0e4a` |
| `common/common.cpp` | `364688ef466dab454e278e4e7c8f85864e39f432` | `e71da00dae2766a165ecdf9fbcccaef01448ef74` |
| `common/common.h` | `9194d3dcb650a37c10dade1f10b7f1430936db9c` | `7bc8a90dc6a2abb359a3cfe84cad262396b6795f` |
| `common/speculative.cpp` | `6fdfa4dc33f0514462f8e8d40cdf093a0029a73e` | `2eb9ce9c722052ef40fdc986e4687ac1354473aa` |
| `ggml/include/ggml-alloc.h` | `a7926a21a9a20f37a4b75cf19fbca5d8d88b7779` | `aa704df1748a76cc20242817bf2abe28c50eeb70` |
| `ggml/include/ggml-backend.h` | `cc3f8cd36e3549c4f3ed3d6ad5f6b577af4b837b` | `7a74515c6df84c85157967e882840c50ff19a4cc` |
| `ggml/include/ggml-cpu.h` | `dc6453c6eaa16667f720f987659ad42d03a403a2` | `d138f1f5ad8bb6cd216f1155989648712a14d38e` |
| `ggml/include/ggml-cuda.h` | `1cd81eeaebcdf4abcd46c87ba1a9a46e275aa12b` | `ce2e5aee685b23a6ecc8c912969b43c352e6389d` |
| `ggml/src/ggml-alloc.c` | `a71838eafc69d1cab66dfb87885690a6e1e464d0` | `7d73bf15df2849ed07b3b370729ffc33874d606e` |
| `ggml/src/ggml-backend.cpp` | `20bf965017e31338f579c1d8d1a64b0ab56c78ee` | `221e903a33d8b40d1ea5399a0f2c2d52846abf4b` |
| `ggml/src/ggml-cpu/ggml-cpu-impl.h` | `5dd9ec8e628acad32537e87eccfdeb01c1c1dc46` | `747ca497185a82b1d9960f033e077fb0081f2e94` |
| `ggml/src/ggml-cpu/ggml-cpu.c` | `8bb0ff7bc3366be957dd20aba3fbe0f50dd249d7` | `e69f8787edf331dd8ed243df304711105e98c200` |
| `ggml/src/ggml-cpu/ggml-cpu.cpp` | `1df0f2bb926894eba96beef691ff2bf520ab70b3` | `7d9ef8cc749e2618c598032f3844eb24fad7c4fb` |
| `ggml/src/ggml-cpu/siliangem_moe_cache.h` | absent | `749c822367ec5a820c9f0c1dfd88c47400d9a0aa` |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `e76ff3128fddd7bf248e16b8469d27c2f441b8e1` | `a27e493276bc405253355a748f41fdec9890c914` |
| `ggml/src/ggml-cuda/top-k.cu` | `c7a0c831788df8547ffd86c8a68e6d53aff61130` | `af47a8172298ca3f0e9746b06b9dd42776d93201` |
| `include/llama.h` | `1805ed0559f92a818dfe951b012608ee2be3115c` | `601fcd1eac17ce398073a00cc83ff856dd6f17dd` |
| `src/CMakeLists.txt` | `afdaddc79de81bc03dadee2707e67d2af4b814c0` | `4ff52fe171c6f3f525997cb3156d095792821b14` |
| `src/llama-batch.cpp` | `89a1f3f37c7b3cce944e4114aa3bbae89f42f109` | `0dd4fec52bcc881db5c6e878ced208b1054fe814` |
| `src/llama-batch.h` | `201d48cce18dd56786162fde75987fd0989b69fd` | `e8eac6aafe48153048fdaf25c3bc92e05b42347d` |
| `src/llama-context.cpp` | `8675f6087336069c43dffff0435002bb4358c6f5` | `00c648eaec5c27aecadbdef7c0c1528c89a94e16` |
| `src/llama-context.h` | `b403b099b76fefea6f3d8a4959bb12287b633876` | `b73d7a7865154cd622f8ad956292f5feb1ede981` |
| `src/llama-cparams.h` | `b592de18c79470243ece446cb7db1ca00b0b803b` | `4dffdc345d10d6cba198831cf040b40191a898fe` |
| `src/llama-graph.cpp` | `0b3bab612375b27b357971287857826dcc8b66e8` | `ffdb9bdb2348be65a21747168b99ae9e154c9eb5` |
| `src/llama-graph.h` | `3daa425bc07bdb2ee9b618124bfa7dfcebc6094f` | `ec0ef82b2ff4ec48193ee24e972d3cc583a7cd07` |
| `src/llama-kv-cache.cpp` | `332d1abe028f23b974e47d9643cdd849afcd75a0` | `f8d0cc47ed67be63587538095c3bd790d9d2fd22` |
| `src/llama-kv-cache.h` | `c4d8699def1216170b3d7d00ba2f8712070ebfcf` | `76f1c7e49149993de7403f1518c403a84e57d9de` |
| `src/llama-memory-hybrid-idx.cpp` | `3972ce9ce29346dc5fb319232a634b01897ec2e0` | `d4a931ca1bce5fc07ebe9af024005b432d398803` |
| `src/llama-memory-hybrid-idx.h` | `705189e7eb5886b3faccefd6e21745aabf0d41fa` | `516fe65c2569c1822426c122964158af017e9862` |
| `src/llama-memory-hybrid.cpp` | `42c7381a9e6fa507c925ba66a35a7e433c4447b8` | `89681ea7cb1f369fc7454b1f27bd7bb523c1d3a1` |
| `src/llama-memory-hybrid.h` | `484eafb749910567b8c50ec36d46bdb678c75426` | `b55d050a5109e086073e88cbe1255de8a2bbad10` |
| `src/llama-memory-recurrent.cpp` | `57919accf09569e129ab35b5b2bd4dab7dbe8a3a` | `412506ca15efada908b7e1a5a7a064213c124635` |
| `src/llama-memory-recurrent.h` | `4abb3f5cf5c09611844dff5361ca36e51ff9c07f` | `cf5ecd27d976fe8164ff2fd2d11a3782771d761d` |
| `src/llama-memory.h` | `db825396645e61c94213ca1e94b78d71d1538d67` | `c68dba585e40599d00f512eaf5d8edb104f0ed03` |
| `src/llama-model-loader.cpp` | `43c396f15af3475e405da8dd73e6a8010ce3d7a5` | `9707a9ada9b7b21cf2fa402554a0df394ab29ad6` |
| `src/llama-model-loader.h` | `9e51d0ce750503788d2236b4892b8fa3e409b478` | `da89930317474ed8a6ad0f103056d9fd09d49786` |
| `src/llama-model.cpp` | `ab5e744b5d1076c2ad9c4700051ae3bdb9edb024` | `371fa9c9750400746f0d02633168194e42ee02a9` |
| `src/llama-model.h` | `a0f9f11423e36699996d1d5f8e56514a0d19b7e8` | `dbc5b6102da3242d25e859cd1c163e3e7cc47a36` |
| `src/llama.cpp` | `ad8e443882ad66e8a68c028f970a0fffd03b92d4` | `42440f6ebedebdd3c8da8eb4651ebf0618772cb9` |
| `src/models/deepseek4.cpp` | `6bf9d34449426504cea436531118eb9b13941f20` | `bf61c024c1cf005a85c9f7e3c8f21c0abb2514b3` |
| `src/models/qwen4exp.cpp` | `f33989de023d377505b0da438f484efb12114410` | `3a53e9bcf7f217726c19341aef676cdac807c7c5` |
| `src/siliang-ds4-front-slab.cpp` | absent | `05636ef8d52cbde133dff107f53bd8c35e05c6da` |
| `src/siliang-ds4-front-slab.h` | absent | `a3c1052be580ed7ceecbd3d64d01350e439a5a74` |
| `src/siliang-expert-source.h` | absent | `72b1d213be6b1d40f687f040e8913e648aeaefd3` |
| `src/siliang-moe-runtime.cpp` | absent | `508d971c25405067716b9f067913582c6aa68631` |
| `src/siliang-moe-runtime.h` | absent | `2e374427813aad8142028a6d8c738d433842dc12` |
| `tests/CMakeLists.txt` | `9b3a4fcc4bbf6afc77cc8554fa14412722b887b4` | `8244dcbb99f2fb414f49a9a4aaf2e6f29bc5906f` |
| `tests/test-arg-parser.cpp` | `e0907631abd8a89e5b6dbadf10d28b65ed483b9a` | `ea164bfe12351845c9bad0e1e3953daafc834017` |
| `tests/test-siliang-prefill.cpp` | absent | `d4237159e5c1f1c284c5c466a13fc93ec4e17db2` |
| `tools/server/server-context.cpp` | `e95fb63ab3cca4352f627738edb077ce8413b31f` | `dabeccf6843172be0e7cc9492818aecd388e6d70` |

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
patch identities and line counts, exact 50-path boundary, base and final Git
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
