# Source provenance

Siliang Engine is maintained directly on a fork of llama.cpp. The upstream
source tree, the Siliang engine delta, and the product documentation therefore
share one repository; there is no vendored `llama.cpp/` directory or submodule.

## Repository identity

| Field | Value |
| --- | --- |
| Siliang fork | [`SpeederX/siliang-engine`](https://github.com/SpeederX/siliang-engine) |
| Official upstream | [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) |
| Pinned upstream base | `07132750825a4f2d27a547cd9cdde1c6f6001885` |
| Upstream tag at base | `b10270` |
| Base root tree | `46f77bf060878e2b3b9d7c43b4d8a3a566ba3384` |

The Siliang branch must descend from that exact upstream commit. Locally,
`origin` points to the Siliang fork and `upstream` points to the official
llama.cpp repository. A CI checkout may omit the convenience `upstream` remote;
the verifier still validates the recorded base object and its ancestry.

## Canonical patch

[`patches/siliang-engine.patch`](../patches/siliang-engine.patch) is the
canonical engine-only delta from the pinned upstream base.

| Identity | Value |
| --- | --- |
| SHA-256 | `F857F01F180724BDEA2A25303BA61774533890085456C3D7E5E41D1EEA16CF20` |
| Git blob | `1ad533211c33348b109ec9bf3d5497e4ac4c6b2b` |

The patch uses full Git object IDs and LF line endings. It contains exactly
**10,548 insertions and 24 deletions** across 34 paths.

## Engine delta

The v0.1.4 engine boundary covers the shared typed argument and context
configuration, CPU/CUDA backend transfer hooks, model-owned expert sources,
the generic K/R/P runtime, the DeepSeek4 FRONT slab, graph and server error
propagation, the C++ argument-parser contract test, and the focused bounded-
prefill/bitmap test. Product documentation,
release workflows, PowerShell helpers, and Python product-contract tests are
maintained in the fork but are deliberately excluded from the canonical engine
patch.

| Path | Base Git blob | Siliang Git blob |
| --- | --- | --- |
| `common/arg.cpp` | `86af0ba10a327283f2500f0bb8e48095df547017` | `21881e865fd7f870b4f25c0cbeeb7b1e707df7dc` |
| `common/common.cpp` | `d9ce5755161edc853dbcf629819ee75971ba413d` | `1ffabac36f8c0382bc284104391fd7f00cbf5106` |
| `common/common.h` | `3444aa157e9b73727ea2ca6107eb0dc9f9b36a74` | `86bbf4a9c35e7abc35e0cd1521a6557d41602ca9` |
| `common/speculative.cpp` | `70dc0ac3b1b74fdd5f08b470308786c3f12411e7` | `3f7c5f87559b224c038b86f958557abbad2c13a4` |
| `ggml/include/ggml-backend.h` | `2924fdbe9884df40abf505fd89d277f5281a835b` | `d04327dbc6ca76b1831ccd1fd8571a107e4982a9` |
| `ggml/include/ggml-cpu.h` | `dc6453c6eaa16667f720f987659ad42d03a403a2` | `8f1634660f907e8e2d52ad8c20c122e576544184` |
| `ggml/include/ggml-cuda.h` | `1cd81eeaebcdf4abcd46c87ba1a9a46e275aa12b` | `9377ba5b12d4ce22025852ecef7b5f15b3cab6cc` |
| `ggml/src/ggml-backend.cpp` | `f6fb91798ca484fd1298d7012be3ae8d73cb0ea4` | `4e3d46d30679d3097395409f0382bcd3ddaea2a3` |
| `ggml/src/ggml-cpu/ggml-cpu-impl.h` | `5d1ca5ffcc368b9f0249d6cf6ccc4549bb9a3ab4` | `97a2a0b1b72df88a9af3cb6ab825ee4b474a40ae` |
| `ggml/src/ggml-cpu/ggml-cpu.c` | `491316f7491252248d6f74a60440d3efa7aa6177` | `83518e829c9f9a72a606514cef178260d3f73a37` |
| `ggml/src/ggml-cpu/ggml-cpu.cpp` | `16cc5116c5451787c6a1dd1988e38b761f20ef12` | `f39002fc95a5f793e2f2ed68187bfa8b3df8201a` |
| `ggml/src/ggml-cpu/siliangem_moe_cache.h` | absent | `f0d97e499332959d836ec8d97aa4f63f13b5629d` |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `561ab7ac599f9e285d2a0296caee0ab0a14ea5c8` | `d3d2cb63b8ec6a9e9621d29a44af510b607423ca` |
| `include/llama.h` | `fb2ca38cee4f8ba84bb6178f1e345e066b0d07e5` | `cdf629e88ebbbff200a44261802e5077b67e7fbd` |
| `src/CMakeLists.txt` | `24f05cc91673217726b919229e1626b7f74a7bcb` | `4b620618f1304164881ac97692d397fa8152688f` |
| `src/llama-context.cpp` | `19cca7df1e9deaafc1e8ee50d0c78ae5ffbc6cfb` | `ecdb8f92ecc2faeee71dee7b460efbf7b194a822` |
| `src/llama-context.h` | `bf91daa8b562aa66d15b08ca559b6baa09ab7855` | `7c566f561669941cec1196ee0fc88519c4b173ab` |
| `src/llama-cparams.h` | `5018170ed85e3b82abad65e6a3c71859067c9f71` | `2de4137861983ca96170eeb71b893ededb9dad29` |
| `src/llama-graph.cpp` | `2be3b75fb9825ccc9aa08cda294f46d6422c61ea` | `4258d05f5ff0f958dafdb621aeecd7d2530e1f48` |
| `src/llama-graph.h` | `32d8d395aa4546ed7e90e7d26d24218fcd37547a` | `08c5d6968e6fde56d207c6607fdc6834e97287ca` |
| `src/llama-model-loader.cpp` | `b31e92e2da7ef42eabbb47173bb1f2088c952f39` | `709db618f656f22e02fbcea1101db185341a8afb` |
| `src/llama-model-loader.h` | `d6b31c2311186608f48e88d1a37c23adc7e1b0c7` | `aa1772d1377724450c2f3b9b3013e771607f8c3f` |
| `src/llama-model.h` | `6b9e94a0a6921745fd20f58aba38490480c36a38` | `f14b607ded2574b98290f9eb5e9237989e9313d6` |
| `src/llama.cpp` | `d6e0bbfefa729329fe6b83e46e603a85dab0f2e3` | `3113ca8d1cacc2a8c42ef815e4aba9a6c22a88bb` |
| `src/models/deepseek4.cpp` | `89cd461765adfe8c32fa2e6c6b6d2e962de4b0ac` | `e0ae251cb500c5eb55d6b2c8b36a24a6f6005b84` |
| `src/siliang-ds4-front-slab.cpp` | absent | `8068e77bc74bc1198b79c80deb50431551a03c32` |
| `src/siliang-ds4-front-slab.h` | absent | `a3c1052be580ed7ceecbd3d64d01350e439a5a74` |
| `src/siliang-expert-source.h` | absent | `01b630212a342f9eb0770985995098a049734064` |
| `src/siliang-moe-runtime.cpp` | absent | `6cf14959d265c2f97b7ed3846d057bffcead8944` |
| `src/siliang-moe-runtime.h` | absent | `5bf98bd7fdaa26a64c66f8c4231ea133d6136cf6` |
| `tests/CMakeLists.txt` | `419e1eba4c2cdb465d20453004eeeca5af28037f` | `b66f6c8fa62a68fcc95194cf5d5689035056adc7` |
| `tests/test-arg-parser.cpp` | `fd5adb740eab632505cd0a4d999fb55a093a5f84` | `d1c58a368d236b6ded05c7353b6108405e1ec6df` |
| `tests/test-siliang-prefill.cpp` | absent | `7d740fe1e709cf8a7985ff418124a732c3f281c8` |
| `tools/server/server-context.cpp` | `5d2798cc14e9295646bb8e570bbec166c9ecc72c` | `1162ccd4b1dc37d203f048d9fbe47f73e0fc997a` |

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
patch identities and line counts, exact 34-path boundary, base and final Git
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
git -C "<fresh-llama-checkout>" checkout 07132750825a4f2d27a547cd9cdde1c6f6001885
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
