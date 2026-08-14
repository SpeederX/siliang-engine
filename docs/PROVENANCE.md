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
| SHA-256 | `ED45181907A0B3AEBC6E2E662486B5729871502FA8B265D7719CC87842D8266B` |
| Git blob | `4875174bd377fe906fb68e3359985088690d34a7` |

The patch uses full Git object IDs and LF line endings. It contains exactly
**2,340 insertions and 4 deletions** across six paths.

## Engine delta

| Path | Base Git blob | Siliang Git blob |
| --- | --- | --- |
| `ggml/include/ggml-cpu.h` | `dc6453c6eaa16667f720f987659ad42d03a403a2` | `326ac4abc9333328f03e2bb0e67668c4c797df08` |
| `ggml/src/ggml-cpu/ggml-cpu.c` | `491316f7491252248d6f74a60440d3efa7aa6177` | `9c18ea720354a1e82be5aa67c286ada4dea96f8d` |
| `ggml/src/ggml-cpu/ggml-cpu.cpp` | `16cc5116c5451787c6a1dd1988e38b761f20ef12` | `4ab2467b5bbe1a95801c65a3b46e0454ec679bbc` |
| `ggml/src/ggml-cpu/siliangem_moe_cache.h` | absent | `7b151a326decaec47585bdadf8ca567b616ab868` |
| `src/llama-model-loader.cpp` | `b31e92e2da7ef42eabbb47173bb1f2088c952f39` | `80994ea1e4d0cc1fb0e3eb8db2dcb5238d106c97` |
| `src/llama-model-loader.h` | `d6b31c2311186608f48e88d1a37c23adc7e1b0c7` | `47da6c71417e9934f7c3ed40824119bfd9970c0e` |

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
patch identities and line counts, exact six-path boundary, base and final Git
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

Siliang intentionally ships only `.github/workflows/ci.yaml`. It validates the
Windows release path plus representative Linux CPU and macOS Metal builds before
packaging tagged Windows CPU/CUDA artifacts. These compatibility jobs prove that
the Siliang delta does not silently narrow the upstream build surface; they do
not claim that the Windows-only expert arena is implemented on those platforms.
Upstream llama.cpp workflows are not carried into the fork because their
publishing, scheduled, and self-hosted automation belongs to the upstream
project rather than Siliang releases.
