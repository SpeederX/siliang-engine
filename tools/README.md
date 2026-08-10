# Expert-major model preparation

An expert-major GGUF is a core input to the accelerated Siliang path. The
repack step places the projections for each routed expert together so the
runtime can fetch that expert with one contiguous storage read into its RAM
arena.

Repacking does not quantize the model or change expert IDs. It changes the
physical layout of routed-expert weights and embeds the geometry the Siliang
loader needs to reconstruct the original logical tensors.

Recreate files produced by pre-Siliang prototypes. The public loader accepts
the current Siliang metadata namespace and intentionally carries no retired
metadata alias.

## Requirements

- 64-bit Windows and PowerShell for the supported wrapper workflow.
- Python 3.
- Enough free storage for the source and destination GGUF at the same time.
- A source GGUF supported by the repository's preflight checks.

Install the repository's `gguf` package into the selected Python environment:

```powershell
python -m pip install -e .\gguf-py
```

## Repack a model

Run the supported wrapper from the repository root:

```powershell
.\scripts\repack-model.ps1 `
    -Source "<source-model.gguf>" `
    -Destination "<new-expert-major-model.gguf>" `
    -Samples 64 `
    -PythonExecutable "python"
```

The wrapper performs the source preflight and dry run before writing, creates a
partial destination, verifies sampled expert bytes and final structure, checks
that the source stayed unchanged, and promotes the partial file only after all
gates pass. It refuses to overwrite an existing destination.

Keep the emitted JSON receipt with the model. It records the accepted source
architecture, stable source identity, output identity, and verification
result.

Do not call `make_expert_major_gguf.py` directly for a release artifact. The
wrapper is the supported entry point because it surrounds conversion with the
required fail-closed checks.

## Verify an existing output

Probe the finalized file before using it:

```powershell
python .\scripts\check_expert_major.py --model "<expert-major-model.gguf>"
```

The required result is `status: expert-major-metadata-ok`. A structural pass is
necessary but does not replace a runtime correctness comparison.

## Files in this folder

- `make_expert_major_gguf.py` writes the self-contained expert-major GGUF.
- `gguf_reader.py` is the minimal GGUF reader used by the converter.

The supported orchestration and probes live in `scripts/`; see
[`../scripts/README.md`](../scripts/README.md).
