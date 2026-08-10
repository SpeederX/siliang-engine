from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from gguf_reader import GGML_TYPES, GGUF  # noqa: E402


EXPERT_PATTERN = re.compile(
    r"^blk\.(?P<layer>\d+)\.ffn_(?P<part>[^.]+)_exps\.weight$"
)
REQUIRED_PARTS = ("gate", "up", "down")


def inspect_source(path: Path) -> dict[str, object]:
    parsed = GGUF(path)
    try:
        architecture = parsed.kv.get("general.architecture")
        if not isinstance(architecture, str):
            raise ValueError("general.architecture must be an explicit string")
        if not architecture:
            raise ValueError("general.architecture must not be empty")
        if architecture != architecture.strip():
            raise ValueError("general.architecture must not contain surrounding whitespace")

        alignment = int(parsed.kv.get("general.alignment", 32))
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError(f"general.alignment must be a positive power of two, found {alignment}")
        if alignment > 512:
            raise ValueError(
                f"general.alignment {alignment} exceeds 512; the unchanged converter "
                "would declare 512 while laying tensors out at the larger alignment"
            )

        unknown_types = sorted({tensor.type_id for tensor in parsed.tensors.values() if tensor.type_id not in GGML_TYPES})
        if unknown_types:
            raise ValueError(
                "the copied reader does not define GGML type ID(s): "
                + ", ".join(str(value) for value in unknown_types)
            )

        tensors_by_layer: dict[int, dict[str, object]] = {}
        expert_like_names: list[str] = []
        for name, tensor in parsed.tensors.items():
            if "_exps.weight" not in name:
                continue
            expert_like_names.append(name)
            match = EXPERT_PATTERN.fullmatch(name)
            if match is None:
                raise ValueError(f"unrecognized expert tensor name: {name}")
            layer = int(match.group("layer"))
            part = match.group("part")
            if part not in REQUIRED_PARTS:
                raise ValueError(f"unexpected expert projection {part!r} in {name}")
            layer_parts = tensors_by_layer.setdefault(layer, {})
            if part in layer_parts:
                raise ValueError(f"duplicate {part} expert projection at layer {layer}")
            layer_parts[part] = tensor

        if not tensors_by_layer:
            raise ValueError("no gate/up/down expert tensors were found")
        layers = sorted(tensors_by_layer)
        expected_layers = list(range(len(layers)))
        if layers != expected_layers:
            raise ValueError(f"MoE layer IDs must be contiguous and zero-based: expected {expected_layers}, found {layers}")

        expert_counts: set[int] = set()
        for layer in layers:
            parts = tensors_by_layer[layer]
            missing = sorted(set(REQUIRED_PARTS) - set(parts))
            extra = sorted(set(parts) - set(REQUIRED_PARTS))
            if missing or extra:
                raise ValueError(
                    f"layer {layer} must have exactly gate/up/down; missing={missing}, extra={extra}"
                )
            for part in REQUIRED_PARTS:
                tensor = parts[part]
                if len(tensor.dims) != 3:
                    raise ValueError(f"{tensor.name} must be three-dimensional, found {tensor.dims}")
                experts = int(tensor.dims[-1])
                if experts <= 0 or tensor.nbytes % experts:
                    raise ValueError(f"{tensor.name} has a nonintegral expert byte stride")
                expert_counts.add(experts)
        if len(expert_counts) != 1:
            raise ValueError(f"inconsistent expert counts: {sorted(expert_counts)}")

        return {
            "source": str(path),
            "sourceArchitecture": architecture,
            "alignment": alignment,
            "layers": layers,
            "layerCount": len(layers),
            "expertCount": next(iter(expert_counts)),
            "parts": list(REQUIRED_PARTS),
            "expertTensorCount": len(expert_like_names),
            "status": "preflight-ok",
        }
    finally:
        parsed.f.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed preflight for the unchanged expert-major converter")
    parser.add_argument("--src", required=True, type=Path)
    arguments = parser.parse_args()
    if not arguments.src.is_file():
        parser.error(f"source file does not exist: {arguments.src}")
    try:
        result = inspect_source(arguments.src.resolve())
    except (EOFError, KeyError, OSError, ValueError) as error:
        print(f"preflight failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
