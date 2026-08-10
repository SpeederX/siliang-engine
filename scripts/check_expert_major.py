from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import struct
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from gguf_reader import GGML_TYPES, GGUF  # noqa: E402


_FIXED_VALUE_FORMATS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}
_REQUIRED_ARRAYS = (
    "siliangem.expert_bytes",
    "siliangem.layer_index",
    "siliangem.part_bytes",
    "siliangem.part_types",
    "siliangem.part_ne0",
    "siliangem.part_ne1",
)
_SUPPORTED_PART_SETS = {
    ("gate", "up", "down"),
    ("gate_up", "down"),
}


def _read_exact(stream, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise EOFError("truncated GGUF metadata")
    return value


def _read_uint32(stream) -> int:
    return struct.unpack("<I", _read_exact(stream, 4))[0]


def _read_uint64(stream) -> int:
    return struct.unpack("<Q", _read_exact(stream, 8))[0]


def _read_string(stream, capture: bool) -> str | None:
    length = _read_uint64(stream)
    if capture:
        return _read_exact(stream, length).decode("utf-8", "replace")
    stream.seek(length, 1)
    return None


def _read_value(stream, value_type: int, capture: bool):
    if value_type == 8:
        return _read_string(stream, capture)
    if value_type == 9:
        element_type = _read_uint32(stream)
        count = _read_uint64(stream)
        if not capture and element_type in _FIXED_VALUE_FORMATS:
            stream.seek(struct.calcsize(_FIXED_VALUE_FORMATS[element_type]) * count, 1)
            return None
        if not capture:
            for _ in range(count):
                _read_value(stream, element_type, False)
            return None
        values = [_read_value(stream, element_type, True) for _ in range(count)]
        return element_type, values
    value_format = _FIXED_VALUE_FORMATS.get(value_type)
    if value_format is None:
        raise ValueError(f"unsupported GGUF metadata type {value_type}")
    raw = _read_exact(stream, struct.calcsize(value_format))
    return struct.unpack(value_format, raw)[0] if capture else None


def _read_required_arrays(path: Path) -> dict[str, list[int]]:
    captured: dict[str, list[int]] = {}
    with path.open("rb") as stream:
        if _read_exact(stream, 4) != b"GGUF":
            raise ValueError("not a GGUF file")
        _version = _read_uint32(stream)
        _tensor_count = _read_uint64(stream)
        metadata_count = _read_uint64(stream)
        for _ in range(metadata_count):
            key = _read_string(stream, True)
            assert key is not None
            value_type = _read_uint32(stream)
            wanted = key in _REQUIRED_ARRAYS
            value = _read_value(stream, value_type, wanted)
            if wanted:
                if key in captured:
                    raise ValueError(f"duplicate metadata key {key}")
                if value_type != 9 or not isinstance(value, tuple):
                    raise ValueError(f"{key} must be an array")
                element_type, entries = value
                if element_type != 4:
                    raise ValueError(f"{key} must be an array of uint32 values")
                captured[key] = [int(entry) for entry in entries]

    missing = [key for key in _REQUIRED_ARRAYS if key not in captured]
    if missing:
        raise ValueError("required expert-major metadata is missing: " + ", ".join(missing))
    return captured


def inspect_expert_major(path: Path) -> dict[str, object]:
    parsed = GGUF(path)
    try:
        if parsed.kv.get("siliangem.expert_major") is not True:
            raise ValueError("siliangem.expert_major is absent or false")

        alignment = int(parsed.kv.get("general.alignment", 32))
        if alignment != 512:
            raise ValueError(f"general.alignment must be 512, found {alignment}")

        expert_count = int(parsed.kv.get("siliangem.n_experts", 0))
        if not 1 <= expert_count <= 65536:
            raise ValueError(
                f"siliangem.n_experts must be between 1 and 65536, found {expert_count}"
            )

        architecture = parsed.kv.get("general.architecture")
        if isinstance(architecture, str):
            architecture_expert_count = parsed.kv.get(f"{architecture}.expert_count")
            if (
                architecture_expert_count is not None
                and int(architecture_expert_count) != expert_count
            ):
                raise ValueError(
                    f"{architecture}.expert_count does not match siliangem.n_experts"
                )

        part_names_value = parsed.kv.get("siliangem.part_names")
        if not isinstance(part_names_value, str):
            raise ValueError("siliangem.part_names is absent or not a string")
        part_names = part_names_value.split(",")
        if (
            not part_names
            or any(not part or part != part.strip() for part in part_names)
            or len(set(part_names)) != len(part_names)
        ):
            raise ValueError(f"siliangem.part_names is invalid: {part_names_value!r}")
        if len(part_names) > 4:
            raise ValueError("siliangem.part_names exceeds the cache limit of four parts")
        if any(
            len(part.encode("utf-8")) > 23
            or "\0" in part
            or any(character.isspace() for character in part)
            for part in part_names
        ):
            raise ValueError(
                "siliangem.part_names entries must be whitespace-free and at most 23 UTF-8 bytes"
            )
        if tuple(part_names) not in _SUPPORTED_PART_SETS:
            raise ValueError(
                "siliangem.part_names is not a layout emitted by the bundled converter"
            )

        arrays = _read_required_arrays(path)
        layer_indices = arrays["siliangem.layer_index"]
        if not layer_indices:
            raise ValueError("siliangem.layer_index is empty")
        if len(layer_indices) > 512:
            raise ValueError("siliangem.layer_index exceeds the cache limit of 512 layers")
        expected_layers = list(range(len(layer_indices)))
        if layer_indices != expected_layers:
            raise ValueError(
                "siliangem.layer_index must be contiguous and zero-based: "
                f"expected {expected_layers}, found {layer_indices}"
            )

        expert_bytes = arrays["siliangem.expert_bytes"]
        if len(expert_bytes) != len(layer_indices):
            raise ValueError(
                "siliangem.expert_bytes length does not match layer_index: "
                f"{len(expert_bytes)} versus {len(layer_indices)}"
            )
        if any(value <= 0 or value % 512 for value in expert_bytes):
            raise ValueError("siliangem.expert_bytes contains a nonpositive or unaligned stride")

        geometry_count = len(layer_indices) * len(part_names)
        geometry_arrays = {
            key: arrays[key]
            for key in (
                "siliangem.part_bytes",
                "siliangem.part_types",
                "siliangem.part_ne0",
                "siliangem.part_ne1",
            )
        }
        for key, values in geometry_arrays.items():
            if len(values) != geometry_count:
                raise ValueError(
                    f"{key} length must equal layers x parts: "
                    f"expected {geometry_count}, found {len(values)}"
                )

        part_bytes = geometry_arrays["siliangem.part_bytes"]
        part_types = geometry_arrays["siliangem.part_types"]
        part_ne0 = geometry_arrays["siliangem.part_ne0"]
        part_ne1 = geometry_arrays["siliangem.part_ne1"]
        if any(value <= 0 for value in part_bytes + part_ne0 + part_ne1):
            raise ValueError("part byte sizes and dimensions must all be positive")

        for index, (byte_count, type_id, ne0, ne1) in enumerate(
            zip(part_bytes, part_types, part_ne0, part_ne1)
        ):
            type_layout = GGML_TYPES.get(type_id)
            if type_layout is None:
                raise ValueError(
                    f"siliangem.part_types[{index}] uses unsupported GGML type {type_id}"
                )
            _type_name, elements_per_block, bytes_per_block = type_layout
            if ne0 % elements_per_block:
                raise ValueError(
                    f"siliangem.part_ne0[{index}] is not aligned to its GGML type block"
                )
            element_count = ne0 * ne1
            expected_bytes = element_count // elements_per_block * bytes_per_block
            if byte_count != expected_bytes:
                raise ValueError(
                    f"siliangem.part_bytes[{index}] is {byte_count}, expected {expected_bytes}"
                )

        file_size = path.stat().st_size
        unknown_tensor_types = sorted({
            tensor.type_id
            for tensor in parsed.tensors.values()
            if tensor.type_id not in GGML_TYPES
        })
        if unknown_tensor_types:
            raise ValueError(
                f"tensor directory uses unsupported GGML types: {unknown_tensor_types}"
            )

        expected_packed_names = {
            f"blk.{layer}.ffn_exps_packed.weight" for layer in layer_indices
        }
        actual_packed_names = {
            name
            for name in parsed.tensors
            if re.fullmatch(r"blk\.\d+\.ffn_exps_packed\.weight", name)
        }
        if actual_packed_names != expected_packed_names:
            missing = sorted(expected_packed_names - actual_packed_names)
            extra = sorted(actual_packed_names - expected_packed_names)
            raise ValueError(
                f"packed tensor-name set is inconsistent (missing={missing}, extra={extra})"
            )
        residual_expert_tensors = sorted(
            name
            for name in parsed.tensors
            if re.fullmatch(r"blk\.\d+\.ffn_.+_exps\.weight", name)
            and name not in expected_packed_names
        )
        if residual_expert_tensors:
            raise ValueError(
                "residual logical expert tensors remain: "
                + ", ".join(residual_expert_tensors[:8])
            )

        pad_target = math.lcm(
            512,
            *(GGML_TYPES[type_id][2] for type_id in part_types),
        )
        for layer_position, layer in enumerate(layer_indices):
            first_part = layer_position * len(part_names)
            raw_expert_bytes = sum(part_bytes[first_part:first_part + len(part_names)])
            stride = expert_bytes[layer_position]
            if raw_expert_bytes > stride:
                raise ValueError(
                    f"layer {layer} part bytes exceed its expert stride: "
                    f"{raw_expert_bytes} > {stride}"
                )
            expected_stride = (
                (raw_expert_bytes + pad_target - 1) // pad_target * pad_target
            )
            if stride != expected_stride:
                raise ValueError(
                    f"layer {layer} expert stride is {stride}, expected canonical "
                    f"bundled-converter stride {expected_stride}"
                )
            row_types = part_types[first_part:first_part + len(part_names)]
            incompatible_type_sizes = sorted({
                GGML_TYPES[type_id][2]
                for type_id in row_types
                if stride % GGML_TYPES[type_id][2]
            })
            if incompatible_type_sizes:
                raise ValueError(
                    f"layer {layer} expert stride is not divisible by part type sizes: "
                    f"{incompatible_type_sizes}"
                )

            packed_name = f"blk.{layer}.ffn_exps_packed.weight"
            packed = parsed.tensors.get(packed_name)
            if packed is None:
                raise ValueError(f"expert-major model is missing {packed_name}")
            expected_packed_bytes = stride * expert_count
            if packed.type_id != 24 or packed.dims != [expected_packed_bytes]:
                raise ValueError(
                    f"{packed_name} must be a one-dimensional I8 tensor of "
                    f"{expected_packed_bytes} bytes"
                )
            if packed.nbytes != expected_packed_bytes:
                raise ValueError(f"{packed_name} has an inconsistent byte length")
            if packed.rel_offset % 512 or packed.abs_offset % 512:
                raise ValueError(f"{packed_name} is not 512-byte aligned")
            if packed.abs_offset + packed.nbytes > file_size:
                raise ValueError(f"{packed_name} extends past the end of the file")

        expected_relative_offset = 0
        for tensor in parsed.tensors.values():
            if tensor.rel_offset != expected_relative_offset:
                raise ValueError(
                    f"{tensor.name} has noncanonical relative offset {tensor.rel_offset}; "
                    f"expected {expected_relative_offset}"
                )
            expected_relative_offset = (
                (expected_relative_offset + tensor.nbytes + alignment - 1)
                // alignment
                * alignment
            )
        expected_file_size = parsed.data_start + expected_relative_offset
        if file_size != expected_file_size:
            raise ValueError(
                f"GGUF file extent is {file_size}, expected canonical extent {expected_file_size}"
            )

        return {
            "model": str(path),
            "alignment": alignment,
            "expertCount": expert_count,
            "expertBytes": expert_bytes,
            "partNames": part_names,
            "partTypeIds": sorted(set(part_types)),
            "canonicalPadTarget": pad_target,
            "layerIndices": layer_indices,
            "layerCount": len(layer_indices),
            "partCount": len(part_names),
            "geometryEntries": geometry_count,
            "packedTensorCount": len(layer_indices),
            "minimumExpertBytes": min(expert_bytes),
            "maximumExpertBytes": max(expert_bytes),
            "status": "expert-major-metadata-ok",
        }
    finally:
        parsed.f.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the expert-major metadata required by Siliang Engine"
    )
    parser.add_argument("--model", required=True, type=Path)
    arguments = parser.parse_args()
    if not arguments.model.is_file():
        parser.error(f"model file does not exist: {arguments.model}")
    try:
        result = inspect_expert_major(arguments.model.resolve())
    except (EOFError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"expert-major metadata check failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
