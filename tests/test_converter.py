from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIRECTORY = REPOSITORY_ROOT / "tools"
GGUF_PY_DIRECTORY = REPOSITORY_ROOT / "gguf-py"
CONVERTER = TOOLS_DIRECTORY / "make_expert_major_gguf.py"
PREFLIGHT = REPOSITORY_ROOT / "scripts" / "preflight_repack.py"
EXPERT_MAJOR_PROBE = REPOSITORY_ROOT / "scripts" / "check_expert_major.py"

sys.path.insert(0, str(GGUF_PY_DIRECTORY))
sys.path.insert(0, str(TOOLS_DIRECTORY))

import numpy as np  # noqa: E402
from gguf import GGUFReader, GGUFValueType, GGUFWriter  # noqa: E402
from gguf_reader import GGML_TYPES, GGUF  # noqa: E402


def load_converter_module():
    spec = importlib.util.spec_from_file_location("siliangem_converter", CONVERTER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {CONVERTER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def close_gguf_reader(reader: GGUFReader) -> None:
    """Release gguf-py's memmap deterministically (required on Windows)."""
    reader.tensors.clear()
    reader.fields.clear()
    mmap_handle = getattr(reader.data, "_mmap", None)
    if mmap_handle is not None:
        mmap_handle.close()


def write_synthetic_moe(
    path: Path,
    *,
    architecture: str | None = "deepseek4",
    alignment: int | None = None,
    layers: tuple[int, ...] = (0, 1),
    incomplete_layer: int | None = None,
) -> None:
    """Write two contiguous, zero-based MoE layers with two experts each."""
    writer = GGUFWriter(path, architecture if architecture else "deepseek4")
    if architecture is None:
        writer.remove_key("general.architecture")
    elif architecture != "deepseek4":
        writer.remove_key("general.architecture")
        writer.add_key_value("general.architecture", architecture, GGUFValueType.STRING)
    writer.add_name("synthetic-expert-major-test")
    writer.add_uint32("synthetic.marker", 0xC0FFEE)
    if alignment is not None:
        writer.add_custom_alignment(alignment)

    # numpy [expert, row, column] becomes GGML [column, row, expert]. Each
    # expert is therefore one contiguous 2 x 4 float32 slice.
    for layer in layers:
        for part_index, part in enumerate(("gate", "up", "down")):
            if layer == incomplete_layer and part == "down":
                continue
            start = 1000 * layer + 100 * part_index
            values = np.arange(start, start + 16, dtype=np.float32).reshape(2, 2, 4)
            writer.add_tensor(f"blk.{layer}.ffn_{part}_exps.weight", values)

    non_expert = np.arange(24, dtype=np.float32).reshape(3, 8)
    writer.add_tensor("token_embd.weight", non_expert)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def write_unknown_type_gguf(path: Path, type_id: int) -> None:
    """Write the smallest GGUF that exposes the copied reader's fallback."""
    def gguf_string(value: str) -> bytes:
        encoded = value.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    header = bytearray()
    header += b"GGUF"
    header += struct.pack("<I", 3)  # GGUF version
    header += struct.pack("<Q", 1)  # tensor count
    header += struct.pack("<Q", 1)  # metadata count
    header += gguf_string("general.alignment")
    header += struct.pack("<I", 4)  # GGUF_TYPE_UINT32
    header += struct.pack("<I", 32)
    header += gguf_string("unknown.weight")
    header += struct.pack("<I", 1)  # dimensions
    header += struct.pack("<Q", 1)
    header += struct.pack("<I", type_id)
    header += struct.pack("<Q", 0)  # data-section-relative offset
    header += b"\0" * ((32 - len(header) % 32) % 32)
    path.write_bytes(header + b"\0\0\0\0")


class ConverterTests(unittest.TestCase):
    maxDiff = None

    def run_converter(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        current_python_path = environment.get("PYTHONPATH")
        path_entries = [str(GGUF_PY_DIRECTORY)]
        if current_python_path:
            path_entries.append(current_python_path)
        environment["PYTHONPATH"] = os.pathsep.join(path_entries)
        return subprocess.run(
            [sys.executable, str(CONVERTER), *arguments],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

    def run_preflight(self, source: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(PREFLIGHT), "--src", str(source)],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def run_expert_major_probe(self, model: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(EXPERT_MAJOR_PROBE), "--model", str(model)],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_module_import_and_contiguous_layer_discovery(self) -> None:
        module = load_converter_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "source.gguf"
            write_synthetic_moe(source)
            parsed = GGUF(source)
            parts, layers, experts, _geometry = module.discover(parsed)
            self.assertEqual(parts, ("gate", "up", "down"))
            self.assertEqual(layers, [0, 1])
            self.assertEqual(experts, 2)
            parsed.f.close()

    def test_cli_help(self) -> None:
        completed = self.run_converter("--help")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--dry-run", completed.stdout)
        self.assertIn("--verify", completed.stdout)

    def test_fail_closed_repack_preflight_accepts_complete_contiguous_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "source.gguf"
            write_synthetic_moe(source)
            completed = self.run_preflight(source)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            evidence = json.loads(completed.stdout)
            self.assertEqual(evidence["status"], "preflight-ok")
            self.assertEqual(evidence["sourceArchitecture"], "deepseek4")

    def test_fail_closed_repack_preflight_rejects_missing_source_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "source.gguf"
            write_synthetic_moe(source, architecture=None)
            completed = self.run_preflight(source)
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
            self.assertIn("general.architecture must be an explicit string", completed.stderr)

    def test_fail_closed_repack_preflight_rejects_empty_source_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "source.gguf"
            write_synthetic_moe(source, architecture="")
            completed = self.run_preflight(source)
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
            self.assertIn("general.architecture must not be empty", completed.stderr)

    def test_fail_closed_repack_preflight_rejects_untrimmed_source_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "source.gguf"
            write_synthetic_moe(source, architecture=" deepseek4 ")
            completed = self.run_preflight(source)
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
            self.assertIn("must not contain surrounding whitespace", completed.stderr)

    def test_fail_closed_repack_preflight_rejects_known_converter_hazards(self) -> None:
        cases = (
            {"alignment": 1024},
            {"incomplete_layer": 1},
            {"layers": (0, 2)},
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary_directory:
                source = Path(temporary_directory) / "source.gguf"
                write_synthetic_moe(source, **case)
                completed = self.run_preflight(source)
                self.assertNotEqual(completed.returncode, 0, completed.stdout)
                self.assertIn("preflight failed:", completed.stderr)

    def test_expert_major_probe_rejects_stock_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "source.gguf"
            write_synthetic_moe(source)
            completed = self.run_expert_major_probe(source)
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
            self.assertIn("siliangem.expert_major", completed.stderr)

    def test_expert_major_probe_rejects_missing_loader_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            source = temporary / "source.gguf"
            destination = temporary / "expert-major.gguf"
            write_synthetic_moe(source)
            create = self.run_converter("--src", str(source), "--dst", str(destination))
            self.assertEqual(create.returncode, 0, create.stderr)

            contents = destination.read_bytes()
            original_key = b"siliangem.part_ne1"
            replacement_key = b"siliangem.part_xe1"
            self.assertEqual(contents.count(original_key), 1)
            destination.write_bytes(contents.replace(original_key, replacement_key, 1))

            completed = self.run_expert_major_probe(destination)
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
            self.assertIn("siliangem.part_ne1", completed.stderr)

    def test_expert_major_probe_rejects_incomplete_conversion_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            source = temporary / "incomplete-source.gguf"
            destination = temporary / "expert-major.gguf"
            write_synthetic_moe(source, incomplete_layer=1)
            create = self.run_converter("--src", str(source), "--dst", str(destination))
            self.assertEqual(create.returncode, 0, create.stderr)

            completed = self.run_expert_major_probe(destination)
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
            self.assertIn("residual logical expert tensors", completed.stderr)

    def test_expert_major_probe_rejects_misdeclared_source_alignment_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            source = temporary / "aligned-source.gguf"
            destination = temporary / "expert-major.gguf"
            write_synthetic_moe(source, alignment=1024)
            create = self.run_converter("--src", str(source), "--dst", str(destination))
            self.assertEqual(create.returncode, 0, create.stderr)

            completed = self.run_expert_major_probe(destination)
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
            self.assertRegex(completed.stderr, r"noncanonical relative offset|canonical extent")

    @unittest.expectedFailure
    def test_copied_reader_tracks_current_forked_ggml_types(self) -> None:
        """Known limitation: GGML type IDs 40-42 are absent from the copy."""
        self.assertEqual(GGML_TYPES[40], ("NVFP4", 64, 36))
        self.assertEqual(GGML_TYPES[41], ("Q1_0", 128, 18))
        self.assertEqual(GGML_TYPES[42], ("Q2_0", 64, 18))

    @unittest.expectedFailure
    def test_copied_reader_rejects_unknown_ggml_types(self) -> None:
        """Known limitation: unknown IDs silently assume four bytes/element."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "unknown-type.gguf"
            write_unknown_type_gguf(source, 43)
            parsed = None
            try:
                with self.assertRaisesRegex(ValueError, "unknown GGML type"):
                    parsed = GGUF(source)
            finally:
                if parsed is not None:
                    parsed.f.close()

    @unittest.expectedFailure
    def test_verify_rejects_nonpositive_sample_count(self) -> None:
        """Known limitation: --samples 0 currently reports a vacuous success."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            source = temporary / "source.gguf"
            destination = temporary / "expert-major.gguf"
            write_synthetic_moe(source)
            create = self.run_converter("--src", str(source), "--dst", str(destination))
            self.assertEqual(create.returncode, 0, create.stderr)
            verify = self.run_converter(
                "--src", str(source), "--dst", str(destination),
                "--verify", "--samples", "0",
            )
            self.assertNotEqual(verify.returncode, 0, verify.stdout)

    @unittest.expectedFailure
    def test_conversion_preserves_source_alignment_above_512(self) -> None:
        """Known limitation: metadata says 512 while offsets use source 1024."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            source = temporary / "aligned-source.gguf"
            destination = temporary / "expert-major.gguf"
            write_synthetic_moe(source, alignment=1024)
            create = self.run_converter("--src", str(source), "--dst", str(destination))
            self.assertEqual(create.returncode, 0, create.stderr)
            parsed = GGUF(destination)
            try:
                self.assertEqual(parsed.kv["general.alignment"], 1024)
            finally:
                parsed.f.close()

    @unittest.expectedFailure
    def test_conversion_rejects_incomplete_moe_layer(self) -> None:
        """Known limitation: an incomplete layer is copied as ordinary tensors."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            source = temporary / "incomplete-source.gguf"
            destination = temporary / "expert-major.gguf"
            write_synthetic_moe(source, incomplete_layer=1)
            create = self.run_converter("--src", str(source), "--dst", str(destination))
            self.assertNotEqual(create.returncode, 0, create.stdout)

    @unittest.expectedFailure
    def test_conversion_rejects_noncontiguous_layer_ids(self) -> None:
        """Known limitation: complete layers 0 and 2 currently convert."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            source = temporary / "gapped-source.gguf"
            destination = temporary / "expert-major.gguf"
            write_synthetic_moe(source, layers=(0, 2))
            create = self.run_converter("--src", str(source), "--dst", str(destination))
            self.assertNotEqual(create.returncode, 0, create.stdout)

    def test_dry_run_create_then_separate_verify_preserves_every_byte(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            source = temporary / "source.gguf"
            destination = temporary / "expert-major.gguf"
            write_synthetic_moe(source)

            dry_run = self.run_converter(
                "--src", str(source), "--dst", str(destination), "--dry-run"
            )
            self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
            self.assertIn("layers        : 2", dry_run.stdout)
            self.assertFalse(destination.exists(), "dry-run unexpectedly wrote output")

            create = self.run_converter(
                "--src", str(source), "--dst", str(destination)
            )
            self.assertEqual(create.returncode, 0, create.stderr)
            self.assertTrue(destination.is_file())
            self.assertIn("done:", create.stdout)

            verify = self.run_converter(
                "--src", str(source),
                "--dst", str(destination),
                "--verify", "--samples", "8",
            )
            self.assertEqual(verify.returncode, 0, verify.stderr)
            self.assertIn("8/8", verify.stdout)

            metadata_probe = self.run_expert_major_probe(destination)
            self.assertEqual(metadata_probe.returncode, 0, metadata_probe.stderr)
            self.assertIn('"status": "expert-major-metadata-ok"', metadata_probe.stdout)
            self.assertIn('"geometryEntries": 6', metadata_probe.stdout)
            self.assertIn('"packedTensorCount": 2', metadata_probe.stdout)

            source_local = GGUF(source)
            destination_local = GGUF(destination)
            destination_reader = GGUFReader(destination)

            source_non_expert = source_local.tensors["token_embd.weight"]
            destination_non_expert = destination_local.tensors["token_embd.weight"]
            with source.open("rb") as source_file, destination.open("rb") as destination_file:
                source_file.seek(source_non_expert.abs_offset)
                destination_file.seek(destination_non_expert.abs_offset)
                self.assertEqual(
                    source_file.read(source_non_expert.nbytes),
                    destination_file.read(destination_non_expert.nbytes),
                    "non-expert tensor bytes changed",
                )

                fields = destination_reader.fields
                self.assertIn("siliangem.expert_major", fields)
                self.assertNotIn("behemoth.expert_major", fields)
                strides = [int(value) for value in fields["siliangem.expert_bytes"].contents()]
                layer_indices = [int(value) for value in fields["siliangem.layer_index"].contents()]
                self.assertEqual(layer_indices, [0, 1])
                self.assertEqual(len(strides), 2)

                packed_by_layer = {
                    int(tensor.name.split(".")[1]): tensor.data.view("uint8").reshape(-1)
                    for tensor in destination_reader.tensors
                    if ".ffn_exps_packed.weight" in tensor.name
                }
                self.assertEqual(sorted(packed_by_layer), [0, 1])

                for layer_position, layer in enumerate(layer_indices):
                    packed = packed_by_layer[layer]
                    stride = strides[layer_position]
                    for expert in range(2):
                        packed_offset = expert * stride
                        raw_bytes = 0
                        for part in ("gate", "up", "down"):
                            name = f"blk.{layer}.ffn_{part}_exps.weight"
                            tensor = source_local.tensors[name]
                            expert_bytes = tensor.nbytes // 2
                            source_file.seek(tensor.abs_offset + expert * expert_bytes)
                            expected = source_file.read(expert_bytes)
                            actual = packed[
                                packed_offset + raw_bytes:
                                packed_offset + raw_bytes + expert_bytes
                            ].tobytes()
                            self.assertEqual(expected, actual, f"changed bytes in {name}, expert {expert}")
                            raw_bytes += expert_bytes
                        self.assertTrue(
                            np.all(packed[packed_offset + raw_bytes:packed_offset + stride] == 0),
                            f"non-zero alignment padding in layer {layer}, expert {expert}",
                        )

            source_local.f.close()
            destination_local.f.close()
            del packed
            packed_by_layer.clear()
            del fields
            close_gguf_reader(destination_reader)


if __name__ == "__main__":
    unittest.main()
