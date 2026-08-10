from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPACK_WRAPPER = REPOSITORY_ROOT / "scripts" / "repack-model.ps1"
POWERSHELL = shutil.which("powershell.exe")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_converter import write_synthetic_moe  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def partial_path(destination: Path) -> Path:
    if destination.suffix:
        return destination.with_name(
            destination.name[: -len(destination.suffix)]
            + ".partial"
            + destination.suffix
        )
    return Path(str(destination) + ".partial")


@unittest.skipUnless(POWERSHELL, "Windows PowerShell is required")
class RepackWrapperTests(unittest.TestCase):
    maxDiff = None

    def test_hashing_does_not_require_get_file_hash_cmdlet(self) -> None:
        source = REPACK_WRAPPER.read_text(encoding="utf-8")
        self.assertNotIn("Get-FileHash", source)
        self.assertIn("[Security.Cryptography.SHA256]::Create()", source)

    def run_wrapper(
        self,
        source: Path,
        destination: Path,
        *,
        python_executable: str = sys.executable,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        process_environment = os.environ.copy()
        if environment:
            process_environment.update(environment)
        return subprocess.run(
            [
                str(POWERSHELL),
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(REPACK_WRAPPER),
                "-Source",
                str(source),
                "-Destination",
                str(destination),
                "-Samples",
                "8",
                "-PythonExecutable",
                python_executable,
            ],
            cwd=REPOSITORY_ROOT,
            env=process_environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )

    def make_fake_python(self, directory: Path) -> Path:
        driver = directory / "fake_python_driver.py"
        driver.write_text(
            textwrap.dedent(
                """
                from __future__ import annotations

                import json
                import os
                from pathlib import Path
                import sys


                script = Path(sys.argv[1]).name
                arguments = sys.argv[2:]
                if script == "preflight_repack.py":
                    action = "preflight"
                elif script == "check_expert_major.py":
                    action = "probe"
                elif script == "make_expert_major_gguf.py":
                    if "--dry-run" in arguments:
                        action = "dry-run"
                    elif "--verify" in arguments:
                        action = "verify"
                    else:
                        action = "create"
                else:
                    print(f"unexpected script: {script}", file=sys.stderr)
                    raise SystemExit(90)

                with Path(os.environ["SILIANG_TEST_FAKE_LOG"]).open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(json.dumps({"action": action, "args": arguments}) + "\\n")

                if action == "preflight":
                    mode = os.environ.get("SILIANG_TEST_PREFLIGHT_MODE", "valid")
                    if mode == "invalid-json":
                        print("not-json")
                    elif mode == "missing-architecture":
                        print(json.dumps({"status": "preflight-ok"}))
                    elif mode == "empty-architecture":
                        print(json.dumps({
                            "status": "preflight-ok",
                            "sourceArchitecture": "",
                        }))
                    elif mode == "untrimmed-architecture":
                        print(json.dumps({
                            "status": "preflight-ok",
                            "sourceArchitecture": " deepseek4 ",
                        }))
                    else:
                        print(json.dumps({
                            "status": "preflight-ok",
                            "sourceArchitecture": "deepseek4",
                        }))
                    raise SystemExit(0)

                if action == "create":
                    destination = Path(arguments[arguments.index("--dst") + 1])
                    destination.write_bytes(b"verified-partial-output")

                if action == "probe":
                    mutation_target = os.environ.get("SILIANG_TEST_MUTATE_SOURCE")
                    if mutation_target:
                        Path(mutation_target).write_bytes(b"source-mutated-during-repack")
                    if os.environ.get("SILIANG_TEST_FAIL_PROBE") == "1":
                        print("deliberate structural-probe failure", file=sys.stderr)
                        raise SystemExit(23)

                print(action)
                """
            ).lstrip(),
            encoding="utf-8",
        )
        command = directory / "fake-python.cmd"
        command.write_text(
            f'@echo off\n"{sys.executable}" "{driver}" %*\n',
            encoding="utf-8",
        )
        return command

    def read_fake_actions(self, log: Path) -> list[str]:
        return [
            json.loads(line)["action"]
            for line in log.read_text(encoding="utf-8").splitlines()
        ]

    def test_real_small_repack_runs_all_gates_and_records_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            source = temporary / "source.gguf"
            destination = temporary / "expert-major.gguf"
            write_synthetic_moe(source)
            source_digest = sha256(source)
            source_size = source.stat().st_size

            completed = self.run_wrapper(source, destination)
            combined = completed.stdout + completed.stderr
            self.assertEqual(completed.returncode, 0, combined)
            self.assertTrue(destination.is_file())
            self.assertFalse(partial_path(destination).exists())
            self.assertEqual(sha256(source), source_digest)

            self.assertIn("Source before repack:", completed.stdout)
            self.assertIn("Source after repack:", completed.stdout)
            self.assertIn("Verified partial output:", completed.stdout)
            self.assertIn("Final output:", completed.stdout)
            self.assertIn('"status": "expert-major-metadata-ok"', completed.stdout)

            receipts = [
                json.loads(line)
                for line in completed.stdout.splitlines()
                if line.startswith('{"status":"repack-finalized"')
            ]
            self.assertEqual(len(receipts), 1, completed.stdout)
            receipt = receipts[0]
            self.assertEqual(receipt["sourceArchitecture"], "deepseek4")
            self.assertEqual(receipt["sourceBytesBefore"], source_size)
            self.assertEqual(receipt["sourceSHA256Before"], source_digest)
            self.assertEqual(receipt["sourceBytesAfter"], source_size)
            self.assertEqual(receipt["sourceSHA256After"], source_digest)
            self.assertEqual(receipt["partialBytes"], destination.stat().st_size)
            self.assertEqual(receipt["partialSHA256"], sha256(destination))
            self.assertEqual(receipt["destinationBytes"], destination.stat().st_size)
            self.assertEqual(receipt["destinationSHA256"], sha256(destination))
            self.assertEqual(receipt["samples"], 8)

    def test_rejects_invalid_preflight_json_or_architecture_before_conversion(self) -> None:
        cases = (
            ("invalid-json", "emitted invalid JSON evidence"),
            ("missing-architecture", "must contain sourceArchitecture as a string"),
            ("empty-architecture", "sourceArchitecture must not be empty"),
            ("untrimmed-architecture", "must not contain surrounding whitespace"),
        )
        for mode, expected_error in cases:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary_directory:
                temporary = Path(temporary_directory)
                source = temporary / "source.gguf"
                destination = temporary / "expert-major.gguf"
                log = temporary / "fake-python.jsonl"
                source.write_bytes(b"immutable-source")
                fake_python = self.make_fake_python(temporary)

                completed = self.run_wrapper(
                    source,
                    destination,
                    python_executable=str(fake_python),
                    environment={
                        "SILIANG_TEST_FAKE_LOG": str(log),
                        "SILIANG_TEST_PREFLIGHT_MODE": mode,
                    },
                )
                combined = completed.stdout + completed.stderr
                self.assertNotEqual(completed.returncode, 0, combined)
                self.assertIn(expected_error, combined)
                self.assertEqual(self.read_fake_actions(log), ["preflight"])
                self.assertFalse(partial_path(destination).exists())
                self.assertFalse(destination.exists())

    def test_refuses_existing_partial_or_final_before_running_tools(self) -> None:
        for occupied_output in ("partial", "final"):
            with self.subTest(occupied_output=occupied_output), tempfile.TemporaryDirectory() as temporary_directory:
                temporary = Path(temporary_directory)
                source = temporary / "source.gguf"
                destination = temporary / "expert-major.gguf"
                source.write_bytes(b"source")
                occupied = (
                    partial_path(destination)
                    if occupied_output == "partial"
                    else destination
                )
                occupied.write_bytes(b"do-not-overwrite")

                completed = self.run_wrapper(source, destination)
                combined = completed.stdout + completed.stderr
                self.assertNotEqual(completed.returncode, 0, combined)
                self.assertIn("Refusing to overwrite existing output", combined)
                self.assertEqual(occupied.read_bytes(), b"do-not-overwrite")

    def test_structural_probe_failure_retains_partial_and_never_finalizes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            source = temporary / "source.gguf"
            destination = temporary / "expert-major.gguf"
            log = temporary / "fake-python.jsonl"
            source.write_bytes(b"immutable-source")
            fake_python = self.make_fake_python(temporary)

            completed = self.run_wrapper(
                source,
                destination,
                python_executable=str(fake_python),
                environment={
                    "SILIANG_TEST_FAKE_LOG": str(log),
                    "SILIANG_TEST_FAIL_PROBE": "1",
                },
            )
            combined = completed.stdout + completed.stderr
            self.assertNotEqual(completed.returncode, 0, combined)
            self.assertIn("failed with exit code 23", combined)
            self.assertEqual(
                self.read_fake_actions(log),
                ["preflight", "dry-run", "create", "verify", "probe"],
            )
            self.assertEqual(
                partial_path(destination).read_bytes(), b"verified-partial-output"
            )
            self.assertFalse(destination.exists())

    def test_source_hash_change_fails_closed_and_retains_verified_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            source = temporary / "source.gguf"
            destination = temporary / "expert-major.gguf"
            log = temporary / "fake-python.jsonl"
            source.write_bytes(b"source-before-repack")
            fake_python = self.make_fake_python(temporary)

            completed = self.run_wrapper(
                source,
                destination,
                python_executable=str(fake_python),
                environment={
                    "SILIANG_TEST_FAKE_LOG": str(log),
                    "SILIANG_TEST_MUTATE_SOURCE": str(source),
                },
            )
            combined = completed.stdout + completed.stderr
            self.assertNotEqual(completed.returncode, 0, combined)
            self.assertIn("Source GGUF changed during repack", combined)
            self.assertEqual(
                self.read_fake_actions(log),
                ["preflight", "dry-run", "create", "verify", "probe"],
            )
            self.assertEqual(
                partial_path(destination).read_bytes(), b"verified-partial-output"
            )
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
