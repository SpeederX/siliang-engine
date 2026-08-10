from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yaml"


class ReleasePackagingContractTests(unittest.TestCase):
    def test_windows_packages_include_and_smoke_openssl_runtime(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("^OPENSSL_INCLUDE_DIR:PATH=(.+)$", text)
        self.assertIn("libssl-3-x64.dll", text)
        self.assertIn("libcrypto-3-x64.dll", text)
        self.assertIn("OPENSSL-Apache-2.0.txt", text)
        self.assertIn('"openssl_runtime=$opensslVersion"', text)
        self.assertIn(
            '$env:PATH = "$binTarget;$env:SystemRoot\\System32;$env:SystemRoot"',
            text,
        )
        self.assertIn("@('llama-server.exe', 'llama-cli.exe')", text)
        self.assertIn("--version", text)


if __name__ == "__main__":
    unittest.main()
