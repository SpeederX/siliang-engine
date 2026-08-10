import json
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "siliang-env.ps1"


@unittest.skipUnless(shutil.which("powershell.exe"), "Windows PowerShell is required")
class SiliangEnvironmentTests(unittest.TestCase):
    def run_ps(self, body: str) -> dict:
        escaped_script = str(SCRIPT).replace("'", "''")
        command = (
            "$ErrorActionPreference='Stop'; "
            f"$script='{escaped_script}'; "
            f"{body}"
        )
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout.strip().splitlines()[-1])

    def test_enable_and_reset(self):
        values = self.run_ps(
            "& $script -CacheMiB 1234 -Verbose -NoMemoryReport | Out-Null; "
            "$enabled=[ordered]@{cache=$env:SILIANGEM_CACHE_MIB;defer=$env:SILIANGEM_DEFER;"
            "verbose=$env:SILIANGEM_VERBOSE;mem=$env:SILIANGEM_MEM_REPORT;"
            "prefetch=$env:GGML_MOE_PREFETCH;disabled=$env:SILIANGEM_DISABLE}; "
            "& $script -Reset | Out-Null; "
            "$enabled.reset=($null -eq $env:SILIANGEM_CACHE_MIB -and "
            "$null -eq $env:SILIANGEM_VERBOSE -and $null -eq $env:GGML_MOE_PREFETCH); "
            "$enabled | ConvertTo-Json -Compress"
        )
        self.assertEqual(values["cache"], "1234")
        self.assertEqual(values["defer"], "1")
        self.assertEqual(values["verbose"], "1")
        self.assertEqual(values["mem"], "0")
        self.assertEqual(values["prefetch"], "0")
        self.assertIsNone(values["disabled"])
        self.assertTrue(values["reset"])

    def test_disabled_control(self):
        values = self.run_ps(
            "& $script -Disable | Out-Null; "
            "[ordered]@{disabled=$env:SILIANGEM_DISABLE;cache=$env:SILIANGEM_CACHE_MIB;"
            "prefetch=$env:GGML_MOE_PREFETCH} | ConvertTo-Json -Compress"
        )
        self.assertEqual(values["disabled"], "1")
        self.assertIsNone(values["cache"])
        self.assertEqual(values["prefetch"], "0")


if __name__ == "__main__":
    unittest.main()
