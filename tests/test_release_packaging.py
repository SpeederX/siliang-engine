from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yaml"
PUBLISH_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
BUILD_SCRIPT = ROOT / "scripts" / "build.ps1"
README = ROOT / "README.md"
SCRIPTS_README = ROOT / "scripts" / "README.md"


class ReleasePackagingContractTests(unittest.TestCase):
    def test_release_packages_ship_cli_configuration_without_environment_helper(self) -> None:
        ci_text = WORKFLOW.read_text(encoding="utf-8")
        publish_text = PUBLISH_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn('default: "v0.1.4"', publish_text)
        for text in (ci_text, publish_text):
            self.assertIn("'.\\docs\\CONFIGURATION.md'", text)
            self.assertIn('$releaseNotesPath = ".\\docs\\releases\\$tag.md"', text)
            self.assertIn("$releaseDocsTarget = Join-Path $docsTarget 'releases'", text)
            self.assertIn("$releaseNotesPath -Destination $releaseDocsTarget", text)
            self.assertNotIn("siliang-" + "env.ps1", text)
            self.assertIn("llama-cli.exe", text)
            self.assertIn("llama-server.exe", text)
            self.assertIn("$requiredExpertCacheOptions", text)
            self.assertIn("--expert-cache-l2-mib", text)
            self.assertIn("--expert-cache-exchange-r", text)
            self.assertIn("--expert-cache-roll", text)
            self.assertIn("--expert-cache-memory-report", text)
            self.assertIn("--expert-cache-deferred-wait", text)
            self.assertIn("[regex]::IsMatch($helpText, $helpPattern)", text)

    def test_build_uses_portable_runtime_selected_cpu_variants(self) -> None:
        text = BUILD_SCRIPT.read_text(encoding="utf-8")

        for definition in (
            "-DBUILD_SHARED_LIBS=ON",
            "-DGGML_NATIVE=OFF",
            "-DGGML_BACKEND_DL=ON",
            "-DGGML_CPU_ALL_VARIANTS=ON",
        ):
            self.assertIn(definition, text)

    def test_windows_release_build_provisions_embedded_web_ui_and_fails_closed(self) -> None:
        text = BUILD_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("-DLLAMA_BUILD_UI=ON", text)
        self.assertIn("-DLLAMA_USE_PREBUILT_UI=ON", text)
        self.assertIn("llama_build_ui = $true", text)
        self.assertIn("llama_use_prebuilt_ui = $true", text)
        self.assertIn("tools\\ui\\ui.h", text)
        self.assertIn("#define LLAMA_UI_HAS_ASSETS 1", text)
        self.assertIn("Release build completed without embedded llama.cpp Web UI assets.", text)

    def test_cuda_architecture_override_is_optional_and_release_uses_upstream_multi_arch(self) -> None:
        build_text = BUILD_SCRIPT.read_text(encoding="utf-8")
        workflow_text = WORKFLOW.read_text(encoding="utf-8")

        self.assertNotIn("-CudaArchitecture is required", build_text)
        self.assertNotIn("$pinnedCudaArchitectures", build_text)
        self.assertIn("if (-not [string]::IsNullOrWhiteSpace($CudaArchitecture))", build_text)
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES={0}", build_text)
        self.assertIn("<llama.cpp upstream default>", build_text)
        self.assertIn("label: CUDA 13.2 multi-arch", workflow_text)
        self.assertIn("slug: cuda-13.2", workflow_text)
        self.assertNotIn("cuda-13.2-sm75", workflow_text)
        self.assertNotIn("matrix.cuda_architecture", workflow_text)
        self.assertIn(
            r".\scripts\build.ps1 -Backend Cuda -BuildDirectory $buildDirectory",
            workflow_text,
        )
        self.assertNotIn("$expectedCudaArchitectures", workflow_text)
        self.assertIn("SILIANG-BUILD-CONFIG.json", build_text)
        self.assertIn("$cudaArchitectureSource = 'llama.cpp-upstream'", build_text)
        self.assertIn("SILIANG-BUILD-CONFIG.json", workflow_text)
        self.assertIn("Tagged CUDA releases must use upstream architecture selection", workflow_text)
        self.assertIn("Expected an upstream multi-architecture CUDA release build", workflow_text)
        self.assertIn("cuda_architectures=$effectiveCudaArchitectures", workflow_text)

        readme_text = README.read_text(encoding="utf-8")
        scripts_readme_text = SCRIPTS_README.read_text(encoding="utf-8")
        self.assertIn("upstream `llama.cpp` multi-architecture selection", readme_text)
        self.assertIn("-CudaArchitecture 75", readme_text)
        self.assertIn("delegates architecture selection", scripts_readme_text)
        self.assertIn("-CudaArchitecture 75", scripts_readme_text)

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

    def test_release_gate_validates_linux_and_macos_upstream_compatibility(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("validate-linux:", text)
        self.assertIn("runs-on: ubuntu-24.04", text)
        self.assertIn("validate-macos:", text)
        self.assertIn("runs-on: macos-14", text)
        self.assertIn("-DGGML_NATIVE=OFF", text)
        self.assertIn("-DGGML_BACKEND_DL=ON", text)
        self.assertIn("--list-devices", text)
        self.assertIn("needs: [validate-windows, validate-linux, validate-macos]", text)

    def test_windows_packages_require_and_record_cpu_dispatch_variants(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")

        expected_variants = {
            "ggml-cpu-alderlake.dll",
            "ggml-cpu-cannonlake.dll",
            "ggml-cpu-cascadelake.dll",
            "ggml-cpu-haswell.dll",
            "ggml-cpu-icelake.dll",
            "ggml-cpu-sandybridge.dll",
            "ggml-cpu-skylakex.dll",
            "ggml-cpu-sse42.dll",
            "ggml-cpu-x64.dll",
        }
        for variant in expected_variants:
            self.assertEqual(text.count(f"'{variant}'"), 1)
        self.assertIn("Portable CPU backend inventory mismatch", text)
        self.assertIn("CMakeCache.txt", text)
        self.assertIn("$expectedCacheEntries", text)
        self.assertIn("--verbose --list-devices", text)
        self.assertIn("loaded CPU backend from", text)
        self.assertIn("function Invoke-PackagedNative", text)
        self.assertIn("[Diagnostics.ProcessStartInfo]::new()", text)
        self.assertIn("RedirectStandardOutput = $true", text)
        self.assertIn("RedirectStandardError = $true", text)
        self.assertIn("$dispatchResult = Invoke-PackagedNative", text)
        self.assertIn("RUNTIME-DISPATCH.log", text)
        self.assertIn("'--log-file', $dispatchLogPath", text)
        self.assertIn("$dispatchLogText = Get-Content -LiteralPath $dispatchLogPath -Raw", text)
        self.assertNotIn("$dispatchOutput = @(&", text)
        self.assertIn("$cpuBackendMatches = [regex]::Matches(", text)
        self.assertIn("loaded CPU backend from .*?(ggml-cpu-[A-Za-z0-9._-]+[.]dll)", text)
        self.assertIn("$cpuBackendNames.Count -ne 1", text)
        self.assertIn("Runtime dispatch log must identify exactly one packaged CPU backend variant", text)
        self.assertNotIn("$cudaBackendMatches = [regex]::Matches(", text)
        self.assertNotIn("ggml-cpu-[A-Za-z0-9._-]+\\.dll)\\s*$", text)
        self.assertIn("ggml_native=OFF", text)
        self.assertIn("ggml_backend_dl=ON", text)
        self.assertIn("ggml_cpu_all_variants=ON", text)
        self.assertIn("cpu_dispatch=runtime-selected", text)
        self.assertIn("cpu_dispatch_evidence=provenance/RUNTIME-DISPATCH.log", text)
        self.assertIn("cpu_dispatch_build_host_selection=$selectedCpuVariant", text)
        self.assertIn("Copy-Item -LiteralPath $buildReceiptPath -Destination $provenanceTarget", text)

    def test_publish_workflow_uses_native_process_capture_for_package_smoke(self) -> None:
        text = PUBLISH_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("function Invoke-PackagedNative", text)
        self.assertIn("[Diagnostics.ProcessStartInfo]::new()", text)
        self.assertIn("RedirectStandardOutput = $true", text)
        self.assertIn("RedirectStandardError = $true", text)
        self.assertIn("$versionResult = Invoke-PackagedNative", text)
        self.assertIn("$helpResult = Invoke-PackagedNative", text)
        self.assertNotIn("$helpText = @(&", text)

    def test_cuda_package_requires_exact_runtime_set_and_loads_it_isolated(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("CUDA backend was not packaged", text)
        self.assertIn("cudart = 'cudart64_*.dll'", text)
        self.assertIn("cublas = 'cublas64_*.dll'", text)
        self.assertIn("cublasLt = 'cublasLt64_*.dll'", text)
        self.assertIn("Expected one $($runtimePattern.Key) runtime DLL name", text)
        self.assertIn("[Runtime.InteropServices.NativeLibrary]::Load", text)
        self.assertIn("[Runtime.InteropServices.NativeLibrary]::Free", text)
        self.assertIn("Packaged CUDA library is not loadable", text)
        self.assertNotIn("$cudaBackendMatches = [regex]::Matches(", text)
        self.assertNotIn("Runtime did not load exactly the packaged ggml-cuda.dll backend", text)
        self.assertNotIn("The CPU package unexpectedly reported a loaded CUDA backend", text)
        self.assertIn("GitHub's Windows host is not required to expose a physical GPU", text)
        self.assertIn("cuda_runtime_cudart=$($cudaRuntimeNames.cudart)", text)
        self.assertIn("cuda_runtime_cublas=$($cudaRuntimeNames.cublas)", text)
        self.assertIn("cuda_runtime_cublaslt=$($cudaRuntimeNames.cublasLt)", text)
        self.assertIn("cuda_backend_loadability=", text)


if __name__ == "__main__":
    unittest.main()
