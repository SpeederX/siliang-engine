from __future__ import annotations

from pathlib import Path
import json
import os
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.request


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (REPOSITORY_ROOT / path).read_text(encoding="utf-8")


ARG_SOURCE = read("common/arg.cpp")
LLAMA_HEADER = read("include/llama.h")
LLAMA_CONTEXT = read("src/llama-context.cpp")
LLAMA_GRAPH = read("src/llama-graph.cpp")
MOE_RUNTIME = read("src/siliang-moe-runtime.cpp")
QWEN4EXP = read("src/models/qwen4exp.cpp")
KV_CACHE = read("src/llama-kv-cache.cpp")
RECURRENT = read("src/llama-memory-recurrent.cpp")
CUDA_SOURCE = read("ggml/src/ggml-cuda/ggml-cuda.cu")
TOPK_SOURCE = read("ggml/src/ggml-cuda/top-k.cu")
CACHE_SOURCE = read("ggml/src/ggml-cpu/siliangem_moe_cache.h")
CPU_SOURCE = read("ggml/src/ggml-cpu/ggml-cpu.c")
SERVER_SOURCE = read("tools/server/server-context.cpp")
CONFIGURATION = read("docs/CONFIGURATION.md")
CLI_HELP_DOC = read("tools/cli/README.md")
SERVER_HELP_DOC = read("tools/server/README.md")


def function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening:index + 1]
    raise AssertionError(f"unterminated body for {signature}")


class SiliangPrefillDecodeContract(unittest.TestCase):
    def test_decode_k_is_explicit_fitted_and_fail_closed(self) -> None:
        self.assertIn("uint32_t l1_k_decode;", LLAMA_HEADER)
        self.assertIn('{"--expert-cache-l1-k-decode"}, "N"', ARG_SOURCE)
        # the extension is fitted to a measured compute buffer, then reserved again with the tail
        self.assertIn("if (siliang_fit_decode_tail()) {", LLAMA_CONTEXT)
        # every decode graph is checked before an extra slot is used
        check = function_body(MOE_RUNTIME, "void check_decode_extension()")
        self.assertIn("plan > k_ext_tail_head", check)
        self.assertIn("k_ext_active = false;", check)
        # bounded prefill never plans into the extra slots
        bounds = function_body(MOE_RUNTIME, "bool policy_bounds(")
        self.assertIn("prefill_range || !k_ext_active", bounds)
        self.assertIn("policy_bounds(layer, policy_first, policy_last, true)", MOE_RUNTIME)
        # entering decode zeroes the extra slots before a kernel can read them
        self.assertIn("clear_decode_extension();", function_body(MOE_RUNTIME, "void enter_phase("))
        # a reallocated compute buffer maps a new head in front of the same tail memory
        self.assertIn("tail_handles", CUDA_SOURCE)
        for doc in (CONFIGURATION, CLI_HELP_DOC, SERVER_HELP_DOC):
            self.assertIn("--expert-cache-l1-k-decode", doc)

    def test_layer_major_replays_placements(self) -> None:
        self.assertIn("bool prefill_layer_major;", LLAMA_HEADER)
        self.assertIn('{"--expert-cache-prefill-layer-major"}', ARG_SOURCE)
        # a second placement would skip the state reset of a new recurrent sequence: replay instead
        recurrent_apply = function_body(RECURRENT, "bool llama_memory_recurrent_context::apply()")
        self.assertIn("if (replay) {", recurrent_apply)
        self.assertIn("cur.rs_z  = mem->rs_z;", recurrent_apply)
        self.assertIn("n_kv = n_kv_by_ubatch[i_cur];", KV_CACHE)
        driver = function_body(LLAMA_CONTEXT, "int llama_context::siliang_prefill_layer_major(")
        self.assertIn("mctx->seek_replay(u)", driver)
        self.assertIn("return -3;", driver)
        # a one-layer graph builds only the inputs its range uses
        self.assertIn("range_attn ? build_inp_pos() : nullptr", QWEN4EXP)
        self.assertIn("const auto allocated = [](const ggml_tensor * t)", LLAMA_GRAPH)
        for doc in (CONFIGURATION, CLI_HELP_DOC, SERVER_HELP_DOC):
            self.assertIn("--expert-cache-prefill-layer-major", doc)

    def test_slot_checkpoints_travel_with_slot_files(self) -> None:
        self.assertIn("static int slot_checkpoints_save(", SERVER_SOURCE)
        self.assertIn("static bool slot_checkpoints_load(", SERVER_SOURCE)
        self.assertIn("ignoring an invalid context checkpoint file", SERVER_SOURCE)
        self.assertIn('{"--no-checkpoint-ubatch"}', ARG_SOURCE)
        self.assertIn("params_base.checkpoint_ubatch ? 4 + n_ubatch : 4", SERVER_SOURCE)
        self.assertIn(".ckpt", SERVER_HELP_DOC)

    def test_cuda_top_k_is_deterministic_by_default(self) -> None:
        # cub::DeviceTopK only offers unsorted output without a determinism guarantee
        self.assertIn("defined(CUB_TOP_K_AVAILABLE) && defined(GGML_CUDA_TOPK_NONDETERMINISTIC)", TOPK_SOURCE)
        self.assertIn("static void top_k_det_cuda(", TOPK_SOURCE)
        write = function_body(TOPK_SOURCE, "static __global__ void top_k_det_write(")
        self.assertNotIn("atomicAdd", write)
        self.assertIn("ExclusiveSum", write)

    def test_prefill_copies_overlap_reads_without_copying_pending_slots(self) -> None:
        self.assertIn("static void siliangem_wait_key(uint32_t key)", CACHE_SOURCE)
        copy_part = function_body(CPU_SOURCE, "int ggml_siliangem_cache_state_copy_cached_part(")
        self.assertIn("if (siliangem_read_pending(slot, transient)) return 0;", copy_part)
        self.assertIn("if (!wait_l2_read(layer, route.experts[union_index])) {", MOE_RUNTIME)
        self.assertIn("if (!finish_l2_reads()) {", MOE_RUNTIME)

    def test_in_batch_checkpoints_match_the_state_format(self) -> None:
        self.assertIn("LLAMA_API bool   llama_siliang_checkpoints_supported(", LLAMA_HEADER)
        # the blob carries the header state_seq_get_data writes before the memory part
        self.assertIn("siliang_seq_state_header(ck.seq_id, ck.blob);", LLAMA_CONTEXT)
        self.assertIn("balloc->set_split_cuts(std::move(cuts));", LLAMA_CONTEXT)
        self.assertIn("balloc->set_split_cuts({});", LLAMA_CONTEXT)
        # draft models keep the split path: a captured checkpoint has no draft state
        self.assertIn("ctx_dft == nullptr && !spec &&", SERVER_SOURCE)
        self.assertIn("add_captured_checkpoint(*slot, (int64_t) pos + 1, pos, data, size);", SERVER_SOURCE)
        self.assertIn("checkpoint_make_room(slot, n_tokens);", SERVER_SOURCE)

    def test_no_development_environment_switches(self) -> None:
        for source in (MOE_RUNTIME, QWEN4EXP, LLAMA_CONTEXT):
            self.assertNotIn("SILIANG_DEV_", source)
            self.assertNotIn("std::getenv(", source)


class RecurrentPromptReuseIntegration(unittest.TestCase):
    context_size = 512
    batch_size = 256
    ubatch_size = 32
    prompt_size = 197
    common_prefix_size = 161
    marker_positions = (16, 80, 144)
    request_timeout = 30
    reuse_pattern = ("a", "b", "b", "a", "a", "b")

    @classmethod
    def setUpClass(cls):
        binary = os.environ.get("SILIANG_TEST_SERVER")
        model = os.environ.get("SILIANG_TEST_QWEN_MODEL")
        if not binary or not model:
            raise unittest.SkipTest("set SILIANG_TEST_SERVER and SILIANG_TEST_QWEN_MODEL to run the generated-model regression")
        cls.binary = Path(binary).resolve()
        cls.model = Path(model).resolve()
        if not cls.binary.is_file() or not cls.model.is_file():
            raise AssertionError("regression binary/model is missing")
        output = os.environ.get("SILIANG_TEST_OUTPUT")
        cls.output = Path(output if output else tempfile.mkdtemp(prefix="siliang-prompt-reuse-"))
        cls.output.mkdir(parents=True, exist_ok=True)

    def check_prompt_reuse(self, checkpoints, message_spans=False):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        alias = f"prompt-reuse-regression-{port}"
        command = [str(self.binary), "--model", str(self.model), "--host", "127.0.0.1",
                   "--port", str(port), "--alias", alias, "--ctx-size", str(self.context_size), "--parallel", "1",
                   "--batch-size", str(self.batch_size), "--ubatch-size", str(self.ubatch_size), "--threads", "1",
                   "--threads-batch", "1", "--ctx-checkpoints", str(checkpoints), "--no-checkpoint-ubatch"]
        extra = json.loads(os.environ.get("SILIANG_TEST_EXTRA_ARGS", "[]"))
        if not isinstance(extra, list) or not all(isinstance(arg, str) for arg in extra):
            raise AssertionError("SILIANG_TEST_EXTRA_ARGS must be a JSON array of arguments")
        command.extend(extra)
        stem = self.output / f"checkpoints-{checkpoints}{'-spans' if message_spans else ''}"
        results = {"command": command, "requests": [], "passed": False, "fresh_reuse_mismatches": []}
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with stem.with_suffix(".log").open("wb") as log:
            process = subprocess.Popen(command, stdout=log, stderr=log, stdin=subprocess.DEVNULL, creationflags=flags)
            def request(method, path, data=None):
                payload = json.dumps(data).encode() if data is not None else None
                req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", payload,
                                             {"Content-Type": "application/json"}, method=method)
                with urllib.request.urlopen(req, timeout=self.request_timeout) as response:
                    return json.load(response)
            try:
                deadline = time.monotonic() + 90
                while True:
                    if process.poll() is not None:
                        self.fail(f"server exited during startup with {process.returncode}; see {stem}.log")
                    try:
                        request("GET", "/health")
                        self.assertEqual(request("GET", "/v1/models")["data"][0]["id"], alias)
                        break
                    except (OSError, TimeoutError):
                        if time.monotonic() > deadline:
                            raise
                        time.sleep(.1)
                prompts = {"a": [3 + i % 4 for i in range(self.prompt_size)],
                           "b": [3 + i % 4 for i in range(self.common_prefix_size)] +
                                [7 + i % 4 for i in range(self.prompt_size - self.common_prefix_size)]}
                delimiters = []
                if message_spans:
                    marker = "<|im_start|>user\n"
                    marker_tokens = request("POST", "/tokenize", {"content": marker, "add_special": False})["tokens"]
                    self.assertTrue(marker_tokens)
                    for prompt in prompts.values():
                        for pos in self.marker_positions:
                            prompt[pos:pos + len(marker_tokens)] = marker_tokens
                    delimiters = [{"role": "user", "delimiter": marker}]
                def complete(name, cached):
                    body = request("POST", "/completion", {"prompt": prompts[name], "id_slot": 0,
                        "cache_prompt": cached, "n_predict": 4, "temperature": 0, "top_k": 1,
                        "seed": 1234, "ignore_eos": True, "return_tokens": True, "message_delimiters": delimiters})
                    self.assertEqual(body["tokens_predicted"], 4)
                    self.assertEqual(len(body["tokens"]), 4)
                    self.assertFalse(body["truncated"])
                    self.assertIsNone(process.poll(), "server must survive the completion")
                    results["requests"].append({"prompt": name, "cache_prompt": cached,
                        "tokens": body["tokens"], "timings": body["timings"]})
                    return body["tokens"]
                control = {name: complete(name, False) for name in prompts}
                # Alternate a changed suffix and identical prompts; a valid prefix trim must preserve greedy tokens.
                for name in self.reuse_pattern:
                    with self.subTest(checkpoints=checkpoints, prompt=name, request=len(results["requests"])):
                        tokens = complete(name, True)
                        if tokens != control[name]:
                            results["fresh_reuse_mismatches"].append(len(results["requests"]) - 1)
                        self.assertEqual(tokens, control[name])
                results["passed"] = not results["fresh_reuse_mismatches"]
            finally:
                results["server_alive_before_cleanup"] = process.poll() is None
                results["cleanup_requested"] = process.poll() is None
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                results["exit_code_after_cleanup"] = process.returncode
                stem.with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    def test_reuse_without_checkpoints(self):
        self.check_prompt_reuse(0)

    def test_reuse_with_checkpoints(self):
        self.check_prompt_reuse(4)

    def test_reuse_with_message_spans(self):
        self.check_prompt_reuse(4, message_spans=True)


if __name__ == "__main__":
    unittest.main()
