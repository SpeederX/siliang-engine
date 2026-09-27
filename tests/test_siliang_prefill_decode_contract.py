from __future__ import annotations

from pathlib import Path
import unittest


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


if __name__ == "__main__":
    unittest.main()
