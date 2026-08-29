#!/usr/bin/env python3
"""Natural DS4 expert-cache policy campaign runner.

The runner is intentionally serial. Each replica is a fresh llama-cli process,
and every checkpoint in one replica comes from the same continuously evolving
cache. No frozen routes, fixed seed, temperature override, top-k override, or
other deterministic sampling override is added here.

Campaign shape:
  * screen64:    all declared cells, 3 fresh replicas, 64 requested tokens
  * qualify512:  an explicit shortlist, 3 fresh replicas, 512 requested tokens
  * curve2048:   an explicit shortlist, 3 fresh replicas, 2048 requested tokens
  * top1024:     an explicit shortlist, 1 fresh replica, 1024 requested tokens

The full screen contains 23 cells:
  L2 = LRU / LFU / W-TinyLFU / SLFU
  L1 = LFU / W-TinyLFU / SLFU{cold on/off x demote on/off}
  minus W-TinyLFU L1 x W-TinyLFU L2, which was already structurally rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_EXE = ROOT / ".agent-local/build-hybrid/bin/Release/llama-cli.exe"
DEFAULT_MODEL = pathlib.Path(
    r"C:/behemoth/0731/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731-siliang-expert-major.gguf"
)
DEFAULT_OUT = ROOT / ".agent-local/natural-policy-full-v2"
DEFAULT_PROMPT = (
    "Write a detailed technical discussion of cache locality, memory hierarchy, "
    "and data movement in local mixture-of-experts inference. Continue for several "
    "paragraphs and include concrete examples."
)

PHASE_DEFAULTS = {
    "screen64": (64, 3),
    "qualify512": (512, 3),
    "curve2048": (2048, 3),
    "top1024": (1024, 1),
}

CHECKPOINT_RE = re.compile(
    r"route_stats checkpoint generated_tokens=(?P<tokens>\d+) routes=(?P<routes>\d+) "
    r"selections=(?P<selections>\d+) L1=(?P<l1>\d+)\((?P<l1_pct>[\d.]+)%\) "
    r"L2=(?P<l2>\d+)\((?P<l2_pct>[\d.]+)%\) uncached=(?P<cold>\d+)\((?P<cold_pct>[\d.]+)%\) "
    r"K_hit=(?P<khit>\d+) K_admit=(?P<kadmit>\d+) R=(?P<r>\d+) CPU=(?P<cpu>\d+) "
    r"L1_evict=(?P<l1_evict>\d+) L2_evict=(?P<l2_evict>\d+) L2_reject=(?P<l2_reject>\d+) "
    r"demotions=(?P<demotions>\d+)"
    r"(?: D_reuse_L2=(?P<d_reuse_l2>\d+) D_reuse_cold=(?P<d_reuse_cold>\d+) "
    r"D_pending=(?P<d_pending>\d+))? unknown=(?P<unknown>\d+)"
)
FINAL_RE = re.compile(
    r"route_stats routes=(?P<routes>\d+) exact_routes=(?P<exact>\d+) unknown_routes=(?P<unknown_routes>\d+) "
    r"selections=(?P<selections>\d+) residency_L1=(?P<l1>\d+)\((?P<l1_pct>[\d.]+)%\) "
    r"residency_L2=(?P<l2>\d+)\((?P<l2_pct>[\d.]+)%\) "
    r"residency_uncached=(?P<cold>\d+)\((?P<cold_pct>[\d.]+)%\)"
)
SLFU_RE = re.compile(r"siliang_moe_runtime: route_stats slfu .*")
DEMOTION_ROUNDS_RE = re.compile(r"siliang_moe_runtime: route_stats demotion_reuse_rounds .*")
THROUGHPUT_RE = re.compile(r"Generation:\s*([\d.]+)\s*t/s")


@dataclass(frozen=True)
class Cell:
    id: str
    l2_policy: str
    l1_policy: str
    admit_k_cold: str | None = None
    demote_k_hot: str | None = None

    def cli_args(self) -> list[str]:
        args = [
            "--expert-cache-l2-policy", self.l2_policy,
            "--expert-cache-l1-policy", self.l1_policy,
        ]
        if self.l1_policy == "slfu":
            assert self.admit_k_cold in {"on", "off"}
            assert self.demote_k_hot in {"on", "off"}
            args += [
                "--admit-k-cold", self.admit_k_cold,
                "--demote-k-hot", self.demote_k_hot,
            ]
        return args


def make_cells() -> list[Cell]:
    l2s = [
        ("lru", "lru"),
        ("lfu", "lfu"),
        ("wtiny", "wtinylfu-w10-slru-p80"),
        ("slfu", "slfu"),
    ]
    cells: list[Cell] = []
    for l2_slug, l2_policy in l2s:
        cells.append(Cell(f"l2-{l2_slug}__l1-lfu", l2_policy, "lfu"))
        if l2_slug != "wtiny":
            cells.append(Cell(f"l2-{l2_slug}__l1-wtiny", l2_policy, "wtinylfu-w10-slru-p80"))
        for cold in ("on", "off"):
            for demote in ("off", "on"):
                cells.append(Cell(
                    f"l2-{l2_slug}__l1-slfu__cold-{cold}__demote-{demote}",
                    l2_policy,
                    "slfu",
                    cold,
                    demote,
                ))
    assert len(cells) == 23, len(cells)
    assert "l2-wtiny__l1-wtiny" not in {cell.id for cell in cells}
    return cells


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def working_tree_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    )
    return bool(result.stdout.strip())


def parse_run(err_text: str, out_text: str) -> dict[str, Any]:
    checkpoints: dict[str, dict[str, Any]] = {}
    for match in CHECKPOINT_RE.finditer(err_text):
        values: dict[str, Any] = match.groupdict()
        for key, value in list(values.items()):
            if value is None:
                continue
            if key.endswith("_pct"):
                values[key] = float(value)
            else:
                values[key] = int(value)
        checkpoints[str(values["tokens"])] = values

    final_matches = list(FINAL_RE.finditer(err_text))
    final: dict[str, Any] | None = None
    if final_matches:
        values = final_matches[-1].groupdict()
        final = {
            key: (float(value) if key.endswith("_pct") else int(value))
            for key, value in values.items()
        }

    slfu = SLFU_RE.findall(err_text)
    rounds = DEMOTION_ROUNDS_RE.findall(err_text)
    throughput = THROUGHPUT_RE.findall(out_text)
    return {
        "checkpoints": checkpoints,
        "final": final,
        "slfu_line": slfu[-1] if slfu else None,
        "demotion_rounds_line": rounds[-1] if rounds else None,
        "generation_tps": float(throughput[-1]) if throughput else None,
    }


def complete_receipt(meta_path: pathlib.Path, err_path: pathlib.Path) -> bool:
    if not meta_path.exists() or not err_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("returncode") != 0:
            return False
        parsed = meta.get("parsed") or {}
        return parsed.get("final") is not None
    except Exception:
        return False


def base_command(exe: pathlib.Path, model: pathlib.Path, tokens: int, prompt: str) -> list[str]:
    return [
        str(exe),
        "-m", str(model),
        "-c", "4096",
        "-b", "512",
        "-ub", "512",
        "-t", "2",
        "-tb", "2",
        "-ngl", "99",
        "-ncmoe", "43",
        "-nkvo",
        "--no-op-offload",
        "--expert-cache",
        "--expert-cache-l2-mib", "8192",
        "--expert-cache-l1-k", "216",
        "--expert-cache-exchange-r", "12",
        "--expert-cache-elevator-p", "12",
        "--expert-cache-roll", "deepseek4",
        "--no-expert-cache-prefill",
        "--no-expert-cache-memory-report",
        "--expert-cache-route-stats",
        "--offline",
        "--single-turn",
        "--no-conversation",
        "-p", prompt,
        "-n", str(tokens),
        "--ignore-eos",
    ]


def write_manifest(
    phase_dir: pathlib.Path,
    phase: str,
    cells: list[Cell],
    exe: pathlib.Path,
    model: pathlib.Path,
    tokens: int,
    reps: int,
    prompt: str,
) -> None:
    phase_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "siliang-natural-policy-matrix-v2",
        "phase": phase,
        "git_head": git_value("rev-parse", "HEAD"),
        "git_describe": git_value("describe", "--always", "--dirty"),
        "binary": str(exe),
        "binary_sha256": sha256(exe) if exe.exists() else None,
        "model": str(model),
        "tokens": tokens,
        "replicas": reps,
        "serial": True,
        "sampling_overrides": [],
        "frozen_route": False,
        "prompt": prompt,
        "fixed_geometry": {
            "context": 4096,
            "batch": 512,
            "ubatch": 512,
            "threads": 2,
            "batch_threads": 2,
            "l2_mib": 8192,
            "K": 216,
            "R": 12,
            "P": 12,
            "front": "deepseek4",
            "routed_prefill_arena": False,
        },
        "excluded": [
            {
                "id": "l2-wtiny__l1-wtiny",
                "reason": "prior natural screening showed admission churn / cold collapse; explicitly excluded by campaign decision",
            }
        ],
        "cells": [asdict(cell) for cell in cells],
    }
    (phase_dir / "campaign-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def refresh_summary(phase_dir: pathlib.Path, cells: list[Cell], reps: int) -> None:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        cell_dir = phase_dir / cell.id
        for rep in range(1, reps + 1):
            meta_path = cell_dir / f"rep{rep}.json"
            if not meta_path.exists():
                continue
            try:
                row = json.loads(meta_path.read_text(encoding="utf-8"))
                rows.append(row)
            except Exception:
                pass
    (phase_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASE_DEFAULTS, default="screen64")
    parser.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
    parser.add_argument("--exe", type=pathlib.Path, default=DEFAULT_EXE)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--reps", type=int)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--cells",
        help="comma-separated cell ids. Required for qualify512/curve2048/top1024; screen64 defaults to all 23 cells.",
    )
    parser.add_argument("--force", action="store_true", help="rerun completed receipts")
    parser.add_argument("--allow-dirty", action="store_true", help="allow a dirty git worktree")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cooldown", type=float, default=2.0)
    args = parser.parse_args()

    default_tokens, default_reps = PHASE_DEFAULTS[args.phase]
    tokens = args.tokens or default_tokens
    reps = args.reps or default_reps
    if tokens <= 0 or reps <= 0:
        parser.error("tokens and reps must be positive")

    exe = args.exe if args.exe.is_absolute() else ROOT / args.exe
    model = args.model
    if not exe.exists():
        parser.error(f"llama-cli not found: {exe}")
    if not model.exists():
        parser.error(f"model not found: {model}")

    all_cells = make_cells()
    by_id = {cell.id: cell for cell in all_cells}
    if args.cells:
        requested = [item.strip() for item in args.cells.split(",") if item.strip()]
        unknown = [item for item in requested if item not in by_id]
        if unknown:
            parser.error(f"unknown cell ids: {', '.join(unknown)}")
        cells = [by_id[item] for item in requested]
    else:
        if args.phase != "screen64":
            parser.error("--cells is required for qualify512/curve2048/top1024")
        cells = all_cells

    if working_tree_dirty() and not args.allow_dirty:
        parser.error("git worktree is dirty; commit/checkpoint first or pass --allow-dirty intentionally")

    phase_dir = args.output / args.phase
    write_manifest(phase_dir, args.phase, cells, exe, model, tokens, reps, args.prompt)

    print(f"phase={args.phase} tokens={tokens} reps={reps} cells={len(cells)} total_runs={len(cells) * reps}")
    for cell in cells:
        print(cell.id)
    if args.dry_run:
        return 0

    progress_path = phase_dir / "progress.log"
    base = base_command(exe, model, tokens, args.prompt)

    for cell in cells:
        cell_dir = phase_dir / cell.id
        cell_dir.mkdir(parents=True, exist_ok=True)
        for rep in range(1, reps + 1):
            out_path = cell_dir / f"rep{rep}.out"
            err_path = cell_dir / f"rep{rep}.err"
            meta_path = cell_dir / f"rep{rep}.json"
            if not args.force and complete_receipt(meta_path, err_path):
                message = f"SKIP {cell.id} rep={rep} complete"
                print(message, flush=True)
                with progress_path.open("a", encoding="utf-8") as progress:
                    progress.write(message + "\n")
                continue

            command = base + cell.cli_args()
            start = time.perf_counter()
            start_epoch = time.time()
            message = f"START {cell.id} rep={rep}"
            print(message, flush=True)
            with progress_path.open("a", encoding="utf-8") as progress:
                progress.write(message + "\n")

            with out_path.open("wb") as stdout, err_path.open("wb") as stderr:
                process = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
            wall = time.perf_counter() - start
            err_text = err_path.read_text(encoding="utf-8", errors="replace")
            out_text = out_path.read_text(encoding="utf-8", errors="replace")
            parsed = parse_run(err_text, out_text)
            meta = {
                "schema": "siliang-natural-policy-run-v2",
                "phase": args.phase,
                "cell": asdict(cell),
                "replica": rep,
                "tokens_requested": tokens,
                "returncode": process.returncode,
                "wall_seconds": round(wall, 3),
                "started_epoch": start_epoch,
                "command": command,
                "parsed": parsed,
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            refresh_summary(phase_dir, cells, reps)

            done = f"DONE {cell.id} rep={rep} rc={process.returncode} wall_s={wall:.2f}"
            print(done, flush=True)
            with progress_path.open("a", encoding="utf-8") as progress:
                progress.write(done + "\n")
                for token in sorted(parsed["checkpoints"], key=int):
                    cp = parsed["checkpoints"][token]
                    progress.write(
                        f"  cp={token} L1={cp['l1_pct']:.2f}% L2={cp['l2_pct']:.2f}% "
                        f"cold={cp['cold_pct']:.2f}% L2_reject={cp['l2_reject']} "
                        f"demotions={cp['demotions']} D_reuse_L2={cp.get('d_reuse_l2')} "
                        f"D_reuse_cold={cp.get('d_reuse_cold')} D_pending={cp.get('d_pending')}\n"
                    )
                if parsed["slfu_line"]:
                    progress.write("  " + parsed["slfu_line"] + "\n")
                if parsed["demotion_rounds_line"]:
                    progress.write("  " + parsed["demotion_rounds_line"] + "\n")

            if process.returncode != 0:
                print(f"FAIL {cell.id} rep={rep}; stopping serial campaign", file=sys.stderr)
                return process.returncode or 1
            if parsed["final"] is None:
                print(f"FAIL {cell.id} rep={rep}; no final route_stats receipt", file=sys.stderr)
                return 2
            if args.cooldown > 0:
                time.sleep(args.cooldown)

    refresh_summary(phase_dir, cells, reps)
    with progress_path.open("a", encoding="utf-8") as progress:
        progress.write("ALL_DONE\n")
    print("ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
