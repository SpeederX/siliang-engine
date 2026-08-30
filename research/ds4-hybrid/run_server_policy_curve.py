#!/usr/bin/env python3
"""Fresh-process llama-server policy curve campaign.

Each replica:
  1. starts a fresh llama-server process;
  2. waits for /health;
  3. sends exactly one natural /completion request with streaming SSE;
  4. timestamps emitted tokens and records cumulative/window decode tok/s at
     32, 64, 128, 256, 512, 1024, 1512, and 2048 generated tokens;
  5. gracefully stops the server with CTRL_BREAK_EVENT on Windows;
  6. joins runtime route telemetry from stderr into a JSON receipt.

No seed, temperature, top-k, top-p, or frozen-route override is supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from typing import Any

import requests

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_SERVER = ROOT / ".agent-local/build-hybrid/bin/Release/llama-server.exe"
DEFAULT_MODEL = pathlib.Path(
    r"C:/behemoth/0731/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731-siliang-expert-major.gguf"
)
DEFAULT_OUTPUT = ROOT / ".agent-local/server-policy-curve-v1"
DEFAULT_PROMPT = (
    "Write a detailed technical discussion of cache locality, memory hierarchy, "
    "and data movement in local mixture-of-experts inference. Continue for several "
    "paragraphs and include concrete examples."
)
CHECKPOINTS = (32, 64, 128, 256, 512, 1024, 1512, 2048)

CHECKPOINT_RE = re.compile(
    r"route_stats checkpoint generated_tokens=(?P<tokens>\d+) routes=(?P<routes>\d+) "
    r"selections=(?P<selections>\d+) L1=(?P<l1>\d+)\((?P<l1_pct>[\d.]+)%\) "
    r"L2=(?P<l2>\d+)\((?P<l2_pct>[\d.]+)%\) uncached=(?P<cold>\d+)\((?P<cold_pct>[\d.]+)%\) "
    r"K_hit=(?P<khit>\d+) K_admit=(?P<kadmit>\d+) R=(?P<r>\d+) CPU=(?P<cpu>\d+) "
    r"L1_evict=(?P<l1_evict>\d+) L2_evict=(?P<l2_evict>\d+) L2_reject=(?P<l2_reject>\d+) "
    r"demotions=(?P<demotions>\d+) D_reuse_L2=(?P<d_reuse_l2>\d+) "
    r"D_reuse_cold=(?P<d_reuse_cold>\d+) D_pending=(?P<d_pending>\d+) "
    r"D_exposed_ms=(?P<d_exposed_ms>[\d.]+) unknown=(?P<unknown>\d+)"
)
SLFU_RE = re.compile(r"siliang_moe_runtime: route_stats slfu .*")
ROUNDS_RE = re.compile(r"siliang_moe_runtime: route_stats demotion_reuse_rounds .*")
L2_FINAL_RE = re.compile(
    r"siliang_moe_runtime: route_stats l2 admissions=(?P<admissions>\d+) "
    r"evictions=(?P<evictions>\d+) rejections=(?P<rejections>\d+)"
)
SILIANGEM_RE = re.compile(
    r"siliangem\[(?P<tag>[^]]+)\]: (?P<lookups>\d+) lookups, (?P<hits>\d+) hits "
    r"\((?P<hit_pct>[\d.]+)%\), (?P<misses>\d+) misses, expert-request hit "
    r"(?P<expert_hit_pct>[\d.]+)% \((?P<expert_hits>\d+)/(?P<expert_requests>\d+)\)"
)


@dataclass(frozen=True)
class Arm:
    id: str
    l2_mib: int
    l2_policy: str
    k: int
    r: int
    p: int
    l1_policy: str | None
    admit_k_cold: str | None
    demote_k_hot: str | None
    roll: str
    reps: int
    route_stats: bool = True
    threads: int = 2

    def cache_args(self) -> list[str]:
        args = [
            "--expert-cache",
            "--expert-cache-l2-mib", str(self.l2_mib),
            "--expert-cache-l2-policy", self.l2_policy,
        ]
        if self.k > 0:
            args += [
                "--expert-cache-l1-k", str(self.k),
                "--expert-cache-exchange-r", str(self.r),
                "--expert-cache-elevator-p", str(self.p),
                "--expert-cache-l1-policy", str(self.l1_policy),
                "--admit-k-cold", str(self.admit_k_cold),
                "--demote-k-hot", str(self.demote_k_hot),
                "--expert-cache-roll", self.roll,
                "--no-expert-cache-prefill",
            ]
            if self.route_stats:
                args.append("--expert-cache-route-stats")
        else:
            # L2-only baseline. K/R/P/roll controls are intentionally absent.
            args += ["--no-expert-cache-memory-report"]
        if self.k > 0:
            args.append("--no-expert-cache-memory-report")
        return args


def main_arms() -> list[Arm]:
    def k8(id_: str, l2: str, cold: str, demote: str) -> Arm:
        return Arm(id_, 8192, l2, 216, 12, 12, "slfu", cold, demote, "deepseek4", 3)

    return [
        k8("l2-lru__l1-slfu__cold-off__demote-on", "lru", "off", "on"),
        k8("l2-lru__l1-slfu__cold-on__demote-on", "lru", "on", "on"),
        k8("l2-lru__l1-slfu__cold-off__demote-off", "lru", "off", "off"),
        k8("l2-lru__l1-slfu__cold-on__demote-off", "lru", "on", "off"),
        k8("l2-slfu__l1-slfu__cold-off__demote-on", "slfu", "off", "on"),
        k8("l2-slfu__l1-slfu__cold-on__demote-off", "slfu", "on", "off"),
        k8("l2-slfu__l1-slfu__cold-off__demote-off", "slfu", "off", "off"),
        k8("l2-slfu__l1-slfu__cold-on__demote-on", "slfu", "on", "on"),
        k8("l2-wtiny__l1-slfu__cold-on__demote-off", "wtinylfu-w10-slru-p80", "on", "off"),
        k8("l2-wtiny__l1-slfu__cold-on__demote-on", "wtinylfu-w10-slru-p80", "on", "on"),
        k8("l2-lfu__l1-slfu__cold-on__demote-off", "lfu", "on", "off"),
    ]


def extra_18g_arms() -> list[Arm]:
    # User-requested capacity controls. The L2-only arm has no K admission
    # control; the two K216 arms intentionally use cold admission OFF.
    return [
        Arm("l2-18g-lru-only__t12", 18432, "lru", 0, 0, 0, None, None, None, "off", 1, False, 12),
        Arm("l2-18g-lru__l1-slfu__k216__roll-off__cold-off__demote-off",
            18432, "lru", 216, 12, 12, "slfu", "off", "off", "off", 1),
        Arm("l2-18g-lru__l1-slfu__k216__roll-off__cold-off__demote-on",
            18432, "lru", 216, 12, 12, "slfu", "off", "on", "off", 1),
    ]


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex((host, port)) != 0


def wait_ready(proc: subprocess.Popen[Any], url: str, timeout_s: float = 180.0) -> None:
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during startup rc={proc.returncode}")
        try:
            r = requests.get(url + "/health", timeout=1.0)
            last = (r.status_code, r.text[:200])
            if r.status_code == 200:
                return
        except Exception as exc:
            last = repr(exc)
        time.sleep(0.25)
    raise TimeoutError(f"server did not become ready: {last}")


def graceful_stop(proc: subprocess.Popen[Any]) -> str:
    if proc.poll() is not None:
        return "server-exited"
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10.0)
        return "graceful"
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=5.0)
        except Exception:
            pass
        return "forced"


def parse_runtime(err_text: str) -> dict[str, Any]:
    checkpoints: dict[str, dict[str, Any]] = {}
    for m in CHECKPOINT_RE.finditer(err_text):
        values: dict[str, Any] = m.groupdict()
        for k, v in list(values.items()):
            values[k] = float(v) if k.endswith("_pct") or k.endswith("_ms") else int(v)
        checkpoints[str(values["tokens"])] = values
    slfu = SLFU_RE.findall(err_text)
    rounds = ROUNDS_RE.findall(err_text)
    l2 = list(L2_FINAL_RE.finditer(err_text))
    l2_final = None
    if l2:
        l2_final = {k: int(v) for k, v in l2[-1].groupdict().items()}
    sm = list(SILIANGEM_RE.finditer(err_text))
    siliangem_final = None
    if sm:
        vals = sm[-1].groupdict()
        siliangem_final = {
            k: (float(v) if k.endswith("_pct") else (v if k == "tag" else int(v)))
            for k, v in vals.items()
        }
    return {
        "checkpoints": checkpoints,
        "slfu_line": slfu[-1] if slfu else None,
        "demotion_rounds_line": rounds[-1] if rounds else None,
        "l2_final": l2_final,
        "siliangem_final": siliangem_final,
    }


def stream_completion(url: str, prompt: str, tokens: int) -> dict[str, Any]:
    body = {
        "prompt": prompt,
        "n_predict": tokens,
        "stream": True,
        "return_tokens": True,
        "cache_prompt": False,
        "ignore_eos": True,
    }
    token_times: list[float] = []
    final_event: dict[str, Any] | None = None
    request_start = time.perf_counter()
    with requests.post(url + "/completion", json=body, stream=True, timeout=(10, 7200)) as response:
        response.raise_for_status()
        for raw in response.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            payload = raw[5:].strip()
            if not payload:
                continue
            event = json.loads(payload)
            now = time.perf_counter()
            if event.get("stop"):
                final_event = event
                continue
            ids = event.get("tokens") or []
            if not isinstance(ids, list):
                ids = []
            for _ in ids:
                token_times.append(now)
    request_end = time.perf_counter()
    if len(token_times) != tokens:
        raise RuntimeError(f"stream emitted {len(token_times)} tokens, expected {tokens}")
    if final_event is None:
        raise RuntimeError("stream ended without final stop event")

    speed: dict[str, dict[str, Any]] = {}
    previous_n = 1
    previous_t = token_times[0]
    first_t = token_times[0]
    for n in CHECKPOINTS:
        if n > len(token_times):
            continue
        t = token_times[n - 1]
        cumulative_dt = t - first_t
        cumulative_tps = (n - 1) / cumulative_dt if n > 1 and cumulative_dt > 0 else None
        window_dt = t - previous_t
        window_tokens = n - previous_n
        window_tps = window_tokens / window_dt if window_tokens > 0 and window_dt > 0 else None
        speed[str(n)] = {
            "emitted_tokens": n,
            "elapsed_from_first_token_s": round(cumulative_dt, 6),
            "cumulative_tps": round(cumulative_tps, 6) if cumulative_tps else None,
            "window_from": previous_n,
            "window_tokens": window_tokens,
            "window_elapsed_s": round(window_dt, 6),
            "window_tps": round(window_tps, 6) if window_tps else None,
        }
        previous_n = n
        previous_t = t

    timings = final_event.get("timings") or {}
    return {
        "speed_checkpoints": speed,
        "final_event": final_event,
        "server_predicted_tps": timings.get("predicted_per_second"),
        "server_prompt_tps": timings.get("prompt_per_second"),
        "server_predicted_n": timings.get("predicted_n"),
        "request_wall_s": round(request_end - request_start, 6),
    }


def receipt_complete(path: pathlib.Path, expected_tokens: int) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("success") is True and data.get("stream", {}).get("server_predicted_n") == expected_tokens
    except Exception:
        return False


def base_server_command(server: pathlib.Path, model: pathlib.Path, host: str, port: int, threads: int) -> list[str]:
    return [
        str(server),
        "-m", str(model),
        "-c", "4096",
        "-b", "512",
        "-ub", "512",
        "-t", str(threads),
        "-tb", str(threads),
        "-ngl", "99",
        "-ncmoe", "43",
        "-nkvo",
        "--no-op-offload",
        "--parallel", "1",
        "--host", host,
        "--port", str(port),
        "--no-webui",
        "--offline",
    ]


def run_replica(
    arm: Arm,
    rep: int,
    server: pathlib.Path,
    model: pathlib.Path,
    out_dir: pathlib.Path,
    prompt: str,
    host: str,
    port: int,
    tokens: int,
) -> dict[str, Any]:
    arm_dir = out_dir / arm.id
    arm_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = arm_dir / f"rep{rep}.server.out"
    stderr_path = arm_dir / f"rep{rep}.server.err"
    receipt_path = arm_dir / f"rep{rep}.json"
    command = base_server_command(server, model, host, port, arm.threads) + arm.cache_args()
    record: dict[str, Any] = {
        "schema": "siliang-server-policy-curve-v1",
        "arm": asdict(arm),
        "replica": rep,
        "tokens_requested": tokens,
        "command": command,
        "started_epoch": time.time(),
        "success": False,
    }
    proc: subprocess.Popen[Any] | None = None
    termination = None
    start = time.perf_counter()
    try:
        if not port_available(host, port):
            raise RuntimeError(f"port {port} is already in use")
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            proc = subprocess.Popen(
                command, cwd=ROOT, stdout=stdout, stderr=stderr, creationflags=creationflags)
            record["server_pid"] = proc.pid
            wait_ready(proc, f"http://{host}:{port}")
            record["ready_wall_s"] = round(time.perf_counter() - start, 3)
            stream = stream_completion(f"http://{host}:{port}", prompt, tokens)
            record["stream"] = stream
            termination = graceful_stop(proc)
            proc = None
        err_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        runtime = parse_runtime(err_text)
        record["runtime"] = runtime
        # K-enabled arms must have the exact runtime checkpoints requested.
        if arm.k > 0:
            expected_checkpoints = [cp for cp in CHECKPOINTS if cp <= tokens]
            missing = [str(cp) for cp in expected_checkpoints if str(cp) not in runtime["checkpoints"]]
            if missing:
                raise RuntimeError(f"missing runtime checkpoints: {missing}")
        record["success"] = True
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if proc is not None:
            termination = graceful_stop(proc)
        record["termination"] = termination
        record["wall_seconds"] = round(time.perf_counter() - start, 3)
        receipt_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def write_manifest(out_dir: pathlib.Path, arms: list[Arm], server: pathlib.Path, model: pathlib.Path, tokens: int, prompt: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "siliang-server-policy-curve-v1",
        "git_head": git_value("rev-parse", "HEAD"),
        "git_describe": git_value("describe", "--always", "--dirty"),
        "server": str(server),
        "server_sha256": sha256(server),
        "model": str(model),
        "tokens": tokens,
        "checkpoints": list(CHECKPOINTS),
        "prompt": prompt,
        "sampling_overrides": [],
        "cache_prompt": False,
        "fresh_server_per_replica": True,
        "arms": [asdict(a) for a in arms],
        "total_runs": sum(a.reps for a in arms),
    }
    (out_dir / "campaign-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=pathlib.Path, default=DEFAULT_SERVER)
    parser.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18180)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include", help="comma-separated arm ids")
    parser.add_argument("--reps", type=int, help="override replicas per selected arm (smoke/debug only)")
    parser.add_argument("--cooldown", type=float, default=3.0)
    args = parser.parse_args()

    server = args.server if args.server.is_absolute() else ROOT / args.server
    model = args.model
    if not server.exists():
        parser.error(f"llama-server not found: {server}")
    if not model.exists():
        parser.error(f"model not found: {model}")
    if args.tokens <= 0:
        parser.error("tokens must be positive")

    arms = main_arms() + extra_18g_arms()
    by_id = {a.id: a for a in arms}
    if args.include:
        ids = [x.strip() for x in args.include.split(",") if x.strip()]
        unknown = [x for x in ids if x not in by_id]
        if unknown:
            parser.error(f"unknown arm ids: {unknown}")
        arms = [by_id[x] for x in ids]
    if args.reps is not None:
        if args.reps <= 0:
            parser.error("--reps must be positive")
        arms = [replace(a, reps=args.reps) for a in arms]

    out_dir = args.output if args.output.is_absolute() else ROOT / args.output
    write_manifest(out_dir, arms, server, model, args.tokens, args.prompt)
    print(f"server curve: arms={len(arms)} total_runs={sum(a.reps for a in arms)} tokens={args.tokens}")
    for arm in arms:
        print(f"{arm.id} reps={arm.reps} L2={arm.l2_mib}MiB K={arm.k} roll={arm.roll}")
    if args.dry_run:
        return 0

    progress = out_dir / "progress.log"
    failures = 0
    for arm in arms:
        for rep in range(1, arm.reps + 1):
            receipt = out_dir / arm.id / f"rep{rep}.json"
            if not args.force and receipt_complete(receipt, args.tokens):
                msg = f"SKIP {arm.id} rep={rep} complete"
                print(msg, flush=True)
                with progress.open("a", encoding="utf-8") as f:
                    f.write(msg + "\n")
                continue
            msg = f"START {arm.id} rep={rep}"
            print(msg, flush=True)
            with progress.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
            result = run_replica(arm, rep, server, model, out_dir, args.prompt, args.host, args.port, args.tokens)
            speed = result.get("stream", {}).get("speed_checkpoints", {})
            end = speed.get("2048", {})
            done = (
                f"DONE {arm.id} rep={rep} success={result.get('success')} "
                f"server_tps={result.get('stream', {}).get('server_predicted_tps')} "
                f"cp2048_tps={end.get('cumulative_tps')} wall_s={result.get('wall_seconds')}"
            )
            print(done, flush=True)
            with progress.open("a", encoding="utf-8") as f:
                f.write(done + "\n")
                for cp in CHECKPOINTS:
                    s = speed.get(str(cp))
                    r = result.get("runtime", {}).get("checkpoints", {}).get(str(cp))
                    if s:
                        f.write(
                            f"  cp={cp} cum_tps={s.get('cumulative_tps')} win_tps={s.get('window_tps')}"
                            + (f" L1={r.get('l1_pct')} L2={r.get('l2_pct')} cold={r.get('cold_pct')} D_exposed_ms={r.get('d_exposed_ms')}" if r else "")
                            + "\n"
                        )
            if not result.get("success"):
                failures += 1
            if args.cooldown > 0:
                time.sleep(args.cooldown)
    with progress.open("a", encoding="utf-8") as f:
        f.write(f"ALL_DONE failures={failures}\n")
    print(f"ALL_DONE failures={failures}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
