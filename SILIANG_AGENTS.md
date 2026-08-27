# Siliang Engine instructions for coding agents

These rules supplement the upstream [`AGENTS.md`](AGENTS.md) and apply to the
complete Siliang Engine fork. A more specific `AGENTS.md` inside a subtree may
add local rules, but it does not weaken either root policy.

## Ground truth

Use this authority order when sources disagree:

1. Live files, current Git state, and reproducible raw measurements.
2. Machine-readable manifests and retained command output.
3. Tests and current implementation comments.
4. Narrative documentation.
5. Agent memory or an earlier handoff.

Do not resolve a disagreement by silently choosing the most convenient source.
Record it, verify the live state, and update stale material when the task
authorizes that scope.

Before every behavior or performance change, capture at least:

- HEAD and branch identity;
- `git status` and the relevant diff;
- the exact command and environment variables;
- build type, backend, model identity, and effective runtime configuration;
- the intended single variable under test.

Keep unrelated user changes intact. Do not rewrite, reset, clean, or otherwise
discard work that is outside the assigned change.

## Behavior and compatibility

- Never silently add or alter flags, defaults, clamps, heuristics, fallbacks,
  environment variables, or GGUF metadata semantics.
- A fallback must be explicit in logs, represented in tests, and disclosed in
  results. It must not be counted as the requested path.
- v0.1.3 uses the typed `--expert-cache...` CLI and context configuration. Do
  not restore expert-cache environment aliases or hidden process-state setup.
  Preserve `siliangem.*` GGUF keys unless an explicit metadata migration is
  part of the task.
- Do not rename public interfaces or add legacy aliases without explicit
  authorization.
- Keep the expert-major mmap refusal. Loading its strided tensors without mmap
  is a correctness hazard.
- Treat documented deferred defects as deferred. Do not repair or mask them
  without explicit authorization, focused tests, and updated docs.
- When behavior changes, update implementation tests, README, validation
  guidance, issue records, and provenance artifacts in the same change.

Changes to the Siliang engine delta must keep these synchronized:

- the engine-delta paths listed in `docs/PROVENANCE.md`;
- `patches/siliang-engine.patch`;
- `docs/source-manifest.json`;
- `docs/PROVENANCE.md`;
- relevant automated and runtime tests.

## Correctness before performance

Do not time a configuration until a deterministic correctness gate passes.
The gate must:

- use an explicit prompt, seed, sampling configuration, context, and token
  count;
- assert that generation is nonempty and completed for the intended reason;
- compare an arena-disabled control with the intended arena path;
- record output or token evidence, not merely a healthy process or HTTP status;
- prove from logs and counters that the intended source armed and served work.

A load success, server-ready event, or plausible output is insufficient on its
own. If a fallback occurs, mark the cell void and investigate it separately.

## Performance evidence

Benchmark only on an idle machine. Downloads, builds, virus scans, other model
servers, storage-heavy jobs, and unrelated GPU workloads invalidate a run when
they can contend for the measured resource.

For every A/B claim:

- change one declared variable;
- use the same binary where possible;
- use at least three independent process starts per arm;
- interleave the arm order with a schedule chosen before measurement;
- retain every valid raw repetition;
- report the median and the observed min-to-max range;
- report units, model hash, storage device class, command, environment, and
  effective arena/path state;
- inspect available RAM, file cache, commit/pagefile pressure, dedicated VRAM,
  and shared VRAM before attributing a mechanism;
- never combine runs from different storage devices into one comparative
  throughput claim.

Any run that falls back, fails to produce nonempty output, changes an
uncontrolled setting, or overlaps a competing workload is void. Keep the raw
record and the reason; do not quietly delete it from the sample set.

Preserve retractions. If later evidence invalidates a claim, mark the old claim
as retracted, state why, and link the replacement evidence. Do not edit history
so the earlier conclusion appears never to have existed.

## Long-running jobs and STOP

Launch long builds, conversions, and benchmarks in a way that survives the
initiating shell when unattended execution is intended. Each job must have:

- a unique output directory;
- stdout and stderr logs;
- an explicit success or failure done marker written only after cleanup;
- enough recorded command and environment state to reproduce it.

Never infer completion from a partial JSON file, a quiet log, or the absence of
new output. Verify the done marker and process exit status.

A status question does not cancel active work. An explicit `STOP` does: stop
launching new work, terminate only the processes owned by the task, restore any
temporarily swapped artifacts, verify that the machine is idle, and report
which outputs are incomplete.

## Public repository hygiene

- Never publish secrets, tokens, credentials, machine-bound absolute paths,
  usernames, session URLs, private repository names, private document names,
  or raw conversational transcripts.
- Do not import historical research ledgers or machine-specific benchmark
  harnesses. Distill only the active, reproducible contract needed here.
- Use placeholders in documented commands and sanitize retained logs.
- Models, generated binaries, build trees, caches, temporary outputs, raw
  runtime logs, and local handoffs stay outside Git.
- Public claims must stand on evidence included in, or reproducible from, this
  repository. Private context is never a public citation.

Before committing, run the repository hygiene and provenance checks and inspect
the staged diff. Scope whitespace checking to the public-authored files plus the
engine-delta paths listed in `docs/PROVENANCE.md`; the pinned
upstream tree contains preexisting whitespace that is not part of this patch.
Do not normalize or reformat unrelated upstream files to make a whole-tree
check clean. Do not add a remote, push, publish, or rewrite history unless the
user explicitly requests that action.
