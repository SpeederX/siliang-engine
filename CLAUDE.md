IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

For Siliang Engine work, also read and follow
[`SILIANG_AGENTS.md`](SILIANG_AGENTS.md).

## Siliang handoff before compaction

Before compacting context during an active Siliang task, write or refresh the
ignored local file `.agent-local/HANDOFF.md`. Do not compact first and
reconstruct it from memory afterward.

The handoff must contain:

- the active objective and exact authorized scope;
- current HEAD, branch, status, and a concise diff summary;
- files changed and user-owned changes that must be preserved;
- exact commands, environment variables, and results gathered so far;
- active process IDs and locations of logs and done markers;
- unresolved correctness or provenance questions;
- the next safe action and any explicit blocker.

The file is local working state, not a public artifact. Do not put secrets,
personal paths, private source names, session links, or transcript content in
it. On resume, compare the handoff with live files and Git state; live evidence
wins if they differ.

Do not commit `.agent-local/HANDOFF.md`. If the path is no longer ignored, stop
and repair the ignore rule before writing a handoff.
