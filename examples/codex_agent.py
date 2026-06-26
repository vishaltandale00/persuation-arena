"""Coding-agent harness whose brain is the Codex CLI (`codex exec`), one session per game by default.

Set ARENA_AGENT_RESET_BETWEEN_GAMES=0 to keep one Codex session for the whole run instead.

Each game opens a Codex session with `codex exec --json`; the first event line carries a
`thread.started.thread_id`, which we keep and feed to `codex exec resume <id>` on every later turn,
so Codex carries its own running context for the whole game instead of starting fresh each turn. The
agent's final message is read from `--output-last-message` (robust across Codex's event-schema
versions); `--json` is parsed only to capture the thread id.

Requires the `codex` CLI on PATH and Codex auth (ChatGPT login or `OPENAI_API_KEY`). Override the
model with `ARENA_CODEX_MODEL` (unset -> Codex's configured default) and reasoning effort with
`ARENA_CODEX_REASONING_EFFORT` (fallback: `ARENA_AGENT_REASONING_EFFORT`, then
`ARENA_REASONING_EFFORT`, then `medium`).

    arena-agent play --run <run> examples/codex_agent.py
"""
from __future__ import annotations

import json
import os
import tempfile

from examples._coding_agent_util import SessionCodingHarness, run_cli


def _thread_id(stdout: str) -> str | None:
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "thread.started" and obj.get("thread_id"):
            return obj["thread_id"]
    return None


class CodexHarness(SessionCodingHarness):
    brain_name = "codex"
    model_env = "ARENA_CODEX_MODEL"
    reasoning_effort_env = "ARENA_CODEX_REASONING_EFFORT"
    reasoning_efforts = {"minimal", "low", "medium", "high", "xhigh"}

    def _call(self, prompt: str, cwd: str, session_id: str | None) -> tuple[str, str | None]:
        # A UNIQUE output file per call. Codex only writes it on success, so if a call fails (bad
        # flag, auth, timeout) the file stays empty and we surface the error instead of silently
        # replaying a previous turn's message from a stale, reused path. `resume` rejects -C/-s/
        # --color, so keep its flags minimal; it inherits the opening call's cwd and sandbox.
        fd, last_message = tempfile.mkstemp(prefix=".codex_out_", suffix=".txt", dir=cwd)
        os.close(fd)
        try:
            if session_id is None:
                cmd = ["codex", "exec", "--json", "--skip-git-repo-check",
                       "-s", "read-only", "-C", cwd, "-o", last_message]
            else:
                cmd = ["codex", "exec", "resume", session_id, "--json", "--skip-git-repo-check",
                       "-o", last_message]
            if self.model:
                cmd += ["-m", self.model]
            if self.reasoning_effort:
                cmd += ["-c", f'model_reasoning_effort="{self.reasoning_effort}"']
            cmd += ["-"]  # read the prompt from stdin (avoids ARG_MAX on long openings)

            stdout = run_cli(cmd, input_text=prompt, cwd=cwd)
            new_session = _thread_id(stdout) or session_id
            try:
                with open(last_message) as f:
                    text = f.read().strip()
            except OSError:
                text = ""
        finally:
            try:
                os.unlink(last_message)
            except OSError:
                pass
        # Empty file (e.g. a CLI error before any message) -> surface stdout so the caller sees the
        # sentinel/error text and repairs/falls back legally rather than replaying a stale message.
        return (text or stdout, new_session)


_default = CodexHarness()
on_event = _default.on_event
act = _default.act
