"""Coding-agent harness whose brain is opencode (`opencode run`), one session per game by default.

Set ARENA_AGENT_RESET_BETWEEN_GAMES=0 to keep one opencode session for the whole run instead.

Each game's first turn runs `opencode run --format json`, whose event stream carries a `sessionID`;
we keep it and pass `--session <id>` on every later turn, so opencode resumes the same session and
carries its own running context for the whole game rather than starting fresh each turn. The
assistant's reply is reassembled from the `type:"text"` events (`.part.text`).

Requires the `opencode` CLI on PATH and a configured provider/auth (`opencode auth login`). Override
the model with `ARENA_OPENCODE_MODEL` in `provider/model` form (unset -> opencode's default model).
Set `ARENA_OPENCODE_REASONING_EFFORT` to pass opencode's documented `--variant` value; supported
values are model/provider-specific, so unset leaves opencode's own default unchanged.

    arena-agent play --run <run> examples/opencode_agent.py
"""
from __future__ import annotations

import os

from examples._coding_agent_util import SessionCodingHarness, collect_jsonl_text, run_cli


class OpencodeHarness(SessionCodingHarness):
    brain_name = "opencode"
    model_env = "ARENA_OPENCODE_MODEL"
    reasoning_effort_env = "ARENA_OPENCODE_REASONING_EFFORT"

    def __init__(self, model: str | None = None, workdir: str | None = None):
        super().__init__(model=model, workdir=workdir)
        self.reasoning_effort = self._explicit_reasoning_effort()

    def _explicit_reasoning_effort(self) -> str | None:
        raw = os.environ.get(self.reasoning_effort_env, "").strip().lower()
        return raw if raw in self.reasoning_efforts else None

    def _call(self, prompt: str, cwd: str, session_id: str | None) -> tuple[str, str | None]:
        cmd = ["opencode", "run", "--format", "json", "--dir", cwd]
        if session_id is not None:
            cmd += ["--session", session_id]
        if self.model:
            cmd += ["-m", self.model]
        if self.reasoning_effort:
            cmd += ["--variant", self.reasoning_effort]
        cmd += [prompt]  # message is a positional arg

        stdout = run_cli(cmd, cwd=cwd)
        text, found_session = collect_jsonl_text(
            stdout, type_key="text", text_path=("part", "text"), session_key="sessionID",
        )
        # On a resume the same id comes back; on a clean error stream none does, so keep the old one.
        return (text or stdout, found_session or session_id)


_default = OpencodeHarness()
on_event = _default.on_event
act = _default.act
