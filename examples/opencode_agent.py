"""Coding-agent harness whose brain is opencode (`opencode run`), one session per game.

Each game's first turn runs `opencode run --format json`, whose event stream carries a `sessionID`;
we keep it and pass `--session <id>` on every later turn, so opencode resumes the same session and
carries its own running context for the whole game rather than starting fresh each turn. The
assistant's reply is reassembled from the `type:"text"` events (`.part.text`).

Requires the `opencode` CLI on PATH and a configured provider/auth (`opencode auth login`). Override
the model with `ARENA_OPENCODE_MODEL` in `provider/model` form (unset -> opencode's default model).

    arena-agent play --run <run> examples/opencode_agent.py
"""
from __future__ import annotations

from examples._coding_agent_util import SessionCodingHarness, collect_jsonl_text, run_cli


class OpencodeHarness(SessionCodingHarness):
    brain_name = "opencode"
    model_env = "ARENA_OPENCODE_MODEL"

    def _call(self, prompt: str, cwd: str, session_id: str | None) -> tuple[str, str | None]:
        cmd = ["opencode", "run", "--format", "json", "--dir", cwd]
        if session_id is not None:
            cmd += ["--session", session_id]
        if self.model:
            cmd += ["-m", self.model]
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
