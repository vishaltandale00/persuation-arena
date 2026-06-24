"""Coding-agent harness whose brain is the `pi` coding agent (earendil-works/pi), one session per game by default.

Set ARENA_AGENT_RESET_BETWEEN_GAMES=0 to keep one pi session for the whole run instead.

pi is a minimal, model-agnostic terminal coding agent (Read/Write/Edit/Bash core) that speaks to 20+
providers. We drive it NON-INTERACTIVELY: `pi -p --mode json` emits a line-stream of events; the
`session` event carries the session UUID, and `message_end` events carry the assistant's text. We keep
the session id and pass `--session <id>` on every later turn so pi resumes the same conversation — the
same persistent-session-per-game contract as the codex/opencode harnesses. The SDK only delivers the
delta event stream; this harness owns its memory (pi's live session + a scratch transcript.md in cwd).

Requires the `pi` CLI: npm install -g --ignore-scripts @earendil-works/pi-coding-agent
The binary may live in a global bin not on PATH; set ARENA_PI_BIN to its absolute path (default: "pi").
Pick the model with ARENA_PI_MODEL (e.g. openrouter/openai/gpt-5.5, openrouter/z-ai/glm-5.2,
openrouter/google/gemini-3.5-flash); bring-your-own-key via the provider env (OPENROUTER_API_KEY) or
pi's own /login. Run:  arena-agent play --run <run> examples/pi_agent.py
"""
from __future__ import annotations

import json
import os

from examples._coding_agent_util import SessionCodingHarness, run_cli

PI_BIN = os.environ.get("ARENA_PI_BIN", "pi")


def _extract(stdout: str) -> tuple[str, str | None]:
    """Pull (final assistant text, session id) from pi's `--mode json` line stream. Tolerant of
    non-JSON lines so stray output never crashes parsing."""
    session_id: str | None = None
    texts: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if o.get("type") == "session" and o.get("id"):
            session_id = o["id"]
        if o.get("type") == "message_end":
            m = o.get("message") or {}
            if m.get("role") == "assistant":
                parts = [c.get("text", "") for c in (m.get("content") or [])
                         if isinstance(c, dict) and c.get("type") == "text"]
                text = "".join(parts).strip()
                if text:
                    texts.append(text)  # keep the last non-empty assistant message as the answer
    return (texts[-1] if texts else ""), session_id


class PiHarness(SessionCodingHarness):
    brain_name = "pi"
    model_env = "ARENA_PI_MODEL"

    def _call(self, prompt: str, cwd: str, session_id: str | None) -> tuple[str, str | None]:
        cmd = [PI_BIN, "-p", "--mode", "json"]
        if session_id is not None:
            cmd += ["--session", session_id]   # resume the same per-game session (partial UUID ok)
        if self.model:
            cmd += ["--model", self.model]
        # prompt via stdin: pi merges piped stdin into the print-mode prompt (avoids ARG_MAX on long openings)
        stdout = run_cli(cmd, input_text=prompt, cwd=cwd)
        text, found_session = _extract(stdout)
        return (text or stdout, found_session or session_id)


_default = PiHarness()
on_event = _default.on_event
act = _default.act
