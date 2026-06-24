"""Shared plumbing for the *coding-agent* harnesses (codex, opencode, Claude Agent SDK).

These harnesses differ from the chat harnesses (`session_agent`, `file_memory_agent`) in their
brain: instead of a single OpenRouter chat call, each turn is decided by a real coding agent —
codex, opencode, or the Claude Agent SDK — driving a model inside its own persistent session.

Memory = the agent's own session, kept alive across the whole game (the `session_agent` model, not
the `file_memory_agent` fresh-read model). The mechanics:

  - These tools are request->response: you cannot push a message and get no reply, so we cannot
    forward each event the instant it arrives. Instead `on_event` ACCUMULATES rendered deltas into a
    per-game buffer.
  - `act` flushes that buffer as the turn's single message and sends it into the RESUMED session for
    that game. The first turn opens the session (with a system preamble + everything observed so
    far); every later turn resumes it and forwards only the new deltas. The agent therefore carries
    its own reasoning, prior turns, and any scratch files across the entire game — it is never handed
    a fresh, amnesiac session mid-game.
  - Each game also gets its own working directory containing a running `transcript.md`. The coding
    agent runs with that directory as its cwd, so it may read those files for reference — these are
    coding agents, after all — but the prompt itself only ever carries the deltas, never a full
    re-dump. (Codex runs read-only and the Claude SDK brain runs with no tools, so the file is
    optional reference context, not a guaranteed-writable scratchpad.)

A subclass implements one method, `_call(prompt, cwd, session_id) -> (text, new_session_id)`: open a
session when `session_id is None`, otherwise resume it, and return the agent's final text plus the
session id to reuse next turn. Everything else — buffering, prompt assembly, validation, one repair,
legal fallback — lives here so each harness file is just its tool's invocation.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile

from examples._harness_util import (
    SYSTEM, REPAIR_MESSAGE, action_request, fallback_action, interpret, render_event,
)

# Coding agents can think and call tools, so a turn may take a while. The arena has its own turn
# deadline; this only bounds how long the harness waits on the tool before falling back to a legal
# action, so a hung tool degrades to a pass rather than blocking the loop forever.
BRAIN_TIMEOUT = float(os.environ.get("ARENA_AGENT_BRAIN_TIMEOUT", "150"))

_WORKSPACE_NOTE = (
    "Your working directory contains transcript.md, a running log of everything you have observed "
    "(public chat plus your private night info). If your tools allow it you may read that file for "
    "reference, but you do not need to — your live session already remembers the game."
)


def opening_prompt(history_text: str, turn) -> str:
    """First message of a game's session: system framing + everything observed before this turn."""
    return (
        f"{SYSTEM}\n\n{_WORKSPACE_NOTE}\n\n"
        f"The game so far (most recent last):\n{history_text or '(no events yet)'}\n\n"
        f"{action_request(turn)}"
    )


def delta_prompt(new_events_text: str, turn) -> str:
    """Every later turn: only what is new since this seat last acted (the session holds the rest)."""
    new = new_events_text.strip()
    head = (f"New events since your last turn (most recent last):\n{new}\n\n"
            if new else "No new public events since your last turn.\n\n")
    return f"{head}{action_request(turn)}"


def run_cli(cmd: list[str], *, input_text: str | None = None, cwd: str | None = None) -> str:
    """Run a coding-agent CLI non-interactively and return its stdout. Errors and timeouts come back
    as a sentinel string so the caller validates -> repairs -> falls back rather than raising."""
    try:
        proc = subprocess.run(
            cmd, input=input_text, cwd=cwd, capture_output=True, text=True, timeout=BRAIN_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return "<brain error: timeout>"
    except FileNotFoundError:
        return f"<brain error: {cmd[0]} not found on PATH>"
    except OSError as e:  # pragma: no cover - environment dependent
        return f"<brain error: {type(e).__name__}: {e}>"
    if proc.returncode != 0 and not (proc.stdout or "").strip():
        tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
        return f"<brain error: exit {proc.returncode}: {tail[0]}>"
    return proc.stdout or ""


def collect_jsonl_text(stdout: str, *, type_key: str, text_path: tuple[str, ...],
                       session_key: str | None = None) -> tuple[str, str | None]:
    """Pull assistant text (and optionally a session id) out of a JSONL event stream. Used by the
    opencode brain; tolerant of non-JSON lines so stray log output never crashes parsing."""
    texts: list[str] = []
    session_id: str | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if session_key and obj.get(session_key) and session_id is None:
            session_id = obj[session_key]
        if obj.get("type") == type_key:
            node = obj
            for key in text_path:
                node = node.get(key, {}) if isinstance(node, dict) else {}
            if isinstance(node, str):
                texts.append(node)
    return "\n".join(texts), session_id


class SessionCodingHarness:
    """Base for a coding-agent harness that keeps one persistent session per game.

    Subclass and implement `_call`. Set `brain_name` (used for the workspace prefix) and `model_env`
    (the env var that overrides the model; unset -> the tool's own default model)."""

    brain_name = "coding-agent"
    model_env: str | None = None

    def __init__(self, model: str | None = None, workdir: str | None = None):
        env_model = os.environ.get(self.model_env) if self.model_env else None
        self.model = model or env_model or None
        self.root = workdir or tempfile.mkdtemp(prefix=f"arena-{self.brain_name}-")
        self._pending: dict[str, list[str]] = {}      # game_instance_id -> rendered events not yet sent
        self._sessions: dict[str, str] = {}           # game_instance_id -> tool-native session id
        self._dirs: dict[str, str] = {}               # game_instance_id -> working directory

    # -- workspace -----------------------------------------------------------------------------
    def _game_dir(self, gid: str) -> str:
        path = self._dirs.get(gid)
        if path is None:
            path = os.path.join(self.root, gid)
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "transcript.md"), "w") as f:
                f.write(f"# Game {gid} — running log of what I have observed\n\n")
            self._dirs[gid] = path
        return path

    # -- SDK handlers --------------------------------------------------------------------------
    def on_event(self, event) -> None:
        gid = getattr(event, "game_instance_id", None)
        if gid is None:
            return  # run-level event, not part of any game's memory
        line = render_event(event)
        self._pending.setdefault(gid, []).append(line)  # the buffer is the real memory; must not be lost
        # transcript.md is only scratch/reference, so a write failure (disk full, permissions) must
        # never propagate out of the SDK's poll loop and abort the seat — best-effort only.
        try:
            with open(os.path.join(self._game_dir(gid), "transcript.md"), "a") as f:
                f.write(f"- {line}\n")
        except OSError:
            pass

    def act(self, turn) -> dict:
        gid = turn.game_instance_id
        cwd = self._game_dir(gid)
        new_events = "\n".join(self._pending.pop(gid, []))
        if gid in self._sessions:
            prompt = delta_prompt(new_events, turn)
        else:
            prompt = opening_prompt(new_events, turn)

        raw = self._run(prompt, cwd, gid)
        action, reasoning, legal = interpret(turn, raw)
        if not legal:
            # One repair attempt. If the first call established a session, resume it with a bare
            # nudge — it already holds the full game context. If it did NOT (the opening call failed,
            # so no session id was captured), a bare nudge would open a fresh, context-free session
            # that cannot produce a legal action; re-send the whole prompt instead, so that fresh
            # session is properly seeded (and later delta-resumes still have the context).
            repair = REPAIR_MESSAGE if gid in self._sessions else f"{prompt}\n\n{REPAIR_MESSAGE}"
            raw2 = self._run(repair, cwd, gid)
            action2, reasoning2, legal2 = interpret(turn, raw2)
            if legal2:
                action, reasoning = action2, reasoning2
            else:
                action = fallback_action(turn)
                reasoning = reasoning or "(fallback: coding agent produced no legal action)"
        return {"action": action, "reasoning": reasoning}

    # -- brain ---------------------------------------------------------------------------------
    def _run(self, prompt: str, cwd: str, gid: str) -> str:
        session_id = self._sessions.get(gid)
        try:
            text, new_session_id = self._call(prompt, cwd, session_id)
        except Exception as e:  # any tool/SDK failure degrades to a legal fallback, never a forfeit
            return f"<brain error: {type(e).__name__}: {e}>"
        if new_session_id:
            self._sessions[gid] = new_session_id
        return text or ""

    def _call(self, prompt: str, cwd: str, session_id: str | None) -> tuple[str, str | None]:
        """Open (session_id is None) or resume a session; return (final_text, session_id_to_reuse)."""
        raise NotImplementedError
