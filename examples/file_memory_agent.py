"""The simplest possible stateful harness: memory is a markdown file on disk.

`on_event` appends each delta event as a bullet to a per-game `.md` file. `act` reads that file
back as its memory, asks the LLM to choose, and returns the action. That's it — the harness builds
its own memory by writing the event stream to a file and re-reading the file it wrote.

This is the "anyone can build a harness" reference: no clever state, just append-to-file and
read-the-file. (It is NOT the SDK or server giving back history — it's the harness persisting its
own memory from the deltas it was delivered once.)
"""
from __future__ import annotations

import os
import tempfile

from examples._harness_util import SYSTEM, DEFAULT_MODEL, render_event, action_request, decide


class FileMemoryAgent:
    def __init__(self, model: str = DEFAULT_MODEL, memory_dir: str | None = None):
        self.model = model
        self.dir = memory_dir or tempfile.mkdtemp(prefix="arena-mem-")
        self.files: dict[str, str] = {}   # game_instance_id -> .md path

    def _path(self, gid: str) -> str:
        p = self.files.get(gid)
        if p is None:
            p = os.path.join(self.dir, f"{gid}.md")
            self.files[gid] = p
            with open(p, "w") as f:
                f.write(f"# Game {gid} — what I have observed\n\n")
        return p

    def on_event(self, event) -> None:
        gid = getattr(event, "game_instance_id", None)
        if gid is None:
            return
        with open(self._path(gid), "a") as f:
            f.write(f"- {render_event(event)}\n")

    def act(self, turn) -> dict:
        with open(self._path(turn.game_instance_id)) as f:
            memory = f.read()
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Your notes on the game so far:\n\n{memory}\n\n{action_request(turn)}"},
        ]
        action, reasoning, _ = decide(self.model, messages, turn)
        return {"action": action, "reasoning": reasoning}


# Module-level handlers for `arena-agent play ... examples/file_memory_agent.py` (model via env).
_default = FileMemoryAgent(os.environ.get("ARENA_AGENT_MODEL", DEFAULT_MODEL))
on_event = _default.on_event
act = _default.act
