"""Coding-agent harness whose brain is the Claude Agent SDK, one session per game.

Each game's first turn calls `query(...)`; the `ResultMessage.session_id` is kept and passed back as
`ClaudeAgentOptions(resume=...)` on every later turn, so the SDK resumes the same session and Claude
carries its own running context for the whole game rather than starting fresh each turn. The reply
is taken from `ResultMessage.result` (falling back to the assistant `TextBlock`s).

The session runs with no tools (`allowed_tools=[]`, `max_turns=1`) and `setting_sources=[]` so it is
a clean game brain — it does NOT load this repo's CLAUDE.md, skills, or settings.

Requires `claude-agent-sdk` (the optional `harness-agents` extra: `uv sync --extra harness-agents`)
and Claude auth (Claude subscription via the `claude` CLI, or `ANTHROPIC_API_KEY`). Override the
model with `ARENA_CLAUDE_AGENT_MODEL` (unset -> the SDK's default model).

    arena-agent play --run <run> examples/claude_agent_sdk_agent.py
"""
from __future__ import annotations

import asyncio

from examples._coding_agent_util import SessionCodingHarness


class ClaudeAgentSdkHarness(SessionCodingHarness):
    brain_name = "claude-agent-sdk"
    model_env = "ARENA_CLAUDE_AGENT_MODEL"

    def _call(self, prompt: str, cwd: str, session_id: str | None) -> tuple[str, str | None]:
        # Import lazily so the harness module stays importable without the optional SDK installed.
        from claude_agent_sdk import (
            AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, query,
        )

        async def _go() -> tuple[str, str | None]:
            kwargs = dict(max_turns=1, allowed_tools=[], setting_sources=[], cwd=cwd)
            if session_id is not None:
                kwargs["resume"] = session_id
            if self.model:
                kwargs["model"] = self.model
            options = ClaudeAgentOptions(**kwargs)

            texts: list[str] = []
            result: str | None = None
            new_session = session_id
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            texts.append(block.text)
                elif isinstance(message, ResultMessage):
                    result = message.result
                    new_session = message.session_id or new_session
            return ((result or "\n".join(texts)).strip(), new_session)

        return asyncio.run(_go())


_default = ClaudeAgentSdkHarness()
on_event = _default.on_event
act = _default.act
