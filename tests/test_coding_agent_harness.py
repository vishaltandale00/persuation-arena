"""The coding-agent harnesses (codex / opencode / Claude Agent SDK) keep ONE session per game and
feed it only the new deltas each turn. These tests pin that contract with a fake brain — no CLIs,
no SDK, no network — plus the small JSONL/thread-id parsers the real brains depend on.
"""
from __future__ import annotations

import json
import sys
import types

from examples._coding_agent_util import (
    SessionCodingHarness, collect_jsonl_text, delta_prompt, opening_prompt,
)
from examples._harness_util import fallback_action
from persuasion_arena_agent.models import Event, Turn

GID = "game_1"


def _event(etype: str, payload: dict, gid: str | None = GID, eid: str = "e") -> Event:
    return Event.from_dict({"event_id": eid, "type": etype, "payload": payload,
                            "game_instance_id": gid})


def _speak_turn() -> Turn:
    return Turn.from_dict({
        "turn_id": "t1", "game_instance_id": GID, "game": "onuw", "seat": 0,
        "phase": "discussion", "action_kind": "onuw.discussion.speak_or_pass",
        "deadline_at": "2026-06-23T17:00:00Z",
        "observation": {"format": "text", "text": "speak"},
        "legal_action": {"schema": {}, "choices": {"players": [{"seat": 1}, {"seat": 2}]}},
    })


class FakeBrain(SessionCodingHarness):
    """Records every (prompt, session_id) it is called with and returns scripted replies."""
    brain_name = "fake"

    def __init__(self, scripted, **kw):
        super().__init__(**kw)
        self.scripted = list(scripted)
        self.calls: list[tuple[str, str | None]] = []

    def _call(self, prompt, cwd, session_id):
        self.calls.append((prompt, session_id))
        text = self.scripted.pop(0) if self.scripted else '{"action":{"pass":true},"declared_reasoning":""}'
        return text, "sess-xyz"


def test_first_turn_opens_session_later_turns_resume_with_only_deltas(tmp_path):
    h = FakeBrain([
        json.dumps({"action": {"speak": "hello"}, "declared_reasoning": "open"}),
        json.dumps({"action": {"speak": "again"}, "declared_reasoning": "delta"}),
    ], workdir=str(tmp_path))

    h.on_event(_event("game_setup", {"n": 5}))
    h.on_event(_event("role_info", {"seat": 0, "role": "Seer"}))
    out1 = h.act(_speak_turn())

    assert out1 == {"action": {"speak": "hello"}, "declared_reasoning": "open"}
    first_prompt, first_session = h.calls[0]
    assert first_session is None                      # opened a fresh session
    assert "The game so far" in first_prompt          # opening prompt shape
    assert "Seer" in first_prompt                     # everything-so-far is included on turn 1

    h.on_event(_event("speech", {"actor_seat": 2, "text": "I am the Robber"}))
    out2 = h.act(_speak_turn())

    assert out2 == {"action": {"speak": "again"}, "declared_reasoning": "delta"}
    second_prompt, second_session = h.calls[1]
    assert second_session == "sess-xyz"               # resumed the SAME session, not a fresh one
    assert "New events since your last turn" in second_prompt
    assert "I am the Robber" in second_prompt         # the new delta is forwarded
    assert "Seer" not in second_prompt                # prior context is NOT re-dumped (session holds it)
    assert h._pending.get(GID, []) == []              # buffer drained after flushing


def test_illegal_first_reply_is_repaired_in_the_same_session(tmp_path):
    h = FakeBrain([
        "not json at all",
        json.dumps({"action": {"speak": "fixed"}, "declared_reasoning": "ok"}),
    ], workdir=str(tmp_path))
    out = h.act(_speak_turn())
    assert out["action"] == {"speak": "fixed"}
    assert len(h.calls) == 2
    repair_prompt, repair_session = h.calls[1]
    assert repair_session == "sess-xyz"               # the repair resumes the same session
    assert "LEGAL action" in repair_prompt


def test_failed_open_repairs_with_full_context_not_a_bare_nudge(tmp_path):
    """If the OPENING call fails (no session captured), the repair must re-seed a fresh session with
    the full prompt — a bare 'that was illegal' nudge into an empty session can't produce a legal
    action for the right turn."""
    class FailOpenBrain(SessionCodingHarness):
        brain_name = "failopen"

        def __init__(self, **kw):
            super().__init__(**kw)
            self.calls: list[tuple[str, str | None]] = []

        def _call(self, prompt, cwd, session_id):
            self.calls.append((prompt, session_id))
            if len(self.calls) == 1:
                return "<brain error: timeout>", None          # open fails, no session id
            return json.dumps({"action": {"speak": "recovered"}, "declared_reasoning": "ok"}), "sess-late"

    h = FailOpenBrain(workdir=str(tmp_path))
    h.on_event(_event("role_info", {"seat": 0, "role": "Robber"}))
    out = h.act(_speak_turn())

    assert out["action"] == {"speak": "recovered"}
    assert len(h.calls) == 2
    _, first_session = h.calls[0]
    repair_prompt, repair_session = h.calls[1]
    assert first_session is None and repair_session is None    # open never established a session
    assert "YOUR TURN" in repair_prompt                        # full action request is re-sent...
    assert "Robber" in repair_prompt                           # ...with the observed game context...
    assert "LEGAL action" in repair_prompt                     # ...plus the repair nudge appended


def test_two_bad_replies_fall_back_to_a_legal_action(tmp_path):
    turn = _speak_turn()
    h = FakeBrain(["garbage", "still garbage"], workdir=str(tmp_path))
    out = h.act(turn)
    assert out["action"] == fallback_action(turn)     # guaranteed-legal, never a forfeit


def test_brain_exception_degrades_to_fallback_without_raising(tmp_path):
    class BoomBrain(SessionCodingHarness):
        brain_name = "boom"

        def _call(self, prompt, cwd, session_id):
            raise RuntimeError("tool exploded")

    turn = _speak_turn()
    out = BoomBrain(workdir=str(tmp_path)).act(turn)
    assert out["action"] == fallback_action(turn)


def test_transcript_file_accumulates_events(tmp_path):
    from examples._harness_util import state_key, state_path_name
    h = FakeBrain([], workdir=str(tmp_path))
    ev = _event("speech", {"actor_seat": 1, "text": "vote me"})
    h.on_event(ev)
    # The workspace dir is keyed by the (collision-resistant) state-path name, not the raw gid.
    transcript = tmp_path / state_path_name(state_key(ev, h.reset_between_games)) / "transcript.md"
    assert transcript.exists()
    assert "vote me" in transcript.read_text()


def test_run_level_events_without_game_id_are_ignored(tmp_path):
    h = FakeBrain([], workdir=str(tmp_path))
    h.on_event(_event("run_announcement", {"text": "starting"}, gid=None))
    assert h._pending == {}


def test_collect_jsonl_text_reads_opencode_shape():
    stream = "\n".join([
        json.dumps({"type": "step_start", "sessionID": "ses_abc", "part": {}}),
        json.dumps({"type": "text", "sessionID": "ses_abc",
                    "part": {"type": "text", "text": '{"action": {"pass": true}}'}}),
        "not-json-noise",
        json.dumps({"type": "step_finish", "part": {}}),
    ])
    text, session = collect_jsonl_text(stream, type_key="text", text_path=("part", "text"),
                                       session_key="sessionID")
    assert session == "ses_abc"
    assert text == '{"action": {"pass": true}}'


def test_codex_thread_id_parser():
    from examples.codex_agent import _thread_id
    stream = "\n".join([
        "Reading additional input from stdin...",
        json.dumps({"type": "thread.started", "thread_id": "uuid-123"}),
        json.dumps({"type": "item.completed", "item": {"type": "assistant", "text": "OK"}}),
    ])
    assert _thread_id(stream) == "uuid-123"
    assert _thread_id("no json here") is None


def test_prompt_builders_shape():
    turn = _speak_turn()
    assert "(no events yet)" in opening_prompt("", turn)
    assert "No new public events" in delta_prompt("", turn)
    assert "fresh delta" in delta_prompt("fresh delta", turn)


def test_model_resolves_from_env_then_constructor(tmp_path, monkeypatch):
    from examples.codex_agent import CodexHarness
    monkeypatch.delenv("ARENA_CODEX_MODEL", raising=False)
    assert CodexHarness(workdir=str(tmp_path)).model is None          # unset -> tool default
    monkeypatch.setenv("ARENA_CODEX_MODEL", "gpt-x")
    assert CodexHarness(workdir=str(tmp_path)).model == "gpt-x"       # env override
    assert CodexHarness(model="explicit", workdir=str(tmp_path)).model == "explicit"  # arg wins


def test_reasoning_effort_resolves_from_provider_env_then_shared_env(tmp_path, monkeypatch):
    from examples.codex_agent import CodexHarness
    from examples.claude_agent_sdk_agent import ClaudeAgentSdkHarness

    monkeypatch.delenv("ARENA_CODEX_REASONING_EFFORT", raising=False)
    monkeypatch.setenv("ARENA_AGENT_REASONING_EFFORT", "high")
    assert CodexHarness(workdir=str(tmp_path / "codex")).reasoning_effort == "high"
    monkeypatch.setenv("ARENA_CODEX_REASONING_EFFORT", "xhigh")
    assert CodexHarness(workdir=str(tmp_path / "codex2")).reasoning_effort == "xhigh"

    monkeypatch.setenv("ARENA_CLAUDE_AGENT_REASONING_EFFORT", "minimal")
    assert ClaudeAgentSdkHarness(workdir=str(tmp_path / "claude")).reasoning_effort == "medium"
    monkeypatch.setenv("ARENA_CLAUDE_AGENT_REASONING_EFFORT", "max")
    assert ClaudeAgentSdkHarness(workdir=str(tmp_path / "claude2")).reasoning_effort == "max"


def test_codex_argv_opens_with_sandbox_resumes_without_and_injects_model(tmp_path, monkeypatch):
    import examples.codex_agent as cx
    calls: list[list[str]] = []
    monkeypatch.setattr(cx, "run_cli", lambda cmd, input_text=None, cwd=None: (
        calls.append(cmd) or '{"action": {"pass": true}, "declared_reasoning": "r"}'))

    h = cx.CodexHarness(model="gpt-x", workdir=str(tmp_path))
    h._call("p", str(tmp_path), None)            # opening
    h._call("p", str(tmp_path), "sess-1")        # resume
    open_cmd, resume_cmd = calls

    for cmd in (open_cmd, resume_cmd):
        assert "--json" in cmd and "-o" in cmd
        assert "-m" in cmd and cmd[cmd.index("-m") + 1] == "gpt-x"   # contiguous flag pair
        assert "-c" in cmd
        assert cmd[cmd.index("-c") + 1] == 'model_reasoning_effort="medium"'
    assert open_cmd[:2] == ["codex", "exec"] and "resume" not in open_cmd
    assert "-C" in open_cmd and "-s" in open_cmd                     # opening sets cwd + sandbox
    assert resume_cmd[:3] == ["codex", "exec", "resume"]
    assert resume_cmd[3] == "sess-1"
    assert "-C" not in resume_cmd and "-s" not in resume_cmd         # resume rejects these flags


def test_claude_agent_sdk_options_include_reasoning_effort(tmp_path, monkeypatch):
    import examples.claude_agent_sdk_agent as cl

    captured = {}

    class TextBlock:
        def __init__(self, text):
            self.text = text

    class AssistantMessage:
        def __init__(self):
            self.content = [TextBlock('{"action":{"pass":true},"declared_reasoning":"r"}')]

    class ResultMessage:
        def __init__(self):
            self.result = '{"action":{"pass":true},"declared_reasoning":"r"}'
            self.session_id = "claude-session"

    class ClaudeAgentOptions:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    async def query(prompt, options):
        yield AssistantMessage()
        yield ResultMessage()

    monkeypatch.setitem(sys.modules, "claude_agent_sdk", types.SimpleNamespace(
        AssistantMessage=AssistantMessage,
        ClaudeAgentOptions=ClaudeAgentOptions,
        ResultMessage=ResultMessage,
        TextBlock=TextBlock,
        query=query,
    ))
    monkeypatch.setenv("ARENA_CLAUDE_AGENT_REASONING_EFFORT", "high")

    text, session = cl.ClaudeAgentSdkHarness(model="claude-x", workdir=str(tmp_path))._call(
        "p", str(tmp_path), None,
    )

    assert text == '{"action":{"pass":true},"declared_reasoning":"r"}'
    assert session == "claude-session"
    assert captured["model"] == "claude-x"
    assert captured["effort"] == "high"
    assert captured["allowed_tools"] == []
    assert captured["setting_sources"] == []


def test_claude_agent_sdk_effort_falls_back_to_extra_args(tmp_path, monkeypatch):
    import examples.claude_agent_sdk_agent as cl

    captured = {}

    class TextBlock:
        def __init__(self, text):
            self.text = text

    class AssistantMessage:
        content = [TextBlock('{"action":{"pass":true}}')]

    class ResultMessage:
        result = '{"action":{"pass":true}}'
        session_id = "claude-session"

    class ClaudeAgentOptions:
        def __init__(self, **kwargs):
            if "effort" in kwargs:
                raise TypeError("unknown effort")
            captured.update(kwargs)

    async def query(prompt, options):
        yield ResultMessage()

    monkeypatch.setitem(sys.modules, "claude_agent_sdk", types.SimpleNamespace(
        AssistantMessage=AssistantMessage,
        ClaudeAgentOptions=ClaudeAgentOptions,
        ResultMessage=ResultMessage,
        TextBlock=TextBlock,
        query=query,
    ))
    monkeypatch.setenv("ARENA_CLAUDE_AGENT_REASONING_EFFORT", "xhigh")

    cl.ClaudeAgentSdkHarness(workdir=str(tmp_path))._call("p", str(tmp_path), "prev")

    assert captured["resume"] == "prev"
    assert captured["extra_args"] == {"effort": "xhigh"}


def test_opencode_argv_adds_session_only_on_resume_and_injects_model(tmp_path, monkeypatch):
    import examples.opencode_agent as oc
    calls: list[list[str]] = []
    monkeypatch.setattr(oc, "run_cli", lambda cmd, input_text=None, cwd=None: (
        calls.append(cmd) or json.dumps(
            {"type": "text", "sessionID": "ses_1",
             "part": {"type": "text", "text": '{"action": {"pass": true}}'}})))
    monkeypatch.delenv("ARENA_OPENCODE_REASONING_EFFORT", raising=False)

    h = oc.OpencodeHarness(model="anthropic/x", workdir=str(tmp_path))
    h._call("p", str(tmp_path), None)
    h._call("p", str(tmp_path), "ses_1")
    open_cmd, resume_cmd = calls

    for cmd in (open_cmd, resume_cmd):
        assert cmd[:2] == ["opencode", "run"] and "--format" in cmd
        assert "-m" in cmd and cmd[cmd.index("-m") + 1] == "anthropic/x"
        assert "--variant" not in cmd
    assert "--session" not in open_cmd
    assert "--session" in resume_cmd and resume_cmd[resume_cmd.index("--session") + 1] == "ses_1"


def test_opencode_argv_injects_explicit_reasoning_variant(tmp_path, monkeypatch):
    import examples.opencode_agent as oc
    calls: list[list[str]] = []
    monkeypatch.setattr(oc, "run_cli", lambda cmd, input_text=None, cwd=None: (
        calls.append(cmd) or json.dumps(
            {"type": "text", "sessionID": "ses_1",
             "part": {"type": "text", "text": '{"action": {"pass": true}}'}})))
    monkeypatch.setenv("ARENA_OPENCODE_REASONING_EFFORT", "xhigh")

    oc.OpencodeHarness(model="openai/gpt-5", workdir=str(tmp_path))._call("p", str(tmp_path), None)

    cmd = calls[0]
    assert "--variant" in cmd
    assert cmd[cmd.index("--variant") + 1] == "xhigh"


def test_pi_argv_injects_model_session_and_explicit_thinking(tmp_path, monkeypatch):
    import examples.pi_agent as pi
    calls: list[list[str]] = []
    monkeypatch.setattr(pi, "run_cli", lambda cmd, input_text=None, cwd=None: (
        calls.append(cmd) or json.dumps({
            "type": "message_end",
            "message": {"role": "assistant", "content": [
                {"type": "text", "text": '{"action": {"pass": true}}'},
            ]},
        })))
    monkeypatch.setenv("ARENA_PI_REASONING_EFFORT", "none")

    h = pi.PiHarness(model="openrouter/openai/gpt-5", workdir=str(tmp_path))
    h._call("p", str(tmp_path), None)
    h._call("p", str(tmp_path), "pi-session")
    open_cmd, resume_cmd = calls

    for cmd in (open_cmd, resume_cmd):
        assert cmd[:3] == [pi.PI_BIN, "-p", "--mode"]
        assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "openrouter/openai/gpt-5"
        assert "--thinking" in cmd and cmd[cmd.index("--thinking") + 1] == "off"
    assert "--session" not in open_cmd
    assert "--session" in resume_cmd and resume_cmd[resume_cmd.index("--session") + 1] == "pi-session"


def test_pi_argv_omits_thinking_when_unset_or_invalid(tmp_path, monkeypatch):
    import examples.pi_agent as pi
    calls: list[list[str]] = []
    monkeypatch.setattr(pi, "run_cli", lambda cmd, input_text=None, cwd=None: (
        calls.append(cmd) or ""))

    monkeypatch.delenv("ARENA_PI_REASONING_EFFORT", raising=False)
    pi.PiHarness(workdir=str(tmp_path / "unset"))._call("p", str(tmp_path), None)
    monkeypatch.setenv("ARENA_PI_REASONING_EFFORT", "turbo")
    pi.PiHarness(workdir=str(tmp_path / "invalid"))._call("p", str(tmp_path), None)

    assert "--thinking" not in calls[0]
    assert "--thinking" not in calls[1]


def test_harness_modules_import_and_wire():
    import examples.codex_agent as cx
    import examples.opencode_agent as oc
    import examples.claude_agent_sdk_agent as cl
    for mod in (cx, oc, cl):
        assert callable(mod.act) and callable(mod.on_event)
