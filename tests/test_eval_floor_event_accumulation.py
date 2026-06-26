"""EVAL FLOOR — a stateful example harness MUST accumulate the events it is delivered into its own
memory BEFORE it acts, and run-level (no game_instance_id) events MUST NOT pollute per-game memory.

Why this is a floor (not a judgment): the SDK delivers each delta event exactly once and the server
never re-serves old ones, so a stateful harness that drops what it was handed plays blind. The two
reference stateful harnesses — examples/session_agent.py (memory = a live chat session) and
examples/file_memory_agent.py (memory = a per-game .md file) — MUST therefore (a) carry every event
they received for a game into the prompt the brain sees on the FIRST act() for that game, and
(b) keep run-level events (game_instance_id is None) out of every per-game memory.

Scoring is exact-match / schema-legality only: we mock each harness's module-bound `decide` brain to
CAPTURE the messages it is handed (so we can assert, by substring, that the role and each speech text
are present) and to return an action checked LEGAL by the REAL SDK oracle examples._harness_util._is_legal
for the turn's action_kind. No network, no API key, no model judgment. The real harness on_event/act
code paths run unchanged; only the brain seam is mocked.
"""
from __future__ import annotations

import pytest

from examples._harness_util import _is_legal, action_request, render_event
from persuasion_arena_agent.models import Event, Turn

GID = "game_acc_1"
OTHER_GID = "game_acc_2"

# The real legal_action for onuw.discussion.speak_or_pass, copied verbatim from
# arena/games/onuw.py (run_discussion). Embedding the authentic schema means the action the mock
# brain returns is asserted legal against the SAME contract the engine enforces.
SPEAK_LEGAL_ACTION = {
    "schema": {
        "oneOf": [
            {
                "type": "object",
                "required": ["speak", "urgency"],
                "properties": {
                    "speak": {"type": "string", "minLength": 1, "maxLength": 1000},
                    "urgency": {"type": "integer", "enum": [1, 2, 3]},
                },
                "additionalProperties": False,
            },
            {
                "type": "object",
                "required": ["pass"],
                "properties": {
                    "pass": {"enum": [True]},
                    "stance": {"enum": ["wait", "done"]},
                },
                "additionalProperties": False,
            },
        ]
    },
    "choices": {"pass": True, "stances": ["wait", "done"]},
}

# A legal speak action under BOTH the engine parser and the SDK oracle _is_legal.
LEGAL_SPEAK = {"speak": "I think Participant 3 is lying.", "urgency": 1}

ROLE = "Seer"
SPEECHES = [
    (1, "I am the Robber and I swapped with seat 4."),
    (2, "No, I was the Seer the whole night."),
    (3, "Lets just vote out Participant 2."),
]


def _event(etype: str, payload: dict, gid: str | None = GID, eid: str = "e") -> Event:
    return Event.from_dict(
        {"event_id": eid, "type": etype, "payload": payload, "game_instance_id": gid}
    )


def _speak_turn(gid: str = GID) -> Turn:
    return Turn.from_dict(
        {
            "turn_id": "t1",
            "game_instance_id": gid,
            "game": "onuw",
            "seat": 0,
            "phase": "discussion",
            "action_kind": "onuw.discussion.speak_or_pass",
            "deadline_at": "2026-06-25T17:00:00Z",
            "observation": {"format": "text", "text": "The table is deciding who speaks next."},
            "legal_action": SPEAK_LEGAL_ACTION,
        }
    )


class CaptureDecide:
    """Stand-in for examples._harness_util.decide, monkeypatched into a harness MODULE (patched where
    it is USED). Captures every `messages` list the harness assembles from its own memory, and returns
    a schema-legal action + an assistant message (matching decide's (action, reasoning, assistant)
    return shape so the real harness code keeps working)."""

    def __init__(self, action: dict):
        self.action = action
        self.calls: list[list[dict]] = []

    def __call__(self, model, messages, turn):
        # Deep-ish snapshot: the harness may keep mutating its live list after this call.
        self.calls.append([dict(m) for m in messages])
        return self.action, "(captured)", {"role": "assistant", "content": "ok"}

    def transcript(self, idx: int = 0) -> str:
        """All message content from one captured call, concatenated — the brain's full view."""
        return "\n".join(str(m.get("content", "")) for m in self.calls[idx])


# --- sanity: the rendered text we assert on actually contains role + speeches ------------------

def test_render_event_surfaces_role_and_speech_text_for_assertions():
    role_line = render_event(_event("role_info", {"seat": 0, "role": ROLE}))
    assert ROLE in role_line
    for seat, text in SPEECHES:
        speech_line = render_event(_event("speech", {"actor_seat": seat, "text": text}))
        assert text in speech_line


def test_embedded_speak_action_is_legal_under_the_real_oracle():
    # Guards against the test asserting on an action the engine would actually reject.
    assert _is_legal(_speak_turn(), LEGAL_SPEAK) is True


# --- session_agent: memory = the live chat session ---------------------------------------------

def test_session_agent_accumulates_events_into_the_prompt_before_acting(monkeypatch):
    import examples.session_agent as sa_mod
    from examples.session_agent import SessionAgent

    cap = CaptureDecide(LEGAL_SPEAK)
    monkeypatch.setattr(sa_mod, "decide", cap)  # patch where decide is USED

    agent = SessionAgent(model="scripted")
    agent.on_event(_event("role_info", {"seat": 0, "role": ROLE}))
    for seat, text in SPEECHES:
        agent.on_event(_event("speech", {"actor_seat": seat, "text": text}))

    out = agent.act(_speak_turn())

    # The brain was called exactly once, with the harness's accumulated memory.
    assert len(cap.calls) == 1
    seen = cap.transcript(0)
    assert ROLE in seen  # the role it was dealt is in memory before it acts
    for _seat, text in SPEECHES:
        assert text in seen  # every speech it received is in memory before it acts

    # The harness returned the brain's legal action verbatim, in the SDK envelope.
    assert out == {"action": LEGAL_SPEAK, "declared_reasoning": "(captured)"}
    assert _is_legal(_speak_turn(), out["action"]) is True


def test_session_agent_run_level_event_does_not_pollute_any_per_game_memory(monkeypatch):
    import examples.session_agent as sa_mod
    from examples.session_agent import SessionAgent

    cap = CaptureDecide(LEGAL_SPEAK)
    monkeypatch.setattr(sa_mod, "decide", cap)

    agent = SessionAgent(model="scripted")
    POISON = "RUN-LEVEL-ANNOUNCEMENT-SHOULD-NOT-LEAK"
    agent.on_event(_event("run_announcement", {"text": POISON}, gid=None))
    agent.on_event(_event("role_info", {"seat": 0, "role": ROLE}))

    # No per-game session was created for the run-level event.
    assert None not in agent.sessions
    assert list(agent.sessions.keys()) == [GID]

    agent.act(_speak_turn())
    assert POISON not in cap.transcript(0)
    assert ROLE in cap.transcript(0)


def test_session_agent_keeps_two_games_memory_separate(monkeypatch):
    import examples.session_agent as sa_mod
    from examples.session_agent import SessionAgent

    cap = CaptureDecide(LEGAL_SPEAK)
    monkeypatch.setattr(sa_mod, "decide", cap)

    agent = SessionAgent(model="scripted")
    agent.on_event(_event("speech", {"actor_seat": 1, "text": "GAME-ONE-SECRET"}, gid=GID))
    agent.on_event(_event("speech", {"actor_seat": 1, "text": "GAME-TWO-SECRET"}, gid=OTHER_GID))

    agent.act(_speak_turn(GID))
    only_game_one = cap.transcript(0)
    assert "GAME-ONE-SECRET" in only_game_one
    assert "GAME-TWO-SECRET" not in only_game_one  # the other game's memory never bleeds in


# --- file_memory_agent: memory = a per-game .md file it writes then re-reads --------------------

def test_file_memory_agent_persists_events_to_disk_and_rereads_them_on_act(monkeypatch, tmp_path):
    import examples.file_memory_agent as fm_mod
    from examples.file_memory_agent import FileMemoryAgent

    cap = CaptureDecide(LEGAL_SPEAK)
    monkeypatch.setattr(fm_mod, "decide", cap)

    agent = FileMemoryAgent(model="scripted", memory_dir=str(tmp_path))
    agent.on_event(_event("role_info", {"seat": 0, "role": ROLE}))
    for seat, text in SPEECHES:
        agent.on_event(_event("speech", {"actor_seat": seat, "text": text}))

    # The per-game memory FILE exists on disk and holds the accumulated events.
    mem_path = tmp_path / f"{GID}.md"
    assert mem_path.exists()
    on_disk = mem_path.read_text()
    assert ROLE in on_disk
    for _seat, text in SPEECHES:
        assert text in on_disk

    out = agent.act(_speak_turn())

    # act() RE-READ the file it wrote and handed that memory to the brain.
    assert len(cap.calls) == 1
    seen = cap.transcript(0)
    assert ROLE in seen
    for _seat, text in SPEECHES:
        assert text in seen
    # The prompt the brain saw is built from the file's contents (memory is embedded in the request).
    assert action_request(_speak_turn()) in seen

    assert out == {"action": LEGAL_SPEAK, "declared_reasoning": "(captured)"}
    assert _is_legal(_speak_turn(), out["action"]) is True


def test_file_memory_agent_run_level_event_writes_no_file_and_never_reaches_a_game(monkeypatch, tmp_path):
    import examples.file_memory_agent as fm_mod
    from examples.file_memory_agent import FileMemoryAgent

    cap = CaptureDecide(LEGAL_SPEAK)
    monkeypatch.setattr(fm_mod, "decide", cap)

    agent = FileMemoryAgent(model="scripted", memory_dir=str(tmp_path))
    POISON = "RUN-LEVEL-ANNOUNCEMENT-SHOULD-NOT-LEAK"
    agent.on_event(_event("run_announcement", {"text": POISON}, gid=None))
    agent.on_event(_event("role_info", {"seat": 0, "role": ROLE}))

    # Only the real game's file was created; no None-keyed / run-level file exists.
    assert agent.files == {GID: str(tmp_path / f"{GID}.md")}
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{GID}.md"]
    assert POISON not in (tmp_path / f"{GID}.md").read_text()

    agent.act(_speak_turn())
    assert POISON not in cap.transcript(0)
    assert ROLE in cap.transcript(0)


def test_file_memory_agent_keeps_two_games_in_separate_files(monkeypatch, tmp_path):
    import examples.file_memory_agent as fm_mod
    from examples.file_memory_agent import FileMemoryAgent

    cap = CaptureDecide(LEGAL_SPEAK)
    monkeypatch.setattr(fm_mod, "decide", cap)

    agent = FileMemoryAgent(model="scripted", memory_dir=str(tmp_path))
    agent.on_event(_event("speech", {"actor_seat": 1, "text": "GAME-ONE-SECRET"}, gid=GID))
    agent.on_event(_event("speech", {"actor_seat": 1, "text": "GAME-TWO-SECRET"}, gid=OTHER_GID))

    assert "GAME-TWO-SECRET" not in (tmp_path / f"{GID}.md").read_text()
    assert "GAME-ONE-SECRET" not in (tmp_path / f"{OTHER_GID}.md").read_text()

    agent.act(_speak_turn(GID))
    only_game_one = cap.transcript(0)
    assert "GAME-ONE-SECRET" in only_game_one
    assert "GAME-TWO-SECRET" not in only_game_one
