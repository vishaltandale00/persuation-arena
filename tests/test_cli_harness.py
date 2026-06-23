"""The participant CLI (`arena-agent play <file>`) must come up to date with delta transport:
wire on_event for stateful harnesses so they can build memory, and only act for stateless ones.
Without on_event a stateful harness receives no events and plays blind."""
import pytest

from persuasion_arena_agent.agent import ArenaAgent
from persuasion_arena_agent.cli import _load_harness


def _wire(path: str) -> tuple[bool, bool]:
    mod = _load_harness(path)
    agent = ArenaAgent("x")
    agent.act(mod.act)
    if hasattr(mod, "on_event"):
        agent.on_event(mod.on_event)
    return agent._act_handler is not None, agent._event_handler is not None


@pytest.mark.parametrize("path,expect_on_event", [
    ("examples/session_agent.py", True),
    ("examples/file_memory_agent.py", True),
    ("examples/pass_agent.py", False),       # a stateless reference harness
])
def test_cli_wires_handlers(path, expect_on_event):
    act_ok, on_event_ok = _wire(path)
    assert act_ok, f"{path}: act not registered"
    assert on_event_ok == expect_on_event, f"{path}: on_event wiring wrong"


def test_load_harness_requires_act(tmp_path):
    bad = tmp_path / "bad_harness.py"
    bad.write_text("x = 1\n")  # defines no act(turn)
    with pytest.raises(RuntimeError):
        _load_harness(str(bad))
