"""$0 per-PR ORCHESTRATION GATE: prove the infra + CLI that runs games still work end-to-end.

Drives the REAL `arena run` -> `arena score` CLI path (parser -> _dispatch -> handler) through the
REAL ONUW engine, batch scheduler, transcript persistence, and scoring — with ONLY the OpenRouter
network call stubbed. So it is hermetic and FREE (no API key, no network, SQLite via conftest),
yet a break anywhere in the orchestration glue (arg parsing, caps wiring, the game loop, transcript
save, or scoring) fails this test. Agent/model QUALITY is out of scope — that is the harness
author's concern; the real-model `arena run` is a manual dev tool, not this gate.
"""
from __future__ import annotations

import pytest

from arena import cli, openrouter, store
from arena.score import score_run

# A canned, schema-shaped reply. For discussion it parses as pass/done (ends discussion fast); for
# vote/night turns it fails to parse and the engine falls back to its safe default — either way the
# game RESOLVES, which is all the orchestration gate needs.
_CANNED = '{"action": {"pass": true, "stance": "done"}, "declared_reasoning": "smoke"}'


class _Usage:
    def model_dump(self):
        return {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


class _Resp:
    choices = [type("C", (), {"message": type("M", (), {"content": _CANNED})()})()]
    usage = _Usage()


class _FakeClient:
    chat = type("Chat", (), {"completions": type("Comp", (), {"create": lambda self, **kw: _Resp()})()})()


@pytest.fixture
def hermetic(tmp_path, monkeypatch):
    """SQLite-only store + a stubbed OpenRouter client => the real orchestration runs for $0."""
    monkeypatch.delenv("DATABASE_URL", raising=False)          # conftest also does this
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "smoke.db")
    store.init_schema()
    fake = _FakeClient()
    monkeypatch.setattr(openrouter, "_client", fake)
    monkeypatch.setattr(openrouter, "client", lambda: fake)


def test_arena_run_then_score_orchestration(hermetic, capsys):
    parser = cli.build_parser()

    # `arena run --game onuw --games 2 --rounds 1 --workers 1 --run-id smoke`
    run_args = parser.parse_args(
        ["run", "--game", "onuw", "--games", "2", "--rounds", "1", "--workers", "1", "--run-id", "smoke"]
    )
    cli._dispatch(run_args)  # parser -> _use_local_store -> _run -> run_batch (real engine, stubbed net)

    # The run orchestrated to completion and persisted.
    run = store.get_run("smoke")
    assert run is not None, "run was not persisted"
    assert run["status"] == "done", f"run did not complete cleanly: {run['status']}"
    assert store.distinct_gids("smoke") == [1, 2], "both games should be saved"
    assert run["agents"], "roster not persisted with the run"

    # Scoring works on the persisted run: one competitor per roster seat, each with >=1 game.
    sc = score_run("smoke")
    assert sc, "score_run returned no competitors"
    assert len(sc) == len(run["agents"]), f"expected {len(run['agents'])} competitors, got {len(sc)}"
    for name, d in sc.items():
        assert d["overall"]["n"] >= 1, f"{name} has no scored games"

    # The `arena score` CLI path itself renders a leaderboard.
    capsys.readouterr()  # drop run-phase output
    score_args = parser.parse_args(["score", "--run", "smoke"])
    cli._dispatch(score_args)
    out = capsys.readouterr().out
    assert "win%" in out, "score CLI did not render the leaderboard header"
    assert any(name in out for name in sc), "score CLI did not list any competitor"
