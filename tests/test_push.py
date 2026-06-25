"""`arena push` — offline unit tests for the pure helpers and the recompute safety guard.

These never touch a live server or prod Neon: the autouse conftest deletes DATABASE_URL, and the
store is a tmp_path SQLite. We exercise payload-building, the gid-diff, token-from-env, the
game_players unique index (idempotent re-push), and the disaster guardrail.
"""
from __future__ import annotations

import pytest

from arena import cli, rating, store


def _sqlite(tmp_path, monkeypatch, name="push.db"):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / name)
    store.init_schema()


def _save_run(run_id, n_players=3, status="done", created="2026-01-01T00:00:00Z"):
    store.save_run({
        "id": run_id, "game": "onuw", "label": "ONUW", "status": status,
        "n_games": 99, "players": n_players, "seed_base": 1, "created": created[:16],
        "agents": [{"name": f"n{i}", "model": "m", "harness": "base"} for i in range(n_players)],
        "created_utc": created, "deck_preset": "arena",
    })


def _save_game(run_id, gid, n_players=3):
    seats = list(range(n_players))
    transcript = {
        "seed": gid, "winner_team": "good", "outcome": {"text": "t"},
        "players": [{"seat": s, "dealt": "Villager", "end": "Villager", "team": "good",
                     "won": s == 0, "calls": 2, "forfeits": 0} for s in seats],
    }
    agents = [{"name": f"n{s}", "model": "m", "harness": "base"} for s in seats]
    return store.save_game(run_id, gid, transcript, agents)


# --- token-from-env ----------------------------------------------------------
def test_push_token_from_env_present():
    assert cli.push_token_from_env({"ARENA_INGEST_TOKEN": "  team=abc  "}) == "team=abc"


def test_push_token_from_env_absent():
    assert cli.push_token_from_env({}) is None
    assert cli.push_token_from_env({"ARENA_INGEST_TOKEN": "   "}) is None


def test_push_fails_closed_when_token_unset(tmp_path, monkeypatch):
    """(d) No $ARENA_INGEST_TOKEN -> exit 2 (fail closed) BEFORE any network call. We arm the HTTP
    helpers to fail loudly so a regression that reaches the network is caught."""
    _sqlite(tmp_path, monkeypatch)
    _save_run("r1")
    _save_game("r1", 0)
    monkeypatch.delenv("ARENA_INGEST_TOKEN", raising=False)
    monkeypatch.setattr(cli, "_worker_get", lambda *a, **k: pytest.fail("hit network with no token"))
    monkeypatch.setattr(cli, "_worker_post", lambda *a, **k: pytest.fail("hit network with no token"))

    class _Args:
        run, all, dry_run, force, no_recompute = "r1", False, True, False, False
        server = "http://127.0.0.1:59999"  # never reached

    with pytest.raises(SystemExit) as ei:
        cli._push(_Args())
    assert ei.value.code == 2


# --- gid diff ----------------------------------------------------------------
def test_gid_diff_uploads_only_missing():
    remote = {"games": [{"gid": 0}, {"gid": 1}]}
    assert cli.gid_diff([0, 1, 2, 3], remote) == [2, 3]


def test_gid_diff_no_remote_uploads_all():
    assert cli.gid_diff([2, 0, 1], None) == [0, 1, 2]


def test_gid_diff_no_op_when_all_present():
    """(c) re-push: every local gid already on the board -> nothing to upload."""
    remote = {"games": [{"gid": 0}, {"gid": 1}]}
    assert cli.gid_diff([0, 1], remote) == []


def test_gid_diff_force_resends_everything():
    remote = {"games": [{"gid": 0}, {"gid": 1}]}
    assert cli.gid_diff([0, 1, 2], remote, force=True) == [0, 1, 2]


def test_gid_diff_tolerates_bad_remote_rows():
    remote = {"games": [{"gid": 0}, {"nope": 1}, {"gid": "x"}]}
    assert cli.gid_diff([0, 1], remote) == [1]


# --- payload building --------------------------------------------------------
def test_build_import_payload_matches_save_game_shape(tmp_path, monkeypatch):
    _sqlite(tmp_path, monkeypatch)
    _save_run("r1")
    _save_game("r1", 0)
    _save_game("r1", 1)
    run = store.get_run("r1")
    payload = cli.build_import_payload(run, [0, 1], store.get_game)

    assert set(payload) == {"run", "games"}
    assert payload["run"]["id"] == "r1"
    assert payload["run"]["status"] == "done"
    assert isinstance(payload["run"]["agents"], list)
    assert [g["gid"] for g in payload["games"]] == [0, 1]
    g0 = payload["games"][0]
    # full transcript (server derives game_players FROM transcript['players']), not a flat array
    assert "players" in g0["transcript"]
    assert g0["agents"] == run["agents"]
    assert {p["seat"] for p in g0["transcript"]["players"]} == {0, 1, 2}


def test_build_import_payload_skips_missing_transcript(tmp_path, monkeypatch):
    _sqlite(tmp_path, monkeypatch)
    _save_run("r1")
    _save_game("r1", 0)
    run = store.get_run("r1")
    payload = cli.build_import_payload(run, [0, 7], store.get_game)  # gid 7 not stored
    assert [g["gid"] for g in payload["games"]] == [0]


# --- unique index / idempotent re-push --------------------------------------
def test_game_players_unique_index_enforced(tmp_path, monkeypatch):
    _sqlite(tmp_path, monkeypatch)
    _save_run("r1")
    assert _save_game("r1", 0) is True
    # re-saving the same (run_id,gid) is first-writer-wins and a no-op
    assert _save_game("r1", 0) is False
    rows = store.player_rows("r1")
    seats = sorted(p["seat"] for p in rows)
    assert seats == [0, 1, 2]  # exactly one row per seat, no duplicates


def test_game_players_unique_index_exists(tmp_path, monkeypatch):
    _sqlite(tmp_path, monkeypatch)
    with store.conn() as c:
        idx = c.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='game_players_rgs_uq'"
        ).fetchone()
    assert idx is not None


# --- recompute safety guard --------------------------------------------------
def test_guard_aborts_on_empty_local():
    with pytest.raises(rating.RecomputeAborted):
        rating.guard_recompute(local_game_players=0, prior_prod_count=0, current_prod_count=0)


def test_guard_aborts_when_prod_would_shrink():
    with pytest.raises(rating.RecomputeAborted):
        rating.guard_recompute(local_game_players=30, prior_prod_count=100, current_prod_count=90)


def test_guard_passes_when_prod_grows():
    # does not raise
    rating.guard_recompute(local_game_players=30, prior_prod_count=100, current_prod_count=130)


def test_guard_passes_when_prod_unchanged():
    rating.guard_recompute(local_game_players=10, prior_prod_count=10, current_prod_count=10)


def test_push_recompute_uses_prior_baseline_for_shrink_guard(monkeypatch):
    """Regression: _push_recompute must feed the PRE-UPLOAD baseline it was handed into the shrink
    guard and compare it against the CURRENT prod count. If the prior dead-guard bug returns (prior
    captured post-upload, so prior==current), this abort never fires."""
    monkeypatch.setattr(store, "game_player_count", lambda: 90)  # current prod (shrunk vs baseline)
    monkeypatch.setattr(rating, "recompute", lambda: pytest.fail("recompute ran despite a shrink"))
    with pytest.raises(rating.RecomputeAborted):
        cli._push_recompute(prior_prod_count=100, local_game_players=50)


def test_push_recompute_runs_when_prod_grows(monkeypatch):
    monkeypatch.setattr(store, "game_player_count", lambda: 130)  # current prod grew vs baseline
    monkeypatch.setattr(rating, "recompute", lambda: {"ok": True})
    assert cli._push_recompute(prior_prod_count=100, local_game_players=50) == {"ok": True}


def test_game_player_count(tmp_path, monkeypatch):
    _sqlite(tmp_path, monkeypatch)
    assert store.game_player_count() == 0
    _save_run("r1")
    _save_game("r1", 0)
    assert store.game_player_count() == 3  # 3 seats
