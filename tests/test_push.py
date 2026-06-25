"""`arena push` — offline unit tests for the pure helpers and the recompute safety guard.

These never touch a live server or prod Neon: the autouse conftest deletes DATABASE_URL, and the
store is a tmp_path SQLite. We exercise payload-building, the gid-diff, token-from-env, the
game_players unique index (idempotent re-push), and the disaster guardrail.
"""
from __future__ import annotations

import os

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


# --- token resolution: env -> cache -> auto-register -------------------------
def test_push_token_env_override(tmp_path, monkeypatch):
    """$ARENA_INGEST_TOKEN wins outright — no CredentialsStore read, no network register."""
    monkeypatch.setenv("ARENA_INGEST_TOKEN", "pa_live_envtoken")
    from persuasion_arena_agent import credentials as creds_mod
    monkeypatch.setattr(creds_mod.CredentialsStore, "get",
                        lambda self, server: pytest.fail("read cache despite env token"))
    assert cli.push_token_or_register("https://x.example") == "pa_live_envtoken"


def test_push_token_uses_cache(tmp_path, monkeypatch):
    """A cached identity for the server is returned without registering."""
    monkeypatch.delenv("ARENA_INGEST_TOKEN", raising=False)
    from persuasion_arena_agent import credentials as creds_mod
    cached = creds_mod.AgentCredentials(server="https://x.example", agent_id="agent_1",
                                        display_name="laptop", agent_token="pa_live_cached")
    monkeypatch.setattr(creds_mod.CredentialsStore, "get", lambda self, server: cached)
    from persuasion_arena_agent import client as client_mod
    monkeypatch.setattr(client_mod, "ArenaHttpClient",
                        lambda *a, **k: pytest.fail("registered despite cached identity"))
    assert cli.push_token_or_register("https://x.example") == "pa_live_cached"


def test_push_token_auto_registers_and_caches(tmp_path, monkeypatch):
    """No env token + no cache -> register_agent mints a pa_live_ token, which is cached and used."""
    monkeypatch.delenv("ARENA_INGEST_TOKEN", raising=False)
    from persuasion_arena_agent import credentials as creds_mod
    from persuasion_arena_agent import client as client_mod
    monkeypatch.setattr(creds_mod.CredentialsStore, "get", lambda self, server: None)

    minted = creds_mod.AgentCredentials(server="https://x.example", agent_id="agent_new",
                                        display_name="myhost", agent_token="pa_live_minted")
    saved = {}
    monkeypatch.setattr(creds_mod.CredentialsStore, "save",
                        lambda self, creds: saved.update(creds=creds))

    class _FakeClient:
        def __init__(self, server):
            self.server = server

        def register_agent(self, display_name):
            assert display_name == "myhost"
            return minted

        def close(self):
            pass

    monkeypatch.setattr(client_mod, "ArenaHttpClient", _FakeClient)
    assert cli.push_token_or_register("https://x.example", "myhost") == "pa_live_minted"
    assert saved["creds"] is minted


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


def test_game_player_count(tmp_path, monkeypatch):
    _sqlite(tmp_path, monkeypatch)
    assert store.game_player_count() == 0
    _save_run("r1")
    _save_game("r1", 0)
    assert store.game_player_count() == 3  # 3 seats


# --- arena run is local-only (never touches the remote DB) -------------------
def test_run_drops_database_url_local_only(monkeypatch):
    """`arena run` must NEVER use a remote DB: it drops DATABASE_URL so the store is local SQLite.
    The only path to the prod leaderboard is `arena push` (the Vercel JS API)."""
    monkeypatch.setenv("DATABASE_URL", "postgres://bogus:bogus@localhost:1/none")
    captured = {}

    def fake_run_batch(**kwargs):
        captured["db_url_at_call"] = os.environ.get("DATABASE_URL")

    monkeypatch.setattr("arena.batch.run_batch", fake_run_batch)

    class _Args:
        game, games, seed, run_id, workers, rounds = "onuw", 1, 9000, "guardtest", 8, 10
        deck, deal_schedule, port = "arena", None, 8000
        reasoning_effort = max_tokens_per_turn = temperature = retries = prior_message_turns = None

    cli._run(_Args())
    assert captured["db_url_at_call"] is None  # DATABASE_URL was set, but `arena run` dropped it


def test_use_local_store_pops_database_url(monkeypatch):
    """The shared guard used by the local commands (run/score/runs) drops DATABASE_URL so they
    never reach the remote DB."""
    monkeypatch.setenv("DATABASE_URL", "postgres://bogus:bogus@localhost:1/none")
    cli._use_local_store()
    assert os.environ.get("DATABASE_URL") is None
