"""P2 — leaderboard API. Shapes for the UI; per-competitor counts reconcile with score.py."""
from __future__ import annotations

from fastapi.testclient import TestClient

from arena import rating, score, store
from arena.server import app


def _sqlite(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "lb.db")
    store.init_schema()


def _save_run(run_id, n_players):
    store.save_run({
        "id": run_id, "game": "onuw", "label": "ONUW", "status": "done", "n_games": 99,
        "players": n_players, "seed_base": 1, "created": "2026-01-01 00:00",
        "agents": [{"name": f"n{i}", "model": "m", "harness": "base"} for i in range(n_players)],
        "created_utc": "2026-01-01T00:00:00Z", "deck_preset": "std",
    })


def _save_game(run_id, gid, seats):
    seats = sorted(seats, key=lambda s: s["seat"])
    transcript = {"seed": gid, "winner_team": "good", "outcome": {"text": "t"},
                  "players": [{"seat": s["seat"], "dealt": s["role"], "end": s["role"],
                               "team": s["team"], "won": bool(s["won"]),
                               "calls": s.get("calls", 1), "forfeits": s.get("forfeits", 0)}
                              for s in seats]}
    meta = [{"name": s["name"], "model": s.get("model", "m"), "harness": "base",
             "agent_id": s["agent_id"], "signup_id": s["agent_id"]} for s in seats]
    store.save_game(run_id, gid, transcript, meta)


def _seed_run(monkeypatch, tmp_path):
    _sqlite(tmp_path, monkeypatch)
    store.register_agent("Winner", "h_w", "arena-agent-v1", "t",
                         agent_id="agent_winner", declared_model="gpt-x", declared_harness="claude")
    _save_run("r1", 4)
    for gid in range(6):
        _save_game("r1", gid, [
            {"seat": 0, "role": "Villager", "team": "good", "won": 1, "agent_id": "agent_winner", "name": "Winner"},
            {"seat": 1, "role": "Werewolf", "team": "evil", "won": 0, "agent_id": f"agent_wolf{gid}", "name": "W"},
            {"seat": 2, "role": "Seer", "team": "good", "won": 1, "agent_id": f"agent_s{gid}", "name": "S"},
            {"seat": 3, "role": "Tanner", "team": "good", "won": gid % 2, "agent_id": f"agent_t{gid}", "name": "T"},
        ])
    rating.recompute()


def test_leaderboard_shape_and_ranking(tmp_path, monkeypatch):
    _seed_run(monkeypatch, tmp_path)
    client = TestClient(app)
    r = client.get("/api/leaderboard")
    assert r.status_code == 200
    comps = r.json()["competitors"]
    assert comps, "leaderboard empty"
    top = comps[0]
    for key in ("identity_key", "agent_id", "elo", "rd", "conservative", "games", "wins",
                "provisional", "forfeit_rate", "overall", "by_objective", "by_role"):
        assert key in top, f"missing {key}"
    assert set(top["overall"]) >= {"w", "n", "rate", "lo", "hi"}
    assert set(top["by_objective"]) == {"village", "werewolf", "tanner"}
    # conservative ranking is non-increasing
    cons = [c["conservative"] for c in comps]
    assert cons == sorted(cons, reverse=True)
    # the always-winning villager tops the board
    assert comps[0]["identity_key"] == "agent_winner"
    role_cell = comps[0]["by_role"]["Villager"]
    assert {"base", "d_r", "vs_spread", "hard"} <= set(role_cell)
    # declared fields surfaced from the agents table
    assert comps[0]["declared_model"] == "gpt-x" and comps[0]["declared_harness"] == "claude"
    # no token field ever leaks
    assert "token" not in top and "token_hash" not in top


def test_leaderboard_counts_reconcile_with_score_py(tmp_path, monkeypatch):
    _seed_run(monkeypatch, tmp_path)
    sc = score.score_run("r1")            # keyed by agent display name
    comps = {c["identity_key"]: c for c in rating.leaderboard()}
    winner = comps["agent_winner"]
    assert winner["overall"]["w"] == sc["Winner"]["overall"]["w"]
    assert winner["overall"]["n"] == sc["Winner"]["overall"]["n"]
    assert winner["by_objective"]["village"]["n"] == sc["Winner"]["good"]["n"]


def test_agent_detail_endpoint(tmp_path, monkeypatch):
    _seed_run(monkeypatch, tmp_path)
    client = TestClient(app)
    r = client.get("/api/agents/agent_winner")
    assert r.status_code == 200
    d = r.json()
    assert d["identity_key"] == "agent_winner"
    assert len(d["history"]) == 6 and d["runs"] == ["r1"]
    assert d["history"][0]["post_elo"] != d["history"][-1]["post_elo"]   # rating moved over time
    assert "token" not in d

    assert client.get("/api/agents/nope_missing").status_code == 404
