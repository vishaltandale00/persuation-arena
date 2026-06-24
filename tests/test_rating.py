"""P1 — rating engine. The unit is per-seat `won`; ratings are a deterministic replay."""
from __future__ import annotations

from arena import rating, store


def _sqlite(tmp_path, monkeypatch, name="rating.db"):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / name)
    store.init_schema()


def _save_run(run_id, n_players, deck="std", created="2026-01-01T00:00:00Z"):
    store.save_run({
        "id": run_id, "game": "onuw", "label": "ONUW", "status": "done",
        "n_games": 99, "players": n_players, "seed_base": 1, "created": created[:16],
        "agents": [{"name": f"n{i}", "model": "m", "harness": "base"} for i in range(n_players)],
        "created_utc": created, "deck_preset": deck,
    })


def _save_game(run_id, gid, seats):
    """seats: list of {seat, role, team, won, agent_id, [model], [calls], [forfeits]} (contiguous seats)."""
    seats = sorted(seats, key=lambda s: s["seat"])
    transcript = {
        "seed": gid, "winner_team": "good", "outcome": {"text": "t"},
        "players": [{"seat": s["seat"], "dealt": s["role"], "end": s["role"], "team": s["team"],
                     "won": bool(s["won"]), "calls": s.get("calls", 1), "forfeits": s.get("forfeits", 0)}
                    for s in seats],
    }
    meta = [{"name": s.get("name", f"a{s['seat']}"), "model": s.get("model", "m"), "harness": "base",
             "agent_id": s.get("agent_id"), "signup_id": s.get("agent_id")} for s in seats]
    store.save_game(run_id, gid, transcript, meta)


def _ratings_by_key(tmp_path=None):
    return {r["identity_key"]: r for r in store.leaderboard_rows()}


def test_recompute_is_deterministic(tmp_path, monkeypatch):
    _sqlite(tmp_path, monkeypatch)
    _save_run("r1", 5)
    for gid in range(4):
        _save_game("r1", gid, [
            {"seat": 0, "role": "Werewolf", "team": "evil", "won": gid % 2, "agent_id": "wolf"},
            {"seat": 1, "role": "Villager", "team": "good", "won": 1 - gid % 2, "agent_id": "v1"},
            {"seat": 2, "role": "Seer", "team": "good", "won": 1 - gid % 2, "agent_id": "v2"},
            {"seat": 3, "role": "Robber", "team": "good", "won": 1 - gid % 2, "agent_id": "v3"},
            {"seat": 4, "role": "Tanner", "team": "good", "won": gid % 2, "agent_id": "tan"},
        ])
    rating.recompute()
    first = store.leaderboard_rows()
    rating.recompute()
    second = store.leaderboard_rows()
    assert first == second  # byte-identical snapshot (data-derived stamp, deterministic ids)
    assert len(first) == 5


def test_tanner_cowin_raises_village_and_tanner(tmp_path, monkeypatch):
    """Wolf voted out (village wins) AND Tanner voted out (Tanner wins): both objectives rise,
    the werewolf falls. This is the case a binary team model cannot represent."""
    _sqlite(tmp_path, monkeypatch)
    _save_run("r1", 4)
    _save_game("r1", 0, [
        {"seat": 0, "role": "Werewolf", "team": "evil", "won": 0, "agent_id": "wolf"},
        {"seat": 1, "role": "Villager", "team": "good", "won": 1, "agent_id": "vil"},
        {"seat": 2, "role": "Seer", "team": "good", "won": 1, "agent_id": "seer"},
        {"seat": 3, "role": "Tanner", "team": "good", "won": 1, "agent_id": "tan"},
    ])
    rating.recompute()
    r = _ratings_by_key()
    assert r["vil"]["skill"] > 0
    assert r["seer"]["skill"] > 0
    assert r["tan"]["skill"] > 0          # Tanner co-wins alongside the village
    assert r["wolf"]["skill"] < 0


def test_consistent_winner_rises_monotonically(tmp_path, monkeypatch):
    _sqlite(tmp_path, monkeypatch)
    _save_run("r1", 3)
    for gid in range(8):
        _save_game("r1", gid, [
            {"seat": 0, "role": "Villager", "team": "good", "won": 1, "agent_id": "winner"},
            {"seat": 1, "role": "Werewolf", "team": "evil", "won": 0, "agent_id": f"loser_a{gid}"},
            {"seat": 2, "role": "Villager", "team": "good", "won": 1, "agent_id": f"ally{gid}"},
        ])
    rating.recompute()
    ev = store.rating_events_for("winner")
    assert len(ev) == 8
    posts = [e["post_skill"] for e in ev]
    assert all(b > a for a, b in zip(posts, posts[1:])), posts  # strictly increasing
    assert _ratings_by_key()["winner"]["skill"] > 0


def test_forfeits_shrink_the_update(tmp_path, monkeypatch):
    """Two villagers win the same game; the one who forfeited every turn gains far less."""
    _sqlite(tmp_path, monkeypatch)
    _save_run("r1", 3)
    _save_game("r1", 0, [
        {"seat": 0, "role": "Villager", "team": "good", "won": 1, "agent_id": "clean",
         "calls": 4, "forfeits": 0},
        {"seat": 1, "role": "Villager", "team": "good", "won": 1, "agent_id": "forfeiter",
         "calls": 4, "forfeits": 4},
        {"seat": 2, "role": "Werewolf", "team": "evil", "won": 0, "agent_id": "wolf",
         "calls": 4, "forfeits": 0},
    ])
    rating.recompute()
    r = _ratings_by_key()
    assert r["clean"]["skill"] > r["forfeiter"]["skill"]
    assert abs(r["forfeiter"]["skill"]) < 1e-9   # full forfeit → damp 0 → no movement
    assert r["forfeiter"]["forfeit_rate"] == 1.0


def test_no_contest_game_excluded_from_rating(tmp_path, monkeypatch):
    """A game with no evil seat and no winner (the ONUW no-contest: no Werewolf/Minion in play,
    the vote killed an innocent) must not enter the replay — rating a loss against nobody skews Elo.
    Only the real game is rated, and the no-contest-only seats never appear in the snapshot."""
    _sqlite(tmp_path, monkeypatch)
    _save_run("r1", 5)
    _save_game("r1", 0, [
        {"seat": 0, "role": "Werewolf", "team": "evil", "won": 0, "agent_id": "wolf"},
        {"seat": 1, "role": "Villager", "team": "good", "won": 1, "agent_id": "vil"},
        {"seat": 2, "role": "Seer", "team": "good", "won": 1, "agent_id": "seer"},
        {"seat": 3, "role": "Robber", "team": "good", "won": 1, "agent_id": "rob"},
        {"seat": 4, "role": "Villager", "team": "good", "won": 1, "agent_id": "vil2"},
    ])
    _save_game("r1", 1, [  # no evil seat, nobody won -> no-contest
        {"seat": i, "role": role, "team": "good", "won": 0, "agent_id": f"nc{i}"}
        for i, role in enumerate(["Seer", "Robber", "Troublemaker", "Villager", "Villager"])
    ])
    summary = rating.recompute()
    assert summary["games"] == 1                          # only the real game is rated
    r = _ratings_by_key()
    assert not any(k.startswith("nc") for k in r)         # no-contest seats earn no rating
    assert "wolf" in r and "vil" in r


def test_hard_role_has_higher_difficulty_than_easy_role(tmp_path, monkeypatch):
    _sqlite(tmp_path, monkeypatch)
    _save_run("r1", 2)
    for gid in range(12):
        _save_game("r1", gid, [
            {"seat": 0, "role": "Villager", "team": "good", "won": 1, "agent_id": f"v{gid}"},
            {"seat": 1, "role": "Werewolf", "team": "evil", "won": 0, "agent_id": f"w{gid}"},
        ])
    rating.recompute()
    diff = store.role_difficulty_map()
    d_wolf = diff[("*", "Werewolf")]["d_r"]
    d_vil = diff[("*", "Villager")]["d_r"]
    assert d_wolf > d_vil    # Werewolf wins ~never here → harder → larger handicap
    assert diff[("*", "Villager")]["base_rate"] > diff[("*", "Werewolf")]["base_rate"]
