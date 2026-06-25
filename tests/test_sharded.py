"""Verifiers for the parallel-shards feature (SPEC-parallel-shards.md).

Step 1 (SPEC §6.1) — schema + store plumbing: the new run_kind/parent_run_id/
shard_index/num_shards columns, save_run threading, child_run_ids, the
list_open_runs discovery filter (INV-4), and init_schema idempotency (INV-5).

Steps 3+4 (SPEC §6.3-6.4, REQ-2) — score aggregation across runs (score_runs),
parent status rollup (rollup_parent_status), and the discovery filter restated as
test_children_not_discoverable (INV-4 / V-2).
"""
from __future__ import annotations

import threading
import time

from fastapi.testclient import TestClient

from arena import score, store
from arena.connected import run_connected_batch
from arena.identity import NO_ONE_REF
from arena.sharded import create_sharded_run, rollup_parent_status
from arena.server import app


def _sqlite_store(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "sharded.db")
    store.init_schema()


def _base_run(run_id: str, **extra) -> dict:
    meta = {
        "id": run_id, "game": "onuw", "label": "ONUW", "status": "open",
        "n_games": 4, "players": 3, "seed_base": 7,
        "created": "2026-06-25 00:00", "created_utc": "2026-06-25T00:00:00Z",
        "agents": [],
    }
    meta.update(extra)
    return meta


def test_normal_run_defaults_run_kind_and_null_shard_cols(tmp_path, monkeypatch):
    _sqlite_store(tmp_path, monkeypatch)
    store.save_run(_base_run("run_normal"))
    row = store.get_run("run_normal")
    assert row is not None
    assert row["run_kind"] == "normal"
    assert row["parent_run_id"] is None
    assert row["shard_index"] is None
    assert row["num_shards"] is None


def test_create_connected_run_is_normal_by_default(tmp_path, monkeypatch):
    _sqlite_store(tmp_path, monkeypatch)
    store.create_connected_run({
        "id": "run_conn", "game": "onuw", "label": "ONUW", "status": "open",
        "n_games": 2, "players": 1, "seed_base": 1,
    })
    row = store.get_run("run_conn")
    assert row["run_kind"] == "normal"
    assert row["parent_run_id"] is None
    assert row["shard_index"] is None
    assert row["num_shards"] is None


def test_create_connected_run_reads_shard_meta(tmp_path, monkeypatch):
    _sqlite_store(tmp_path, monkeypatch)
    store.create_connected_run({
        "id": "child_meta", "game": "onuw", "label": "ONUW", "status": "open",
        "n_games": 6, "players": 3, "seed_base": 99,
        "run_kind": "child", "parent_run_id": "parent_meta",
        "shard_index": 1, "num_shards": 2,
    })
    row = store.get_run("child_meta")
    assert row["run_kind"] == "child"
    assert row["parent_run_id"] == "parent_meta"
    assert row["shard_index"] == 1
    assert row["num_shards"] == 2


def test_parent_and_child_excluded_from_list_open_runs(tmp_path, monkeypatch):
    _sqlite_store(tmp_path, monkeypatch)
    store.save_run(_base_run("run_normal_open"))
    store.save_run(_base_run("run_parent", run_kind="parent", num_shards=2))
    store.save_run(_base_run(
        "run_child", run_kind="child", parent_run_id="run_parent",
        shard_index=0, num_shards=2))

    open_ids = {r["run_id"] for r in store.list_open_runs()}
    assert "run_normal_open" in open_ids
    assert "run_parent" not in open_ids
    assert "run_child" not in open_ids

    # game-filtered path must apply the same filter
    open_ids_filtered = {r["run_id"] for r in store.list_open_runs(game="onuw")}
    assert open_ids_filtered == {"run_normal_open"}


def test_child_run_ids_ordered_by_shard_index(tmp_path, monkeypatch):
    _sqlite_store(tmp_path, monkeypatch)
    store.save_run(_base_run("parent_x", run_kind="parent", num_shards=3))
    # insert out of shard order to prove ORDER BY shard_index
    store.save_run(_base_run(
        "child_x2", run_kind="child", parent_run_id="parent_x",
        shard_index=2, num_shards=3))
    store.save_run(_base_run(
        "child_x0", run_kind="child", parent_run_id="parent_x",
        shard_index=0, num_shards=3))
    store.save_run(_base_run(
        "child_x1", run_kind="child", parent_run_id="parent_x",
        shard_index=1, num_shards=3))
    # an unrelated child of a different parent must not leak in
    store.save_run(_base_run(
        "child_other", run_kind="child", parent_run_id="parent_other",
        shard_index=0, num_shards=1))

    assert store.child_run_ids("parent_x") == ["child_x0", "child_x1", "child_x2"]
    assert store.child_run_ids("parent_other") == ["child_other"]
    assert store.child_run_ids("no_such_parent") == []


def test_init_schema_idempotent_preserves_rows(tmp_path, monkeypatch):
    """INV-5: opening the store twice (re-running init_schema) is a no-op and rows survive."""
    _sqlite_store(tmp_path, monkeypatch)
    store.save_run(_base_run("run_persist", run_kind="parent", num_shards=2))
    store.save_run(_base_run(
        "child_persist", run_kind="child", parent_run_id="run_persist",
        shard_index=0, num_shards=2))

    # Re-run init_schema (idempotent migrations) and re-open a conn — must not error.
    store.init_schema()
    with store.conn():
        pass

    parent = store.get_run("run_persist")
    child = store.get_run("child_persist")
    assert parent is not None and parent["run_kind"] == "parent"
    assert parent["num_shards"] == 2
    assert child is not None and child["run_kind"] == "child"
    assert child["parent_run_id"] == "run_persist"
    assert child["shard_index"] == 0
    assert store.child_run_ids("run_persist") == ["child_persist"]


# --- Steps 3+4: rollup, score aggregation, discovery (REQ-2 / V-2; score half of V-4/V-5) ----


def _seat(seat: int, role: str, team: str, won: bool) -> dict:
    return {"seat": seat, "dealt": role, "end": role, "team": team, "won": won}


def _save_two_seat_game(run_id: str, gid: int, *, a_won: bool, b_won: bool) -> None:
    """Save one 2-seat game: seat 0 = agent A (good), seat 1 = agent B (evil)."""
    transcript = {
        "seed": 1000 + gid,
        "winner_team": "good" if a_won else "evil",
        "outcome": {"text": f"game {gid}"},
        "players": [
            _seat(0, "Villager", "good", a_won),
            _seat(1, "Werewolf", "evil", b_won),
        ],
    }
    agents = [
        {"name": "A", "model": "connected-agent"},
        {"name": "B", "model": "connected-agent"},
    ]
    store.save_game(run_id, gid, transcript, agents)


def test_rollup_parent_status(tmp_path, monkeypatch):
    """REQ-2 / V-2: rollup reflects children — all done->done, any partial->partial,
    any not-yet-done->running, child-with-failures->partial, no children->open."""
    _sqlite_store(tmp_path, monkeypatch)

    def _parent_with_children(parent_id: str, child_statuses: list[str]) -> str:
        store.save_run(_base_run(parent_id, run_kind="parent",
                                 num_shards=len(child_statuses)))
        for k, st in enumerate(child_statuses):
            store.save_run(_base_run(
                f"{parent_id}_c{k}", status=st, run_kind="child",
                parent_run_id=parent_id, shard_index=k,
                num_shards=len(child_statuses)))
        return parent_id

    assert rollup_parent_status(
        _parent_with_children("p_dd", ["done", "done"])) == "done"
    assert rollup_parent_status(
        _parent_with_children("p_dp", ["done", "partial"])) == "partial"
    assert rollup_parent_status(
        _parent_with_children("p_dr", ["done", "running"])) == "running"
    # a single partial child rolls the parent up to partial even if others run
    assert rollup_parent_status(
        _parent_with_children("p_pr", ["partial", "running"])) == "partial"

    # no children started yet -> open
    store.save_run(_base_run("p_empty", run_kind="parent", num_shards=2))
    assert rollup_parent_status("p_empty") == "open"


def test_score_runs_aggregates_across_disjoint_runs(tmp_path, monkeypatch):
    """Score half of V-4/V-5: score_runs unions player_rows across run_ids and runs the
    existing per-agent aggregation. Child runs carry DISJOINT global gids, so per-agent
    overall.n sums across both runs."""
    _sqlite_store(tmp_path, monkeypatch)
    store.create_connected_run(_base_run("shard_a", run_kind="child",
                                         parent_run_id="P", shard_index=0, num_shards=2))
    store.create_connected_run(_base_run("shard_b", run_kind="child",
                                         parent_run_id="P", shard_index=1, num_shards=2))
    # disjoint global gids: even gids on shard_a, odd gids on shard_b
    _save_two_seat_game("shard_a", 2, a_won=True, b_won=False)
    _save_two_seat_game("shard_a", 4, a_won=False, b_won=True)
    _save_two_seat_game("shard_b", 1, a_won=True, b_won=False)
    _save_two_seat_game("shard_b", 3, a_won=True, b_won=False)
    _save_two_seat_game("shard_b", 5, a_won=False, b_won=True)

    a_only = score.score_run("shard_a")
    b_only = score.score_run("shard_b")
    assert a_only["A"]["overall"]["n"] == 2
    assert b_only["A"]["overall"]["n"] == 3

    combined = score.score_runs(["shard_a", "shard_b"])
    # overall.n sums across both shards (2 + 3 = 5 games per agent)
    assert combined["A"]["overall"]["n"] == 5
    assert combined["B"]["overall"]["n"] == 5
    # A is good in every game; won 3 of 5 (gids 2,1,3) -> w == 3
    assert combined["A"]["good"]["n"] == 5
    assert combined["A"]["good"]["w"] == 3
    assert combined["A"]["overall"]["w"] == 3
    # B is evil in every game; won 2 of 5 (gids 4,5) -> w == 2
    assert combined["B"]["evil"]["n"] == 5
    assert combined["B"]["evil"]["w"] == 2

    # single-run call is unchanged: score_run(id) == score_runs([id])
    assert score.score_runs(["shard_a"]) == a_only


def test_children_not_discoverable(tmp_path, monkeypatch):
    """INV-4 / V-2: list_open_runs excludes parents and children; only normal runs surface."""
    _sqlite_store(tmp_path, monkeypatch)
    store.save_run(_base_run("normal_open"))
    store.save_run(_base_run("parent_open", run_kind="parent", num_shards=2))
    store.save_run(_base_run(
        "child_open", run_kind="child", parent_run_id="parent_open",
        shard_index=0, num_shards=2))

    open_ids = {r["run_id"] for r in store.list_open_runs()}
    assert open_ids == {"normal_open"}
    open_ids_filtered = {r["run_id"] for r in store.list_open_runs(game="onuw")}
    assert open_ids_filtered == {"normal_open"}


# --- Steps 5+6: end-to-end, K=1 no-op, and the equivalence keystone (REQ-3/9/4; V-3/V-9/V-4) ----
#
# Driven by the scripted-responder pattern from tests/test_connected_runner.py:135-223 — no LLMs,
# no network, no real concurrency: each child's run_connected_batch is coordinated SEQUENTIALLY
# (D6) by spawning a responder thread that polls pending_turn_for_signup and replies deterministically.


def _scripted_action(turn: dict):
    """A DETERMINISTIC policy: same observation -> same wire action. (No randomness, so the saved
    transcript for a given seed is reproducible across runs — required for the equivalence keystone.)"""
    kind = turn["action_kind"]
    legal = turn["legal_action"]
    players = legal.get("choices", {}).get("players") or []
    if kind == "onuw.discussion.speak_or_pass":
        return {"pass": True}
    if kind == "onuw.vote":
        return {"target": NO_ONE_REF}
    if kind == "onuw.seer.inspect":
        return {"mode": "center", "indices": [0, 1]}
    if kind == "onuw.troublemaker.swap_two_or_decline":
        return {"a": None, "b": None}
    if kind in {"onuw.doppelganger.copy_player", "onuw.robber.swap_or_decline"}:
        return {"target": players[0].get("ref", players[0].get("seat")) if players else None}
    if kind == "onuw.drunk.swap_center":
        return {"index": 0}
    raise AssertionError(kind)


def _seat_run(run_id: str, creds: list[tuple[str, str]]) -> dict[str, str]:
    """Register/sign up identities (in the given fixed order) and ready them.

    `creds` is an ordered list of (display_name, agent_id); the same agent_id may be reused across
    runs (shared credentials, REQ-5/D9). Signing up in the SAME fixed order gives the SAME
    arrival-order seat assignment in every run (controls V-4's seats without an explicit-seat
    feature, which is a later step). Returns {signup_id: agent_id}."""
    # Shard child runs require the per-parent join_token (INV-4); read it from the run row. Normal
    # runs have join_token=None, so this is a no-op for them.
    join_token = (store.get_run(run_id) or {}).get("join_token")
    agent_by_signup: dict[str, str] = {}
    for name, agent_id in creds:
        if store.get_agent(agent_id) is None:
            store.register_agent(name, f"hash_{agent_id}", "arena-agent-v1", "test", agent_id=agent_id)
        signup, err = store.create_signup(run_id, agent_id, join_token=join_token)
        assert err is None, err
        agent_by_signup[signup["id"]] = agent_id
    for signup_id, agent_id in agent_by_signup.items():
        _, err = store.mark_signup_ready(signup_id, agent_id)
        assert err is None or err == "not_ready_required"
    return agent_by_signup


def _coordinate(run_id: str, agent_by_signup: dict[str, str], rounds: int = 1) -> None:
    """Run one child/normal run to completion against a scripted responder thread."""
    stop = threading.Event()

    def responder():
        while not stop.is_set():
            for signup_id, agent_id in agent_by_signup.items():
                turn = store.pending_turn_for_signup(signup_id)
                if turn:
                    store.reply_to_turn(turn["id"], agent_id, _scripted_action(turn), "scripted", 1)
            if {s["status"] for s in store.list_run_signups(run_id)} == {"completed"}:
                return
            time.sleep(0.005)

    thread = threading.Thread(target=responder)
    thread.start()
    try:
        run_connected_batch(run_id, discussion_rounds=rounds)
    finally:
        stop.set()
    thread.join(timeout=5)


def _distinct_gids(run_id: str) -> set[int]:
    return {r["gid"] for r in store.player_rows(run_id)}


def _outcome_by_gid(run_id: str) -> dict[int, dict]:
    """{gid: {winner_team, line, seats:{agent_id:(dealt_role,end_role,won)}}} for one run's saved games."""
    out: dict[int, dict] = {}
    rows = store.player_rows(run_id)
    for g in store.get_run(run_id)["games"]:
        seats = {
            r["agent_id"]: (r["dealt_role"], r["end_role"], int(r["won"]))
            for r in rows if r["gid"] == g["gid"]
        }
        out[g["gid"]] = {"winner_team": g["winner_team"], "line": g["line"], "seats": seats}
    return out


def test_sharded_run_end_to_end(tmp_path, monkeypatch):
    """REQ-3 / V-3: a K=2 sharded run completes; every global game appears exactly once across
    children; parent rolls up to done; score_runs(children) yields one scorecard per agent."""
    _sqlite_store(tmp_path, monkeypatch)
    N = 5
    parent_cfg = _base_run("e2e", n_games=N, players=5, seed_base=11)
    children = create_sharded_run(parent_cfg, 2)
    assert children == ["e2e_shard_0", "e2e_shard_1"]

    # parent + 2 children written with correct run_kind / shard cols
    parent = store.get_run("e2e")
    assert parent["run_kind"] == "parent" and parent["num_shards"] == 2
    for k, cid in enumerate(children):
        ch = store.get_run(cid)
        assert ch["run_kind"] == "child"
        assert ch["parent_run_id"] == "e2e"
        assert ch["shard_index"] == k and ch["num_shards"] == 2
        assert int(ch["n_games"]) == N  # GLOBAL N stored on the child (D8)

    # shared creds: the SAME 5 agent_ids are seated in BOTH children (one competitor per identity)
    creds = [(f"P{i}", f"agent_{i}") for i in range(5)]
    for cid in children:
        agent_by_signup = _seat_run(cid, creds)
        _coordinate(cid, agent_by_signup)

    gids0, gids1 = _distinct_gids("e2e_shard_0"), _distinct_gids("e2e_shard_1")
    assert store.get_run("e2e_shard_0")["status"] == "done"
    assert store.get_run("e2e_shard_1")["status"] == "done"
    assert gids0 | gids1 == set(range(1, N + 1))  # every global game present
    assert gids0 & gids1 == set()                 # no gid in two children
    assert gids0 == {1, 3, 5} and gids1 == {2, 4}  # stride slice g%2==k on gid-1

    assert rollup_parent_status("e2e") == "done"

    scorecard = score.score_runs(children)
    assert set(scorecard) == {f"P{i}" for i in range(5)}
    for i in range(5):
        # each identity played every global game once across the two shards
        assert scorecard[f"P{i}"]["overall"]["n"] == N


def test_k1_matches_plain_run(tmp_path, monkeypatch):
    """REQ-9 / V-9 (INV-1): a K=1 sharded run yields the SAME saved games as a direct
    run_connected_batch of the same N/seed/roster — same gids, winner_team, per-seat
    dealt_role/end_role/won."""
    _sqlite_store(tmp_path, monkeypatch)
    N, seed = 3, 11
    creds = [(f"P{i}", f"agent_{i}") for i in range(5)]

    # plain run
    store.create_connected_run({
        "id": "plain", "game": "onuw", "label": "ONUW", "status": "open",
        "n_games": N, "players": 5, "seed_base": seed,
    })
    _coordinate("plain", _seat_run("plain", creds))

    # K=1 sharded run with the same roster/seed/N
    children = create_sharded_run(
        _base_run("k1", n_games=N, players=5, seed_base=seed), 1)
    assert children == ["k1_shard_0"]
    _coordinate("k1_shard_0", _seat_run("k1_shard_0", creds))

    assert store.get_run("k1_shard_0")["status"] == "done"
    assert rollup_parent_status("k1") == "done"
    assert _distinct_gids("plain") == _distinct_gids("k1_shard_0") == set(range(1, N + 1))
    assert _outcome_by_gid("plain") == _outcome_by_gid("k1_shard_0")


def test_equivalence_sharded_vs_unsharded(tmp_path, monkeypatch):
    """REQ-4 / V-4 KEYSTONE: with identical roster, seed, deck, rounds and a deterministic scripted
    policy, sharded(K) == unsharded(N) in game outcomes AND scores.

    Seats are controlled by signing identities up in the SAME fixed order for the unsharded run and
    for each child (arrival-order seating; no explicit-seat feature — that is a later step). Elo
    equality is NOT asserted (replay order differs by design, D4)."""
    _sqlite_store(tmp_path, monkeypatch)
    N, seed = 6, 23
    creds = [(f"P{i}", f"agent_{i}") for i in range(5)]

    # Run A: one unsharded run of N games
    store.create_connected_run({
        "id": "A", "game": "onuw", "label": "ONUW", "status": "open",
        "n_games": N, "players": 5, "seed_base": seed,
    })
    _coordinate("A", _seat_run("A", creds))

    # Run B: K=3 sharded run (children sum to N), same roster/seed/rounds/policy
    children = create_sharded_run(
        _base_run("B", n_games=N, players=5, seed_base=seed), 3)
    for cid in children:
        _coordinate(cid, _seat_run(cid, creds))

    # every child done; union of gids == {1..N}, pairwise disjoint
    child_gids = [_distinct_gids(c) for c in children]
    assert all(store.get_run(c)["status"] == "done" for c in children)
    assert set().union(*child_gids) == set(range(1, N + 1))
    union_count = sum(len(s) for s in child_gids)
    assert union_count == N  # disjoint

    # per global gid the saved transcript outcome is IDENTICAL between A and B
    a_outcomes = _outcome_by_gid("A")
    b_outcomes: dict[int, dict] = {}
    for c in children:
        b_outcomes.update(_outcome_by_gid(c))
    assert a_outcomes == b_outcomes

    # scores identical per agent across overall/good/evil/by_role (w, n)
    a_scores = score.score_run("A")
    b_scores = score.score_runs(children)
    assert set(a_scores) == set(b_scores)
    for agent in a_scores:
        for bucket in ("overall", "good", "evil"):
            assert (a_scores[agent][bucket]["w"], a_scores[agent][bucket]["n"]) == \
                   (b_scores[agent][bucket]["w"], b_scores[agent][bucket]["n"])
        assert {r: (c["w"], c["n"]) for r, c in a_scores[agent]["by_role"].items()} == \
               {r: (c["w"], c["n"]) for r, c in b_scores[agent]["by_role"].items()}


# --- Step 9 (SPEC §6.9, D7, observer): server parent read-paths + run-creation `shards` field ---
#
# The observer must treat a sharded parent as ONE normal run: /api/runs hides children and lists the
# parent; /api/runs/{parent} aggregates the children's games and scorecard; run-creation routes
# shards>1 to create_sharded_run. Normal-run responses stay byte-identical (INV-2).


def _save_child_game(child_id: str, gid: int) -> None:
    """One 2-seat game on a child shard (seat 0 good agent A, seat 1 evil agent B)."""
    transcript = {
        "seed": 2000 + gid,
        "winner_team": "good",
        "outcome": {"text": f"shard game {gid}"},
        "players": [
            _seat(0, "Villager", "good", True),
            _seat(1, "Werewolf", "evil", False),
        ],
    }
    agents = [
        {"name": "A", "model": "connected-agent", "agent_id": "agent_A"},
        {"name": "B", "model": "connected-agent", "agent_id": "agent_B"},
    ]
    store.save_game(child_id, gid, transcript, agents)


def _seed_parent_with_children(parent_id: str, k: int, games_per_child: int) -> list[str]:
    """Seed a parent + K children, each with a couple of saved games on disjoint global gids
    (stride slice g%K==shard on gid-1, matching the runner)."""
    store.create_connected_run({
        "id": parent_id, "game": "onuw", "label": "ONUW", "status": "open",
        "n_games": k * games_per_child, "players": 2, "seed_base": 7,
        "created": "2026-06-25 00:00", "created_utc": "2026-06-25T00:00:00Z",
        "run_kind": "parent", "num_shards": k,
    })
    child_ids = []
    for shard in range(k):
        cid = f"{parent_id}_shard_{shard}"
        store.create_connected_run({
            "id": cid, "game": "onuw", "label": "ONUW", "status": "done",
            "n_games": k * games_per_child, "players": 2, "seed_base": 7,
            "created": "2026-06-25 00:00", "created_utc": "2026-06-25T00:00:00Z",
            "run_kind": "child", "parent_run_id": parent_id,
            "shard_index": shard, "num_shards": k,
        })
        # global gids striped to this shard: 1-indexed gid with (gid-1) % k == shard
        for i in range(games_per_child):
            gid = shard + 1 + i * k
            _save_child_game(cid, gid)
        child_ids.append(cid)
    return child_ids


def test_api_runs_index_shows_parent_hides_children(tmp_path, monkeypatch):
    """SPEC D7 / §6.9(a): GET /api/runs lists the parent (presented as one normal run) and never the
    children; a plain normal run still appears, unchanged."""
    _sqlite_store(tmp_path, monkeypatch)
    store.create_connected_run({
        "id": "plain_idx", "game": "onuw", "label": "ONUW", "status": "open",
        "n_games": 2, "players": 2, "seed_base": 1,
    })
    children = _seed_parent_with_children("parent_idx", 2, 2)

    with TestClient(app) as client:
        rows = client.get("/api/runs").json()
    ids = {r["id"] for r in rows}
    assert "parent_idx" in ids          # parent shows
    assert "plain_idx" in ids           # normal run shows
    assert ids.isdisjoint(set(children))  # children hidden


def test_api_run_parent_aggregates_children(tmp_path, monkeypatch):
    """SPEC D7 / §6.9(a): GET /api/runs/{parent} returns the UNION of the children's games and an
    aggregated scorecard (score_runs across children)."""
    _sqlite_store(tmp_path, monkeypatch)
    children = _seed_parent_with_children("parent_agg", 2, 2)  # gids 1,3 on shard0; 2,4 on shard1

    with TestClient(app) as client:
        detail = client.get("/api/runs/parent_agg").json()

    # union of games across both children: global gids 1..4, no duplicates, ordered
    gids = [g["gid"] for g in detail["games"]]
    assert gids == [1, 2, 3, 4]
    # aggregated scorecard: each agent played every global game once across shards
    assert detail["scores"]["A"]["overall"]["n"] == 4
    assert detail["scores"]["B"]["overall"]["n"] == 4
    # status rolls up across children (both done -> done)
    assert detail["status"] == "done"


def test_api_run_normal_unchanged(tmp_path, monkeypatch):
    """INV-2: a normal run's /api/runs/{id} response is unaffected by the parent-aggregation path."""
    _sqlite_store(tmp_path, monkeypatch)
    store.create_connected_run({
        "id": "plain_detail", "game": "onuw", "label": "ONUW", "status": "open",
        "n_games": 2, "players": 2, "seed_base": 1,
    })
    _save_child_game("plain_detail", 1)
    _save_child_game("plain_detail", 2)

    with TestClient(app) as client:
        detail = client.get("/api/runs/plain_detail").json()
    assert [g["gid"] for g in detail["games"]] == [1, 2]
    assert detail["scores"]["A"]["overall"]["n"] == 2
    assert detail["status"] == "open"


def test_api_run_parent_game_transcript(tmp_path, monkeypatch):
    """SPEC §6.9(a): a parent's per-game transcript resolves across children (the game lives on a
    child run_id, but the observer asks the parent)."""
    _sqlite_store(tmp_path, monkeypatch)
    _seed_parent_with_children("parent_tx", 2, 2)  # gid 2 lives on shard_1
    with TestClient(app) as client:
        tx = client.get("/api/runs/parent_tx/games/2").json()
    assert tx["winner_team"] == "good"
    assert tx["outcome"]["text"] == "shard game 2"


def test_api_create_run_routes_shards_to_sharded(tmp_path, monkeypatch):
    """SPEC §6.9(a): POST /api/runs with connected + shards>1 creates a parent + K children via
    create_sharded_run; shards<=1 keeps today's single-run behavior (INV-2)."""
    _sqlite_store(tmp_path, monkeypatch)
    with TestClient(app) as client:
        resp = client.post("/api/runs", json={
            "connected": True, "game": "onuw", "players": 5, "games": 6,
            "seed": 42, "run_id": "shardy", "shards": 3,
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
    assert body["run_id"] == "shardy"
    assert body.get("shards") == 3
    parent = store.get_run("shardy")
    assert parent["run_kind"] == "parent" and parent["num_shards"] == 3
    children = store.child_run_ids("shardy")
    assert children == ["shardy_shard_0", "shardy_shard_1", "shardy_shard_2"]
    for k, cid in enumerate(children):
        ch = store.get_run(cid)
        assert ch["run_kind"] == "child" and ch["shard_index"] == k
        assert int(ch["n_games"]) == 6  # GLOBAL N on each child (D8)
    # the parent is never publicly discoverable / joinable (INV-4)
    open_ids = {r["run_id"] for r in store.list_open_runs()}
    assert "shardy" not in open_ids and open_ids.isdisjoint(set(children))


def test_api_create_run_shards_one_is_single_run(tmp_path, monkeypatch):
    """INV-2: shards omitted or 1 -> a plain single connected run, no parent/child rows."""
    _sqlite_store(tmp_path, monkeypatch)
    with TestClient(app) as client:
        resp = client.post("/api/runs", json={
            "connected": True, "game": "onuw", "players": 5, "games": 4,
            "seed": 7, "run_id": "single",
        })
        assert resp.status_code == 200, resp.text
    row = store.get_run("single")
    assert row["run_kind"] == "normal"
    assert row["parent_run_id"] is None and row["num_shards"] is None
    assert store.child_run_ids("single") == []


# --- Step 9 (SPEC §6.9(b)): launcher --shards wiring (light; real bring-up is the Modal smoke) ---


def test_launcher_shards_creates_parent_and_children(tmp_path, monkeypatch):
    """SPEC §6.9(b): tools/coding_agent_run with --shards K calls create_sharded_run with K and the
    K children exist; the per-shard subprocess bring-up is covered by the manual Modal smoke, so we
    stub the spawn/monitor and assert only the orchestration plumbing."""
    _sqlite_store(tmp_path, monkeypatch)
    import tools.coding_agent_run as launcher

    seen = {}
    real_create = launcher.create_sharded_run

    def spy_create(parent_cfg, k):
        seen["k"] = k
        seen["parent_id"] = parent_cfg["id"]
        return real_create(parent_cfg, k)

    monkeypatch.setattr(launcher, "create_sharded_run", spy_create)
    # never touch the network / spawn real subprocesses / block-monitor in a light test; the real
    # cred pre-registration + subprocess bring-up is covered by the manual Modal smoke (V-10).
    monkeypatch.setattr(launcher, "_pre_register_creds", lambda *a, **k: {})
    monkeypatch.setattr(launcher, "_spawn_shard_hosts", lambda *a, **k: [])
    monkeypatch.setattr(launcher, "_monitor_shards", lambda *a, **k: "done")

    rc = launcher.main([
        "--run-id", "L", "--games", "6", "--rounds", "2", "--shards", "3",
        "--server", "http://127.0.0.1:8000", "--ready-timeout", "0.1",
    ])
    assert rc == 0
    assert seen["k"] == 3 and seen["parent_id"] == "L"
    parent = store.get_run("L")
    assert parent["run_kind"] == "parent" and parent["num_shards"] == 3
    assert store.child_run_ids("L") == ["L_shard_0", "L_shard_1", "L_shard_2"]


# --- P1 (INV-4 / SPEC D7): direct signups gated to parents and children -------------------------
#
# The discovery filter only HIDES shard rows from list_open_runs; create_signup must also make them
# UNJOINABLE. Parents are NEVER joinable; children require the per-parent join_token. Normal runs are
# unaffected (token ignored), keeping INV-2 byte-identical.


def _register_agent(agent_id: str = "agent_join", name: str = "Joiner") -> str:
    if store.get_agent(agent_id) is None:
        store.register_agent(name, f"hash_{agent_id}", "arena-agent-v1", "test", agent_id=agent_id)
    return agent_id


def test_signup_to_parent_always_rejected(tmp_path, monkeypatch):
    """INV-4: a parent run is never joinable, even with the correct token."""
    _sqlite_store(tmp_path, monkeypatch)
    create_sharded_run(_base_run("gp", n_games=4, players=3, seed_base=7), 2)
    token = store.get_run("gp")["join_token"]
    assert token  # a token was minted on the parent
    agent_id = _register_agent()

    signup, err = store.create_signup("gp", agent_id)
    assert signup is None and err == "run_not_joinable"
    # even presenting the real token does not make a parent joinable
    signup, err = store.create_signup("gp", agent_id, join_token=token)
    assert signup is None and err == "run_not_joinable"


def test_signup_to_child_requires_token(tmp_path, monkeypatch):
    """INV-4: a child is joinable ONLY with the parent's join_token; absent/wrong -> run_not_joinable."""
    _sqlite_store(tmp_path, monkeypatch)
    children = create_sharded_run(_base_run("gc", n_games=4, players=3, seed_base=7), 2)
    child = children[0]
    token = store.get_run(child)["join_token"]
    assert token
    agent_id = _register_agent()

    # no token -> rejected (today this SUCCEEDS; the bug)
    signup, err = store.create_signup(child, agent_id)
    assert signup is None and err == "run_not_joinable"
    # wrong token -> rejected
    signup, err = store.create_signup(child, agent_id, join_token="not-the-token")
    assert signup is None and err == "run_not_joinable"
    # correct token -> accepted
    signup, err = store.create_signup(child, agent_id, join_token=token)
    assert err is None and signup is not None
    assert signup["agent_id"] == agent_id


def test_signup_to_normal_run_ignores_token(tmp_path, monkeypatch):
    """INV-2: a normal run signs up identically with or without a token (token ignored)."""
    _sqlite_store(tmp_path, monkeypatch)
    store.create_connected_run({
        "id": "n_notoken", "game": "onuw", "label": "ONUW", "status": "open",
        "n_games": 2, "players": 3, "seed_base": 1,
    })
    store.create_connected_run({
        "id": "n_withtoken", "game": "onuw", "label": "ONUW", "status": "open",
        "n_games": 2, "players": 3, "seed_base": 1,
    })
    a1 = _register_agent("agent_n1", "N1")
    a2 = _register_agent("agent_n2", "N2")

    s_no, err_no = store.create_signup("n_notoken", a1)
    assert err_no is None and s_no is not None
    s_tok, err_tok = store.create_signup("n_withtoken", a2, join_token="irrelevant")
    assert err_tok is None and s_tok is not None
    # byte-identical signup shape (modulo the run/agent/ids that legitimately differ)
    ignore = {"id", "run_id", "agent_id", "created_utc", "updated_utc",
              "waiting_expires_utc", "ready_deadline_utc"}
    assert {k: v for k, v in s_no.items() if k not in ignore} == \
           {k: v for k, v in s_tok.items() if k not in ignore}


def test_http_signup_parent_and_tokenless_child_rejected(tmp_path, monkeypatch):
    """INV-4 over HTTP: the signup endpoint rejects a parent and a tokenless child with a 4xx."""
    import hashlib
    _sqlite_store(tmp_path, monkeypatch)
    children = create_sharded_run(_base_run("gh", n_games=4, players=3, seed_base=7), 2)
    child = children[0]
    raw_token = "pa_live_test_join_secret"
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    store.register_agent("HJoiner", token_hash, "arena-agent-v1", "test", agent_id="agent_hjoin")

    with TestClient(app) as client:
        headers = {"Authorization": f"Bearer {raw_token}"}
        # parent: rejected
        resp = client.post("/api/runs/gh/signups",
                           json={"protocol_version": "arena-agent-v1"}, headers=headers)
        assert resp.status_code >= 400, resp.text
        # child without token: rejected
        resp = client.post(f"/api/runs/{child}/signups",
                           json={"protocol_version": "arena-agent-v1"}, headers=headers)
        assert resp.status_code >= 400, resp.text


# --- P2 (codex): parent observer aggregation — the parent renders as ONE normal run (D7) -------
#
# GET /api/runs/{parent} and the /api/runs index must source agents/roster, connected, wins/
# teamSplit, recent events AND status from the CHILDREN, not the (empty) parent row. Children remain
# hidden from the index, and normal/child responses stay byte-identical (INV-2).


def _seed_parent_children_signups_events(parent_id: str, k: int, games_per_child: int) -> list[str]:
    """Seed a parent + K children with saved games, child SIGNUPS (shared creds across shards), and
    child run events — i.e. everything a real running sharded run accrues on the children."""
    join_token = "tok_" + parent_id
    store.create_connected_run({
        "id": parent_id, "game": "onuw", "label": "ONUW", "status": "open",
        "n_games": k * games_per_child, "players": 2, "seed_base": 7,
        "created": "2026-06-25 00:00", "created_utc": "2026-06-25T00:00:00Z",
        "run_kind": "parent", "num_shards": k, "join_token": join_token,
    })
    # one shared identity per logical competitor, signed up on every child (REQ-5 / D9)
    creds = [("A", "agent_A"), ("B", "agent_B")]
    for name, agent_id in creds:
        if store.get_agent(agent_id) is None:
            store.register_agent(name, f"hash_{agent_id}", "arena-agent-v1", "test", agent_id=agent_id)
    child_ids = []
    for shard in range(k):
        cid = f"{parent_id}_shard_{shard}"
        store.create_connected_run({
            "id": cid, "game": "onuw", "label": "ONUW", "status": "open",
            "n_games": k * games_per_child, "players": 2, "seed_base": 7,
            "created": "2026-06-25 00:00", "created_utc": "2026-06-25T00:00:00Z",
            "run_kind": "child", "parent_run_id": parent_id,
            "shard_index": shard, "num_shards": k, "join_token": join_token,
        })
        # sign up the shared identities WHILE the child is open, then add events + games + finish.
        for name, agent_id in creds:
            signup, err = store.create_signup(cid, agent_id, join_token=join_token)
            assert err is None, err
            store.mark_signup_ready(signup["id"], agent_id)
        store.append_event(cid, "round_started", {"shard": cid})
        for i in range(games_per_child):
            gid = shard + 1 + i * k  # global gids striped to this shard
            _save_child_game(cid, gid)
        store.save_run({**store.get_run(cid), "status": "done"})
        child_ids.append(cid)
    return child_ids


def test_api_run_parent_aggregates_signups_events_status(tmp_path, monkeypatch):
    """P2(i) / D7: GET /api/runs/{parent} sources agents, connected, teamSplit, recentEvents and
    status from the CHILDREN — not the empty parent row. RED today: agents=[], connected=False,
    teamSplit 0-0, no events."""
    _sqlite_store(tmp_path, monkeypatch)
    children = _seed_parent_children_signups_events("parent_full", 2, 2)  # 4 good wins total

    with TestClient(app) as client:
        detail = client.get("/api/runs/parent_full").json()

    # roster derived from a child's signups (the identities are shared across shards)
    names = {a["name"] for a in detail["agents"]}
    assert names == {"A", "B"}, detail["agents"]
    # connected flag reflects that children have signups
    assert detail["connected"] is True
    # teamSplit aggregates the children's game outcomes (4 good wins, 0 evil), not 0-0
    assert detail["teamSplit"] == {"good": 4, "evil": 0}, detail["teamSplit"]
    # recent events are the union across children, not the parent's empty stream
    assert len(detail["recentEvents"]) >= 2
    # status rolls up across children (both done -> done)
    assert detail["status"] == "done"
    # the games union is still present and ordered
    assert [g["gid"] for g in detail["games"]] == [1, 2, 3, 4]


def test_api_runs_index_parent_rollup_status_and_split(tmp_path, monkeypatch):
    """P2(ii) / D7: the /api/runs index row for a parent shows the ROLLED-UP status (not its own
    stale 'open') and the aggregated teamSplit (not 0-0). RED today: status 'open', split 0-0."""
    _sqlite_store(tmp_path, monkeypatch)
    _seed_parent_children_signups_events("parent_idx_full", 2, 2)  # children done; 4 good wins

    with TestClient(app) as client:
        rows = client.get("/api/runs").json()
    by_id = {r["id"]: r for r in rows}
    assert "parent_idx_full" in by_id
    prow = by_id["parent_idx_full"]
    assert prow["status"] == "done", prow
    assert prow["teamSplit"] == {"good": 4, "evil": 0}, prow["teamSplit"]


def test_api_runs_index_normal_row_unchanged(tmp_path, monkeypatch):
    """INV-2: the index row for a normal run is unaffected by the parent-rollup path."""
    _sqlite_store(tmp_path, monkeypatch)
    store.create_connected_run({
        "id": "plain_idx_row", "game": "onuw", "label": "ONUW", "status": "open",
        "n_games": 2, "players": 2, "seed_base": 1,
    })
    _save_child_game("plain_idx_row", 1)  # one good win

    with TestClient(app) as client:
        rows = client.get("/api/runs").json()
    prow = {r["id"]: r for r in rows}["plain_idx_row"]
    assert prow["status"] == "open"
    assert prow["teamSplit"] == {"good": 1, "evil": 0}


def test_create_connected_run_shards_response_returns_usable_join_token(tmp_path, monkeypatch):
    """codex P1 / SPEC D7: POSTing a connected creation with shards>1 must hand back the
    per-parent join token so an API-driven orchestrator can sign agents into the child shards.
    The returned token must (a) match the secret on the child rows and (b) actually authorize a
    child signup, while a wrong/absent token is rejected."""
    _sqlite_store(tmp_path, monkeypatch)
    with TestClient(app) as client:
        created = client.post("/api/runs", json={
            "connected": True, "run_id": "run_shards_tok", "game": "onuw",
            "players": 5, "games": 4, "seed": 4242, "shards": 2,
        })
        assert created.status_code == 200, created.text
        body = created.json()
        assert body["shards"] == 2
        child_ids = body["child_run_ids"]
        assert len(child_ids) == 2
        # the response carries the join token
        token = body["join_token"]
        assert token
        # it matches the secret actually stored on every child row (and the parent)
        for cid in child_ids:
            assert store.get_run(cid)["join_token"] == token
        assert store.get_run("run_shards_tok")["join_token"] == token

        # the returned token authorizes a child signup; a wrong/absent token is rejected
        store.register_agent("Shard Agent", "hash_shard_agent", "arena-agent-v1", "test",
                             agent_id="agent_shard")
        child = child_ids[0]
        bad, err_bad = store.create_signup(child, "agent_shard", join_token="not-the-token")
        assert bad is None and err_bad == "run_not_joinable"
        missing, err_missing = store.create_signup(child, "agent_shard")
        assert missing is None and err_missing == "run_not_joinable"
        ok, err_ok = store.create_signup(child, "agent_shard", join_token=token)
        assert ok is not None and err_ok is None


def test_create_connected_run_normal_response_unchanged(tmp_path, monkeypatch):
    """INV-2: a normal (shards=1 / absent) connected creation response is byte-identical and must
    NOT leak any join_token / shard fields."""
    _sqlite_store(tmp_path, monkeypatch)
    with TestClient(app) as client:
        absent = client.post("/api/runs", json={
            "connected": True, "run_id": "run_plain_noshard", "game": "onuw",
            "players": 5, "games": 4, "seed": 4242,
        })
        one = client.post("/api/runs", json={
            "connected": True, "run_id": "run_plain_oneshard", "game": "onuw",
            "players": 5, "games": 4, "seed": 4242, "shards": 1,
        })
    assert absent.status_code == 200 and one.status_code == 200
    for body in (absent.json(), one.json()):
        assert "join_token" not in body
        assert "shards" not in body
        assert "child_run_ids" not in body
        assert set(body) == {"run_id", "status", "game", "games", "players", "rounds",
                             "deck_preset", "run_config"}


def _ready_seated_run(run_id: str, creds_with_seats):
    """Sign up identities in the given arrival order, each with an explicit seat (roster_index),
    then ready them so _maybe_ready_required assigns seats and rebuilds the persisted roster.

    `creds_with_seats` is an ordered list of (display_name, agent_id, seat); arrival order is the
    list order, the explicit seat is honored deterministically (SPEC D5/REQ-7)."""
    agent_by_signup: dict[str, str] = {}
    for name, agent_id, seat in creds_with_seats:
        if store.get_agent(agent_id) is None:
            store.register_agent(name, f"hash_{agent_id}", "arena-agent-v1", "test", agent_id=agent_id)
        signup, err = store.create_signup(run_id, agent_id, seat=seat)
        assert err is None, err
        agent_by_signup[signup["id"]] = agent_id
    for signup_id, agent_id in agent_by_signup.items():
        _, err = store.mark_signup_ready(signup_id, agent_id)
        assert err is None or err == "not_ready_required"
    return agent_by_signup


def test_persisted_roster_ordered_by_assigned_seat(tmp_path, monkeypatch):
    """Codex round-5 / FINDING #1 (CORRECTNESS): when explicit seats differ from arrival order,
    the persisted runs.agents_json (get_run -> agents) must be ordered by ASSIGNED SEAT, so
    run.agents[seat] is the agent seated there. Previously _refresh_run_roster rebuilt the roster in
    arrival order, mis-attributing agents to seats in detail/push/import paths."""
    _sqlite_store(tmp_path, monkeypatch)
    store.save_run(_base_run("seatorder", players=3))
    # arrival order A, B, C but explicit seats reversed: A->2, B->1, C->0.
    _ready_seated_run("seatorder", [("A", "agent_A", 2), ("B", "agent_B", 1), ("C", "agent_C", 0)])

    agents = store.get_run("seatorder")["agents"]
    names = [a["name"] for a in agents]
    assert names == ["C", "B", "A"], f"roster must be seat-ordered, got {names}"
    assert agents[0]["name"] == "C" and agents[2]["name"] == "A"


def test_persisted_roster_arrival_order_when_no_explicit_seats(tmp_path, monkeypatch):
    """INV-2: with NO explicit seats, seats are assigned in arrival order, so the seat-ordered roster
    equals the arrival-order roster — byte-identical to pre-sharding behavior."""
    _sqlite_store(tmp_path, monkeypatch)
    store.save_run(_base_run("arrivalorder", players=3))
    _seat_run("arrivalorder", [("A", "agent_A"), ("B", "agent_B"), ("C", "agent_C")])

    names = [a["name"] for a in store.get_run("arrivalorder")["agents"]]
    assert names == ["A", "B", "C"], f"arrival-order roster expected, got {names}"


def test_create_sharded_run_retry_reuses_join_token(tmp_path, monkeypatch):
    """Codex round-4: retrying create_sharded_run for an existing parent must REUSE the persisted
    join token, not mint a fresh one. save_run upserts join_token=COALESCE(excluded, existing), so a
    new token would ROTATE the secret out from under any token an orchestrator already handed to
    agents (they'd then be rejected 403 run_not_joinable)."""
    _sqlite_store(tmp_path, monkeypatch)
    parent_cfg = _base_run("retry", n_games=5, players=5, seed_base=7)

    create_sharded_run(parent_cfg, 2)
    token1 = store.get_run("retry")["join_token"]
    assert token1 and store.get_run("retry_shard_0")["join_token"] == token1

    create_sharded_run(parent_cfg, 2)  # retry, e.g. after a partial write
    assert store.get_run("retry")["join_token"] == token1, "retry must not rotate the join token"
    tokens = {store.get_run(r)["join_token"] for r in ("retry", "retry_shard_0", "retry_shard_1")}
    assert tokens == {token1}


def test_create_sharded_run_shard_count_change_rejected(tmp_path, monkeypatch):
    """Codex round-5 / FINDING #2 (CORRECTNESS): re-creating an existing parent with a DIFFERENT
    num_shards must be REJECTED with a ValueError — shard-count changes are not supported. Today the
    mismatch is silently accepted: shrinking K only upserts the new range and leaves stale orphan
    children (e.g. parent_shard_2) still pointing at the parent, so child_run_ids() and every parent
    rollup/score keep counting them."""
    _sqlite_store(tmp_path, monkeypatch)
    parent_cfg = _base_run("scchange", n_games=6, players=3, seed_base=7)

    create_sharded_run(parent_cfg, 3)
    assert store.child_run_ids("scchange") == [
        "scchange_shard_0", "scchange_shard_1", "scchange_shard_2"]

    # shrinking K must raise, not silently leave scchange_shard_2 orphaned
    import pytest
    with pytest.raises(ValueError, match="shard"):
        create_sharded_run(parent_cfg, 2)
    # growing K is likewise rejected
    with pytest.raises(ValueError, match="shard"):
        create_sharded_run(parent_cfg, 1)

    # the original 3 children are untouched (no rows deleted, no orphans created)
    assert store.child_run_ids("scchange") == [
        "scchange_shard_0", "scchange_shard_1", "scchange_shard_2"]
    assert store.get_run("scchange")["num_shards"] == 3


def test_create_sharded_run_same_count_recreate_idempotent(tmp_path, monkeypatch):
    """The same-K re-create must stay idempotent (the round-4 retry test relies on it): re-creating
    with the SAME num_shards re-upserts the same children, leaves no orphan rows, and child_run_ids
    is exactly the K expected ids."""
    _sqlite_store(tmp_path, monkeypatch)
    parent_cfg = _base_run("samek", n_games=4, players=3, seed_base=7)

    first = create_sharded_run(parent_cfg, 2)
    second = create_sharded_run(parent_cfg, 2)
    assert first == second == ["samek_shard_0", "samek_shard_1"]
    assert store.child_run_ids("samek") == ["samek_shard_0", "samek_shard_1"]
    assert store.get_run("samek")["num_shards"] == 2


def test_create_sharded_run_same_count_changed_config_rejected(tmp_path, monkeypatch):
    """Codex round-8 / FINDING P2 (DATA-INTEGRITY): a same-K re-create of an existing parent is
    treated as an idempotent retry, but today it does NOT validate the OTHER immutable run fields.
    If n_games / seed_base / players / game / deck_preset differ from the existing parent, the
    upserts rewrite parent+child metadata while old games/signups/events stay under the same child
    run ids (save_game is first-write-wins by (run_id, gid)) — a reused --run-id can then silently
    serve STALE transcripts/scores under NEW settings. A same-K re-create whose immutable config
    differs must raise a clear ValueError; a truly identical same-K retry stays idempotent."""
    import pytest
    _sqlite_store(tmp_path, monkeypatch)
    base = _base_run("cfgchg", n_games=6, players=3, seed_base=7,
                     deck_preset="standard")
    create_sharded_run(base, 2)
    orig_token = store.get_run("cfgchg")["join_token"]

    # changed n_games (same K) -> reject
    with pytest.raises(ValueError, match="config"):
        create_sharded_run(
            _base_run("cfgchg", n_games=8, players=3, seed_base=7,
                      deck_preset="standard"), 2)
    # changed seed_base (same K) -> reject
    with pytest.raises(ValueError, match="config"):
        create_sharded_run(
            _base_run("cfgchg", n_games=6, players=3, seed_base=99,
                      deck_preset="standard"), 2)
    # changed players (same K) -> reject
    with pytest.raises(ValueError, match="config"):
        create_sharded_run(
            _base_run("cfgchg", n_games=6, players=5, seed_base=7,
                      deck_preset="standard"), 2)
    # changed game (same K) -> reject
    with pytest.raises(ValueError, match="config"):
        create_sharded_run(
            _base_run("cfgchg", game="werewolf", n_games=6, players=3,
                      seed_base=7, deck_preset="standard"), 2)
    # changed deck_preset (same K) -> reject
    with pytest.raises(ValueError, match="config"):
        create_sharded_run(
            _base_run("cfgchg", n_games=6, players=3, seed_base=7,
                      deck_preset="alt"), 2)

    # the existing parent is untouched (no clobber, token preserved)
    parent = store.get_run("cfgchg")
    assert int(parent["n_games"]) == 6
    assert int(parent["players"]) == 3
    assert int(parent["seed_base"]) == 7
    assert parent["game"] == "onuw"
    assert parent["deck_preset"] == "standard"
    assert parent["join_token"] == orig_token
    assert store.child_run_ids("cfgchg") == ["cfgchg_shard_0", "cfgchg_shard_1"]

    # an identical-config same-K retry stays idempotent (returns same ids, reuses token)
    again = create_sharded_run(
        _base_run("cfgchg", n_games=6, players=3, seed_base=7,
                  deck_preset="standard"), 2)
    assert again == ["cfgchg_shard_0", "cfgchg_shard_1"]
    assert store.get_run("cfgchg")["join_token"] == orig_token


def test_create_sharded_run_same_count_changed_metadata_rejected(tmp_path, monkeypatch):
    """Codex round-9 / FINDING P2: the round-8 same-K immutable-field guard validates
    n_games/seed_base/players/game/deck_preset but NOT metadata (the run_config). A same-K
    re-create with identical numerics/game/deck but DIFFERENT metadata.run_config (e.g. rounds,
    temperature, prior_message_turns) leaves the mismatch set empty, so the upserts overwrite
    metadata_json while old games/signups stay under the SAME child ids — a reused run id serves
    old transcripts/scores under NEW run settings. The metadata must be part of the same-K
    immutable-field validation: a changed-metadata re-create raises 'changed config'; a truly
    identical-metadata same-K retry stays idempotent (same child ids, token reused)."""
    import pytest
    _sqlite_store(tmp_path, monkeypatch)
    base = _base_run("metachg", n_games=6, players=3, seed_base=7,
                     deck_preset="standard",
                     metadata={"run_config": {"rounds": 5, "temperature": 0.7,
                                              "prior_message_turns": 2}})
    create_sharded_run(base, 2)
    orig_token = store.get_run("metachg")["join_token"]

    # changed metadata.run_config (rounds 5 -> 9), everything else identical -> reject
    with pytest.raises(ValueError, match="config"):
        create_sharded_run(
            _base_run("metachg", n_games=6, players=3, seed_base=7,
                      deck_preset="standard",
                      metadata={"run_config": {"rounds": 9, "temperature": 0.7,
                                               "prior_message_turns": 2}}), 2)

    # the existing parent is untouched (metadata preserved, token preserved)
    parent = store.get_run("metachg")
    assert parent["metadata"] == {"run_config": {"rounds": 5, "temperature": 0.7,
                                                 "prior_message_turns": 2}}
    assert parent["join_token"] == orig_token
    assert store.child_run_ids("metachg") == ["metachg_shard_0", "metachg_shard_1"]

    # an identical-metadata same-K retry stays idempotent (same ids, token reused)
    again = create_sharded_run(
        _base_run("metachg", n_games=6, players=3, seed_base=7,
                  deck_preset="standard",
                  metadata={"run_config": {"rounds": 5, "temperature": 0.7,
                                           "prior_message_turns": 2}}), 2)
    assert again == ["metachg_shard_0", "metachg_shard_1"]
    assert store.get_run("metachg")["join_token"] == orig_token
    assert store.get_run("metachg")["metadata"] == {
        "run_config": {"rounds": 5, "temperature": 0.7, "prior_message_turns": 2}}


def test_create_sharded_run_rejects_reused_normal_run_id(tmp_path, monkeypatch):
    """Codex round-6 / FINDING #1 (DATA-INTEGRITY): create_sharded_run must REJECT a parent_id that
    already belongs to a NORMAL run. Today existing_k (num_shards) is None for a normal run, so the
    shard-count guard is skipped and save_run rewrites that normal run as a sharded parent — hiding
    the original run's games/signups (parent read paths source only child runs). A reused --run-id or
    payload run_id triggers this silent clobber; it must instead raise a clear ValueError and leave
    the existing normal run's row untouched."""
    import pytest
    _sqlite_store(tmp_path, monkeypatch)
    # an existing NORMAL run that an orchestrator must not silently clobber
    store.save_run(_base_run("collide", n_games=4, players=3, seed_base=7))
    assert store.get_run("collide")["run_kind"] == "normal"

    parent_cfg = _base_run("collide", n_games=4, players=3, seed_base=7)
    with pytest.raises(ValueError, match="(?i)in use|not a parent"):
        create_sharded_run(parent_cfg, 2)

    # the original normal run is unchanged — still a normal run, no parent rewrite, no children
    row = store.get_run("collide")
    assert row["run_kind"] == "normal", f"normal run was clobbered: {row['run_kind']}"
    assert row["num_shards"] is None
    assert store.child_run_ids("collide") == []

    # a fresh id still works
    fresh = create_sharded_run(_base_run("fresh", n_games=4, players=3, seed_base=7), 2)
    assert fresh == ["fresh_shard_0", "fresh_shard_1"]
    assert store.get_run("fresh")["run_kind"] == "parent"

    # a same-K parent re-create stays idempotent (the round-4/round-5 retry path)
    again = create_sharded_run(_base_run("fresh", n_games=4, players=3, seed_base=7), 2)
    assert again == fresh
    assert store.get_run("fresh")["num_shards"] == 2


def test_create_sharded_run_rejects_reused_child_run_id(tmp_path, monkeypatch):
    """Codex round-6 / FINDING #1 (DATA-INTEGRITY): the same guard must reject a parent_id that is
    already a CHILD shard of some other parent — it is not a parent, so it must not be rewritten."""
    import pytest
    _sqlite_store(tmp_path, monkeypatch)
    create_sharded_run(_base_run("realparent", n_games=4, players=3, seed_base=7), 2)
    assert store.get_run("realparent_shard_0")["run_kind"] == "child"

    with pytest.raises(ValueError, match="(?i)in use|not a parent"):
        create_sharded_run(_base_run("realparent_shard_0", n_games=4, players=3, seed_base=7), 2)
    # the child row is unchanged
    child = store.get_run("realparent_shard_0")
    assert child["run_kind"] == "child"
    assert child["parent_run_id"] == "realparent"


def test_create_sharded_run_rejects_derived_child_id_colliding_normal_run(tmp_path, monkeypatch):
    """Codex round-7 / FINDING #1 (DATA-INTEGRITY): round-6 guarded the PARENT id, but the DERIVED
    child ids ({parent}_shard_k) are still upserted blindly via create_connected_run. If a NORMAL run
    already exists named like "c_shard_0", creating sharded parent "c" rewrites that existing row into
    a child of the new parent — hiding/corrupting the old run and folding its games into the new
    parent's rollup. create_sharded_run must validate each derived child id is absent OR already a
    child of THIS parent, and otherwise raise a clear ValueError and write NOTHING new."""
    import pytest
    _sqlite_store(tmp_path, monkeypatch)
    # a pre-existing NORMAL run that happens to be named like a derived child id of parent "c"
    store.save_run(_base_run("c_shard_0", n_games=4, players=3, seed_base=7))
    assert store.get_run("c_shard_0")["run_kind"] == "normal"

    parent_cfg = _base_run("c", n_games=4, players=3, seed_base=7)
    with pytest.raises(ValueError, match="(?i)shard|in use|collid"):
        create_sharded_run(parent_cfg, 2)

    # the colliding normal run is NOT rewritten into a child; the parent "c" was not created either
    row = store.get_run("c_shard_0")
    assert row["run_kind"] == "normal", f"normal run was clobbered: {row['run_kind']}"
    assert row["parent_run_id"] is None
    assert row["num_shards"] is None
    # nothing new was written: no parent row, no other child
    assert store.get_run("c") is None
    assert store.get_run("c_shard_1") is None
    assert store.child_run_ids("c") == []


def test_create_sharded_run_rejects_derived_child_id_owned_by_other_parent(tmp_path, monkeypatch):
    """Codex round-7 / FINDING #1: a derived child id ({parent}_shard_k) that is already a CHILD of a
    DIFFERENT parent must also be rejected — creating it would steal/re-home that child."""
    import pytest
    _sqlite_store(tmp_path, monkeypatch)
    # parent "other" owns "other_shard_0"; now someone tries parent id that derives "other_shard_0"?
    # Construct a parent whose derived child id collides with another parent's existing child.
    create_sharded_run(_base_run("p1", n_games=4, players=3, seed_base=7), 2)  # p1_shard_0, p1_shard_1
    # pre-create a stray child owned by a different parent, named like p2's derived child
    store.save_run(_base_run(
        "p2_shard_0", run_kind="child", parent_run_id="someone_else",
        shard_index=0, num_shards=2))
    assert store.get_run("p2_shard_0")["parent_run_id"] == "someone_else"

    with pytest.raises(ValueError, match="(?i)shard|in use|collid"):
        create_sharded_run(_base_run("p2", n_games=4, players=3, seed_base=7), 2)
    # the stray child is unchanged and no new p2 parent/children were written
    assert store.get_run("p2_shard_0")["parent_run_id"] == "someone_else"
    assert store.get_run("p2") is None
    assert store.get_run("p2_shard_1") is None


def test_create_sharded_run_clean_children_and_same_k_recreate_still_pass(tmp_path, monkeypatch):
    """Codex round-7 / FINDING #1: the new derived-child guard must NOT break the happy paths — a
    clean parent id with no colliding children works, and a same-K re-create (children already belong
    to THIS parent) stays idempotent."""
    _sqlite_store(tmp_path, monkeypatch)
    clean = create_sharded_run(_base_run("clean", n_games=4, players=3, seed_base=7), 2)
    assert clean == ["clean_shard_0", "clean_shard_1"]
    assert store.get_run("clean_shard_0")["parent_run_id"] == "clean"

    # same-K re-create: children already belong to THIS parent -> idempotent, still passes
    again = create_sharded_run(_base_run("clean", n_games=4, players=3, seed_base=7), 2)
    assert again == clean
    assert store.child_run_ids("clean") == ["clean_shard_0", "clean_shard_1"]


def test_create_signup_rejects_out_of_range_seat(tmp_path, monkeypatch):
    """Codex round-5 / FINDING #3 (CORRECTNESS): an explicit seat must be bounds-checked against
    run.players BEFORE insert. A seat outside [0, players) (or a non-integer) can never be assigned
    by _maybe_ready_required (which only seats roster_index < players), so the signup would sit
    unseated forever and wedge the shard in waiting/ready_required. create_signup must instead reject
    it with the 'invalid_seat' reason and insert nothing."""
    _sqlite_store(tmp_path, monkeypatch)
    store.save_run(_base_run("seatbounds", players=5))
    store.register_agent("solo", "hash_solo", "arena-agent-v1", "test", agent_id="agent_solo")

    # too-large seat (>= players)
    sg, err = store.create_signup("seatbounds", "agent_solo", seat=99)
    assert sg is None and err == "invalid_seat", (sg, err)
    # negative seat
    sg, err = store.create_signup("seatbounds", "agent_solo", seat=-1)
    assert sg is None and err == "invalid_seat", (sg, err)
    # exactly players is out of range (seats are 0..players-1)
    sg, err = store.create_signup("seatbounds", "agent_solo", seat=5)
    assert sg is None and err == "invalid_seat", (sg, err)
    # non-integer seat
    sg, err = store.create_signup("seatbounds", "agent_solo", seat="x")
    assert sg is None and err == "invalid_seat", (sg, err)

    # nothing was inserted by any of the rejected attempts
    assert store._active_signups.__module__  # sanity: helper exists
    with store.conn() as c:
        ph = store._ph()
        rows = c.execute(
            f"SELECT COUNT(*) AS n FROM run_signups WHERE run_id={ph}", ("seatbounds",)
        ).fetchone()
        assert rows["n"] == 0, "rejected signups must not insert a row"


def test_create_signup_accepts_in_range_seat_and_no_seat(tmp_path, monkeypatch):
    """In-range explicit seats are accepted; a signup with NO seat (normal runs) is unaffected
    (INV-2)."""
    _sqlite_store(tmp_path, monkeypatch)
    store.save_run(_base_run("seatok", players=5))
    store.register_agent("a0", "hash_a0", "arena-agent-v1", "test", agent_id="agent_0")
    store.register_agent("a1", "hash_a1", "arena-agent-v1", "test", agent_id="agent_1")

    sg, err = store.create_signup("seatok", "agent_0", seat=0)
    assert err is None and sg is not None
    sg, err = store.create_signup("seatok", "agent_1")  # no seat: INV-2
    assert err is None and sg is not None


def test_create_signup_rejects_duplicate_seat(tmp_path, monkeypatch):
    """Codex round-6 / FINDING #3 (CORRECTNESS): two ACTIVE signups in the same run must not be able
    to request the SAME explicit seat. create_signup bounds-checks an explicit seat but did NOT check
    uniqueness, so two agents could both claim seat 0; _maybe_ready_required would then assign
    duplicate seats and corrupt the deterministic identity->seat contract (SPEC D5/V-7). The second
    same-seat request must be rejected with 'invalid_seat' and insert nothing."""
    _sqlite_store(tmp_path, monkeypatch)
    store.save_run(_base_run("seatdup", players=5))
    store.register_agent("a0", "hash_a0", "arena-agent-v1", "test", agent_id="agent_0")
    store.register_agent("a1", "hash_a1", "arena-agent-v1", "test", agent_id="agent_1")
    store.register_agent("a2", "hash_a2", "arena-agent-v1", "test", agent_id="agent_2")

    sg, err = store.create_signup("seatdup", "agent_0", seat=1)
    assert err is None and sg is not None, (sg, err)

    # a different agent requesting the SAME seat must be rejected
    sg, err = store.create_signup("seatdup", "agent_1", seat=1)
    assert sg is None and err == "invalid_seat", (sg, err)

    # a distinct seat is still fine
    sg, err = store.create_signup("seatdup", "agent_2", seat=2)
    assert err is None and sg is not None, (sg, err)

    # the duplicate attempt inserted nothing: exactly two signups exist (seats 1 and 2)
    with store.conn() as c:
        ph = store._ph()
        rows = c.execute(
            f"SELECT roster_index FROM run_signups WHERE run_id={ph} ORDER BY roster_index",
            ("seatdup",),
        ).fetchall()
        assert [r["roster_index"] for r in rows] == [1, 2], rows


def test_create_signup_rejects_bool_seat(tmp_path, monkeypatch):
    """Codex round-6 / FINDING #3: a JSON boolean (True/False) is an int subclass in Python, so it
    would silently coerce to seat 1/0. An explicit boolean seat must be rejected as 'invalid_seat'."""
    _sqlite_store(tmp_path, monkeypatch)
    store.save_run(_base_run("seatbool", players=5))
    store.register_agent("a0", "hash_a0", "arena-agent-v1", "test", agent_id="agent_0")

    sg, err = store.create_signup("seatbool", "agent_0", seat=True)
    assert sg is None and err == "invalid_seat", (sg, err)
    sg, err = store.create_signup("seatbool", "agent_0", seat=False)
    assert sg is None and err == "invalid_seat", (sg, err)
    sg, err = store.create_signup("seatbool", "agent_0", seat=1.7)
    assert sg is None and err == "invalid_seat", (sg, err)

    with store.conn() as c:
        ph = store._ph()
        n = c.execute(
            f"SELECT COUNT(*) AS n FROM run_signups WHERE run_id={ph}", ("seatbool",)
        ).fetchone()
        assert n["n"] == 0, "rejected bool/float seats must not insert a row"


def test_signup_endpoint_malformed_seat_is_400_not_500(tmp_path, monkeypatch):
    """Codex round-6 / FINDING #2: api_signup_run must NOT int()-coerce the raw seat before store
    validation. A non-integer seat ("x") previously raised ValueError -> HTTP 500; a JSON
    boolean/float was silently coerced. All malformed explicit seats must surface as HTTP 400
    'invalid_seat' through the store's validation path."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("INGEST_TOKENS", raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "seat_malformed.db")
    with TestClient(app) as client:
        created = client.post("/api/runs", json={
            "connected": True, "run_id": "run_seatbad", "game": "onuw",
            "players": 5, "games": 1, "seed": 9101,
        })
        assert created.status_code == 200, created.text
        reg = client.post("/api/agents/register", json={
            "display_name": "omega", "protocol_version": "arena-agent-v1",
            "harness": "test", "public_key": "pk_omega",
        })
        assert reg.status_code == 200, reg.text
        token = reg.json()["agent_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # non-integer string seat: must be 400 (was 500 via int("x"))
        bad = client.post("/api/runs/run_seatbad/signups", json={
            "protocol_version": "arena-agent-v1", "seat": "x",
        }, headers=headers)
        assert bad.status_code == 400, bad.text
        assert "invalid_seat" in bad.text

        # JSON boolean: must be 400 (was silently coerced to seat 1)
        boolbad = client.post("/api/runs/run_seatbad/signups", json={
            "protocol_version": "arena-agent-v1", "seat": True,
        }, headers=headers)
        assert boolbad.status_code == 400, boolbad.text
        assert "invalid_seat" in boolbad.text


def test_signup_endpoint_rejects_duplicate_seat(tmp_path, monkeypatch):
    """Codex round-6 / FINDING #3: two agents requesting the SAME explicit seat in one run -> the
    second must get HTTP 400 'invalid_seat'."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("INGEST_TOKENS", raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "seat_dup_ep.db")
    with TestClient(app) as client:
        created = client.post("/api/runs", json={
            "connected": True, "run_id": "run_seatdupep", "game": "onuw",
            "players": 5, "games": 1, "seed": 9102,
        })
        assert created.status_code == 200, created.text
        tokens = []
        for name in ("dup_a", "dup_b"):
            reg = client.post("/api/agents/register", json={
                "display_name": name, "protocol_version": "arena-agent-v1",
                "harness": "test", "public_key": f"pk_{name}",
            })
            assert reg.status_code == 200, reg.text
            tokens.append(reg.json()["agent_token"])

        first = client.post("/api/runs/run_seatdupep/signups", json={
            "protocol_version": "arena-agent-v1", "seat": 3,
        }, headers={"Authorization": f"Bearer {tokens[0]}"})
        assert first.status_code == 200, first.text

        dup = client.post("/api/runs/run_seatdupep/signups", json={
            "protocol_version": "arena-agent-v1", "seat": 3,
        }, headers={"Authorization": f"Bearer {tokens[1]}"})
        assert dup.status_code == 400, dup.text
        assert "invalid_seat" in dup.text


def test_signup_endpoint_rejects_out_of_range_seat(tmp_path, monkeypatch):
    """The /api/runs/{id}/signups endpoint must surface 'invalid_seat' as HTTP 400 (server maps the
    new store reason). An in-range seat is accepted (200)."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("INGEST_TOKENS", raising=False)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "seat_endpoint.db")
    with TestClient(app) as client:
        created = client.post("/api/runs", json={
            "connected": True, "run_id": "run_seatep", "game": "onuw",
            "players": 5, "games": 1, "seed": 9100,
        })
        assert created.status_code == 200, created.text
        reg = client.post("/api/agents/register", json={
            "display_name": "zeta", "protocol_version": "arena-agent-v1",
            "harness": "test", "public_key": "pk_zeta",
        })
        assert reg.status_code == 200, reg.text
        token = reg.json()["agent_token"]
        headers = {"Authorization": f"Bearer {token}"}

        bad = client.post("/api/runs/run_seatep/signups", json={
            "protocol_version": "arena-agent-v1", "seat": 99,
        }, headers=headers)
        assert bad.status_code == 400, bad.text
        assert "invalid_seat" in bad.text

        ok = client.post("/api/runs/run_seatep/signups", json={
            "protocol_version": "arena-agent-v1", "seat": 0,
        }, headers=headers)
        assert ok.status_code == 200, ok.text
