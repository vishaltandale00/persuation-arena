"""End-to-end smoke test for the Modal per-run coordinator (arena/modal_app.py).

Spawns a per-run Modal container, waits for its Tunnel URL, connects N scripted agents directly to it,
lets the game(s) play, and reports the container's result. Uses the FREE scripted random_agent, so it
proves the TOPOLOGY — per-run server + tunnel + agent connectivity + coordination + Neon persistence —
with no LLM cost.

The Modal container does all Neon writes (creates the run, serves, coordinates, persists); this process
only spawns + drives agents, so it never queries prod Neon directly.

Prereqs:
  .venv/bin/modal deploy arena/modal_app.py
  modal secret `neon-database-url` holding DATABASE_URL
Run (single coordinator):
  PYTHONPATH=. .venv/bin/python tools/connected_modal_smoke.py --games 1 --rounds 2

V-10 — manual parallel-shards gate (SPEC-parallel-shards.md REQ-10). Spawns K real coordinator
containers and K shard host bundles (process per shard, D9), with each agent identity a SINGLE
rating competitor across ALL K shards (REQ-5 / D9).

Three correctness requirements this path used to get wrong (now enforced):
  - Shared identity (REQ-5): each shard has a DIFFERENT coordinator URL, and CredentialsStore keys
    profiles by server URL — so reusing one cred file across shards would register a SEPARATE
    agent_id per shard for the same displayed name. We pre-register each identity ONCE against the
    CENTRAL API (`--central-server`, the Neon-backed app the coordinators also write to) and reuse
    that one credential (agent_id) for every shard, independent of the per-shard coordinator URL.
  - Shared store (D1/D8): the parent/child shard rows are written through the LOCAL store, so they
    MUST land in the SAME database the Modal coordinators read. The sharded path requires
    DATABASE_URL to be set (and to match the coordinators' `neon-database-url` secret) and fails
    fast otherwise, instead of silently writing to local SQLite (which would make each coordinator
    replay the FULL schedule and the parent rollup never complete).
  - Join token (INV-4 / D7): create_sharded_run gates each child's /signups behind a per-parent
    `join_token`, so a child returns 403 run_not_joinable to any agent that does not present it. We
    read each child's token from its (shared-Neon) run row and pass it through to that shard's
    agents' signup; otherwise every smoke agent would be rejected.

Requires a deployed Modal app + a shared DATABASE_URL (this process and the Modal side); NOT in CI.
  # deployed Modal + DATABASE_URL set to the SAME Neon as the coordinators' neon-database-url secret
  DATABASE_URL=$NEON_URL PYTHONPATH=. .venv/bin/python tools/connected_modal_smoke.py \
      --shards 2 --games 4 --rounds 2 \
      --central-server https://persuasion-arena--api.modal.run
Assertions to eyeball (REQ-10): both children reach `running` before either reaches `done`
(structural concurrency); final rollup_parent_status == 'done'; the union of games == N; the
observer renders the parent as one run; and each identity has ONE agent_id spanning both shards
(its overall n on the leaderboard equals the global N, not ~N/K).
"""
from __future__ import annotations

import argparse
import os
import tempfile
import threading
import time

import modal

from persuasion_arena_agent.agent import ArenaAgent
from persuasion_arena_agent.client import ArenaHttpClient
from persuasion_arena_agent.credentials import AgentCredentials, CredentialsStore, DEFAULT_SERVER
from examples import random_agent
from arena import store
from arena.modal_app import APP_NAME, coordinator_url
from arena.sharded import create_sharded_run, rollup_parent_status

N_PLAYERS = 5


def _run_agent(name: str, server: str, run_id: str, cred: str | None = None,
               join_token: str | None = None) -> None:
    # cred path must NOT pre-exist (an empty file makes CredentialsStore json.loads("") crash).
    cred = cred or os.path.join(tempfile.mkdtemp(prefix="arena-cred-"), "cred.json")
    try:
        agent = ArenaAgent(name=name, server=server, credentials=CredentialsStore(cred))
        agent.act(random_agent.act)
        # Shard children are created via create_sharded_run with a per-parent join_token, so they
        # gate /signups (INV-4 / SPEC D7): without the token the child returns 403 run_not_joinable.
        # Single-coordinator (run_kind='normal') runs ignore the token (None is fine).
        signup = agent.signup(run_id=run_id, join_token=join_token)
        print(f"  {name}: {signup.status} seat={signup.seat}", flush=True)
        agent.run_forever([signup])
        print(f"  {name}: done", flush=True)
    except Exception as e:
        print(f"  {name}: ERROR {type(e).__name__}: {e}", flush=True)


def _default_register(name: str, server: str) -> AgentCredentials:
    """Register one identity against `server` (the central API) and return its credential."""
    return ArenaHttpClient(server).register_agent(name)


def _shared_identity_creds(names, *, central_server: str, cred_paths: list[str],
                           register=None) -> list[AgentCredentials]:
    """Pre-register each identity ONCE against the CENTRAL API and persist its credential (REQ-5/D9).

    CredentialsStore keys profiles by SERVER URL, and each shard has a DIFFERENT coordinator URL, so
    naively reusing one cred file across shards would register a SEPARATE agent_id per shard for the
    same displayed name (fracturing the rating competitor). Registering once here against the central
    API yields ONE credential (one agent_id) per identity, which `_seed_cred_for_coordinator` then
    reuses for every shard regardless of its coordinator URL — one competitor across all K shards.
    """
    register = register or _default_register  # resolved at call time (so it stays monkeypatchable)
    creds: list[AgentCredentials] = []
    for name, cred_path in zip(names, cred_paths):
        c = register(name, central_server)
        # Persist under the CENTRAL server key so a re-run reuses the same agent_id idempotently.
        store = CredentialsStore(cred_path)
        store.save(c)
        creds.append(c)
    return creds


def _seed_cred_for_coordinator(creds: AgentCredentials, coordinator_url: str, cred_path: str) -> None:
    """Make the pre-registered identity resolve to the SAME agent_id when its agent connects to a
    per-shard coordinator URL. CredentialsStore.get(server) is keyed by URL, so we save the SAME
    agent_id/token under the coordinator URL's key; ArenaAgent.ensure_registered then returns the
    pre-registered credential instead of registering a new agent_id against that coordinator."""
    CredentialsStore(cred_path).save(
        AgentCredentials(server=coordinator_url, agent_id=creds.agent_id,
                         display_name=creds.display_name, agent_token=creds.agent_token))


def _shared_cred_paths(n: int) -> list[str]:
    """One cred path per identity (each holds that identity's single agent_id, reused per shard)."""
    return [os.path.join(tempfile.mkdtemp(prefix="arena-shardcred-"), f"id_{i}.json")
            for i in range(n)]


def _require_shared_database_url() -> None:
    """Fail fast unless DATABASE_URL is set (D1/D8). The parent/child shard rows are written through
    the LOCAL store here, so they MUST land in the SAME Neon the Modal coordinators read. With it
    unset the rows go to a local SQLite the coordinators never see — each coordinator would then
    replay the FULL schedule and the parent rollup would never complete. Set DATABASE_URL to the same
    Neon as the coordinators' `neon-database-url` secret."""
    if not os.environ.get("DATABASE_URL"):
        raise SystemExit(
            "sharded smoke (--shards > 1) requires DATABASE_URL to be set to the SAME Neon database "
            "as the Modal coordinators' `neon-database-url` secret. Without it, the parent/child "
            "shard rows would be written only to a local SQLite the coordinators never read (so each "
            "coordinator replays the full schedule and the parent rollup never completes). "
            "Re-run with DATABASE_URL=$NEON_URL.")


def _sharded_smoke(args) -> int:
    """V-10 (manual): K real coordinator containers + K shard host bundles, parent rolled up on read.

    Each of the N identities is pre-registered ONCE against the central API and reused on EVERY shard
    (one rating competitor across shards, REQ-5/D9). The parent is presentational; rollup_parent_status
    computes its state on read (D1). Requires DATABASE_URL set to the SAME Neon the coordinators read.
    """
    # D1/D8: refuse to write shard rows to a store the coordinators don't share. Check BEFORE writing
    # any rows or spawning Modal.
    _require_shared_database_url()

    parent_id = args.run_id or f"modal_smoke_shards_{args.seed}"
    parent_cfg = {
        "id": parent_id, "game": "onuw", "label": "One Night Ultimate Werewolf",
        "status": "open", "n_games": args.games, "players": N_PLAYERS,
        "seed_base": args.seed, "submitter": "modal-smoke", "deck_preset": "arena",
    }
    child_ids = create_sharded_run(parent_cfg, args.shards)
    print(f"parent {parent_id} -> {len(child_ids)} shard(s): {', '.join(child_ids)}", flush=True)

    # REQ-5/D9: register each identity ONCE against the central API -> one agent_id per identity,
    # reused on every shard regardless of its (distinct) coordinator URL = one competitor across shards.
    central = args.central_server or DEFAULT_SERVER
    cred_paths = _shared_cred_paths(N_PLAYERS)
    identities = [f"rand-{i}" for i in range(N_PLAYERS)]
    shared_creds = _shared_identity_creds(identities, central_server=central, cred_paths=cred_paths)
    print(f"pre-registered {N_PLAYERS} shared identities against {central}: "
          f"{', '.join(c.agent_id for c in shared_creds)}", flush=True)

    fn = modal.Function.from_name(APP_NAME, "run_server")
    calls = {}
    for cid in child_ids:
        child_cfg = {**parent_cfg, "id": cid}
        calls[cid] = fn.spawn(child_cfg, args.rounds, args.url_timeout)
        print(f"  spawned coordinator for {cid}, call_id={calls[cid].object_id}", flush=True)

    threads = []
    for cid in child_ids:
        deadline = time.time() + args.url_timeout
        url = None
        while time.time() < deadline:
            url = coordinator_url(cid)
            if url:
                break
            time.sleep(1.0)
        if not url:
            raise SystemExit(f"no coordinator URL for {cid} — check modal app logs")
        print(f"  {cid} coordinator URL: {url}", flush=True)
        # INV-4 / D7: create_sharded_run gates each child's /signups behind a per-parent join_token.
        # Read THIS child's token from its (shared-Neon) run row and present it on signup, else the
        # coordinator rejects every agent with 403 run_not_joinable.
        child_row = store.get_run(cid)
        join_token = (child_row or {}).get("join_token")
        if not join_token:
            raise SystemExit(f"no join_token on child row {cid} — create_sharded_run must set one (D7)")
        for i in range(N_PLAYERS):
            # Reuse the pre-registered agent_id on THIS coordinator URL (one competitor across shards).
            _seed_cred_for_coordinator(shared_creds[i], url, cred_paths[i])
            t = threading.Thread(target=_run_agent,
                                 args=(identities[i], url, cid, cred_paths[i], join_token),
                                 daemon=True)
            t.start()
            threads.append(t)
            time.sleep(0.3)

    for t in threads:
        t.join(timeout=600)

    status = rollup_parent_status(parent_id)
    print(f"parent {parent_id} rollup status = {status}", flush=True)
    for cid in child_ids:
        try:
            print(f"  {cid} result: {calls[cid].get(timeout=120)}", flush=True)
        except Exception as e:
            print(f"  {cid} could not get result: {type(e).__name__}: {e}", flush=True)
    return 0 if status == "done" else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default=None)
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--seed", type=int, default=31337)
    p.add_argument("--url-timeout", type=float, default=180.0)
    p.add_argument("--shards", type=int, default=1,
                   help="V-10 parallel-shards gate (SPEC-parallel-shards.md REQ-10): spawn K real "
                        "coordinator containers + K shard host bundles. Each identity is pre-registered "
                        "ONCE against --central-server and reused on every shard (one rating competitor "
                        "across shards). Requires DATABASE_URL set to the coordinators' Neon. "
                        "Default 1 = today's single-coordinator smoke.")
    p.add_argument("--central-server", default=DEFAULT_SERVER,
                   help="central API the shared identities are registered against (REQ-5): the same "
                        "Neon-backed app the Modal coordinators write to. Default: the SDK default "
                        "server. Used only by --shards > 1.")
    args = p.parse_args()

    if args.shards and args.shards > 1:
        return _sharded_smoke(args)

    run_id = args.run_id or f"modal_smoke_{args.seed}"
    run_config = {
        "id": run_id, "game": "onuw", "label": "One Night Ultimate Werewolf",
        "status": "open", "n_games": args.games, "players": N_PLAYERS,
        "seed_base": args.seed, "submitter": "modal-smoke", "deck_preset": "arena",
    }

    print(f"spawning Modal coordinator for {run_id} ...", flush=True)
    fn = modal.Function.from_name(APP_NAME, "run_server")
    call = fn.spawn(run_config, args.rounds, args.url_timeout)
    print(f"  spawned, call_id={call.object_id}", flush=True)

    print("waiting for tunnel URL ...", flush=True)
    url = None
    deadline = time.time() + args.url_timeout
    while time.time() < deadline:
        url = coordinator_url(run_id)
        if url:
            break
        time.sleep(1.0)
    if not url:
        raise SystemExit("no coordinator URL — check: .venv/bin/modal app logs persuasion-arena-coordinator")
    print(f"  coordinator URL: {url}", flush=True)

    print(f"connecting {N_PLAYERS} scripted agents ...", flush=True)
    threads = []
    for i in range(N_PLAYERS):
        t = threading.Thread(target=_run_agent, args=(f"rand-{i}", url, run_id), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.3)  # stagger so seat order is stable
    for t in threads:
        t.join(timeout=600)

    print("waiting for coordinator result ...", flush=True)
    try:
        result = call.get(timeout=120)
        print(f"RESULT: {result}", flush=True)
    except Exception as e:
        print(f"could not get result: {type(e).__name__}: {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
