"""Run a batch of games concurrently (a 'run').

Games are independent, so we play many at once in a thread pool — the OpenAI client is thread-safe
and each game is internally sequential. Worker threads only PLAY (and return transcripts); the main
thread does all SQLite writes, avoiding concurrent-write locking.

Role balancing: games are grouped into BLOCKS of n_players games that share one seed (hence one
deal). Within a block the agent→seat assignment cycles through all n_players rotations, so every
agent plays each seat — and therefore each of that deal's dealt roles — exactly once per block.
This makes per-agent role mix balanced by construction (on full blocks), instead of relying on
seat rotation over independent random deals where role assignment is only balanced in expectation.

Resilience: a single game's exception is caught and the run continues; the run is marked 'partial'
(not 'done') so a hole in the rotation matrix can't silently bias the leaderboard.
"""
from __future__ import annotations

import datetime as _dt
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import SETTINGS
from .games.onuw import ONUW
from .games.avalon import Avalon
from .games.secret_mafia import SecretMafia
from .openrouter import OpenRouterAgent
from . import store

GAME_CORES = {"onuw": ONUW, "avalon": Avalon, "secret_mafia": SecretMafia}


def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def role_balanced_schedule(n_games: int, n_players: int, seed_base: int) -> list[tuple[int, int, int]]:
    """Return [(gid, seed, rot)] for a run.

    Block b = games [b*n_players, (b+1)*n_players); all games in a block use seed_base+b (one deal),
    and rot cycles 0..n_players-1 so each agent visits each seat once per block. A final partial
    block (when n_games is not a multiple of n_players) is still rotated but not perfectly balanced.
    """
    sched = []
    for g in range(n_games):
        block, rot = divmod(g, n_players)
        sched.append((g + 1, seed_base + block, rot))
    return sched


def _play_one(core_cls, specs, n_players, seed, gid, rot, discussion_rounds):
    seat_to_spec = [specs[(i + rot) % n_players] for i in range(n_players)]
    names = {i: seat_to_spec[i].name for i in range(n_players)}
    agents = {i: OpenRouterAgent(seat_to_spec[i].name, seat_to_spec[i].model, seat_to_spec[i].harness)
              for i in range(n_players)}
    t = core_cls(names, seed=seed, discussion_rounds=discussion_rounds).play(agents)
    for p in t["players"]:
        s = seat_to_spec[p["seat"]]
        p["name"], p["model"] = s.name, s.model
    meta = [{"name": seat_to_spec[i].name, "model": seat_to_spec[i].model} for i in range(n_players)]
    return gid, t, meta


def run_batch(game: str = "onuw", n_games: int = 20, seed_base: int = 9000,
              run_id: str | None = None, roster=None, workers: int = 8,
              discussion_rounds: int = 5, skip_gids=None, on_game_saved=None) -> str:
    specs = list(roster or SETTINGS.roster())
    core_cls = GAME_CORES[game]
    n_players = len(specs)  # the roster IS the table — one agent per seat, no fixed count
    lo, hi = core_cls.MIN_PLAYERS, core_cls.MAX_PLAYERS
    if not (lo <= n_players <= hi):
        raise ValueError(f"{core_cls.TITLE} supports {lo}–{hi} players, got {n_players}")
    run_id = run_id or f"run_{seed_base}"
    agents_meta = [{"name": s.name, "model": s.model, "harness": s.harness} for s in specs]
    if n_games % n_players:
        print(f"[{run_id}] note: n_games={n_games} is not a multiple of {n_players}; "
              f"the final block is partially rotated (role balance is exact only on full blocks).",
              flush=True)

    store.save_run({
        "id": run_id, "game": game, "label": core_cls.TITLE, "status": "running",
        "n_games": n_games, "players": n_players, "seed_base": seed_base,
        "created": _now(), "agents": agents_meta,
    })

    skip_gids = set(skip_gids or [])
    sched = [(gid, seed, rot) for (gid, seed, rot) in role_balanced_schedule(n_games, n_players, seed_base)
             if gid not in skip_gids]
    failed: list[int] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_to_gid = {
            ex.submit(_play_one, core_cls, specs, n_players, seed, gid, rot, discussion_rounds): gid
            for (gid, seed, rot) in sched
        }
        done = len(skip_gids)
        for f in as_completed(fut_to_gid):
            gid = fut_to_gid[f]
            try:
                rgid, t, meta = f.result()
                # Persist each game the moment it finishes (main thread = no concurrent-write races),
                # so the run overview shows games appear one by one instead of all at the end.
                store.save_game(run_id, rgid, t, meta)
                if on_game_saved:
                    on_game_saved(rgid, t, meta)
            except Exception as e:  # one bad game must not abort the whole run
                failed.append(gid)
                print(f"[{run_id}] game {gid} FAILED ({type(e).__name__}: {e}) — continuing", flush=True)
                continue
            done += 1
            print(f"[{run_id}] {done}/{n_games} done (game {rgid}: {t['outcome']['text']})", flush=True)

    status = "done" if not failed else "partial"
    store.update_run_status(run_id, status)
    if failed:
        print(f"[{run_id}] PARTIAL: {done}/{n_games} saved; {len(failed)} failed: {sorted(failed)}. "
              f"Score with --allow-partial to override.", flush=True)
    else:
        print(f"[{run_id}] complete: {n_games} games saved.", flush=True)
    return run_id
