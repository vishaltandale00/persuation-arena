"""Run a batch of games concurrently (a 'run').

Games are independent, so we play many at once in a thread pool — the OpenAI client is thread-safe
and each game is internally sequential. Worker threads only PLAY (and return transcripts); the main
thread does all SQLite writes, avoiding concurrent-write locking.

Scheduling: every game gets a fresh seed/deal. Seat assignment still rotates by game so no agent is
stuck in one seat, but the run never repeats the same hidden-role deal across a whole block.

Resilience: a single game's exception is caught and the run continues; the run is marked 'partial'
(not 'done') so a hole in the rotation matrix can't silently bias the leaderboard.
"""
from __future__ import annotations

import datetime as _dt
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import Caps, SETTINGS, caps_with_overrides
from .games.onuw import ONUW, normalize_deck_preset
from .games.avalon import Avalon
from .games.secret_mafia import SecretMafia
from .openrouter import OpenRouterAgent
from . import store

GAME_CORES = {"onuw": ONUW, "avalon": Avalon, "secret_mafia": SecretMafia}


def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def _agent_meta(spec, caps: Caps, discussion_rounds: int) -> dict:
    return {
        "name": spec.name,
        "model": spec.model,
        "harness": spec.harness,
        "provider": "openrouter",
        "reasoning_effort": caps.reasoning_effort,
        "max_tokens": caps.max_tokens_per_turn,
        "max_tokens_per_turn": caps.max_tokens_per_turn,
        "temperature": caps.temperature,
        "retries": caps.retries,
        "prior_message_turns": caps.prior_message_turns,
        "discussion_rounds": discussion_rounds,
        "sessionful": False,
    }


def _run_config_meta(caps: Caps, discussion_rounds: int) -> dict:
    return {
        "discussion_rounds": discussion_rounds,
        "reasoning_effort": caps.reasoning_effort,
        "max_tokens_per_turn": caps.max_tokens_per_turn,
        "temperature": caps.temperature,
        "retries": caps.retries,
        "prior_message_turns": caps.prior_message_turns,
    }


def fresh_deal_schedule(n_games: int, n_players: int, seed_base: int) -> list[tuple[int, int, int]]:
    """Return [(gid, seed, rot)] for a run.

    Seeds increment per game so each game gets an independent deal. Rot cycles 0..n_players-1 so
    each agent still visits each seat regularly, without replaying the same hidden-role state.
    """
    sched = []
    for g in range(n_games):
        sched.append((g + 1, seed_base + g, g % n_players))
    return sched


def _play_one(core_cls, specs, n_players, seed, gid, rot, discussion_rounds, deck_preset=None,
              caps: Caps | None = None):
    caps = caps or SETTINGS.caps
    seat_to_spec = [specs[(i + rot) % n_players] for i in range(n_players)]
    names = {i: seat_to_spec[i].name for i in range(n_players)}
    agents = {i: OpenRouterAgent(seat_to_spec[i].name, seat_to_spec[i].model, seat_to_spec[i].harness,
                                  caps=caps)
              for i in range(n_players)}
    kwargs = {"discussion_rounds": discussion_rounds}
    if core_cls is ONUW:
        kwargs["deck_preset"] = deck_preset
    t = core_cls(names, seed=seed, **kwargs).play(agents)
    for p in t["players"]:
        s = seat_to_spec[p["seat"]]
        p["name"], p["model"] = s.name, s.model
    meta = [_agent_meta(seat_to_spec[i], caps, discussion_rounds) for i in range(n_players)]
    return gid, t, meta


def run_batch(game: str = "onuw", n_games: int = 20, seed_base: int = 9000,
              run_id: str | None = None, roster=None, workers: int = 8,
              discussion_rounds: int = 5, deck_preset: str | None = None,
              skip_gids=None, on_game_saved=None, caps: Caps | None = None) -> str:
    if discussion_rounds <= 0:
        raise ValueError("discussion_rounds must be positive")
    caps = caps_with_overrides(caps or SETTINGS.caps, discussion_rounds=discussion_rounds)
    specs = list(roster or SETTINGS.roster())
    core_cls = GAME_CORES[game]
    n_players = len(specs)  # the roster IS the table — one agent per seat, no fixed count
    lo, hi = core_cls.MIN_PLAYERS, core_cls.MAX_PLAYERS
    if not (lo <= n_players <= hi):
        raise ValueError(f"{core_cls.TITLE} supports {lo}–{hi} players, got {n_players}")
    deck_preset = normalize_deck_preset(deck_preset) if game == "onuw" else None
    run_id = run_id or f"run_{seed_base}"
    agents_meta = [_agent_meta(s, caps, discussion_rounds) for s in specs]
    store.save_run({
        "id": run_id, "game": game, "label": core_cls.TITLE, "status": "running",
        "n_games": n_games, "players": n_players, "seed_base": seed_base,
        "created": _now(), "agents": agents_meta, "deck_preset": deck_preset,
        "metadata": {"run_config": _run_config_meta(caps, discussion_rounds)},
    })

    skip_gids = set(skip_gids or [])
    sched = [(gid, seed, rot) for (gid, seed, rot) in fresh_deal_schedule(n_games, n_players, seed_base)
             if gid not in skip_gids]
    failed: list[int] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_to_gid = {
            ex.submit(_play_one, core_cls, specs, n_players, seed, gid, rot,
                      discussion_rounds, deck_preset, caps): gid
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
