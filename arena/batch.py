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
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .config import Caps, SETTINGS, caps_with_overrides
from .games.onuw import ONUW, default_deck, normalize_deck_preset
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


DealScheduleEntry = tuple[int, int, int, list[str] | None]


def fresh_deal_schedule(
    n_games: int,
    n_players: int,
    seed_base: int,
    *,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> list[tuple[int, int, int]]:
    """Return [(gid, seed, rot)] for a run.

    Seeds increment per game so each game gets an independent deal. Rot cycles 0..n_players-1 so
    each agent still visits each seat regularly, without replaying the same hidden-role state.

    Sharding (SPEC REQ-1): when both ``shard_index`` and ``num_shards`` are given, return only the
    stride slice of the GLOBAL schedule — entries whose 0-indexed position ``g`` (i.e. ``gid-1``)
    satisfies ``g % num_shards == shard_index``. The global ``gid``/``seed``/``rot`` are preserved
    (no renumbering). When either is ``None``, behave exactly as before (INV-2).
    """
    sliced = shard_index is not None and num_shards is not None
    sched = []
    for g in range(n_games):
        if sliced and g % num_shards != shard_index:
            continue
        sched.append((g + 1, seed_base + g, g % n_players))
    return sched


def _pick_balanced_role(
    remaining: Counter[str],
    counts_by_spec: list[Counter[str]],
    spec_idx: int,
    *,
    rng: random.Random,
) -> str:
    roles = list(remaining)
    rng.shuffle(roles)
    return min(
        roles,
        key=lambda role: (counts_by_spec[spec_idx][role], -remaining[role], role),
    )


def _role_balance_score(counts_by_spec: list[Counter[str]]) -> int:
    roles = sorted({role for counts in counts_by_spec for role in counts})
    return sum(counts[role] * counts[role] for counts in counts_by_spec for role in roles)


def _delta_role(
    counts_by_spec: list[Counter[str]],
    spec_idx: int,
    old: str,
    new: str,
    delta: int,
) -> None:
    counts_by_spec[spec_idx][old] -= delta
    counts_by_spec[spec_idx][new] += delta
    if counts_by_spec[spec_idx][old] <= 0:
        del counts_by_spec[spec_idx][old]


def _improve_balanced_deal(
    dealt: list[str],
    center: list[str],
    counts_by_spec: list[Counter[str]],
    rot: int,
) -> tuple[list[str], list[str]]:
    """Locally improve a completed deal against the global model-role count matrix."""
    n_players = len(dealt)
    while True:
        current_score = _role_balance_score(counts_by_spec)
        best: tuple[int, str, int, int] | None = None

        for a in range(n_players):
            spec_a = (a + rot) % n_players
            for b in range(a + 1, n_players):
                if dealt[a] == dealt[b]:
                    continue
                spec_b = (b + rot) % n_players
                _delta_role(counts_by_spec, spec_a, dealt[a], dealt[b], 1)
                _delta_role(counts_by_spec, spec_b, dealt[b], dealt[a], 1)
                score = _role_balance_score(counts_by_spec)
                _delta_role(counts_by_spec, spec_a, dealt[b], dealt[a], 1)
                _delta_role(counts_by_spec, spec_b, dealt[a], dealt[b], 1)
                if score < current_score and (best is None or score < best[0]):
                    best = (score, "seat", a, b)

            for c in range(len(center)):
                if dealt[a] == center[c]:
                    continue
                _delta_role(counts_by_spec, spec_a, dealt[a], center[c], 1)
                score = _role_balance_score(counts_by_spec)
                _delta_role(counts_by_spec, spec_a, center[c], dealt[a], 1)
                if score < current_score and (best is None or score < best[0]):
                    best = (score, "center", a, c)

        if best is None:
            return dealt, center
        _, kind, a, b = best
        if kind == "seat":
            spec_a = (a + rot) % n_players
            spec_b = (b + rot) % n_players
            _delta_role(counts_by_spec, spec_a, dealt[a], dealt[b], 1)
            _delta_role(counts_by_spec, spec_b, dealt[b], dealt[a], 1)
            dealt[a], dealt[b] = dealt[b], dealt[a]
        else:
            spec_a = (a + rot) % n_players
            _delta_role(counts_by_spec, spec_a, dealt[a], center[b], 1)
            dealt[a], center[b] = center[b], dealt[a]


def balanced_onuw_deal_schedule(
    n_games: int,
    n_players: int,
    seed_base: int,
    deck: list[str],
) -> list[DealScheduleEntry]:
    """Return a reproducible model-role balanced schedule for ONUW.

    Seat rotation remains the existing fairness primitive. This adds a deterministic greedy deal
    pass that minimizes the current deficit in the model-index x dealt-role matrix while respecting
    the deck multiset.
    """
    counts_by_spec = [Counter() for _ in range(n_players)]
    sched: list[DealScheduleEntry] = []
    for g in range(n_games):
        gid = g + 1
        seed = seed_base + g
        rot = g % n_players
        rng = random.Random(seed)
        remaining = Counter(deck)
        dealt: list[str | None] = [None] * n_players
        seat_order = list(range(n_players))
        rng.shuffle(seat_order)
        for seat in seat_order:
            spec_idx = (seat + rot) % n_players
            role = _pick_balanced_role(
                remaining,
                counts_by_spec,
                spec_idx,
                rng=rng,
            )
            dealt[seat] = role
            counts_by_spec[spec_idx][role] += 1
            remaining[role] -= 1
            if remaining[role] <= 0:
                del remaining[role]

        center = list(remaining.elements())
        rng.shuffle(center)
        final_dealt = [role for role in dealt if role is not None]
        final_dealt, center = _improve_balanced_deal(final_dealt, center, counts_by_spec, rot)
        sched.append((gid, seed, rot, final_dealt + center))
    return sched


def run_schedule(
    game: str,
    n_games: int,
    n_players: int,
    seed_base: int,
    *,
    deal_schedule: str = "random",
    deck: list[str] | None = None,
) -> list[DealScheduleEntry]:
    if deal_schedule == "random":
        return [(gid, seed, rot, None) for gid, seed, rot in fresh_deal_schedule(n_games, n_players, seed_base)]
    if deal_schedule == "balanced":
        if game != "onuw":
            raise ValueError("balanced deal scheduling currently supports onuw only")
        if not deck:
            raise ValueError("balanced deal scheduling requires an ONUW deck")
        return balanced_onuw_deal_schedule(n_games, n_players, seed_base, deck)
    raise ValueError(f"unknown deal_schedule: {deal_schedule}")


def default_deal_schedule(game: str) -> str:
    return "balanced" if game == "onuw" else "random"


def schedule_role_exposure(schedule: list[DealScheduleEntry], n_players: int, specs) -> dict[str, dict[str, int]]:
    exposure: dict[str, Counter[str]] = {s.name: Counter() for s in specs}
    for _, _, rot, deal in schedule:
        if not deal:
            continue
        for seat, role in enumerate(deal[:n_players]):
            spec = specs[(seat + rot) % n_players]
            exposure[spec.name][role] += 1
    return {name: dict(sorted(counts.items())) for name, counts in exposure.items()}


class StreamingAgent:
    """Emit local static model turn events around a normal in-process agent call."""

    def __init__(self, agent, *, run_id: str, game_instance_id: str, seat: int):
        self.agent = agent
        self.name = agent.name
        self.model = agent.model
        self.harness = getattr(agent, "harness", "base")
        self.run_id = run_id
        self.game_instance_id = game_instance_id
        self.seat = seat

    @property
    def calls(self):
        return getattr(self.agent, "calls", [])

    def act(self, observation: str, parse_action, default_action: Any, **turn_meta: Any):
        phase = turn_meta.get("phase")
        action_kind = turn_meta.get("action_kind")
        store.append_event(
            self.run_id,
            "model_turn_started",
            {"seat": self.seat, "model": self.model, "action_kind": action_kind},
            game_instance_id=self.game_instance_id,
            phase=phase,
        )
        resp = self.agent.act(observation, parse_action, default_action, **turn_meta)
        payload = {
            "seat": self.seat,
            "model": self.model,
            "action_kind": action_kind,
            "ok": bool(getattr(resp, "ok", True)),
            "ms": getattr(resp, "ms", None),
            "action": getattr(resp, "action", None),
            "reasoning": getattr(resp, "reasoning", ""),
        }
        raw = getattr(resp, "raw", None)
        if raw:
            payload["raw"] = raw
        provider_reasoning = getattr(resp, "provider_reasoning", None)
        if provider_reasoning is not None:
            payload["provider_reasoning"] = provider_reasoning
        provider_reasoning_details = getattr(resp, "provider_reasoning_details", None)
        if provider_reasoning_details is not None:
            payload["provider_reasoning_details"] = provider_reasoning_details
        store.append_event(
            self.run_id,
            "model_turn_completed",
            payload,
            game_instance_id=self.game_instance_id,
            phase=phase,
        )
        return resp


class StaticEventSink:
    def __init__(self, run_id: str, game_instance_id: str):
        self.run_id = run_id
        self.game_instance_id = game_instance_id

    def __call__(self, event_type: str, payload: dict, *, phase=None,
                 visibility: str = "public", target_seat: int | None = None):
        data = dict(payload)
        if target_seat is not None:
            data.setdefault("target_seat", target_seat)
        store.append_event(
            self.run_id,
            event_type,
            data,
            visibility=visibility,
            game_instance_id=self.game_instance_id,
            phase=phase,
        )


def _play_one(core_cls, specs, n_players, seed, gid, rot, discussion_rounds, deck_preset=None,
              caps: Caps | None = None, run_id: str | None = None, stream_events: bool = False,
              deal_override: list[str] | None = None):
    caps = caps or SETTINGS.caps
    game_instance_id = f"{run_id}_game_{gid:03d}" if run_id else None
    seat_to_spec = [specs[(i + rot) % n_players] for i in range(n_players)]
    names = {i: seat_to_spec[i].name for i in range(n_players)}
    agents = {}
    for i in range(n_players):
        base_agent = OpenRouterAgent(seat_to_spec[i].name, seat_to_spec[i].model, seat_to_spec[i].harness,
                                     caps=caps)
        agents[i] = (
            StreamingAgent(base_agent, run_id=run_id, game_instance_id=game_instance_id, seat=i)
            if stream_events and run_id and game_instance_id else base_agent
        )
    kwargs = {"discussion_rounds": discussion_rounds}
    if core_cls is ONUW:
        kwargs["deck_preset"] = deck_preset
        if deal_override is not None:
            kwargs["deal_override"] = deal_override
        if stream_events and run_id and game_instance_id:
            kwargs["event_sink"] = StaticEventSink(run_id, game_instance_id)
            kwargs["event_driven"] = False
            store.append_event(run_id, "game_started", {"gid": gid, "seed": seed},
                               game_instance_id=game_instance_id, phase="setup")
    t = core_cls(names, seed=seed, **kwargs).play(agents)
    for p in t["players"]:
        s = seat_to_spec[p["seat"]]
        p["name"], p["model"] = s.name, s.model
    meta = [_agent_meta(seat_to_spec[i], caps, discussion_rounds) for i in range(n_players)]
    return gid, t, meta


def run_batch(game: str = "onuw", n_games: int = 20, seed_base: int = 9000,
              run_id: str | None = None, roster=None, workers: int = 8,
              discussion_rounds: int = 5, deck_preset: str | None = None,
              skip_gids=None, on_game_saved=None, caps: Caps | None = None,
              run_config_overrides: dict | None = None, stream_events: bool = False,
              deal_schedule: str | None = None) -> str:
    if discussion_rounds <= 0:
        raise ValueError("discussion_rounds must be positive")
    caps = caps_with_overrides(caps or SETTINGS.caps, discussion_rounds=discussion_rounds)
    specs = list(roster or SETTINGS.roster())
    core_cls = GAME_CORES[game]
    n_players = len(specs)  # the roster IS the table — one agent per seat, no fixed count
    lo, hi = core_cls.MIN_PLAYERS, core_cls.MAX_PLAYERS
    if not (lo <= n_players <= hi):
        raise ValueError(f"{core_cls.TITLE} supports {lo}–{hi} players, got {n_players}")
    deal_schedule = (deal_schedule or default_deal_schedule(game)).strip().lower()
    deck_preset = normalize_deck_preset(deck_preset) if game == "onuw" else None
    deck = default_deck(n_players, deck_preset) if game == "onuw" else None
    full_sched = run_schedule(
        game, n_games, n_players, seed_base, deal_schedule=deal_schedule, deck=deck
    )
    run_id = run_id or f"run_{seed_base}"
    agents_meta = [_agent_meta(s, caps, discussion_rounds) for s in specs]
    schedule_meta = {"mode": deal_schedule}
    if deal_schedule == "balanced":
        schedule_meta.update({
            "algorithm": "role_deficit_v1",
            "role_exposure": schedule_role_exposure(full_sched, n_players, specs),
        })
    store.save_run({
        "id": run_id, "game": game, "label": core_cls.TITLE, "status": "running",
        "n_games": n_games, "players": n_players, "seed_base": seed_base,
        "created": _now(), "agents": agents_meta, "deck_preset": deck_preset,
        "metadata": {
            "run_config": _run_config_meta(caps, discussion_rounds),
            "run_config_overrides": run_config_overrides or {},
            "deal_schedule": schedule_meta,
        },
    })

    skip_gids = set(skip_gids or [])
    sched = [(gid, seed, rot, deal) for (gid, seed, rot, deal) in full_sched if gid not in skip_gids]
    failed: list[int] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_to_gid = {
            ex.submit(_play_one, core_cls, specs, n_players, seed, gid, rot,
                      discussion_rounds, deck_preset, caps, run_id, stream_events, deal): gid
            for (gid, seed, rot, deal) in sched
        }
        done = len(skip_gids)
        for f in as_completed(fut_to_gid):
            gid = fut_to_gid[f]
            try:
                rgid, t, meta = f.result()
                # Persist each game the moment it finishes (main thread = no concurrent-write races),
                # so the run overview shows games appear one by one instead of all at the end.
                store.save_game(run_id, rgid, t, meta)
                if stream_events:
                    store.append_event(
                        run_id,
                        "game_result",
                        {"gid": rgid, "winner_team": t["winner_team"], "text": t["outcome"]["text"]},
                        game_instance_id=f"{run_id}_game_{rgid:03d}",
                        phase="result",
                    )
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
