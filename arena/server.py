"""Local FastAPI server: serves runs/games JSON in the observer's shapes and the observer page.

  GET /api/runs                      -> run summaries for the index
  GET /api/runs/{id}                 -> run overview (agents, settings, results, games list)
  GET /api/runs/{id}/games/{gid}     -> full game transcript (observer shape)
  GET /                              -> serves web/observer.html and assets (same-origin fetch)

Run with:  uvicorn arena.server:app --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import copy
import hashlib
import hmac
import os
import re
import secrets
import threading
import time
import uuid

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .config import ROOT, SETTINGS, OPENROUTER_BASE_URL, AgentSpec, caps_with_overrides
from . import openrouter as _or
from . import store
from .batch import GAME_CORES, default_deal_schedule
from .games.onuw import DEFAULT_DECK_PRESET, deck_for_preset, deck_preset_options, normalize_deck_preset
from .identity import NO_ONE_REF, participant, validate_public_name, validate_unique_public_names
from .score import score_run, score_runs
from . import sharded
from .sharded import rollup_parent_status

WEB = ROOT / "web"
GAME_LABELS = {
    "onuw": "One Night Ultimate Werewolf",
    "avalon": "The Resistance: Avalon",
    "secret_mafia": "Secret Mafia",
}
PROTOCOL_VERSION = "arena-agent-v1"
_local_run_lock = threading.Lock()
_local_active_runs: set[str] = set()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    store.init_schema()
    recovered = store.mark_orphaned_local_runs_partial() if not os.environ.get("DATABASE_URL") else 0
    if recovered:
        print(f"[startup] marked {recovered} orphaned local run(s) partial", flush=True)
    try:
        yield
    finally:
        if not os.environ.get("DATABASE_URL"):
            with _local_run_lock:
                active = list(_local_active_runs)
            for run_id in active:
                store.update_run_status(run_id, "partial")
            if active:
                print(f"[shutdown] marked {len(active)} active local run(s) partial", flush=True)


app = FastAPI(title="Persuasion Arena", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _token_map() -> dict[str, str]:
    raw = os.environ.get("INGEST_TOKENS", "").strip()
    if not raw:
        return {}
    out: dict[str, str] = {}
    for part in re.split(r"[,\s;]+", raw):
        if not part:
            continue
        sep = "=" if "=" in part else ":"
        if sep in part:
            owner, token = part.split(sep, 1)
            if owner.strip() and token.strip():
                out[owner.strip()] = token.strip()
    return out


def _bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


def _hash_agent_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _issue_agent_token() -> str:
    return "pa_live_" + secrets.token_urlsafe(32)


def _authorized_agent(authorization: str | None) -> dict:
    token = _bearer(authorization)
    if not token:
        raise HTTPException(401, "missing bearer token")
    agent = store.get_agent_by_token_hash(_hash_agent_token(token))
    if not agent:
        raise HTTPException(403, "invalid bearer token")
    store.touch_agent(agent["id"])
    return agent


def _authorized_owner(authorization: str | None, requested_owner: str | None = None) -> str:
    tokens = _token_map()
    requested_owner = (requested_owner or "").strip() or None
    if not tokens:
        return requested_owner or "local"
    token = _bearer(authorization)
    if not token:
        raise HTTPException(401, "missing bearer token")
    for owner, expected in tokens.items():
        if hmac.compare_digest(token, expected):
            if requested_owner and requested_owner != owner:
                raise HTTPException(403, "token cannot act for requested owner")
            return owner
    raise HTTPException(403, "invalid bearer token")


def _job_owner(payload: dict) -> str:
    owner = (payload.get("owner") or payload.get("submitter") or "default").strip()
    return owner or "default"


def _roster_from_payload(payload: dict) -> list[dict]:
    """Build the run's roster from the request (one agent per seat). Falls back to the
    configured roster when none is supplied. The player count IS len(roster); per-game
    bounds are enforced by the caller, which knows which game is being played."""
    agents = payload.get("agents") or []
    roster = [
        {"name": (a.get("name") or "").strip(),
         "model": (a.get("model") or "").strip(),
         "harness": (a.get("harness") or "base").strip() or "base"}
        for a in agents if (a.get("name") or "").strip() and (a.get("model") or "").strip()
    ]
    if not roster:
        roster = [{"name": s.name, "model": s.model, "harness": s.harness} for s in SETTINGS.roster()]
    return roster


def _deck_preset_from_payload(game: str, payload: dict) -> str | None:
    if game != "onuw":
        return None
    try:
        return normalize_deck_preset(payload.get("deck_preset") or payload.get("deckPreset"))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def _deck_for_api(game: str, players: int, deck_preset: str | None) -> list[str] | None:
    if game != "onuw":
        return None
    try:
        return deck_for_preset(players, deck_preset or DEFAULT_DECK_PRESET)
    except ValueError:
        return None  # out-of-range player count (e.g. a partially-configured run) -> no deck preview


def _deal_schedule_from_payload(game: str, payload: dict, *, default: str | None = None) -> str:
    raw = _payload_get(payload, "deal_schedule", "dealSchedule")
    value = (raw or default or default_deal_schedule(game)).strip().lower()
    if value not in {"random", "balanced"}:
        raise HTTPException(400, "deal_schedule must be random or balanced")
    if value == "balanced" and game != "onuw":
        raise HTTPException(400, "balanced deal scheduling currently supports onuw only")
    return value


def _payload_get(payload: dict, *names: str):
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    return None


def _payload_int(payload: dict, names: tuple[str, ...], default: int | None = None,
                 *, positive: bool = False, nonnegative: bool = False) -> int | None:
    raw = _payload_get(payload, *names)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as e:
        raise HTTPException(400, f"{names[0]} must be an integer") from e
    if positive and value <= 0:
        raise HTTPException(400, f"{names[0]} must be positive")
    if nonnegative and value < 0:
        raise HTTPException(400, f"{names[0]} must be nonnegative")
    return value


def _payload_float(payload: dict, names: tuple[str, ...], default: float | None = None,
                   *, nonnegative: bool = False) -> float | None:
    raw = _payload_get(payload, *names)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError) as e:
        raise HTTPException(400, f"{names[0]} must be a number") from e
    if nonnegative and value < 0:
        raise HTTPException(400, f"{names[0]} must be nonnegative")
    return value


def _payload_prior_message_turns(payload: dict) -> int | None:
    raw = _payload_get(payload, "prior_message_turns", "priorMessageTurns")
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in {"", "all", "infinite", "inf"}:
        return -1
    try:
        value = int(raw)
    except (TypeError, ValueError) as e:
        raise HTTPException(400, "prior_message_turns must be -1, all, or a nonnegative integer") from e
    if value < -1:
        raise HTTPException(400, "prior_message_turns must be -1 for all history, or nonnegative")
    return value


def _caps_from_payload(payload: dict, rounds: int):
    try:
        return caps_with_overrides(
            reasoning_effort=_payload_get(payload, "reasoning_effort", "reasoningEffort"),
            max_tokens_per_turn=_payload_int(
                payload, ("max_tokens_per_turn", "maxTokensPerTurn", "max_tokens", "maxTokens"),
                positive=True,
            ),
            temperature=_payload_float(payload, ("temperature",), nonnegative=True),
            retries=_payload_int(payload, ("retries",), nonnegative=True),
            prior_message_turns=_payload_prior_message_turns(payload),
            discussion_rounds=rounds,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def _run_config_meta(caps, rounds: int) -> dict:
    return {
        "discussion_rounds": rounds,
        "reasoning_effort": caps.reasoning_effort,
        "max_tokens_per_turn": caps.max_tokens_per_turn,
        "temperature": caps.temperature,
        "retries": caps.retries,
        "prior_message_turns": caps.prior_message_turns,
    }


def _run_config_overrides(payload: dict) -> dict:
    return {
        "temperature": _payload_get(payload, "temperature") is not None,
        "prior_message_turns": _payload_get(payload, "prior_message_turns", "priorMessageTurns") is not None,
    }


def _annotate_run_agents(agents: list[dict], caps, rounds: int) -> list[dict]:
    config = _run_config_meta(caps, rounds)
    return [
        {
            **a,
            "provider": a.get("provider", "openrouter"),
            "reasoning_effort": config["reasoning_effort"],
            "max_tokens": config["max_tokens_per_turn"],
            "max_tokens_per_turn": config["max_tokens_per_turn"],
            "temperature": config["temperature"],
            "retries": config["retries"],
            "prior_message_turns": config["prior_message_turns"],
            "discussion_rounds": config["discussion_rounds"],
            "sessionful": a.get("sessionful", False),
        }
        for a in agents
    ]


def _queue_run(payload: dict, owner: str) -> dict:
    game = payload.get("game", "onuw")
    if game not in GAME_LABELS:
        raise HTTPException(400, f"unknown game: {game}")
    games = max(1, int(payload.get("games", payload.get("n_games", 6))))
    rounds = _payload_int(payload, ("rounds",), 20, positive=True)
    caps = _caps_from_payload(payload, rounds)
    seed = int(payload.get("seed") or (time.time() * 1000) % 1000000)
    run_id = (payload.get("run_id") or f"run_{seed}_{uuid.uuid4().hex[:6]}").strip()
    raw_agents = _roster_from_payload(payload)
    identity_err = validate_unique_public_names([a["name"] for a in raw_agents])
    if identity_err:
        raise HTTPException(400, identity_err)
    agents = _annotate_run_agents(raw_agents, caps, rounds)
    core = GAME_CORES[game]
    n_players = len(agents)  # the roster IS the table — no fixed player count
    if not (core.MIN_PLAYERS <= n_players <= core.MAX_PLAYERS):
        raise HTTPException(400, f"{core.TITLE} supports {core.MIN_PLAYERS}–{core.MAX_PLAYERS} "
                                 f"players, got {n_players}")
    if _deal_schedule_from_payload(game, payload, default="random") != "random":
        raise HTTPException(400, "balanced deal scheduling is currently only supported for local runs")
    deck_preset = _deck_preset_from_payload(game, payload)
    job = store.enqueue_job({
        "id": f"job_{uuid.uuid4().hex[:12]}",
        "run_id": run_id,
        "owner": owner,
        "game": game,
        "label": GAME_LABELS[game],
        "n_games": games,
        "players": n_players,
        "seed_base": seed,
        "rounds": rounds,
        "agents": agents,
        "deck_preset": deck_preset,
        "metadata": {
            "run_config": _run_config_meta(caps, rounds),
            "run_config_overrides": _run_config_overrides(payload),
        },
    })
    return {"run_id": run_id, "job_id": job["id"], "status": "queued", "owner": owner,
            "game": game, "games": games, "rounds": rounds, "deck_preset": deck_preset,
            "run_config": _run_config_meta(caps, rounds)}


# --- Modal per-run coordinator (opt-in via ARENA_MODAL_COORDINATOR) ------------------------------
# When enabled, creating a connected run fire-and-forgets a per-run Modal container (arena.modal_app)
# that serves + coordinates the run; agents discover its tunnel URL via the signup response and
# re-point ready/poll/reply at it. Best-effort: any Modal error leaves the run coordinatable the old
# way (a worker invoking run_connected_batch). The central process needs Modal auth (MODAL_TOKEN_* or
# ~/.modal.toml) to spawn / read the URL.
_coord_url_cache: dict[str, str] = {}


def _modal_coordinator_enabled() -> bool:
    return os.environ.get("ARENA_MODAL_COORDINATOR", "").strip().lower() not in ("", "0", "false", "no")


def _spawn_coordinator(run_config: dict, rounds: int = 5) -> None:
    if not _modal_coordinator_enabled():
        return
    try:
        from .modal_app import spawn_run_server
        call_id = spawn_run_server(run_config, rounds)
        print(f"[modal] spawned coordinator for {run_config['id']} ({call_id})", flush=True)
    except Exception as e:  # never let a coordinator-spawn failure break run creation
        print(f"[modal] spawn failed for {run_config.get('id')}: {type(e).__name__}: {e}", flush=True)


def _coordinator_url(run_id: str) -> str | None:
    if not _modal_coordinator_enabled():
        return None
    cached = _coord_url_cache.get(run_id)
    if cached:
        return cached
    try:
        from .modal_app import coordinator_url
        url = coordinator_url(run_id)
        if url:
            _coord_url_cache[run_id] = url
        return url
    except Exception:
        return None


def _create_connected_run(payload: dict) -> dict:
    game = payload.get("game", "onuw")
    if game not in GAME_LABELS:
        raise HTTPException(400, f"unknown game: {game}")
    core = GAME_CORES[game]
    players = int(payload.get("players") or core.MIN_PLAYERS)
    if not (core.MIN_PLAYERS <= players <= core.MAX_PLAYERS):
        raise HTTPException(400, f"{core.TITLE} supports {core.MIN_PLAYERS}–{core.MAX_PLAYERS} "
                                 f"players, got {players}")
    games = max(1, int(payload.get("games", payload.get("n_games", 6))))
    rounds = _payload_int(payload, ("rounds",), 20, positive=True)
    caps = _caps_from_payload(payload, rounds)
    seed = int(payload.get("seed") or (time.time() * 1000) % 1000000)
    run_id = (payload.get("run_id") or f"run_{seed}_{uuid.uuid4().hex[:6]}").strip()
    deck_preset = _deck_preset_from_payload(game, payload)
    if _deal_schedule_from_payload(game, payload, default="random") != "random":
        raise HTTPException(400, "balanced deal scheduling is currently only supported for local static runs")
    run_config = {
        "id": run_id,
        "game": game,
        "label": GAME_LABELS[game],
        "status": "open",
        "n_games": games,
        "players": players,
        "seed_base": seed,
        "submitter": payload.get("owner") or payload.get("submitter") or "connected",
        "deck_preset": deck_preset,
        "metadata": {
            "run_config": _run_config_meta(caps, rounds),
            "run_config_overrides": _run_config_overrides(payload),
        },
    }
    # SPEC §6.9(a) / D3: shards>1 fans the run out into a parent + K child shards; default/1 keeps
    # today's single connected run byte-identical (INV-2). K is validated/capped in create_sharded_run.
    shards = _payload_int(payload, ("shards", "num_shards"), 1, positive=True)
    if shards > 1:
        try:
            child_ids = sharded.create_sharded_run(run_config, shards)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        join_token = None
        for cid in child_ids:
            child_run = store.get_run(cid)
            # Every child carries the SAME per-parent secret (arena/sharded.py:create_sharded_run).
            # Surface it so an API-driven orchestrator can sign agents into the child shards — the
            # response only ever exposed child run ids, leaving externally-driven runs unjoinable.
            join_token = child_run.get("join_token")
            _spawn_coordinator(child_run, rounds)  # one coordinator per shard (no-op unless Modal)
        return {"run_id": run_id, "status": "open", "game": game, "games": games,
                "players": players, "rounds": rounds, "deck_preset": deck_preset,
                "shards": shards, "child_run_ids": child_ids, "join_token": join_token,
                "run_config": _run_config_meta(caps, rounds)}
    store.create_connected_run(run_config)
    _spawn_coordinator(run_config, rounds)  # no-op unless ARENA_MODAL_COORDINATOR
    return {"run_id": run_id, "status": "open", "game": game, "games": games,
            "players": players, "rounds": rounds, "deck_preset": deck_preset,
            "run_config": _run_config_meta(caps, rounds)}


@app.get("/api/runs")
def api_runs():
    out = []
    for r in store.list_runs():
        # SPEC D7 / INV-4: child shards are invisible in the observer — the parent renders as one
        # normal run. (Parents and plain runs are listed; only run_kind='child' is hidden.)
        run_kind = (r.get("run_kind") or "normal")
        if run_kind == "child":
            continue
        status, team_split = r["status"], r["team_split"]
        if run_kind == "parent":
            # SPEC D7 / §6.9(a): the index renders the parent as one normal run — its OWN row never
            # gets child progress written back, so roll status up and aggregate the children's
            # team-split here (the parent's own status is stale 'open' and its split is 0-0).
            status = rollup_parent_status(r["id"])
            team_split = {"good": 0, "evil": 0}
            for cid in store.child_run_ids(r["id"]):
                for team, n in (store.get_run(cid)["team_split"]).items():
                    team_split[team] = team_split.get(team, 0) + n
        out.append({
            "id": r["id"], "game": r["game"], "label": r["label"], "status": status,
            "nGames": r["n_games"], "players": r["players"], "seed": r["seed_base"],
            "when": r["created"], "teamSplit": team_split,
            "deckPreset": r.get("deck_preset") or (DEFAULT_DECK_PRESET if r["game"] == "onuw" else None),
        })
    return out


@app.post("/api/runs")
def api_submit_run(payload: dict):
    """Queue a central run for a laptop worker. Model keys remain on the worker machine."""
    if payload.get("connected"):
        return _create_connected_run(payload)
    return _queue_run(payload, _job_owner(payload))


@app.get("/api/decks")
def api_decks(game: str = "onuw", players: int = 5):
    if game != "onuw":
        return {"game": game, "presets": []}
    if not (ONUW_MIN := GAME_CORES["onuw"].MIN_PLAYERS) <= players <= GAME_CORES["onuw"].MAX_PLAYERS:
        raise HTTPException(400, f"onuw supports {ONUW_MIN}–{GAME_CORES['onuw'].MAX_PLAYERS} players")
    return {"game": game, "players": players, "default": DEFAULT_DECK_PRESET,
            "presets": deck_preset_options(players)}


@app.get("/api/runs/open")
def api_open_runs(game: str | None = None):
    return {"runs": store.list_open_runs(game)}


@app.post("/api/agents/register")
def api_register_agent(payload: dict):
    display_name, name_err = validate_public_name(payload.get("display_name") or payload.get("name"))
    if name_err:
        raise HTTPException(400, name_err)
    assert display_name is not None
    protocol_version = (payload.get("protocol_version") or PROTOCOL_VERSION).strip()
    if protocol_version != PROTOCOL_VERSION:
        raise HTTPException(400, "unsupported protocol_version")
    token = _issue_agent_token()
    declared_model = (payload.get("model") or payload.get("declared_model") or "").strip() or None
    declared_harness = (payload.get("harness") or payload.get("declared_harness") or "").strip() or None
    agent = store.register_agent(display_name=display_name,
                                 token_hash=_hash_agent_token(token),
                                 protocol_version=protocol_version,
                                 sdk_version=payload.get("sdk_version"),
                                 declared_model=declared_model,
                                 declared_harness=declared_harness)
    return {
        "agent_id": agent["id"],
        "agent_token": token,
        "protocol_version": PROTOCOL_VERSION,
        "public_name": display_name,
        "public_ref": participant(display_name)["ref"],
    }


@app.get("/api/leaderboard")
def api_leaderboard():
    """Cross-run rating board, conservatively ranked, each competitor expandable to per-role stats."""
    from . import rating
    return {"competitors": rating.leaderboard()}


@app.get("/api/agents/{agent_id:path}")
def api_agent_detail(agent_id: str):
    """One competitor's scorecard + rating history + runs played. Keyed by bearer-token agent_id
    for connected agents, or the static:model:harness identity (which contains slashes) otherwise."""
    from . import rating
    detail = rating.agent_detail(agent_id)
    if not detail:
        raise HTTPException(404, "unknown competitor")
    return detail


def _signup_response(signup: dict) -> dict:
    status = signup["status"]
    poll_after = 5000 if status == "waiting" else 1000 if status in {"ready_required", "ready"} else 250
    msg = {
        "waiting": "Waiting for more agents.",
        "ready_required": "Seat assigned. Ready required before gameplay.",
        "ready": "Ready. Waiting for the run to start.",
        "active": "Run is active.",
        "completed": "Run completed.",
        "expired": "Signup expired.",
        "cancelled": "Run cancelled.",
        "rejected": "Signup rejected.",
    }.get(status, status)
    out = {
        "signup_id": signup["id"],
        "run_id": signup["run_id"],
        "agent_id": signup["agent_id"],
        "status": status,
        "seat": signup.get("seat"),
        "message": msg,
        "poll_after_ms": poll_after,
        "heartbeat_after_ms": 15000,
        "waiting_expires_at": signup.get("waiting_expires_utc"),
        "ready_deadline_at": signup.get("ready_deadline_utc"),
        "coordinator_url": _coordinator_url(signup["run_id"]),  # None unless the Modal coordinator is up
    }
    return out


@app.post("/api/runs/{run_id}/signups")
def api_signup_run(run_id: str, payload: dict, authorization: str | None = Header(None)):
    agent = _authorized_agent(authorization)
    if (payload.get("protocol_version") or PROTOCOL_VERSION) != PROTOCOL_VERSION:
        raise HTTPException(400, "unsupported protocol_version")
    signup, err = store.create_signup(
        run_id, agent["id"], int(payload.get("max_concurrent_turns") or 1),
        join_token=payload.get("join_token"),
    )
    if err == "run_not_found":
        raise HTTPException(404, "run not found")
    if err == "run_full":
        raise HTTPException(409, "run_full")
    if err == "run_not_joinable":
        # INV-4: shard parents/children are not directly joinable by a stray agent.
        raise HTTPException(403, "run_not_joinable")
    if err:
        raise HTTPException(409, err)
    return _signup_response(signup)


@app.get("/api/signups/{signup_id}")
def api_get_signup(signup_id: str, authorization: str | None = Header(None)):
    agent = _authorized_agent(authorization)
    signup = store.get_signup(signup_id, agent["id"])
    if not signup:
        raise HTTPException(404, "signup not found")
    return _signup_response(signup)


@app.post("/api/signups/{signup_id}/ready")
def api_signup_ready(signup_id: str, payload: dict, authorization: str | None = Header(None)):
    agent = _authorized_agent(authorization)
    if (payload.get("protocol_version") or PROTOCOL_VERSION) != PROTOCOL_VERSION:
        raise HTTPException(400, "unsupported protocol_version")
    signup, err = store.mark_signup_ready(signup_id, agent["id"])
    if err == "not_found":
        raise HTTPException(404, "signup not found")
    if err and err != "not_ready_required":
        raise HTTPException(409, err)
    return {"ok": True, **_signup_response(signup)}


def _fallback_participant(seat: int | None) -> dict:
    idx = 0 if seat is None else int(seat) + 1
    return participant(f"Participant {idx}")


def _participant_maps(run_id: str) -> dict[str | None, dict[int, dict]]:
    rosters = store.list_game_rosters(run_id)
    out: dict[str | None, dict[int, dict]] = {
        gid: {seat: participant(name) for seat, name in roster.items()}
        for gid, roster in rosters.items()
    }
    signups = store.list_run_signups(run_id)
    lobby = {
        int(s["seat"]): participant(s["display_name"])
        for s in signups if s.get("seat") is not None
    }
    if lobby:
        out[None] = lobby
    return out


def _participant_at(participants: dict[int, dict], seat: int | None) -> dict:
    if seat is None:
        return _fallback_participant(None)
    return participants.get(int(seat)) or _fallback_participant(int(seat))


def _project_event_payload(payload: dict, participants: dict[int, dict]) -> dict:
    out: dict = {}
    for key, value in payload.items():
        if key == "roster":
            out["participants"] = [_participant_at(participants, int(seat))
                                   for seat in sorted(value, key=lambda s: int(s))]
        elif key == "actor_seat":
            out["actor"] = _participant_at(participants, value)
        elif key == "target_seat":
            out["target"] = _participant_at(participants, value)
        elif key == "seat":
            out["participant"] = _participant_at(participants, value)
        elif key == "target" and isinstance(value, int) and not isinstance(value, bool):
            out["target"] = {"name": "No one", "ref": NO_ONE_REF} if value < 0 else _participant_at(participants, value)
        elif key in {"copied_seat", "swapped_with"} and isinstance(value, int):
            out[key.replace("_seat", "")] = _participant_at(participants, value)
        elif key in {"swapped", "deaths"} and isinstance(value, list):
            out[key] = [
                _participant_at(participants, v) if isinstance(v, int) and not isinstance(v, bool) else v
                for v in value
            ]
        else:
            out[key] = value
    return out


def _api_event(event: dict, participant_maps: dict[str | None, dict[int, dict]] | None = None,
               *, agent_visible: bool = False) -> dict:
    payload = event["payload"] if isinstance(event.get("payload"), dict) else {}
    participants = (participant_maps or {}).get(event.get("game_instance_id")) or (participant_maps or {}).get(None) or {}
    projected_payload = _project_event_payload(payload, participants) if agent_visible else event["payload"]
    actor = payload.get("actor_seat") if isinstance(payload, dict) else None
    out = {
        "event_id": event["event_id"],
        "game_instance_id": event.get("game_instance_id"),
        "seq": event["seq"],
        "visibility": event["visibility"],
        "phase": event.get("phase"),
        "type": event["type"],
        "payload": projected_payload,
    }
    if actor is not None:
        out["actor"] = _participant_at(participants, actor)
    if not agent_visible:
        out["actor_seat"] = actor
    return out


def _api_turn(turn: dict | None,
              participant_maps: dict[str | None, dict[int, dict]] | None = None) -> dict | None:
    if not turn:
        return None
    participants = (participant_maps or {}).get(turn.get("game_instance_id")) or (participant_maps or {}).get(None) or {}
    return {
        "turn_id": turn["id"],
        "game_instance_id": turn["game_instance_id"],
        "game": store.get_run(turn["run_id"])["game"],
        "seat": turn["seat"],
        "participant": _participant_at(participants, turn["seat"]),
        "phase": turn["phase"],
        "action_kind": turn["action_kind"],
        "deadline_at": turn["deadline_utc"],
        "observation": turn["observation"],
        "legal_action": turn["legal_action"],
    }


@app.post("/api/signups/{signup_id}/poll")
def api_signup_poll(signup_id: str, payload: dict, authorization: str | None = Header(None)):
    agent = _authorized_agent(authorization)
    signup = store.get_signup(signup_id, agent["id"])
    if not signup:
        raise HTTPException(404, "signup not found")
    events = store.list_events_for_signup(signup_id, payload.get("after_event_id"),
                                          int(payload.get("max_events") or 50))
    turn = store.pending_turn_for_signup(signup_id) if signup["status"] == "active" else None
    participant_maps = _participant_maps(signup["run_id"])
    return {
        "signup_id": signup_id,
        "run_id": signup["run_id"],
        "run_status": signup["status"],
        "events": [_api_event(e, participant_maps, agent_visible=True) for e in events],
        "turn": _api_turn(turn, participant_maps),
        "poll_after_ms": 100 if turn else 250 if signup["status"] == "active" else 1000,
    }


@app.post("/api/turns/{turn_id}/reply")
def api_turn_reply(turn_id: str, payload: dict, authorization: str | None = Header(None)):
    agent = _authorized_agent(authorization)
    action = payload.get("action")
    if action is None:
        raise HTTPException(400, "action required")
    reply, err = store.reply_to_turn(turn_id, agent["id"], action, payload.get("reasoning"),
                                     payload.get("client_ms"))
    if err == "not_found":
        raise HTTPException(404, "turn not found")
    if err == "forbidden":
        raise HTTPException(403, "turn does not belong to this agent")
    if err == "invalid_action":
        raise HTTPException(422, f"invalid action: {reply.get('reason') if reply else 'schema'}")
    if err == "turn_expired":
        raise HTTPException(409, "turn deadline expired")
    if err == "reply_already_recorded":
        return {"ok": True, "accepted": False, "reason": "reply_already_recorded"}
    return {"ok": True, "accepted": True}


def _source_run_ids(run: dict) -> list[str]:
    """The run ids whose games/scores/events back a run's observer view (SPEC D7).

    A normal/child run is backed by itself. A sharded PARENT is backed by its children: the observer
    sees a parent as one run whose games are the union across shards and whose scorecard is
    score_runs(children). (A parent never has games of its own.)"""
    if (run.get("run_kind") or "normal") == "parent":
        return store.child_run_ids(run["id"]) or []
    return [run["id"]]


def _parent_signups(source_ids: list[str]) -> list[dict]:
    """The roster for a sharded parent: the children share identities (REQ-5), so the same agent is
    signed up on every shard. Present ONE logical signup per agent_id (the observer renders the
    parent as one normal run, D7), picking the most-advanced status across shards so the roster row
    reflects whether that competitor has reached ready/active/completed anywhere."""
    rank = {"waiting": 0, "ready_required": 1, "ready": 2, "active": 3,
            "completed": 4, "expired": -1, "cancelled": -1}
    best: dict[str, dict] = {}
    for cid in source_ids:
        for s in store.list_run_signups(cid):
            key = s["agent_id"]
            cur = best.get(key)
            if cur is None or rank.get(s["status"], 0) > rank.get(cur["status"], 0):
                best[key] = s
    return list(best.values())


def _parent_recent_events(source_ids: list[str]) -> list[dict]:
    """Union of the children's recent events, globally ordered (created time, then per-shard seq) and
    capped like a single run's stream. The parent owns no events of its own (D7)."""
    events = [e for cid in source_ids for e in store.list_run_events(cid, max_events=120)]
    events.sort(key=lambda e: (e.get("created_utc") or "", e.get("seq") or 0))
    return events[-120:]


@app.get("/api/runs/{run_id}")
def api_run(run_id: str):
    r = store.get_run(run_id)
    if not r:
        raise HTTPException(404, "run not found")
    is_parent = (r.get("run_kind") or "normal") == "parent"
    source_ids = _source_run_ids(r)
    if is_parent:
        # SPEC D7: present the parent as ONE normal run by sourcing EVERY aggregate field from the
        # children (the parent row owns no games/signups/events/wins of its own). The children carry
        # the same identities (shared creds, REQ-5), disjoint global gids (D8), and per-shard events.
        child_runs = [store.get_run(cid) for cid in source_ids]
        agg_games = [g for cr in child_runs for g in cr["games"]]
        agg_games.sort(key=lambda g: g["gid"])
        # per-name wins/team-split summed across shards (gids are disjoint, so no double counting).
        agg_wins: dict[str, int] = {}
        agg_split = {"good": 0, "evil": 0}
        for cr in child_runs:
            for name, w in cr["wins"].items():
                agg_wins[name] = agg_wins.get(name, 0) + (w or 0)
            for team, n in cr["team_split"].items():
                agg_split[team] = agg_split.get(team, 0) + n
        r = {**r, "games": agg_games, "wins": agg_wins, "team_split": agg_split,
             "status": rollup_parent_status(run_id)}
    has_games = len(r["games"]) > 0
    signups = _parent_signups(source_ids) if is_parent else store.list_run_signups(run_id)
    signup_by_id = {s["id"]: s for s in signups}
    agents_src = r["agents"] or [
        {"name": s["display_name"], "model": "connected-agent", "harness": "connected",
         "agent_id": s["agent_id"], "signup_id": s["id"]}
        for s in signups
    ]
    agents = []
    for a in agents_src:
        s = signup_by_id.get(a.get("signup_id"))
        agents.append({
            "name": a["name"],
            "model": a["model"],
            "harness": a.get("harness", "base"),
            "wins": r["wins"].get(a["name"], 0) if has_games else None,
            "agent_id": a.get("agent_id"),
            "signup_id": a.get("signup_id"),
            "status": s.get("status") if s else None,
            "seat": s.get("seat") if s else None,
            "last_poll_utc": s.get("last_poll_utc") if s else None,
            "last_event_id": s.get("last_event_id") if s else None,
            "ready_deadline_at": s.get("ready_deadline_utc") if s else None,
        })
    games = [{"gid": g["gid"], "seed": g["seed"], "win": g["winner_team"], "line": g["line"], "full": True}
             for g in r["games"]]
    # partial scores while a run is in progress, full when done; a parent aggregates across shards.
    scores = (score_runs(source_ids) if is_parent else score_run(run_id)) if has_games else {}
    recent_events = (_parent_recent_events(source_ids) if is_parent
                     else store.list_run_events(run_id, max_events=120)) \
        if r["status"] == "running" or signups else []
    return {
        "id": r["id"], "game": r["game"], "label": r["label"], "status": r["status"],
        "nGames": r["n_games"], "players": r["players"], "seedBase": r["seed_base"],
        "deckPreset": r.get("deck_preset") or (DEFAULT_DECK_PRESET if r["game"] == "onuw" else None),
        "deck": _deck_for_api(r["game"], int(r["players"]), r.get("deck_preset")),
        "created": r["created"], "agents": agents, "teamSplit": r["team_split"], "games": games,
        "runConfig": (r.get("metadata") or {}).get("run_config", {}),
        "runConfigOverrides": (r.get("metadata") or {}).get("run_config_overrides", {}),
        "dealSchedule": (r.get("metadata") or {}).get("deal_schedule", {"mode": default_deal_schedule(r["game"])}),
        "recentEvents": [_api_event(e) for e in recent_events],
        "connected": bool(signups),
        "connectedSummary": {
            "signups": len(signups),
            "ready": sum(1 for s in signups if s["status"] in {"ready", "active", "completed"}),
            "active": sum(1 for s in signups if s["status"] == "active"),
            "completed": sum(1 for s in signups if s["status"] == "completed"),
        },
        "scores": scores,
    }


@app.get("/api/runs/{run_id}/debug")
def api_run_debug(run_id: str):
    r = store.get_run(run_id)
    if not r:
        raise HTTPException(404, "run not found")
    signups = store.list_run_signups(run_id)
    turns = store.list_run_turn_debug(run_id)
    events = store.list_run_events(run_id, max_events=300)
    return {
        "run_id": run_id,
        "status": r["status"],
        "connected": bool(signups),
        "signups": [{
            "signup_id": s["id"],
            "agent_id": s["agent_id"],
            "display_name": s["display_name"],
            "agent_status": s.get("agent_status"),
            "status": s["status"],
            "seat": s.get("seat"),
            "created_utc": s.get("created_utc"),
            "updated_utc": s.get("updated_utc"),
            "waiting_expires_at": s.get("waiting_expires_utc"),
            "ready_deadline_at": s.get("ready_deadline_utc"),
            "last_poll_utc": s.get("last_poll_utc"),
            "last_event_id": s.get("last_event_id"),
        } for s in signups],
        "pending_turns": [{
            "turn_id": t["id"],
            "signup_id": t["signup_id"],
            "seat": t["seat"],
            "phase": t["phase"],
            "action_kind": t["action_kind"],
            "deadline_at": t["deadline_utc"],
            "claimed_at": t.get("claimed_utc"),
            "status": t["status"],
        } for t in turns if t["status"] == "pending"],
        "turns": [{
            "turn_id": t["id"],
            "signup_id": t["signup_id"],
            "game_instance_id": t["game_instance_id"],
            "seat": t["seat"],
            "phase": t["phase"],
            "action_kind": t["action_kind"],
            "status": t["status"],
            "deadline_at": t["deadline_utc"],
            "accepted": t.get("accepted"),
            "reply_action": t.get("reply_action"),
            "reasoning": t.get("reasoning"),
        } for t in turns],
        "recent_events": [_api_event(e) for e in events],
        "forfeits": [e for e in (_api_event(ev) for ev in events) if e["type"] == "forfeit"],
    }


def _enrich_transcript_turn_reasoning(run_id: str, gid: int, transcript: dict) -> dict:
    """Attach exact per-turn model telemetry to stored replay events when available."""
    enriched = copy.deepcopy(transcript)
    game_instance_id = f"{run_id}_game_{gid:03d}"
    queues: dict[tuple[str, int], list[dict]] = {}
    for event in store.list_run_events(run_id, max_events=1000):
        if event.get("game_instance_id") != game_instance_id or event.get("type") != "model_turn_completed":
            continue
        payload = event.get("payload") or {}
        seat = payload.get("seat")
        phase = str(event.get("phase") or "").lower()
        if seat is None or not phase:
            continue
        queues.setdefault((phase, int(seat)), []).append(payload)

    def normalized_text(value) -> str:
        return " ".join(str(value or "").split())

    def action_matches_event(payload: dict, replay_event: dict) -> bool:
        action = payload.get("action")
        event_type = replay_event.get("t")
        if event_type == "say":
            return isinstance(action, dict) and normalized_text(action.get("speak")) == normalized_text(replay_event.get("text"))
        if event_type == "pass":
            return isinstance(action, dict) and action.get("pass") is True
        if event_type == "vote":
            try:
                return int(action) == int(replay_event.get("tgt"))
            except (TypeError, ValueError):
                return False
        if event_type == "act":
            return payload.get("ok") is True
        return False

    for phase in enriched.get("phases", []):
        phase_key = str(phase.get("name") or phase.get("kind") or "").lower()
        for event in phase.get("events", []):
            pid = event.get("pid")
            if pid is None or event.get("t") not in {"act", "say", "pass", "vote"}:
                continue
            queue = queues.get((phase_key, int(pid))) or []
            if not queue:
                continue
            match_idx = next((i for i, candidate in enumerate(queue) if action_matches_event(candidate, event)), None)
            if match_idx is None:
                continue
            payload = queue.pop(match_idx)
            if payload.get("reasoning"):
                event["declared_reasoning"] = payload["reasoning"]
            if payload.get("provider_reasoning") is not None:
                event["provider_reasoning"] = payload["provider_reasoning"]
            if payload.get("provider_reasoning_details") is not None:
                event["provider_reasoning_details"] = payload["provider_reasoning_details"]
            if payload.get("raw"):
                event["raw_model_output"] = payload["raw"]
            if payload.get("action_kind"):
                event["action_kind"] = payload["action_kind"]
            if payload.get("model"):
                event["model"] = payload["model"]
    return enriched


@app.get("/api/runs/{run_id}/games/{gid}")
def api_game(run_id: str, gid: int):
    r = store.get_run(run_id)
    if not r:
        raise HTTPException(404, "run not found")
    # SPEC D7: a sharded parent's game lives on whichever child holds that global gid; resolve it
    # there. A normal/child run resolves against itself.
    for sid in _source_run_ids(r):
        t = store.get_game(sid, gid)
        if t:
            return _enrich_transcript_turn_reasoning(sid, gid, t)
    raise HTTPException(404, "game not found")



def _roster_models() -> list[str]:
    seen, out = set(), []
    for s in SETTINGS.roster():
        if s.model not in seen:
            seen.add(s.model)
            out.append(s.model)
    return out


_models_cache: dict = {"at": 0.0, "data": []}


def _hosted() -> bool:
    return bool(os.environ.get("VERCEL"))


def _browser_key_management_enabled() -> bool:
    """Local dev may save/test a browser-submitted key. Hosted deployments may only use
    server-side env secrets and must not accept browser-submitted keys."""
    return not os.environ.get("VERCEL")


def _fetch_models() -> list[dict]:
    """OpenRouter's full model catalog as [{id, name}], cached ~10 min.

    The /models endpoint is public, so a key is optional; on any failure we return the
    last good cache (or [] on the very first miss) so the picker degrades gracefully.
    """
    now = time.time()
    if _models_cache["data"] and now - _models_cache["at"] < 600:
        return _models_cache["data"]
    headers = {}
    if config.has_api_key():
        headers["Authorization"] = f"Bearer {config.get_api_key()}"
    try:
        r = httpx.get(OPENROUTER_BASE_URL + "/models", headers=headers, timeout=15)
        r.raise_for_status()
        models = [{"id": m["id"], "name": m.get("name") or m["id"]}
                  for m in r.json().get("data", []) if m.get("id")]
        models.sort(key=lambda m: m["id"].lower())
        _models_cache.update(at=now, data=models)
        return models
    except Exception:
        return _models_cache["data"]


def _test_openrouter() -> dict:
    """Live-validate the configured key against OpenRouter.

    Hosted deployments intentionally redact account metadata (usage/limit/label) because
    this endpoint is public. Local dev keeps the richer response for convenience.
    """
    if not config.has_api_key():
        return {"ok": False, "error": "no key configured"}
    try:
        r = httpx.get(OPENROUTER_BASE_URL + "/key",
                      headers={"Authorization": f"Bearer {config.get_api_key()}"}, timeout=10)
        if r.status_code == 200:
            if _hosted():
                return {"ok": True, "configured": True, "source": "server_env"}
            d = r.json().get("data", {})
            return {"ok": True, "label": d.get("label"),
                    "usage": d.get("usage"), "limit": d.get("limit"),
                    "is_free_tier": d.get("is_free_tier")}
        if r.status_code in (401, 403):
            return {"ok": False, "error": "invalid key"}
        return {"ok": False, "error": f"status {r.status_code}"}
    except Exception as e:
        return {"ok": False, "error": type(e).__name__}


@app.get("/api/keys")
def api_keys():
    configured = config.has_api_key()
    return {"provider": "openrouter", "base_url": OPENROUTER_BASE_URL,
            "configured": configured, "models": _roster_models(),
            "server_key_management": _browser_key_management_enabled(),
            "key_source": "server_env" if _hosted() and configured else "local_env" if configured else None}


@app.get("/api/models")
def api_models():
    """Full OpenRouter model catalog for the new-run picker. Falls back to the roster's
    models if OpenRouter is unreachable so the picker is never empty."""
    models = _fetch_models()
    if not models:
        models = [{"id": m, "name": m} for m in _roster_models()]
    return {"models": models}


@app.post("/api/keys/test")
def api_keys_test():
    return _test_openrouter()


@app.post("/api/keys")
def api_keys_set(payload: dict):
    if not _browser_key_management_enabled():
        raise HTTPException(403, "browser-submitted keys are disabled on hosted deployments")
    key = (payload.get("api_key") or "").strip()
    if not key:
        raise HTTPException(400, "empty key")
    config.set_api_key(key)   # persists to local .env; value never logged
    _or._client = None        # reset cached OpenAI client so it uses the new key
    return {"configured": True, **_test_openrouter()}


@app.post("/api/run")
def api_launch(payload: dict):
    """Launch a run in the background. Body: {game, games, agents?:[{name,model,harness}]}."""
    if os.environ.get("DATABASE_URL") or payload.get("remote"):
        return _queue_run(payload, _job_owner(payload))

    game = payload.get("game", "onuw")
    games = int(payload.get("games", 6))
    rounds = _payload_int(payload, ("rounds",), 20, positive=True)
    caps = _caps_from_payload(payload, rounds)
    deck_preset = _deck_preset_from_payload(game, payload)
    deal_schedule = _deal_schedule_from_payload(game, payload)
    seed = int(payload.get("seed") or int(time.time()) % 1000000)
    run_id = (payload.get("run_id") or f"run_{seed}_{uuid.uuid4().hex[:6]}").strip()
    agents = payload.get("agents")
    roster = [AgentSpec(name=a["name"], model=a["model"], harness=a.get("harness", "base"))
              for a in agents] if agents else None

    def go():
        from .batch import run_batch
        with _local_run_lock:
            _local_active_runs.add(run_id)
        try:
            run_batch(game=game, n_games=games, seed_base=seed, run_id=run_id,
                      roster=roster, workers=8, discussion_rounds=rounds,
                      deck_preset=deck_preset, caps=caps,
                      run_config_overrides=_run_config_overrides(payload),
                      stream_events=True, deal_schedule=deal_schedule)
        except Exception as e:  # surface a failed run rather than vanishing
            store.update_run_status(run_id, "partial")
            print(f"[{run_id}] run failed: {e}", flush=True)
        finally:
            with _local_run_lock:
                _local_active_runs.discard(run_id)

    threading.Thread(target=go, daemon=True).start()
    return {"run_id": run_id, "status": "running", "game": game, "games": games,
            "deck_preset": deck_preset, "deal_schedule": deal_schedule,
            "run_config": _run_config_meta(caps, rounds)}


@app.post("/api/jobs/claim")
def api_claim_job(payload: dict, authorization: str | None = Header(None)):
    owner = _authorized_owner(authorization, payload.get("owner"))
    worker_id = (payload.get("worker_id") or "").strip()
    if not worker_id:
        raise HTTPException(400, "worker_id required")
    lease_seconds = int(payload.get("lease_seconds") or 300)
    return {"job": store.claim_job(owner, worker_id, lease_seconds)}


@app.post("/api/jobs/heartbeat")
def api_heartbeat_job(payload: dict, authorization: str | None = Header(None)):
    _authorized_owner(authorization, payload.get("owner"))
    job_id = (payload.get("job_id") or "").strip()
    worker_id = (payload.get("worker_id") or "").strip()
    if not job_id or not worker_id:
        raise HTTPException(400, "job_id and worker_id required")
    ok = store.heartbeat_job(job_id, worker_id, int(payload.get("lease_seconds") or 300))
    if not ok:
        raise HTTPException(409, "heartbeat rejected")
    return {"ok": True}


@app.post("/api/ingest")
def api_ingest(payload: dict, authorization: str | None = Header(None)):
    owner = _authorized_owner(authorization, payload.get("owner"))
    job_id = (payload.get("job_id") or "").strip()
    if job_id:
        job = store.get_job(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        if job["owner"] != owner:
            raise HTTPException(403, "job owner mismatch")
    run_id = (payload.get("run_id") or "").strip()
    gid = int(payload.get("gid") or 0)
    transcript = payload.get("transcript")
    agents = payload.get("agents")
    if not run_id or gid <= 0 or not isinstance(transcript, dict) or not isinstance(agents, list):
        raise HTTPException(400, "run_id, gid, transcript, and agents are required")
    inserted = store.save_game(run_id, gid, transcript, agents)
    return {"ok": True, "inserted": bool(inserted)}


@app.post("/api/jobs/complete")
def api_complete_job(payload: dict, authorization: str | None = Header(None)):
    owner = _authorized_owner(authorization, payload.get("owner"))
    job_id = (payload.get("job_id") or "").strip()
    worker_id = (payload.get("worker_id") or "").strip()
    status = (payload.get("status") or "done").strip()
    if not job_id or not worker_id:
        raise HTTPException(400, "job_id and worker_id required")
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job["owner"] != owner:
        raise HTTPException(403, "job owner mismatch")
    ok = store.finish_job(job_id, worker_id, status, payload.get("error"))
    if not ok:
        raise HTTPException(409, "completion rejected")
    return {"ok": True, "status": status}


@app.get("/")
def index():
    return FileResponse(
        WEB / "observer.html",
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
        },
    )


app.mount("/", StaticFiles(directory=str(WEB)), name="web")
