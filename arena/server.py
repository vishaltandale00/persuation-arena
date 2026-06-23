"""Local FastAPI server: serves runs/games JSON in the observer's shapes and the observer page.

  GET /api/runs                      -> run summaries for the index
  GET /api/runs/{id}                 -> run overview (agents, settings, results, games list)
  GET /api/runs/{id}/games/{gid}     -> full game transcript (observer shape)
  GET /                              -> serves web/observer.html and assets (same-origin fetch)

Run with:  uvicorn arena.server:app --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import hmac
import os
import re
import threading
import time
import uuid

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .config import ROOT, SETTINGS, OPENROUTER_BASE_URL, AgentSpec
from . import openrouter as _or
from . import store
from .batch import GAME_CORES
from .score import score_run

WEB = ROOT / "web"
GAME_LABELS = {
    "onuw": "One Night Ultimate Werewolf",
    "avalon": "The Resistance: Avalon",
    "secret_mafia": "Secret Mafia",
}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    store.init_schema()
    yield


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


def _queue_run(payload: dict, owner: str) -> dict:
    game = payload.get("game", "onuw")
    if game not in GAME_LABELS:
        raise HTTPException(400, f"unknown game: {game}")
    games = max(1, int(payload.get("games", payload.get("n_games", 6))))
    rounds = max(1, int(payload.get("rounds", 5)))
    seed = int(payload.get("seed") or (time.time() * 1000) % 1000000)
    run_id = (payload.get("run_id") or f"run_{seed}_{uuid.uuid4().hex[:6]}").strip()
    agents = _roster_from_payload(payload)
    core = GAME_CORES[game]
    n_players = len(agents)  # the roster IS the table — no fixed player count
    if not (core.MIN_PLAYERS <= n_players <= core.MAX_PLAYERS):
        raise HTTPException(400, f"{core.TITLE} supports {core.MIN_PLAYERS}–{core.MAX_PLAYERS} "
                                 f"players, got {n_players}")
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
    })
    return {"run_id": run_id, "job_id": job["id"], "status": "queued", "owner": owner,
            "game": game, "games": games, "rounds": rounds}


@app.get("/api/runs")
def api_runs():
    out = []
    for r in store.list_runs():
        out.append({
            "id": r["id"], "game": r["game"], "label": r["label"], "status": r["status"],
            "nGames": r["n_games"], "players": r["players"], "seed": r["seed_base"],
            "when": r["created"], "teamSplit": r["team_split"],
        })
    return out


@app.post("/api/runs")
def api_submit_run(payload: dict):
    """Queue a central run for a laptop worker. Model keys remain on the worker machine."""
    return _queue_run(payload, _job_owner(payload))


@app.get("/api/runs/{run_id}")
def api_run(run_id: str):
    r = store.get_run(run_id)
    if not r:
        raise HTTPException(404, "run not found")
    has_games = len(r["games"]) > 0
    agents = [{"name": a["name"], "model": a["model"], "harness": a.get("harness", "base"),
               "wins": r["wins"].get(a["name"], 0) if has_games else None}
              for a in r["agents"]]
    games = [{"gid": g["gid"], "seed": g["seed"], "win": g["winner_team"], "line": g["line"], "full": True}
             for g in r["games"]]
    # partial scores while a run is in progress, full when done
    scores = score_run(run_id) if has_games else {}
    return {
        "id": r["id"], "game": r["game"], "label": r["label"], "status": r["status"],
        "nGames": r["n_games"], "players": r["players"], "seedBase": r["seed_base"],
        "created": r["created"], "agents": agents, "teamSplit": r["team_split"], "games": games,
        "scores": scores,
    }


@app.get("/api/runs/{run_id}/games/{gid}")
def api_game(run_id: str, gid: int):
    t = store.get_game(run_id, gid)
    if not t:
        raise HTTPException(404, "game not found")
    return t


def _roster_models() -> list[str]:
    seen, out = set(), []
    for s in SETTINGS.roster():
        if s.model not in seen:
            seen.add(s.model)
            out.append(s.model)
    return out


_models_cache: dict = {"at": 0.0, "data": []}


def _server_keys_enabled() -> bool:
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
    if _server_keys_enabled() and config.has_api_key():
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
    """Live-validate the configured key against OpenRouter (never echoes the key)."""
    if not _server_keys_enabled():
        return {"ok": False, "error": "server key management is disabled on hosted deployments"}
    if not config.has_api_key():
        return {"ok": False, "error": "no key configured"}
    try:
        r = httpx.get(OPENROUTER_BASE_URL + "/key",
                      headers={"Authorization": f"Bearer {config.get_api_key()}"}, timeout=10)
        if r.status_code == 200:
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
    return {"provider": "openrouter", "base_url": OPENROUTER_BASE_URL,
            "configured": _server_keys_enabled() and config.has_api_key(), "models": _roster_models(),
            "server_key_management": _server_keys_enabled()}


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
    if not _server_keys_enabled():
        raise HTTPException(403, "server key management is disabled on hosted deployments")
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
    rounds = int(payload.get("rounds", 5))
    seed = int(time.time()) % 1000000
    run_id = f"run_{seed}"
    agents = payload.get("agents")
    roster = [AgentSpec(**a) for a in agents] if agents else None

    def go():
        from .batch import run_batch
        try:
            run_batch(game=game, n_games=games, seed_base=seed, run_id=run_id,
                      roster=roster, workers=8, discussion_rounds=rounds)
        except Exception as e:  # surface a failed run rather than vanishing
            store.update_run_status(run_id, "done")
            print(f"[{run_id}] run failed: {e}", flush=True)

    threading.Thread(target=go, daemon=True).start()
    return {"run_id": run_id, "status": "running", "game": game, "games": games}


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
    return FileResponse(WEB / "observer.html")


app.mount("/", StaticFiles(directory=str(WEB)), name="web")
