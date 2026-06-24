"""Per-run coordinator on Modal — one isolated, scale-to-zero container per run.

Topology:
  central API (Vercel+Neon) creates a run and fire-and-forgets `run_server.spawn(run_config)`.
  Modal cold-starts a container FOR THAT RUN which:
    1. serves that run's API (arena.server:app) under uvicorn,
    2. exposes it via a Modal Tunnel -> a UNIQUE public URL for this container,
    3. publishes the URL in a modal.Dict keyed by run_id (how agents discover where to play),
    4. waits for the agents to seat + ready, runs the game engine to completion (persisting to Neon),
    5. returns -> Modal scales the container to zero.
The agents (external SDK clients) talk directly to the container's tunnel URL; Neon is persistence.

Deploy:   .venv/bin/modal deploy arena/modal_app.py
Secret:   create `neon-database-url` holding DATABASE_URL (the coordinator never needs an LLM key — agents do):
          grep '^DATABASE_URL=' .env > /tmp/db.env && modal secret create neon-database-url --from-dotenv /tmp/db.env

Verified against modal 1.5.1 (modal.forward tunnels, Function.from_name(...).spawn, Image.add_local_*,
Secret.from_name, scaledown_window/timeout, modal.Dict).
"""
from __future__ import annotations

import modal

APP_NAME = "persuasion-arena-coordinator"
URL_DICT_NAME = "arena-coordinator-urls"
PORT = 8000

app = modal.App(APP_NAME)

# Image: install runtime deps from pyproject, then ship the local packages (.py only) + agents.yaml,
# which config.ROOT (=/root) reads for the batch roster. The connected coordinator doesn't use the
# roster, but shipping it keeps `import arena.*` total.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_file("agents.yaml", "/root/agents.yaml")
    # arena.server mounts StaticFiles(ROOT/"web") at import, so the web/ dir must exist in the
    # container even though the coordinator only serves /api/*.
    .add_local_dir("web", "/root/web")
    .add_local_python_source("arena", "persuasion_arena_agent")
)

# The Neon connection string (DATABASE_URL) — the coordinator reads/writes Neon; it makes no model
# calls, so no LLM key is ever shipped to Modal. This is just the credential, not a database we create.
db_secret = modal.Secret.from_name("neon-database-url", required_keys=["DATABASE_URL"])

# run_id -> public tunnel URL. The central API / agents read this to find the run's container.
coordinator_urls = modal.Dict.from_name(URL_DICT_NAME, create_if_missing=True)


@app.function(image=image, secrets=[db_secret], timeout=7200, scaledown_window=10)
def run_server(run_config: dict, rounds: int = 5, wait_timeout: float = 300.0,
               drain_seconds: float = 6.0) -> dict:
    """Serve + coordinate one run inside its own container. One spawn == one run == one container."""
    import time
    import modal as _modal  # tunnel is only available inside the container

    from arena import coordinator_service as cs
    from arena import store

    run_id = run_config["id"]
    players = int(run_config["players"])

    cs.ensure_run(run_config)
    server = cs.start_api_server("0.0.0.0", PORT)
    try:
        # The tunnel URL is live only for the duration of this `with` — keep the whole run inside it.
        with _modal.forward(PORT) as tunnel:
            coordinator_urls.put(run_id, tunnel.url)
            store.set_coordinator_url(run_id, tunnel.url)   # Neon handoff so the JS registry signup returns it
            print(f"[run_server] {run_id} serving at {tunnel.url}", flush=True)

            n = cs.wait_for_active(run_id, players, wait_timeout)
            if n < players:
                print(f"[run_server] {run_id} ABORT: only {n}/{players} agents active by deadline",
                      flush=True)
            else:
                print(f"[run_server] {run_id} all {n} agents active — coordinating {run_config['n_games']} games",
                      flush=True)
                cs.coordinate(run_id, rounds)
                # Grace so agents poll the 'completed' status and exit cleanly before we tear the
                # tunnel + server down (otherwise their in-flight final poll gets a disconnect).
                time.sleep(drain_seconds)
    finally:
        try:
            coordinator_urls.pop(run_id)
        except KeyError:
            pass
        store.set_coordinator_url(run_id, None)   # clear the Neon handoff on teardown
        cs.stop_api_server(server)

    run = store.get_run(run_id)
    result = {
        "run_id": run_id,
        "status": run["status"] if run else "unknown",
        "games_saved": len(store.distinct_gids(run_id)),
    }
    print(f"[run_server] {run_id} done: {result}", flush=True)
    return result


@app.function(image=image, secrets=[db_secret], min_containers=0, timeout=600)
@modal.concurrent(max_inputs=50)
@modal.asgi_app(label="api")
def api():
    """The central Arena API (arena.server:app), served on Modal — register/signup/ready/poll/reply,
    runs, models, keys. This is where ALL the Python lives now: the static observer (on Vercel) talks
    to this URL, and creating a connected run spawns a per-run coordinator Modal->Modal (no SDK or
    tokens on the frontend). DATABASE_URL comes from the neon secret."""
    import os
    os.environ["ARENA_MODAL_COORDINATOR"] = "1"   # enable coordinator spawning from the API
    from arena.server import app as web
    return web


@app.function(image=image, secrets=[db_secret], timeout=120)
def cleanup_test_runs(prefix: str = "modal_smoke_") -> dict:
    """Delete runs whose id starts with `prefix` (and their child rows) from the store.

    Runs IN the container against Neon, so it's a Neon-cleanup path that doesn't route a prod query
    through a local shell. Invoke:
      .venv/bin/modal run arena/modal_app.py::cleanup_test_runs --prefix modal_smoke_
    """
    from arena import store

    ph = store._ph()
    ids = [r["id"] for r in store.list_runs() if r["id"].startswith(prefix)]
    if not ids:
        print(f"[cleanup] no runs matching {prefix!r}", flush=True)
        return {"deleted_runs": 0, "ids": []}
    placeholders = ",".join([ph] * len(ids))
    with store.conn() as c:
        c.execute(f"DELETE FROM turn_replies WHERE turn_id IN "
                  f"(SELECT id FROM turns WHERE run_id IN ({placeholders}))", tuple(ids))
        for t in ("run_signups", "run_events", "game_players", "games", "jobs", "turns"):
            c.execute(f"DELETE FROM {t} WHERE run_id IN ({placeholders})", tuple(ids))
        c.execute(f"DELETE FROM runs WHERE id IN ({placeholders})", tuple(ids))
    print(f"[cleanup] deleted {len(ids)} runs: {ids}", flush=True)
    return {"deleted_runs": len(ids), "ids": ids}


def spawn_run_server(run_config: dict, rounds: int = 5, wait_timeout: float = 300.0) -> str:
    """Fire-and-forget a coordinator container for a run, from an EXTERNAL process (the central API).

    Returns the Modal FunctionCall id immediately (does NOT wait for the run). The caller needs Modal
    auth (MODAL_TOKEN_ID/MODAL_TOKEN_SECRET, or ~/.modal.toml) and this app must be `modal deploy`-ed.
    """
    fn = modal.Function.from_name(APP_NAME, "run_server")
    call = fn.spawn(run_config, rounds, wait_timeout)
    return call.object_id


def coordinator_url(run_id: str) -> str | None:
    """Look up where a run is being served (None until its container has opened its tunnel)."""
    return modal.Dict.from_name(URL_DICT_NAME, create_if_missing=True).get(run_id)
