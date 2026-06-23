"""Serve one connected run's API and coordinate its games in a single long-lived process.

This is the payload the Modal per-run server (arena/modal_app.py) wraps, kept Modal-agnostic so it's
also runnable/testable locally. The Modal wrapper opens a Tunnel and registers the public URL around
these calls; nothing here imports modal.

Flow inside one process/container:
  ensure_run(cfg)            create the connected run row if it doesn't exist (status 'open')
  start_api_server(...)      run arena.server:app under uvicorn in a daemon thread (agents hit this)
  wait_for_active(...)       block until all agents have signed up + readied (status 'active')
  coordinate(...)            run the game engine to completion (run_connected_batch)
  stop_api_server(...)       shut the server down so the process/container can exit (scale to zero)

The API server and the coordinator share the same store backend (Neon when DATABASE_URL is set,
else local SQLite) — they rendezvous through it exactly as today.
"""
from __future__ import annotations

import threading
import time

import uvicorn

from . import store
from .connected import run_connected_batch


def ensure_run(run_config: dict) -> None:
    """Create the connected run row if absent. run_config keys: id, game, label, status, n_games,
    players, seed_base, submitter, deck_preset (see store.create_connected_run)."""
    if not store.get_run(run_config["id"]):
        store.create_connected_run(run_config)


def start_api_server(host: str = "0.0.0.0", port: int = 8000,
                     startup_timeout: float = 60.0) -> uvicorn.Server:
    """Start arena.server:app under uvicorn in a daemon thread; return once it's accepting connections.

    Off the main thread uvicorn cannot install signal handlers (it would raise 'set wakeup fd only
    works in main thread'), so we disable them.
    """
    config = uvicorn.Config("arena.server:app", host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None
    threading.Thread(target=server.run, daemon=True, name="uvicorn").start()
    deadline = time.time() + startup_timeout
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError(f"uvicorn did not start within {startup_timeout:.0f}s")
        time.sleep(0.05)
    return server


def _seated_active(run_id: str) -> int:
    return sum(1 for s in store.list_run_signups(run_id, statuses={"active"}) if s.get("seat") is not None)


def wait_for_active(run_id: str, need: int, timeout_s: float = 300.0) -> int:
    """Block until `need` agents are seated+active, or timeout. The coordinator DRIVES activation
    (a single race-free promotion of ready->active) rather than waiting for the agents to each
    self-promote — that agent-side promotion is racy across servers on Neon. Returns the active
    count (may be < need on timeout — the caller decides whether to proceed)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        n = store.activate_run_if_ready(run_id)
        if n >= need:
            return n
        time.sleep(0.5)
    return store.activate_run_if_ready(run_id)


def coordinate(run_id: str, rounds: int = 5) -> dict:
    """Drive the game engine for the run to completion, then return the final run record."""
    run_connected_batch(run_id, discussion_rounds=rounds)
    return store.get_run(run_id)


def stop_api_server(server: uvicorn.Server) -> None:
    server.should_exit = True


def serve_and_coordinate(run_config: dict, *, host: str = "0.0.0.0", port: int = 8000,
                         rounds: int = 5, wait_timeout: float = 300.0, on_serving=None) -> dict:
    """Local (no-tunnel) all-in-one, for testing without Modal. `on_serving(local_url)` fires once the
    server is up (so a caller can point agents at it). The Modal wrapper does NOT use this — it drives
    the lower-level functions itself so it can hold the tunnel open across wait+coordinate."""
    ensure_run(run_config)
    server = start_api_server(host, port)
    try:
        if on_serving:
            on_serving(f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}")
        n = wait_for_active(run_config["id"], int(run_config["players"]), wait_timeout)
        if n < int(run_config["players"]):
            raise RuntimeError(f"only {n}/{run_config['players']} agents active before timeout")
        return coordinate(run_config["id"], rounds)
    finally:
        stop_api_server(server)
