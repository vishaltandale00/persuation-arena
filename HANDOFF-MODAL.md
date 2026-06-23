# Handoff — Modal per-run hosting (since the fork)

**Workstream split:** this agent owned **Modal per-run hosting**; the other agent owns the
**delta / pure-event-sourcing statefulness contract**. Hosting is orthogonal to the wire contract —
the container just serves `arena.server:app` and runs `run_connected_batch`; both evolve under the
contract work and the Modal wrapper is unchanged as long as `run_connected_batch(run_id,
discussion_rounds=...)` keeps its signature.

## TL;DR
A per-run Modal coordinator is **built, deployed, and proven** (a full game ran through a per-run
container → tunnel → agents → persisted to Neon → scale-to-zero). The production **spawn-trigger +
SDK discovery wiring** is added and unit-verified. The **joint end-to-end** (central spawns → agents
discover + re-point → play, with the *real stateful agents*) is intentionally **not run yet** — it's
the sync point that verifies both paths together.

---

## New files (purely additive — no conflict)
| File | Purpose |
|---|---|
| `arena/coordinator_service.py` | Modal-agnostic serve+coordinate core: `ensure_run`, `start_api_server` (uvicorn-in-thread, signal handlers off, waits `server.started`), `wait_for_active`, `coordinate` (→ `run_connected_batch`), `stop_api_server`. Locally testable without Modal. |
| `arena/modal_app.py` | The Modal app `persuasion-arena-coordinator`. `run_server(run_config, rounds, wait_timeout, drain_seconds)` = the per-run container; `spawn_run_server()` / `coordinator_url()` = external hooks; `cleanup_test_runs(prefix)` = in-container Neon cleanup. |
| `tools/connected_modal_smoke.py` | End-to-end topology smoke using the free scripted `random_agent` (no LLM cost, decoupled from `model_agent`). |

## Shared-file edits — **reconcile with the contract work**
All Modal behavior is **gated behind `ARENA_MODAL_COORDINATOR`** (off by default) and best-effort, so
these are non-breaking when the flag is unset.

| File | Change | Merge note |
|---|---|---|
| `arena/server.py` | Added `_modal_coordinator_enabled()` / `_spawn_coordinator()` / `_coordinator_url()` helpers; `_create_connected_run` now builds a `run_config` var and calls `_spawn_coordinator(...)` after create; `_signup_response` includes `coordinator_url`. | Additive; the contract work didn't touch these functions as of the fork. |
| `persuasion_arena_agent/agent.py` | `ArenaAgent` gains `self._coord_client` + `_maybe_repoint(signup)`; `signup()` calls it; **`step_signup` routes `ready`/`poll`/`reply` through `play = self._coord_client or self.client`** while status stays on the central client. | ⚠️ **MERGE POINT.** `step_signup` is also where the contract work added the `on_event` delivery loop. The current working tree already has **both** (re-pointing + on_event) coexisting — verify that merge is what you expect. |
| `persuasion_arena_agent/models.py` | `Signup` gains `coordinator_url: str | None`. | Additive. |

## Architecture (the topology)
```
create connected run (central API, ARENA_MODAL_COORDINATOR=1)
    └─ store.create_connected_run(run_config)        # run row -> Neon (status=open)
    └─ spawn_run_server(run_config)                  # fire-and-forget Modal Function.spawn (returns instantly)
           │
           ▼  Modal cold-starts ONE container for this run
       run_server(run_config):
         ensure_run (idempotent) → start uvicorn(arena.server:app) on 0.0.0.0:8000
         with modal.forward(8000) as tunnel:         # per-container HTTPS URL
             modal.Dict["arena-coordinator-urls"][run_id] = tunnel.url
             wait_for_active(run_id, players)         # agents seat + ready
             run_connected_batch(run_id, rounds)      # the game engine, against Neon
             sleep(drain_seconds)                     # let agents poll 'completed' + exit clean
         # tunnel closes, function returns → Modal scales container to zero

agents (external SDK):
    register + signup on the CENTRAL API → signup response carries coordinator_url (from the Dict)
    SDK _maybe_repoint → ready/poll/reply now go DIRECTLY to the container's tunnel URL
    (cold-start tolerant: until the URL appears, those ops fall back to central — both share Neon)
```

## Key decisions (from the grilling)
- **v1 = Neon-backed** (compute isolation + scale-to-zero; both central + container rendezvous via
  Neon). The "in-memory live state, flush to Neon at end" optimization is **deferred** (it overlaps
  the contract work's storage surface).
- **Per-run container via Modal Tunnels (`modal.forward`)** — verified from the installed 1.5.1 source
  to be the *only* mechanism giving a **per-invocation** URL (the `@modal.asgi_app`/`web_server`
  decorators give one stable per-*deployment* URL, wrong for per-run).
- **Spawn-at-creation**, URL handoff via `modal.Dict` (no `store.py` schema change), **cold-start
  tolerant** because central + container share Neon.
- Coordinator needs **only `DATABASE_URL`** (it makes no LLM calls) → secret **`neon-database-url`**.
- **No `@modal.concurrent`** — tunnel traffic bypasses Modal's input routing; one container per run,
  uvicorn handles HTTP concurrency.

## Deployed state
- Modal app **`persuasion-arena-coordinator`** (workspace `vishaltandale00`) — functions `run_server`,
  `cleanup_test_runs`.
- Modal secret **`neon-database-url`** (holds `DATABASE_URL` only).
- ⚠️ **Do not `modal deploy` again until the contract changes settle** — the live deployment is a
  working snapshot, and the working tree currently has the other agent's in-flight edits to
  `arena/connected.py`, `arena/games/onuw.py`, `persuasion_arena_agent/agent.py`, `examples/`.

## Verified ✅
- **Topology smoke** (`tools/connected_modal_smoke.py`, scripted agents): spawn → per-container tunnel
  → 5 agents connect *directly* → game coordinated → **persisted to Neon** (`status=done,
  games_saved=1`) → clean agent exit (drain grace) → scale-to-zero. Test runs cleaned up.
- **Wiring unit checks**: SDK re-point (no-url→central, url→re-point, first-URL-sticks/idempotent);
  server gating (off when unset → spawn/url are no-ops; on when `=1`).

## NOT yet verified — the sync point
- **Joint prod e2e**: central API with `ARENA_MODAL_COORDINATOR=1` → `POST /api/runs {connected}`
  spawns the container → agents pointed at the *central* API discover `coordinator_url`, re-point,
  and play on the container. **Run this after merging the contract work, with the real stateful
  `model_agent` agents** — it then verifies BOTH paths at once (Modal hosting *and* delta/event-
  sourced stateful play). The harness for it is ready (see below).

### How to run the joint e2e (when both paths are merged)
```bash
# 1. central API: Neon-backed + coordinator spawning ON
ARENA_MODAL_COORDINATOR=1 .venv/bin/python -m arena.cli serve --port 8090   # uses DATABASE_URL from .env

# 2. create a connected run (this spawns the per-run Modal container)
curl -s -XPOST localhost:8090/api/runs -H 'content-type: application/json' \
  -d '{"connected":true,"game":"onuw","players":5,"games":10,"rounds":10,"run_id":"joint_e2e_1"}'

# 3. point the real model agents at the CENTRAL API (server=http://localhost:8090, run_id=joint_e2e_1);
#    the SDK discovers coordinator_url from the signup response and re-points ready/poll/reply.
#    (tools/connected_sample.py already wires ModelAgent via agent.on_event/agent.act.)

# cleanup afterward (in-container, no local prod query):
.venv/bin/modal run arena/modal_app.py::cleanup_test_runs --prefix joint_e2e_
```
Requires Modal auth in the central process env (`~/.modal.toml` locally; `MODAL_TOKEN_ID` /
`MODAL_TOKEN_SECRET` on Vercel).

## Remaining Modal work (after the sync)
1. **Joint e2e** with `model_agent` (verifies both paths).
2. **Vercel env config** so the *live* site auto-spawns: set `ARENA_MODAL_COORDINATOR=1`,
   `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, then redeploy Vercel.
3. **In-memory live state** (container holds the event/turn log in memory, flush to Neon at end) — the
   perf follow-on that removes Neon from the per-poll hot path.
4. **Coordinator-crash → run restart** handling (deliberately deferred for v1).

## Modal 1.5.1 API facts (verified against the installed package source)
- `modal.forward(port)` — context manager → `Tunnel.url` (per-container HTTPS); bind `0.0.0.0`, start
  the server first (`server.started`), keep the loop **inside** the `with`. Only works **in-container**.
- Trigger: `modal.Function.from_name(app, fn).spawn(*args)` (fire-and-forget). **`.lookup` is removed.**
  External caller needs `MODAL_TOKEN_*`; app must be `modal deploy`-ed first.
- `modal.Secret.from_name(name, required_keys=[...])` → env vars in the container.
- Image: `Image.debian_slim().pip_install_from_pyproject("pyproject.toml").add_local_python_source(...)`
  — `add_local_python_source` ships **`.py` only**, so non-`.py` deps need `add_local_file` /
  `add_local_dir` (we ship `agents.yaml` + `web/` because `arena.server` mounts `StaticFiles(web)` at
  import). `modal.Mount` is gone.
- `@app.function(timeout=7200, scaledown_window=…, min_containers=0)` — default is scale-to-zero;
  default `timeout=300` **must** be raised. `container_idle_timeout`/`keep_warm`/`concurrency_limit`/
  `allow_concurrent_inputs` are hard-errors now.
