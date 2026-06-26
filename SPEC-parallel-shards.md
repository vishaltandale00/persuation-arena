# SPEC — Parallel games within a run, via sharded sub-runs (Verifier-Driven)

**Goal:** let one logical run of N games execute as K concurrent child runs ("shards"), each a
normal connected run playing a disjoint slice of one global schedule, presented to the world as a
single parent run. Wall-clock drops ~K×; correctness, fairness, scoring, and rating are unchanged.

---

## §0 — How to read this spec (Verifier-Driven Development)

Every requirement `REQ-n` is bound to an **executable verifier** `V-n` (a named test or a real
command). The requirement is **not "done" until its verifier is green when actually run.** Prose is
not evidence; a passing verifier is. Verifiers are specified concretely enough to be written
**before or alongside** the implementation. The single source of truth for "the feature works" is
the **Acceptance Gate** in §5 — a real, executed command whose exit code is the verdict.

Three rules:
1. **Write the verifier first.** For each REQ, the test in `V-n` should exist and *fail* before the
   implementation, then *pass* after. No requirement ships without its verifier committed.
2. **No verifier may weaken an invariant.** The global invariants in §2 have standing guards that
   must stay green for the entire build, not just at the end.
3. **The gate is executed, not described.** "Done" means §5 was run and exited 0 (and the
   Postgres-gated and manual gates were run where applicable), with output captured.

---

## §1 — Locked decisions (settled; do not relitigate)

| ID | Decision |
|----|----------|
| D1 | **Decentralized topology.** Orchestrator fires K children and exits; each child is an independent coordinator. **Parent status is computed-on-read** (`rollup_parent_status`); no process owns the parent. |
| D2 | **`run_kind` enum** column on `runs`: `'normal'` (default) / `'parent'` / `'child'`. The orchestrator is the **sole writer**. Discovery (`list_open_runs`) returns **only `run_kind='normal'`**. |
| D3 | **K is explicit** (`--shards K` / `shards` payload field), **default 1**, **hard cap = 4** (config/env, tunable). **No divisibility constraint** on N, K, or players. |
| D4 | **Stride slicing** (`g % K == k`). **Completed games of a partially-failed logical run still feed global Elo.** Rating stays sharding-ignorant. |
| D5 | **Deterministic seats** via an *optional explicit seat index* on signup, supplied by the orchestrator. **Normal/discovered runs keep arrival-order assignment, untouched.** |
| D6 | Logic verified **sequentially on SQLite** in CI; concurrent-writer safety verified on **Postgres (gated on `DATABASE_URL`)**; real multi-process bring-up verified by a **manual Modal smoke**. |
| D7 | **Sharding is invisible in the observer.** Parent renders as one normal run; its overview aggregates children (games union, `score_runs(children)`, rolled-up status). |
| D8 | Child carries **shard coordinates** (`shard_index`, `num_shards`, shared `seed_base`); the runner slices via `fresh_deal_schedule`. **Global N is stored in the child's `n_games`.** |
| D9 | **Process per shard** in production (K host processes × N agent threads). Credential-sharing across an identity's K shards is **file-based** (pre-register once → cred file → each shard process reads it → same `agent_id`). |
| D10 | **Independent startup** (no cross-shard barrier). Top-level launcher **blocks-and-monitors** as a *non-load-bearing* convenience (poll `rollup`, report per-shard health, print aggregated scorecard). |

---

## §2 — Global invariants (standing guards, must stay green throughout)

| ID | Invariant | Guard (verifier) |
|----|-----------|------------------|
| INV-1 | **K=1 no-op.** A sharded run with K=1 produces byte-identical games to a plain `run_connected_batch`. | `V-9` |
| INV-2 | **Normal runs unchanged.** Non-sharded runs (no `run_kind`/parent/shard cols set) behave exactly as today, including arrival-order seat assignment. | existing `tests/test_connected_runner.py`, `tests/test_runs.py` pass **unchanged**; `V-7` normal-path case |
| INV-3 | **`rating.py` is not modified.** Cross-run Elo keeps keying on `agent_id`; no parent/child awareness added. | `git diff --stat` shows `arena/rating.py` unchanged; `tests/test_rating.py` passes unchanged |
| INV-4 | **Children are never publicly discoverable or joinable by a stray agent.** | `V-2` |
| INV-5 | **Migrations are additive and idempotent on both backends.** Re-running `init_schema()` / opening a SQLite conn twice is a no-op; existing rows survive. | `V-8` (PG) + `V-1`-adjacent SQLite open-twice assertion |

---

## §3 — Implementation surface

| File | Change |
|------|--------|
| `arena/store.py` | Add `run_kind` (default `'normal'`), `parent_run_id`, `shard_index`, `num_shards` to `runs` in **all three sites**: PG `CREATE TABLE` (~L105), `PG_MIGRATION_STMTS` (~L180), `_MIGRATIONS["runs"]` (~L170). Thread through `save_run`/`create_connected_run`. Add `child_run_ids(parent_id)`. Add optional `seat` to `create_signup`, honored in `_maybe_ready_required` (sort by explicit seat when all present; else arrival order). Filter `list_open_runs` to `run_kind='normal'`. |
| `arena/batch.py` | `fresh_deal_schedule(n_games, n_players, seed_base, *, shard_index=None, num_shards=None)` → when both set, return the stride slice (`g % num_shards == shard_index`) with **global** `gid/seed/rot` preserved. |
| `arena/connected.py` | `run_connected_batch`: when the run row has `shard_index`/`num_shards`, build the schedule from `n_games`(=global N) + `seed_base` sliced to this shard. No other change. |
| `arena/sharded.py` *(new)* | `create_sharded_run(parent_config, num_shards) -> list[str]` (writes parent `run_kind='parent'` + K children `run_kind='child'`, K validated ≤ cap); `rollup_parent_status(parent_id) -> str`. Prod: spawn K `modal_app.spawn_run_server`. |
| `arena/score.py` | `score_runs(run_ids: list[str]) -> dict` (union `player_rows`, existing per-agent aggregation). `score_run(id)` → `score_runs([id])`. |
| `arena/server.py` | `/api/runs` (index) shows parents, hides children; parent read-paths (`/api/runs/{id}`, games list, transcript) query across `child_run_ids`; run-creation accepts `shards`. |
| `tools/coding_agent_run.py` | `--shards K`: create parent+children; pre-register N identities → N cred files; spawn K shard subprocesses (single-shard host each, pointed at `child_k`, reading the cred files); block-monitor `rollup` and print `score_runs(children)`. |
| `arena/rating.py` | **UNCHANGED** (INV-3). |

---

## §4 — Requirements & Verifiers

> Test convention (from `tests/test_connected_runner.py`): `_sqlite_store(tmp_path, monkeypatch)`
> points the store at a temp SQLite + `init_schema()`; connected runs are driven by a scripted
> `responder()` thread that polls `pending_turn_for_signup` and replies via `reply_to_turn` — **no
> LLMs, no network**. All §4 verifiers except V-8/V-10 run under default `pytest` on SQLite.

### REQ-1 — Schedule slicing is correct and fair
`fresh_deal_schedule` sliced by `(shard_index, num_shards)` partitions the global schedule with no
loss, no overlap, and global `gid/seed/rot` preserved.
- **V-1** — `tests/test_runs.py::test_shard_slice_partitions_global_schedule`
  - `union(slice(k) for k in range(K)) == fresh_deal_schedule(N,n,seed)` as a set of `(gid,seed,rot)`.
  - Slices are pairwise disjoint on `gid`.
  - For every entry in every slice, `(seed,rot)` equals the global schedule's entry for that `gid`.
  - Edge cases: `K=1` ⇒ slice == full; `K=N` ⇒ each slice length 1; `N % K != 0` ⇒ slice sizes differ by ≤ 1.
  - Pure function; no DB.
- **Run:** `pytest tests/test_runs.py::test_shard_slice_partitions_global_schedule -q`

### REQ-2 — Parent status rollup
`rollup_parent_status` reflects children: all `done`→`done`; any not-done→`partial` if any child is
`partial`/has failures, else `running`/`open`.
- **V-2** — `tests/test_sharded.py::test_rollup_parent_status` and `::test_children_not_discoverable`
  - `{done,done}→done`; `{done,partial}→partial`; `{done,running}→running`; child-with-failures→`partial`.
  - `list_open_runs()` excludes `run_kind in {'parent','child'}`; a `discover_runs`-style call never returns a child or parent (INV-4).
- **Run:** `pytest tests/test_sharded.py -q -k "rollup or discoverable"`

### REQ-3 — Sharded run completes end-to-end
A K=2 sharded run, driven by scripted responders, completes with every global game present exactly
once and the parent rolled up to `done`.
- **V-3** — `tests/test_sharded.py::test_sharded_run_end_to_end`
  - `create_sharded_run(parent_cfg, 2)` writes 1 parent + 2 children with correct `run_kind`/shard cols.
  - N=5 identities registered with **shared creds** (same `agent_id` across both children); seated; scripted responders reply per child; each child's `run_connected_batch` run (sequentially per D6).
  - Assert: both children `done`; `{g for child in children for g in distinct_gids(child)} == {1..N}` with no gid appearing in two children; `rollup_parent_status(parent)=='done'`; events partition by `game_instance_id`; `score_runs(children)` returns one scorecard per agent with `overall.n` summing across shards.
- **Run:** `pytest tests/test_sharded.py::test_sharded_run_end_to_end -q`

### REQ-4 — **Equivalence (keystone)**: sharded == unsharded in game outcomes & scores
With identical roster (deterministic seats), seed, deck, rounds, and deterministic scripted policy,
a sharded run produces the **same games and the same scores** as an unsharded run of the same N.
- **V-4** — `tests/test_sharded.py::test_equivalence_sharded_vs_unsharded`
  - Run A: one `run_connected_batch` of N games (unsharded).
  - Run B: `create_sharded_run` with K (children summing to N), each child coordinated with the *same* scripted policy and *same* explicit seat indices.
  - Assert per global `gid`: saved transcript outcome is identical between A and B — `winner_team`, outcome `line`, and each seat's `dealt_role`/`end_role`/`won`.
  - Assert `score_runs(B.children) == score_run(A.run)` for every agent: identical `overall`/`good`/`evil`/`by_role` `(w,n)`.
  - **Do NOT assert Elo equality** (replay-order differs by design, D4).
- **Run:** `pytest tests/test_sharded.py::test_equivalence_sharded_vs_unsharded -q`

### REQ-5 — Identity linkage: one competitor across K shards
Shared credentials make an identity a single rating competitor spanning all shards; unshared creds
fracture it (documented failure mode).
- **V-5** — `tests/test_identity_linkage.py::test_shared_creds_one_competitor`, `::test_unshared_creds_fracture`
  - Shared creds: one `agent_id` appears in **every** child's `game_players`; `score_runs(children)` shows that agent with `overall.n == N` (its full game count).
  - Negative: distinct creds per shard ⇒ K distinct `agent_id`s, each with `~N/K` games (proves the linchpin is load-bearing).
- **Run:** `pytest tests/test_identity_linkage.py -q -k "shared_creds or fracture"`

### REQ-6 — Per-process coordinator routing (process isolation correctness)
A single-shard agent host holds signups for **exactly one** run and routes poll/reply to that run's
coordinator URL only. (Process isolation per D9 is what makes the SDK's single `_coord_client` safe.)
- **V-6** — `tests/test_agent_sdk.py::test_single_shard_host_routes_to_own_coordinator`
  - An `ArenaAgent` given one signup + that run's `coordinator_url` repoints (`_maybe_repoint`) to that URL and never to another run's.
  - Assert the host's signup set has cardinality 1 (it never accumulates cross-run signups).
- **Run:** `pytest tests/test_agent_sdk.py::test_single_shard_host_routes_to_own_coordinator -q`

### REQ-7 — Deterministic seats (additive; normal runs untouched)
An explicit seat index at signup is honored; identity→seat is identical across shards; absent an
explicit seat, assignment is arrival-order as today.
- **V-7** — `tests/test_connected_runner.py::test_explicit_seat_honored`, `::test_arrival_order_unchanged`
  - `create_signup(..., seat=i)` ⇒ after `_maybe_ready_required`, that signup's `seat == i`.
  - Two children with the same roster indices ⇒ identical identity→seat map (supports INV-1/V-4).
  - No explicit seat ⇒ seats assigned by `created_utc` order, byte-identical to current behavior (INV-2).
- **Run:** `pytest tests/test_connected_runner.py -q -k "seat"`

### REQ-8 — Dual-backend migration + concurrent distinct-`run_id` writers (Postgres)
New columns apply on Postgres; K threads writing **different** `run_id`s concurrently never
cross-contaminate. (Verifies B's structural premise: shards are safe because each is its own run.)
- **V-8** — `tests/test_store_pg.py::test_sharded_columns_on_pg`, `::test_concurrent_distinct_run_writers`
  - Gated on `DATABASE_URL` (skipped otherwise, per existing `test_store_pg.py`).
  - After `init_schema()`, `runs` has `run_kind`/`parent_run_id`/`shard_index`/`num_shards`; `run_kind` defaults `'normal'`.
  - K threads each `save_game`/`append_event` to a distinct `run_id` concurrently ⇒ every row lands, each `run_id` has exactly its own games/events, no exceptions.
  - `score_runs` and `rollup_parent_status` queries execute on PG.
- **Run:** `DATABASE_URL=$NEON_TEST_URL pytest tests/test_store_pg.py -q -k "sharded or concurrent"`

### REQ-9 — K=1 regression / no-op (INV-1)
- **V-9** — `tests/test_sharded.py::test_k1_matches_plain_run` **and** the existing
  `tests/test_connected_runner.py` suite **passing unchanged**.
  - A K=1 `create_sharded_run` coordinated end-to-end produces the same games as a direct
    `run_connected_batch` of the same N/seed/roster.
- **Run:** `pytest tests/test_sharded.py::test_k1_matches_plain_run tests/test_connected_runner.py -q`

### REQ-10 — Real multi-process Modal smoke (manual gate)
Two real coordinator containers + two shard host processes run a small sharded run.
- **V-10** — `tools/connected_modal_smoke.py --shards 2` (or documented equivalent). **Not in CI.**
  - Both children reach `running` before either reaches `done` (structural concurrency).
  - Final `rollup_parent_status == 'done'`; union games == N; observer parent overview renders one run.
- **Run (manual):** documented in the tool's `--help`; requires deployed Modal + `DATABASE_URL`.

---

## §5 — Acceptance Gate (executed, not described)

**Primary gate (must exit 0):**
```
PYTHONPATH=. .venv/bin/pytest tests/ -q
```
This includes the new `tests/test_sharded.py` and the extended `test_runs.py`,
`test_identity_linkage.py`, `test_agent_sdk.py`, `test_connected_runner.py`, and must show the
existing suite still green (INV-1/INV-2/INV-3 regression).

**Invariant spot-check (must hold):**
```
git diff --stat arena/rating.py        # expect: no changes (INV-3)
```

**Postgres gate (run when a test Neon URL is available; must exit 0):**
```
DATABASE_URL=$NEON_TEST_URL PYTHONPATH=. .venv/bin/pytest tests/test_store_pg.py -q
```

**Manual gate (run once before any production sharded run):**
```
# deployed Modal + DATABASE_URL set
PYTHONPATH=. .venv/bin/python tools/connected_modal_smoke.py --shards 2
```

The feature is **DONE** iff the primary gate exits 0, the invariant spot-check shows
`arena/rating.py` unchanged, the Postgres gate exits 0 on a reachable Neon, and V-10 has been run
manually with its assertions met. Capture and report the actual command output for each.

---

## §6 — Build order (each step gated by its verifier before the next)

1. **Schema** (store.py, 3 sites) + INV-5 idempotency check + V-8 SQLite half.
2. **Schedule slice** (`fresh_deal_schedule`) → **V-1** + **V-9** (K=1 no-op holds immediately).
3. **`score_runs`** (score.py) → score half of V-4/V-5.
4. **`rollup_parent_status`** + discovery filter (sharded.py, store.py) → **V-2**.
5. **`create_sharded_run`** + `run_connected_batch` slicing → **V-3** with scripted responders.
6. **Equivalence keystone → V-4.** Nothing proceeds to prod paths until this is green.
7. **Shared-creds linkage → V-5**; per-process routing → **V-6**.
8. **Deterministic seats** (store.py) → **V-7** (and re-confirm INV-2).
9. **Launcher `--shards`** + K-subprocess host + Modal spawn fan-out (tools/, server.py).
10. **V-8 on Postgres** before any prod sharded run; **V-10 manual smoke**.

---

## §7 — Out of scope (v1)

- External-participant sharding (the `--external` flow signing a human-run agent into all K shards):
  host-run frontier suite only for v1.
- Changing rating replay order to make sharded Elo equal a hypothetical unsharded run (D4 keeps
  rating sharding-ignorant; the difference is test-scope only).
- Auto-tuning K from N or live account limits (K stays explicit, capped).
