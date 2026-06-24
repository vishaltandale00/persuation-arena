# Persuasion Arena — Rating & Leaderboard (execution goal)

North-star prompt that drives the build. Work **phase by phase**. A phase is **done** only when
its acceptance test passes — verified by a command or a screenshot, never by reading code. See
`web/observer.html` for the exact UI idiom and data shapes, `arena/score.py` for the existing
descriptive scorer this builds on, and `arena/games/base.py` for win resolution.

## Objective

Build a **cross-run rating system** that ranks the models/harnesses competing in the arena, keyed
by their **bearer-token identity** (`agent_id`), and surface it as a **leaderboard** that replaces
the runs-centric landing page. The rating must respect the real outcome structure of these games:
**not** a binary good/evil team result, but the **per-seat, role-specific win condition** that the
engine already resolves.

Two deliverables:
1. **The metric** — a rating per competitor, updated game-by-game across *all* runs, computed by a
   deterministic replay so it is reproducible and auditable.
2. **The UX** — a collapsible leaderboard (per-row expandable to a per-role breakdown) as the new
   landing page; the runs list demoted to a second collapsible section but still reachable.

## Why binary teams are wrong (the core modeling fact)

`compute_winners` (`arena/games/base.py:62-93`) returns **three independent booleans** —
`village`, `werewolf`, `tanner` — which are **not** a partition. Village + Tanner can co-win the
same game; both factions can lose to a Tanner-only win. So any rating built on "two teams, one
winner" (OpenSkill 2-team, head-to-head Elo) is a lossy, wrong projection.

The honest unit already exists: **`game_players.won ∈ {0,1}`** is the per-seat outcome of *that
seat's* objective (`player_won`, `base.py:96-101`). Rate that, conditioned on the difficulty of the
role the seat was dealt.

## The metric (locked)

**Objective-handicap Elo on per-seat `won`.** Skill is carried in **logits** (natural for the
sigmoid); displayed as a familiar Elo number `elo = 1500 + 173·skill` (173 ≈ 400/ln 10).

For each seat *i* dealt role *r* in a game:

```
expected  pᵢ = σ( sᵢ − d_r − ρᵢ )
update    sᵢ ← sᵢ + k·dampᵢ·(wonᵢ − pᵢ)
```

- `sᵢ` — competitor skill (the thing we rank on). Init 0.
- `d_r` — **role/objective difficulty handicap**, population-shared (NOT per agent). `d_r = −logit(w_r)`,
  `w_r` = baseline win rate of a seat dealt role *r*.
- `ρᵢ` — **resistance**: mean current skill of the seats opposing *i*'s objective.
  village ↔ werewolf-team resist each other; Tanner is resisted by the whole table.
- `k` — base step in logits (default `0.08`, tunable), scaled by an uncertainty factor `k_rd(rd)`
  (bigger when the competitor is provisional) and by `dampᵢ` (forfeit damping, below).
- **Within one game, compute every `pᵢ` from pre-game skills, then apply all deltas** — seats in the
  same game must not see each other's mid-game updates.

**`d_r` — Level 0 (ship this):** pooled base rate over all `game_players` rows grouped by
`dealt_role`, with a Beta prior `α≈2` (essential for rare roles like Tanner; without it a `0/3`
blows up to `±∞`). Bucket by `(game, players, deck_preset)`; if a `(bucket, role)` cell has too few
trials, **fall back** to the game-level then global cell. Recompute `d_r` in **batch**; skills update
**online** during replay. (Level 1 — joint logistic MLE of `sᵢ`+`d_r` with ridge + reference role
`d_Villager≡0` — is a documented later upgrade, not in scope here.)

**Uncertainty & ranking.** Carry a Glicko-style **rating deviation `rd`** per competitor: starts wide
(`rd0`), shrinks with games played. Leaderboard sorts by the **conservative lower bound**
`elo − 2·(173·rd)`. Competitors under **30 games** are flagged `provisional`.

**Forfeit damping.** Forfeits (`game_players.forfeits/calls`) are non-skill noise. `dampᵢ ∈ (0,1]`
shrinks the update when a seat forfeited heavily that game, so rating can't be farmed off defaulted
actions. Surface `forfeit_rate` on the board regardless.

**Identity resolution** (`identity_key` for a seat):
- connected agent → its **`agent_id`** (the bearer-token identity; `agents.token_hash = SHA256(token)`).
- static `agents.yaml` agent (no token) → synthetic stable key **`static:{model}:{harness}`**.
- Both kinds are rated in the same replay so `ρᵢ` is always well-defined and the board includes
  everyone; connected rows are the primary target.

## Execution protocol — run to completion (do not stop until built)

Execute autonomously with the full toolset. **Do not stop at phase boundaries, do not hand back for
approval between steps, and do not end the turn until the Definition of Done is met and verified —
or a listed blocker is hit.**

**Operating loop.** Maintain phase state with the task tools (one task per phase P0–P3). Repeat:
1. Read current repo state and the next unmet acceptance criterion.
2. Implement it (Write/Edit), small and runnable.
3. **Verify with tools — never claim done from reading code:**
   - Logic/migrations/engine: **Bash** (`pytest`, `python -m arena.rating recompute`, `sqlite3`
     queries, `curl`). Read real output.
   - UI: start the server (**Bash** `run_in_background`), `curl` the endpoints to confirm JSON
     shapes, then open `web/observer.html` via the **`claude-in-chrome`** tools and **screenshot**
     it. A phase that touches the interface is not done without a screenshot showing real data and a
     clean console.
4. If red, fix root cause and re-verify. If green, mark the task complete and **start the next phase
   in the same turn.**
5. When P0–P3 are green, do a final end-to-end pass, then summarize.

**Use a Workflow** only if a sub-problem benefits from parallel construction/adversarial review;
routine sequential coding is done directly.

**Only stop for a real blocker** (else pick a sensible default, note it, continue):
- A genuine product decision not answered by this brief and not defaultable.
- The same step fails >3 times after distinct fixes (report what was tried).
- A destructive/irreversible action needs consent.

## Definition of done (the gate)

Each verified by a command or screenshot, not by inspection:
- [ ] `pytest` green, including the new rating tests below.
- [ ] **P0:** `game_players` has `agent_id`/`signup_id` columns (SQLite + Postgres migrations applied
      on open); `save_game` persists them. A connected ONUW game writes rows with a non-null
      `agent_id` — confirmed by a `sqlite3` query. `/api/agents/register` accepts and stores a
      self-declared `model`/`harness`. `store.list_agents()` returns registered agents.
- [ ] **P1:** `python -m arena.rating recompute` rebuilds `ratings` + `rating_events` from the store
      by deterministic replay. **Running it twice yields byte-identical `ratings`** (determinism).
      Unit tests prove: a Tanner co-win raises *both* the village seats' and the Tanner seat's skill;
      a competitor that always wins its dealt role rises monotonically; heavy forfeits shrink the
      delta; `d_r` for a hard role (low base rate) exceeds an easy role's.
- [ ] **P2:** `GET /api/leaderboard` and `GET /api/agents/{agent_id}` return the documented shapes
      (`curl` confirms), and the per-run numbers reconcile with `score.py` for a known run.
- [ ] **P3:** `observer.html` opens on the **leaderboard** as the landing page. Both Leaderboard and
      Runs are **collapsible** sections (Leaderboard open, Runs collapsed by default; state persists).
      A leaderboard **row expands** to the three objective subtotals (village/werewolf/tanner) + a
      per-role table with win%, n, base rate, and **vs-spread**. Runs remains one click from the
      observer. **Verified by screenshot** with real data; console clean.
- [ ] No raw bearer token is ever printed, logged, returned, or committed (only `token_hash` stored).

## Schema (additive migrations — never rewrite shipped tables)

Follow the existing pattern: SQLite via `_MIGRATIONS` (ALTER on open), Postgres via
`PG_MIGRATION_STMTS` (`ADD COLUMN IF NOT EXISTS`).

- `game_players` += `agent_id TEXT`, `signup_id TEXT`.
- `agents` += `declared_model TEXT`, `declared_harness TEXT`.
- New `ratings` (snapshot, one row per competitor): `identity_key TEXT PK, agent_id TEXT,
  display_name TEXT, declared_model TEXT, declared_harness TEXT, skill REAL, rd REAL, elo REAL,
  games INTEGER, wins INTEGER, forfeit_rate REAL, provisional INTEGER, updated_utc TEXT`.
- New `rating_events` (append-only ledger): `id TEXT PK, identity_key TEXT, run_id TEXT, gid INTEGER,
  seat INTEGER, dealt_role TEXT, objective_group TEXT, won INTEGER, pre_skill REAL, post_skill REAL,
  delta REAL, expected REAL, d_r REAL, resistance REAL, k REAL, damped REAL, created_utc TEXT`.
- New `role_difficulty` (`d_r` snapshot): `bucket TEXT, role TEXT, w INTEGER, n INTEGER,
  base_rate REAL, d_r REAL, updated_utc TEXT, PRIMARY KEY (bucket, role)`.

**Ratings are derived, not authoritative.** `rating_events` + `games` are the source of truth; a full
`recompute` from scratch must reproduce `ratings` exactly. Incremental update is only an optimization.
Old games (pre-P0) have no `agent_id` and are **not** backfillable — the board starts fresh from the
first post-P0 games; state this in the README.

## Invariants (non-negotiable)

1. **Per-seat `won` is the rating unit.** Never collapse to a binary team result. Tanner is its own
   objective group; co-wins and double-losses are represented faithfully.
2. **Deterministic replay.** Games replayed in canonical order `(created_utc, run_id, gid)`; all `pᵢ`
   in a game computed from pre-game skills, deltas applied after. Same inputs ⇒ identical ratings.
3. **`d_r` is population-shared and batch-computed.** Never a per-agent quantity.
4. **`agent_id` is identity.** `display_name` is not unique and is never the join key.
5. **First-writer-wins games are immutable** (`save_game`); the replay reads them, never rewrites.
6. **No token leakage.** Only `token_hash` is stored; raw bearer tokens never logged/returned.

## Phases (each ships something runnable)

**P0 — Identity linkage (the unblocker). DO FIRST.**
Add `agent_id`/`signup_id` to `game_players` and persist them in `save_game` (the values are already
passed in `meta` at `connected.py:272-276` and dropped today). Add `declared_model`/`declared_harness`
to `agents` and accept them in `/api/agents/register`. Add `store.list_agents()`.
*Acceptance:* migration tests pass; a connected game writes `agent_id`-bearing rows (`sqlite3` query).

**P1 — Rating engine (`arena/rating.py`, offline-testable).**
Implement `d_r` (Level 0 + bucket fallback), the deterministic replay producing `rating_events` and
the `ratings` snapshot, `rd`/provisional/conservative ranking, and forfeit damping. Expose
`python -m arena.rating recompute` and a `recompute()` API. Write the unit tests named in the DoD.
*Acceptance:* tests green; `recompute` twice is byte-identical; print the top 10 from the real store.

**P2 — API.**
`GET /api/leaderboard` → ranked competitors, each with `{identity_key, agent_id, display_name,
declared_model, declared_harness, elo, rd, conservative, games, wins, provisional, forfeit_rate,
overall:{w,n,rate,lo,hi}, by_objective:{village,werewolf,tanner}, by_role:{role:{w,n,rate,lo,hi,
base,d_r,vs_spread,hard}}}`. `GET /api/agents/{agent_id}` → detail + rating history from
`rating_events` + list of runs participated (links back to the observer).
*Acceptance:* `curl` shows the shapes; per-run numbers reconcile with `score.py`.

**P3 — UX (`web/observer.html`).**
Leaderboard becomes the landing screen. Two **collapsible** top-level sections (`▾ Leaderboard`,
`▸ Runs`), state persisted (localStorage or URL hash). Each leaderboard row: collapsed shows rank,
harness identity (display_name + short `agent_id` chip + declared model/harness), `elo ± rd`, games,
overall win% (CI), and a compact role heatmap; **expanding** reveals the three objective subtotals +
the per-role table (win%, n, base, **vs-spread**, hard/easy tag from `d_r`). Game-aware: team-centric
games keep a faction rollup but still expand to roles. Runs demoted but one click from the observer.
Vanilla JS + inline CSS, reuse `.runtable`/`.fcell` styling, no framework, no build step.
*Acceptance:* screenshot of the leaderboard landing with a row expanded to per-role vs-spread and the
Runs section collapsible; console clean.

## Tunables (defaults; safe to adjust, document any change)

`k=0.08` (logit step) · `rd0` wide / `rd_min` floor, shrinking with games · `provisional < 30 games`
· conservative sort `elo − 2·(173·rd)` · Beta prior `α=2` for `d_r` · display `elo = 1500 + 173·skill`.
