# AGENTS.md

Development patterns for coding agents contributing to Persuasion Arena.

This repo has enough shared infrastructure that the main job is not just "edit code." The expected
pattern is: anchor the work in an issue or PR, prove the smallest local surface first, use manual
GitHub Actions for shared Neon/Modal/Vercel resources, and leave durable evidence on the PR.

## Working Agreement

- Use GitHub issues for shared intent: bugs, infra changes, protocol changes, production repairs,
  and anything that another agent may pick up later.
- Use PRs for shared proof: root cause, scoped diff, checks run, workflow links, preview URLs, cleanup
  status, and remaining risk.
- Prefer draft PRs until the local proof and relevant remote workflow proof are both present.
- Keep one root cause per branch/PR. Do not mix schema work, Modal deploy work, UI changes, and docs
  cleanup unless the issue explicitly requires that combination.
- Before publishing a branch, inspect the worktree and stage only the intended files. This repo often
  has unrelated untracked proof files or another agent's WIP.
- If another agent is active in the same checkout, update `AGENT_COORDINATION.md` before editing
  overlapping files.

## Issue And PR Pattern

Use this flow when taking work from idea to mergeable change:

1. Check for an existing issue, branch, or PR that already covers the work.
2. If the work is new and non-trivial, create or update an issue with the user-visible problem,
   suspected surfaces, and acceptance checks.
3. Create a focused branch from current `origin/main`.
4. Make the smallest defensible change.
5. Run focused local proof first, then the broader local gate when the touched surface warrants it.
6. Open a draft PR with:
   - linked issue;
   - root cause;
   - implementation summary;
   - verification commands and results;
   - remote workflow links or preview URL when infra was exercised;
   - cleanup notes for any Neon branch, Modal app, or Vercel preview.
7. Keep driving the same PR through review, mergeability, and check failures. If `main` is red, compare
   against `main` before attributing a failure to the branch.

Useful live-state commands:

```bash
git fetch origin main
git status --short --branch
gh issue list --state open
gh pr list --state open --json number,title,headRefName,isDraft,mergeStateStatus,reviewDecision
gh pr view <number> --json isDraft,mergeStateStatus,reviewDecision,statusCheckRollup,mergeable,url
gh pr checks <number>
```

## Local Verification Ladder

Start with the cheapest proof that can catch the class of bug you touched. Escalate only when the
change crosses a real boundary.

### 0. Always-free structural checks

```bash
uv run python -m pytest tests/test_orchestration_smoke.py -q
uv run python -m pytest tests/test_harness_smoke.py -q
uv run python -m pytest tests/test_cost_calibration.py -q
uv run python -m pytest tests/test_schema_drift.py tests/test_status_single_source.py -q
uv run python -m pytest tests/test_modal_preview_config.py tests/test_modal_smoke_shards.py -q
```

These are designed to validate real orchestration, SDK loopback behavior, cost-budget assumptions,
schema/status drift, and Modal preview isolation without OpenRouter spend, Modal spawn, or Neon writes.

### 1. Standard local gate

Run this before treating broad work as ready:

```bash
uv run python -m pytest tests/ -q

git ls-files 'api/*.js' 'api/**/*.js' 'web/*.js' 'web/**/*.js' \
  | sort -u \
  | xargs -n1 node --check

DATABASE_URL=postgres://u:p@localhost/db bash -c 'for f in tests/*.test.mjs; do node "$f"; done'
```

`DATABASE_URL` is a dummy for JS tests. Those tests import Vercel/Neon modules but do not open a real
connection.

### 2. Optional Postgres proof

Use this only when touching store/schema behavior that SQLite cannot prove:

```bash
ARENA_TEST_DATABASE_URL=postgresql://postgres:arena@localhost:5433/arena \
  uv run --with pytest python -m pytest tests/test_store_pg.py -q
```

The default pytest path deletes `DATABASE_URL` so a local `.env` cannot accidentally hit Neon.

### 3. Local connected run

Use loopback first when debugging connected-agent behavior:

```bash
uv run python -m arena.cli serve --port 8000
uv run python tools/connected_sample.py --run-id connected_demo --games 1 --rounds 2 \
  --server http://127.0.0.1:8000
```

This uses the local store if `DATABASE_URL` is unset. Only set `OPENROUTER_API_KEY` when you intend to
make real model calls.

### 4. Cheap real-model and paid proof

`tests/test_eval_floor_cheap_models.py` uses real OpenRouter calls but skips when `OPENROUTER_API_KEY`
is absent. Use it for model-facing competence floors after the hermetic floors pass.

The paid end-to-end smoke is intentionally double opt-in:

```bash
OPENROUTER_API_KEY=... ARENA_RUN_PAID_SMOKE=1 \
  uv run python -m pytest tests/test_cost_smoke.py -q -s
```

Do not run paid proof unless the user asked for it or the PR explicitly needs it. Keep the hard ceiling
low and re-derive the cost model before raising it.

## GitHub Actions Runbook

The shared infra workflows are part of the development surface. Use them instead of asking every
collaborator to hold Neon, Modal, or Vercel credentials locally.

| Workflow | Trigger | Purpose | When to use |
| --- | --- | --- | --- |
| `.github/workflows/ci.yml` | PRs and `main` pushes | Blocking Python tests, JS syntax checks, JS parity tests; advisory lint | Every PR |
| `.github/workflows/migrate-neon.yml` | `main` changes to `arena/store.py`, or manual | Apply idempotent production Neon schema/migrations via `store.init_schema()` | Schema/store changes after merge, or manual secret bootstrap |
| `.github/workflows/deploy-modal.yml` | `main` changes to Modal image inputs, or manual | Deploy production Modal coordinator app | Modal/coordinator/runtime dependency changes |
| `.github/workflows/full-stack-preview.yml` | Manual only | Provision or clean up a PR-scoped Neon branch, Modal app, and Vercel preview | Production-like testing that truly needs all three providers |
| `.github/workflows/cost-smoke.yml` | Manual, plus optional weekly cron | Real-money OpenRouter smoke with a hard ceiling | Explicit paid validation only |
| `.github/workflows/prune-run.yml` | Manual only | Delete a production run from Neon and recompute ratings | Admin repair only |

Vercel preview deployments are cheap and automatic through the Vercel GitHub integration. Full-stack
previews are different: Neon branches and Modal apps cost shared resources and require cleanup, so
they must stay `workflow_dispatch` only.

### Full-stack preview pattern

Use **Actions -> Full Stack Preview -> Run workflow** only when a PR needs a production-like stack:

1. `operation=provision`
2. `preview_id=pr-<number>`
3. `pr_number=<number>` so the workflow comments the preview URL on the PR
4. `run_smoke=true` unless you have a specific reason not to
5. After review/testing, rerun with `operation=cleanup` and the same `preview_id`

The PR is not done until the cleanup action is linked or the remaining preview resources are called out
explicitly.

### Production workflow rules

- `migrate-neon.yml` is idempotent, but it is still production-mutating. Use it for schema changes, not
  for exploratory debugging.
- `deploy-modal.yml` deploys Modal only. Vercel production deploys through its GitHub integration on
  `main`.
- `prune-run.yml` deletes production data. Use an issue/PR/comment trail that names the run id and why
  it is safe to remove.
- `cost-smoke.yml` spends money. If it skipped due to a missing secret, record that as "not run", not as
  proof.

## Infra-Specific Patterns

### Neon and store changes

- `arena/store.py` is the schema/migration source of truth.
- Keep SQLite DDL, Postgres DDL, Python insert columns, and JS insert columns aligned.
- Run `tests/test_schema_drift.py` for every schema-adjacent change.
- For production, rely on `migrate-neon.yml` after merge rather than local prod DB access.

### Modal coordinator changes

- Prove coordinator behavior locally first with `tests/test_harness_smoke.py` or
  `tools/connected_sample.py`.
- Use `tests/test_modal_preview_config.py` when touching app names, secret names, or preview isolation.
- Use `tools/connected_modal_smoke.py` only when Modal is actually needed; it requires shared
  `DATABASE_URL` semantics between the central API and coordinator.
- Production Modal deploys must go through `deploy-modal.yml` or a clearly documented manual
  `modal deploy arena/modal_app.py`.

### Vercel API and observer changes

- For `api/` changes, run `node --check` and all `tests/*.test.mjs` with dummy `DATABASE_URL`.
- For `web/observer.html`, test the visible UI when behavior or layout changes, not just syntax.
- Do not expose production `DATABASE_URL` to preview deployments. Preview database access should use
  the full-stack preview workflow with a Neon branch.

### Ratings and leaderboard changes

- Ratings are derived from `rating_events`; do not hand-edit rating snapshots as the primary fix.
- Python rating logic in `arena/rating.py` and JS projection/ports in `api/_rating.js` must stay in
  lockstep.
- If the rating algorithm changes, run `make fixtures` and the JS parity tests.
- Local run publication uses `arena push`; keep `ARENA_INGEST_TOKEN` in the environment, never in CLI
  arguments or logs.

## Core Repo Invariants

- Preserve hidden-information boundaries. A seat can receive public events and its own private
  observations, not another player's private state.
- Public UI and transcript references should use semantic participant names, not internal seat/model
  labels, unless the surface is explicitly diagnostic.
- Connected-agent harnesses own memory; the SDK delivers deltas and actions, not hidden global state.
- The project name is misspelled as `persuation` in package and remote names. Do not rename it as
  incidental cleanup.
- Do not commit secrets, local SQLite DBs, local `.env` files, logs, generated build outputs, or
  one-off proof artifacts unless the issue explicitly asks for a committed artifact. `.env.example`
  is the committed template.
