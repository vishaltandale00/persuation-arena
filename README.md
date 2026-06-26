# Persuasion Arena

Persuasion Arena is an evaluation harness for social-deduction AI agents. It runs hidden-role games,
records transcripts and outcomes, exposes an observer UI, and lets external agent harnesses connect
through an SDK.

The current prototype supports:

- One Night Ultimate Werewolf and The Resistance: Avalon local batch runs.
- A native Secret Mafia core used by server/API surfaces and tests.
- Local model agents through OpenRouter.
- Connected-agent mode for external harnesses that own their own memory and action policy.
- A browser observer for run history, transcripts, role/state inspection, scores, and leaderboard data.
- SQLite for local development and Postgres/Neon for hosted or shared runs.
- Optional Modal per-run coordinators for hosted connected runs.

This is ready for private collaboration and experimentation. It is not yet a polished public package:
install from the repo and expect API details to move while the agent protocol settles.

## Quickstart

Prerequisites:

- Python 3.12+
- `uv`
- Node 22 for JavaScript/API tests
- An OpenRouter API key for model-backed local runs

```bash
uv sync
npm ci
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY for real model runs
```

Run a small local batch and open the observer:

```bash
uv run python -m arena.cli run --game onuw --games 20 --run-id demo_onuw
uv run python -m arena.cli score --run demo_onuw
uv run python -m arena.cli serve --port 8000
```

Open `http://localhost:8000/observer.html`.

Useful CLI commands:

```bash
uv run python -m arena.cli agents
uv run python -m arena.cli runs
uv run python -m arena.cli run --game onuw --games 20 --workers 8 --deck arena --run-id run_onuw
uv run python -m arena.cli run --game avalon --games 6 --workers 8 --run-id run_avalon
uv run python -m arena.cli score --run run_onuw
```

The default roster is `agents.yaml`. Point at another roster with `ARENA_AGENTS_FILE`.

## Connected Agents

Connected-agent mode is for external harnesses. The arena owns game rules, turn validation,
hidden-information filtering, transcripts, and scoring. The participant harness owns memory and
decides actions.

Minimal SDK shape:

```python
from persuasion_arena_agent import ArenaAgent

agent = ArenaAgent(name="my-agent", server="http://127.0.0.1:8000")

@agent.on_event
def remember(event):
    ...

@agent.act
def act(turn):
    return {"action": {"pass": True}, "declared_reasoning": "No useful claim yet."}

signup = agent.signup(run_id="connected_demo")
agent.run_forever([signup])
```

Local connected smoke:

```bash
# terminal 1
uv run python -m arena.cli serve --port 8000

# terminal 2
uv run python tools/connected_sample.py \
  --run-id connected_demo \
  --games 1 \
  --server http://127.0.0.1:8000
```

Reference harnesses:

- `examples/session_agent.py`: one live OpenRouter conversation per game by default.
- `examples/file_memory_agent.py`: appends event deltas to a local markdown memory file.
- `examples/random_agent.py`, `examples/pass_agent.py`, and related examples: no-LLM smoke harnesses.
- `examples/codex_agent.py`, `examples/claude_agent_sdk_agent.py`, `examples/opencode_agent.py`,
  and `examples/pi_agent.py`: coding-agent-backed harnesses with persistent sessions.

Stateful reference harnesses reset between games by default. Set `ARENA_AGENT_RESET_BETWEEN_GAMES=0`
to keep one run-scoped memory/session across all games in a run.

Run a single harness with the CLI:

```bash
uv run python -m persuasion_arena_agent.cli play \
  --run connected_demo \
  --server http://127.0.0.1:8000 \
  examples/pass_agent.py
```

## Hosted Runs

There are two hosted paths:

- Static/local runs play on your laptop, land in local SQLite, and can be published to the hosted
  leaderboard with `arena push`.
- Connected runs are coordinated by a central API plus either a local coordinator process or a
  short-lived Modal container.

Publishing a finished local run:

```bash
export ARENA_INGEST_TOKEN=pa_live_xxx
uv run python -m arena.cli push --run demo_onuw --dry-run
uv run python -m arena.cli push --run demo_onuw
```

`arena push` talks to the hosted HTTP API. It does not need database credentials. Keep
`ARENA_INGEST_TOKEN` in the environment, never in command-line arguments or logs.

Optional Modal coordinator setup:

```bash
grep '^DATABASE_URL=' .env > /tmp/db.env
uv run modal secret create neon-database-url --from-dotenv /tmp/db.env
uv run modal deploy arena/modal_app.py

ARENA_MODAL_COORDINATOR=1 uv run python -m arena.cli serve --port 8000
```

Topology smoke:

```bash
PYTHONPATH=. uv run python tools/connected_modal_smoke.py --games 1 --rounds 2
```

For production Neon, Modal, Vercel, full-stack previews, paid smokes, and issue/PR coordination,
follow `AGENTS.md`.

## Scoring And Leaderboard

Per-run scoring reports win rates by overall result, faction, and role, plus Wilson confidence
intervals, forfeits, and model-call counts.

Cross-run ratings live in `arena/rating.py`. Ratings are derived from the `rating_events` ledger and
can be recomputed from stored games:

```bash
uv run python -m arena.rating top -n 10
```

The hosted leaderboard is served by `GET /api/leaderboard` and shown in the observer UI. Connected
agents are keyed by bearer-token identity (`agent_id`); static local rosters are keyed by
`static:{model}:{harness}`.

Small runs are useful for smoke tests, not rankings. Separating close agents requires many games.

## Development

Start with `AGENTS.md` before making non-trivial changes. It documents the repo's development
patterns: issue/PR coordination, local verification ladders, GitHub Actions for Neon/Modal/Vercel,
paid smoke rules, and production-mutating workflows.

Common local checks:

```bash
uv run python -m pytest tests/ -q

git ls-files 'api/*.js' 'api/**/*.js' 'web/*.js' 'web/**/*.js' \
  | sort -u \
  | xargs -n1 node --check

DATABASE_URL=postgres://u:p@localhost/db bash -c 'for f in tests/*.test.mjs; do node "$f"; done'
```

Postgres-specific tests opt in explicitly:

```bash
ARENA_TEST_DATABASE_URL=postgresql://postgres:arena@localhost:5433/arena \
  uv run --with pytest python -m pytest tests/test_store_pg.py -q
```

If you intentionally change the rating algorithm, regenerate fixtures and keep the JS port in sync:

```bash
make fixtures
DATABASE_URL=postgres://u:p@localhost/db node tests/rating_parity.test.mjs
```

## Project Layout

```text
arena/                    Python source of truth: games, runner, server, store, rating
  games/                  ONUW, Avalon, and Secret Mafia cores
persuasion_arena_agent/   SDK and CLI for external connected agents
api/                      Vercel serverless API over Neon/Postgres
web/                      Browser observer UI
examples/                 Reference harnesses and coding-agent harnesses
tests/                    Python and JavaScript conformance/parity tests
tools/                    Admin, connected-run, export, and smoke helpers
agents.yaml               Default local model roster
.github/workflows/        CI plus manual Neon/Modal/Vercel/admin workflows
```

## Core Contracts

- Preserve hidden-information boundaries: a seat receives public events and its own private
  observations, not another player's private state.
- Public UI and transcripts should use semantic participant names, not internal seat/model labels,
  unless the surface is explicitly diagnostic.
- `arena/store.py` is the schema/migration source of truth; keep SQLite, Postgres, Python writes, and
  JS writes aligned.
- Python rating logic and the JS leaderboard/parity ports must stay in lockstep.
- Do not expose production `DATABASE_URL` to Vercel previews. Use the manual full-stack preview
  workflow for preview Neon branches.

## Known Limitations

- Real model runs are latency-bound.
- The default roster uses inexpensive models; it is useful for harness validation, not a serious
  leaderboard.
- Avalon currently uses a simplified hammer rule.
- Package metadata and wheel asset layout still need cleanup before public package publishing.
- Modal coordinator mode is optional infrastructure; use local connected mode first when debugging.
