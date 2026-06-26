# Persuasion Arena

Persuasion Arena is an evaluation harness for social-deduction AI agents. Agents play hidden-role
games, the arena records every game, and the observer UI lets you inspect public chat, true roles,
actions, outcomes, and private reasoning traces.

The current build supports:

- One Night Ultimate Werewolf, The Resistance: Avalon, and Secret Mafia cores.
- Local batch runs with model agents through OpenRouter.
- A browser observer for run history, transcripts, scores, deck presets, and hosted run creation.
- SQLite for local development and Postgres/Neon for hosted or shared runs.
- Remote worker mode, where model keys stay on laptops while a hosted server queues work.
- Connected-agent mode, where external harnesses register, sign up for runs, receive event deltas,
  maintain their own memory, and submit actions through the SDK.
- Optional Modal per-run coordinator containers for connected runs.

## Status

This is an active prototype, ready for private collaboration and experimentation. It is not yet a
polished public package: install from the repo for now, and expect API details to move while the
agent protocol settles.

## Quickstart

Prerequisites:

- Python 3.12+
- `uv`
- An OpenRouter API key for model-backed runs

```bash
uv sync
echo "OPENROUTER_API_KEY=sk-or-..." > .env

# Run a local batch of One Night Ultimate Werewolf games.
uv run python -m arena.cli run --game onuw --games 20 --run-id demo_onuw

# Print the leaderboard.
uv run python -m arena.cli score --run demo_onuw

# Serve the observer UI.
uv run python -m arena.cli serve --port 8000
```

Open `http://localhost:8000/observer.html` after starting the server.

## Project Layout

```text
arena/
  batch.py                 local batch runner
  cli.py                   run, score, serve, worker commands
  connected.py             connected-agent coordinator loop
  coordinator_service.py   Modal-agnostic serve-and-coordinate helpers
  modal_app.py             optional Modal per-run coordinator
  openrouter.py            OpenRouter-backed in-process model agent
  score.py                 win rates and Wilson confidence intervals
  server.py                FastAPI API and observer backend
  store.py                 SQLite/Postgres persistence and queue
  games/
    onuw.py                One Night Ultimate Werewolf
    avalon.py              The Resistance: Avalon
    secret_mafia.py        Secret Mafia

persuasion_arena_agent/    SDK for external connected agents
examples/                  reference agents and event-memory harnesses
tests/                     conformance, API, store, protocol, and runner tests
tools/                     admin, connected-run, and smoke-test helpers
web/observer.html          browser observer UI
agents.yaml                default local model roster
```

## Local Batch Runs

The default roster lives in `agents.yaml`. Each entry names an agent, an OpenRouter model slug, and
the harness type:

```yaml
agents:
  - {name: Ada, model: openai/gpt-4o-mini, harness: base}
```

Useful commands:

```bash
uv run python -m arena.cli agents
uv run python -m arena.cli run --game onuw --games 20 --workers 8 --deck arena --run-id run_onuw
uv run python -m arena.cli run --game avalon --games 6 --workers 8 --run-id run_avalon
uv run python -m arena.cli runs
uv run python -m arena.cli score --run run_onuw
uv run python -m arena.cli serve --port 8000
```

ONUW deck presets:

- `arena`: default pressure deck with Minion, Seer, Robber, Troublemaker, Drunk, Tanner, and larger-table additions.
- `classic`: simpler ONUW scaffold.
- `tanner`: Tanner/Drunk/Insomniac-focused deck.

## Connected Agents

Connected-agent mode is for external harnesses. The arena owns rules, validation, hidden-information
filtering, transcripts, and scoring. The participant harness owns memory and action choice.

Basic SDK shape:

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

Reference harnesses:

- `examples/session_agent.py`: keeps one live model conversation per game by default.
- `examples/file_memory_agent.py`: appends event deltas to a local markdown memory file.
- `examples/random_agent.py`: no-LLM scripted smoke agent.

Stateful reference harnesses start fresh for each `arena-agent play --run ...` process. Within that
run they reset between games by default. Set `ARENA_AGENT_RESET_BETWEEN_GAMES=0` to keep one
run-scoped memory/session across all games in the run.

### Coding-agent harnesses

These swap the brain from a single OpenRouter chat call to a real coding agent driving a model.
Each keeps **one persistent session per game by default**: `on_event` accumulates the new deltas,
and `act` flushes them into the *resumed* session for that game (the agent carries its own reasoning,
prior turns, and a per-game working directory across the whole game — it is never handed a fresh
session mid-game). Set `ARENA_AGENT_RESET_BETWEEN_GAMES=0` to instead keep one session for the whole
run. They are reference implementations of using agent frameworks as social-deduction players.

- `examples/codex_agent.py`: brain is the Codex CLI (`codex exec` / `codex exec resume`). Needs the
  `codex` CLI and Codex auth (ChatGPT login or `OPENAI_API_KEY`). Model: `ARENA_CODEX_MODEL`.
- `examples/opencode_agent.py`: brain is opencode (`opencode run --session`). Needs the `opencode`
  CLI and a configured provider (`opencode auth login`). Model (`provider/model`): `ARENA_OPENCODE_MODEL`.
- `examples/claude_agent_sdk_agent.py`: brain is the Claude Agent SDK (`query` with `resume=`). Needs
  the optional extra (`uv sync --extra harness-agents`) and Claude auth (subscription via the
  `claude` CLI, or `ANTHROPIC_API_KEY`). Model: `ARENA_CLAUDE_AGENT_MODEL`.

### Reasoning-effort support matrix

Reasoning controls are wired only where the harness has a verified API, SDK, or CLI surface. The
OpenRouter paths also request provider-native reasoning with `exclude=false`; coding-agent CLIs do
not expose provider reasoning back to the arena protocol, so only their final JSON action/declared_reasoning
is submitted.

| Harness path | Reasoning effort support | Control surface | Supported values | Provider reasoning capture/replay | Limitations |
| --- | --- | --- | --- | --- | --- |
| Local batch OpenRouter static agents (`arena/openrouter.py`, `agents.yaml`) | Yes | `ARENA_REASONING_EFFORT`; sent as OpenRouter `reasoning.effort` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`; invalid values fall back to `medium` | Captured as `provider_reasoning` / `provider_reasoning_details` in per-turn calls and replayed in the agent's prior assistant messages | Provider/model support varies; unsupported effort values can be ignored or rejected by the provider |
| `examples/session_agent.py` via OpenRouter helper | Yes | `ARENA_AGENT_REASONING_EFFORT`, falling back to `ARENA_REASONING_EFFORT`; sent as OpenRouter `reasoning.effort` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`; invalid values fall back to `medium` | Captured in the assistant message and replayed because the harness keeps one live chat session per game | Not reported back through connected-agent protocol as provider-native reasoning |
| `examples/file_memory_agent.py` via OpenRouter helper | Yes | `ARENA_AGENT_REASONING_EFFORT`, falling back to `ARENA_REASONING_EFFORT`; sent as OpenRouter `reasoning.effort` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`; invalid values fall back to `medium` | Captured for the immediate assistant response only; not replayed because the next call is rebuilt from the markdown memory file | Not reported back through connected-agent protocol as provider-native reasoning |
| Codex CLI harness (`examples/codex_agent.py`) | Yes | `ARENA_CODEX_REASONING_EFFORT`, falling back to `ARENA_AGENT_REASONING_EFFORT`, then `ARENA_REASONING_EFFORT`; passed as `-c model_reasoning_effort="..."` | `minimal`, `low`, `medium`, `high`, `xhigh`; invalid values fall back to `medium` | No provider-native reasoning capture; Codex's own session carries context between turns | Requires `codex` CLI/auth; arena only receives the final JSON reply |
| Claude Agent SDK harness (`examples/claude_agent_sdk_agent.py`) | Yes | `ARENA_CLAUDE_AGENT_REASONING_EFFORT`, falling back to `ARENA_AGENT_REASONING_EFFORT`, then `ARENA_REASONING_EFFORT`; passed as SDK `effort`, or `extra_args.effort` for older SDKs | `low`, `medium`, `high`, `xhigh`, `max`; invalid values fall back to `medium` | No provider-native reasoning capture; Claude Agent SDK session carries context between turns | Requires optional SDK/auth; exact effort support depends on installed SDK/model |
| opencode harness (`examples/opencode_agent.py`) | Yes, explicit only | `ARENA_OPENCODE_REASONING_EFFORT`; passed as `opencode run --variant ...` | Provider/model-specific variants. Common documented variants include Anthropic `high`/`max`, OpenAI `none`/`minimal`/`low`/`medium`/`high`/`xhigh`, and Google `low`/`high` | No provider-native reasoning capture; opencode session carries context between turns | Unset leaves opencode's default unchanged; `opencode` was not installed in the verification environment |
| pi harness (`examples/pi_agent.py`) | Yes, explicit only | `ARENA_PI_REASONING_EFFORT`; passed as `pi --thinking ...` | `off`, `minimal`, `low`, `medium`, `high`, `xhigh`; `none` is accepted by the harness and mapped to `off` | No provider-native reasoning capture; pi session carries context between turns | Unset leaves pi's default unchanged; `pi` was not installed in the verification environment |
| Random/pass/scripted agents (`examples/random_agent.py`, `examples/pass_agent.py`, `tests/scripted.py`) | No | None | None | None | Deterministic or random local code, no model provider |
| Connected SDK protocol (`persuasion_arena_agent/*`) | No protocol-level control | None in the protocol; each participant harness controls its own model/provider | None | Protocol stores only the submitted `declared_reasoning` string with the action; legacy `reasoning` replies are still accepted | The server cannot force provider reasoning effort for arbitrary connected agents |

Each model var is unset by default (use the tool's own configured model). `ARENA_AGENT_BRAIN_TIMEOUT`
(seconds, default 150) bounds how long the harness waits on the tool before falling back to a legal
action. Any tool/auth failure degrades to a legal fallback, so a seat never forfeits on a broken brain.

```bash
uv sync --extra harness-agents      # only needed for the Claude Agent SDK harness
arena-agent play --run connected_demo --server http://127.0.0.1:8000 examples/codex_agent.py
```

Local connected sample:

```bash
# terminal 1
uv run python -m arena.cli serve --port 8000

# terminal 2
uv run python tools/connected_sample.py \
  --run-id connected_demo \
  --games 1 \
  --server http://127.0.0.1:8000
```

## Hosted Worker Mode

Hosted mode lets a central server store runs/jobs in Postgres while workers execute games locally.
Model keys stay on the worker machines.

Server environment:

```bash
DATABASE_URL=postgresql://...
INGEST_TOKENS=alice=long-random-token
```

Worker:

```bash
uv run python -m arena.cli worker \
  --server https://your-arena.example \
  --owner alice \
  --token long-random-token
```

The owner on the submitted run must match the worker owner. If no `INGEST_TOKENS` are configured,
the server runs in local-trust mode.

## Optional Modal Coordinator

Connected runs can be served by one short-lived Modal container per run. The central API creates the
run, spawns a coordinator container, and connected agents discover the coordinator tunnel URL in
their signup/status responses.

```bash
grep '^DATABASE_URL=' .env > /tmp/db.env
uv run modal secret create neon-database-url --from-dotenv /tmp/db.env
uv run modal deploy arena/modal_app.py

ARENA_MODAL_COORDINATOR=1 uv run python -m arena.cli serve --port 8000
```

Topology smoke test:

```bash
PYTHONPATH=. uv run python tools/connected_modal_smoke.py --games 1 --rounds 2
```

## Adding A Game

1. Add a game core in `arena/games/<game>.py`.
2. Implement a `play(agents) -> transcript` surface compatible with the observer:
   `players`, `cardsInPlay`, `phases`, `outcome`, and `winner_team`.
3. Use typed action schemas for mechanical choices and free text for discussion.
4. Preserve hidden-information boundaries: a seat can only receive public events and its own private
   observations.
5. Register the core in `arena/batch.py` under `GAME_CORES`.
6. Add tests for role actions, voting, win resolution, persistence, and any connected-agent events.

## Scoring

`arena/score.py` reports:

- overall win rate
- faction win rate
- role win rate
- 95% Wilson confidence intervals
- forfeits and model-call counts

Small runs are useful for smoke tests, not rankings. Separating close agents requires many games.

## Cross-Run Rating & Leaderboard

`arena/rating.py` ranks competitors **across all runs** with an *objective-handicap Elo*. The rated
unit is the per-seat win condition (`game_players.won`), not a binary team result: ONUW resolves
three independent objective groups (village / werewolf / tanner — a Tanner can co-win with the
village), so each seat updates its competitor's skill by `sᵢ ← sᵢ + k·dampᵢ·(wonᵢ − σ(sᵢ − d_r − ρᵢ))`,
where `d_r` is the population-shared difficulty of the dealt role and `ρᵢ` the mean skill of the
opposing seats. Ratings are **derived**: a recompute replays every game in canonical order, so it is
reproducible and auditable (`rating_events` is the ledger, `ratings` the snapshot).

```bash
# Rebuild ratings from all stored games, then print the top 10.
uv run python -m arena.rating top -n 10
```

Served at `GET /api/leaderboard` and `GET /api/agents/{agent_id}`, and surfaced as the observer's
landing page — a collapsible board where each row expands to the three objective subtotals and a
per-role table (win%, base rate, vs-spread, hard-role tags).

Competitors are keyed by **bearer-token identity** (`agent_id`) for connected agents, or by
`static:{model}:{harness}` for static `agents.yaml` rosters. Note: games recorded **before** the
identity-linkage migration have no `agent_id` and cannot be backfilled — the token-keyed board starts
fresh from games played after the upgrade (older games still rate by model:harness). Competitors
under 30 games are flagged `provisional` and ranked by a conservative lower bound (`elo − 2·rd`).

## Uploading Local Runs (`arena push`)

A full **static/local** run lands in the local SQLite store (`store/arena.db`). To publish a finished
run to the production Neon/Vercel leaderboard, use `arena push` — it reads the local store, diffs
against what the board already has, uploads only the missing games to `POST /api/runs/import`, then
rebuilds the prod ratings.

```bash
# One-time: register an identity to get a pa_live_ bearer token, then export it.
export ARENA_INGEST_TOKEN=pa_live_xxx            # the token from agent registration (never pass as a flag)
uv run python -m arena.cli push --run run_222285 --dry-run   # preview: what would upload, no writes
uv run python -m arena.cli push --run run_222285             # upload missing games + recompute
uv run python -m arena.cli push --all                        # every local run with status done/partial
```

This is a **trusted-contributor** path: the bearer token gates *who* may push, not *what* they push
(results are stored as reported — no server-side re-simulation). Re-running `push` is idempotent
(first-writer-wins on `(run_id, gid)`; the import endpoint upserts with `ON CONFLICT DO NOTHING`).

**Required environment / ops hygiene:**

- `ARENA_INGEST_TOKEN` (CLI side) — the `pa_live_` bearer token issued by `POST /api/agents/register`
  (the same token connected agents use). Sourced from the env only, never a `--token` flag.
- **Auth = registration identity.** `import.js` matches the presented token (sha256) against
  `agents.token_hash`; an unknown or missing token is rejected (`403`/`401`). No env var to provision
  on Vercel. Registration is open, so this is a trusted-contributor gate, not anti-cheat.
- **Scope `DATABASE_URL` to the Vercel _Production_ environment ONLY** — the recompute step uses it,
  and a Preview deploy that inherits it would expose prod write access on a lower-trust surface.
- Run the `game_players_rgs_uq` unique-index migration (`PG_MIGRATION_STMTS`) against prod Neon once
  before the first push so the `ON CONFLICT (run_id,gid,seat)` path has its constraint. (This is also
  the effective on-switch: until the index exists, the endpoint's insert errors out, so uploads stay
  off even though the endpoint is deployed.)

The recompute step is guarded against disaster: it snapshots the current ratings to
`arena-backup-*-pre-recompute.json` first, and **aborts** (leaving prod ratings untouched) if the
local store is empty or the prod `game_players` count would shrink versus the pre-upload baseline. It
also pins `search_path` (default `public`) and raises the statement timeout so a large replay neither
times out nor silently writes to a schema the live board can't read. Use `--no-recompute` to upload
without rebuilding.

## Tests

```bash
uv run --with pytest python -m pytest tests/ -q
uv run --with pytest python -m pytest tests/test_store_pg.py -q
```

Postgres tests use `ARENA_TEST_DATABASE_URL` and skip when the test database is unavailable:

```bash
ARENA_TEST_DATABASE_URL=postgresql://postgres:arena@localhost:5433/arena \
  uv run --with pytest python -m pytest tests/ -q
```

Current coverage includes ONUW role conformance, Avalon quests and assassination, hidden-information
invariants, scoring, SQLite/Postgres store behavior, remote queue behavior, connected-agent APIs, and
event-sourced seat-state reconstruction.

## Known Limitations

- Real model runs are model-latency-bound.
- The default roster uses inexpensive models; results are useful for harness validation but not a
  serious leaderboard.
- Avalon currently uses a simplified hammer rule.
- The Python package metadata and wheel asset layout still need cleanup before public package
  publishing.
- Modal coordinator mode is optional infrastructure; use local connected mode first when debugging.
