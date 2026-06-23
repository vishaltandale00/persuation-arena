# Persuasion Arena

An evaluation harness for **social-intelligence games**: LLM agents play hidden-role games
(One Night Ultimate Werewolf, Avalon, Secret Mafia), every game is recorded in full, agents are
scored by **per-role win-rate with 95% confidence intervals**, and any game is **watchable**
in a browser — public chat, true roles, actions, and each agent's **private reasoning**.

See `DESIGN.md` for rationale and `GOAL.md` for the build brief.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest pyyaml python-dotenv
echo "OPENROUTER_API_KEY=sk-or-..." > .env        # your OpenRouter key

# play a run of games (cheap models by default)
python -m arena.cli run --game onuw --games 20

# print its leaderboard
python -m arena.cli score --run run_9000

# serve the observer, then open http://localhost:8000/observer.html
python -m arena.cli serve --port 8000
```

Models run via **OpenRouter** (one key, any model). The default roster (`agents.yaml`) uses cheap
models so a 20-game run costs a few cents.

## What's here

```
arena/
  config.py        # loads OPENROUTER_API_KEY from .env; caps; roster
  openrouter.py    # agent: prompt -> {reasoning, action} JSON (retry -> default on failure)
  games/
    base.py        # team map, tally_votes, compute_winners (Hunter chain, Tanner)
    onuw.py        # One Night Ultimate Werewolf (full role set)
    avalon.py      # The Resistance: Avalon
    secret_mafia.py# Secret Mafia
  batch.py         # run N games concurrently (a "run"); fresh deal each game, rotating seats
  score.py         # per-role + overall win-rate with 95% Wilson CIs
  store.py         # SQLite/Postgres persistence + remote job queue
  server.py        # FastAPI: serves observer JSON, queues remote runs, ingests worker results
  cli.py           # run / score / runs / agents / worker / serve
web/observer.html# the observer UI (fetches from the server)
tests/             # pytest: ONUW + Avalon conformance + invariants (no API calls)
```

## Add an agent

Edit `agents.yaml` — one entry per agent:

```yaml
- {name: Ada, model: mistralai/mistral-small-2603, harness: base}
```

`model` is any [OpenRouter](https://openrouter.ai/models) slug. That's it — the agent is now in the
default roster. You can also pass a custom roster per run via the New-run screen or the POST API.

## Add a game

1. Write a Game Core class in `arena/games/<game>.py` with a `play(agents) -> transcript` method.
   Reuse the agent contract: each turn call `agent.act(prompt, parse_action, default_action)` and read
   `resp.reasoning` (private, logged) and `resp.action` (public). Emit the transcript shape the
   observer renders (see `onuw.py` — `players`, `cardsInPlay`, `phases` with `events`/`reason`/`synth`,
   `outcome`, `winner_team`). Honour the invariants in `DESIGN.md §13` (per-seat info filtering,
   atomic vote, win on end-of-night state).
2. Register it in `arena/batch.py` → `GAME_CORES`.
3. The observer auto-renders ONUW and Avalon with custom panels; other games use a generic state panel.

## Start a run

- **CLI:** `python -m arena.cli run --game avalon --games 6 --workers 8`
- **UI:** open the observer → **New run** → pick game / games / models → **Launch run** (calls `POST /api/run`).
- A run plays N games concurrently; seats rotate each game so agents play each position equally.

## Remote Vercel + Neon mode

The hosted site stores runs/jobs in Postgres. Model keys stay on laptops: each worker claims only
jobs for its owner, runs games locally, and publishes transcripts back to the site.

Server env:

```bash
DATABASE_URL=postgresql://...              # Neon pooled URL
INGEST_TOKENS=alice=long-random-token      # optional, comma-separated owner=token pairs
```

Worker:

```bash
python -m arena.cli worker \
  --server https://your-vercel-app.vercel.app \
  --owner alice \
  --token long-random-token
```

The New-run screen has a **Worker owner** field. It must match the worker's `--owner`.

## Scoring

`score.py` reports per-agent **overall** and **per-faction** (village/good vs werewolf/evil) win-rate,
each with a 95% Wilson confidence interval. At n=20 the CIs are ~±20%; separating close agents needs
more games (see `DESIGN.md §9`). The observer's run overview shows the bars and CIs.

## Tests

```bash
python -m pytest tests/ -q
ARENA_TEST_DATABASE_URL=postgresql://postgres:arena@localhost:5433/arena python -m pytest tests/ -q
```

Covers ONUW role actions / swaps / Doppelgänger, Avalon quests + assassination, invariants
(all-tied-die, no-kill, Hunter chain, Tanner, atomic vote, no-leak filtering), batch resilience,
dual SQLite/Postgres store behavior, and remote queue claim/ingest/complete semantics.

## Known limitations (v1)

- **Speed:** real games are model-latency-bound; runs play games concurrently (default 8 workers).
- **Balance:** with cheap models, the deceiving side tends to win (village/good coordinates poorly) —
  a realistic signal, not a bug.
- **Avalon hammer rule** is simplified (re-proposals cap at 3 then auto-pass).
- Mirror/CRN pairing, TrueSkill, and offline ToM/deception metrics are designed-for but not yet built
  (`DESIGN.md §9`).
