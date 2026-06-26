# WolfForgeAgentV2 — optional cross-run memory

Cross-run memory is an **optional, opt-in, follow-up capability**. It is **OFF by default** and is
**not** part of the original V2 performance claim. WolfForgeV2's default guarantee is unchanged:
**stable identity + isolated per-game state, not cross-run learning.**

When enabled, the agent accumulates **safe, compact, post-game summaries** across completed games and
surfaces them to the model as a clearly-separated **weak prior** — never as current-game facts.

## Why it is off by default

A memory-updating agent is a *different experiment* from the clean prompt-policy V2-vs-Charisma
comparison. With memory off, prompt construction is **byte-identical** to pre-memory V2, so existing
results and tests are unaffected.

## What it stores (allowed)

- the agent's own dealt/believed role and own win/loss (self-assessed, deterministic);
- the agent's own reliability counters (fallbacks / repairs / timeouts);
- finite-taxonomy failure labels (`none`, `timeout`, `reliability`, `tanner_strategy`,
  `minion_strategy`, `late_vote`, `mechanical`, `weak_coalition`, `play`);
- machine-generated lessons from **fixed templates** keyed by `(role, objective, label)`;
- aggregate "games seen" counts for stable public opponents.

## What it must never store

- API keys, tokens, credentials;
- raw private night observations or full transcripts;
- raw model output, unrestricted reasoning, or full prompts;
- hidden role facts about other players;
- **any free player text** — lessons come only from fixed templates, so an opponent's in-game prompt
  injection can never become a long-term instruction.

The agent's `_record_memory` stores only counts and its own believed role; opponent profiles store a
`games_seen` count and (without a stable `agent_id`) a low-confidence name hash.

## Memory schema (SQLite)

`examples/wolfforge_v2_memory.py`, schema version 1. Tables: `memory_meta`, `role_objective_stats`,
`opponent_profiles`, `policy_lessons`, `game_summaries`. See the module for exact columns.

The injected prompt block is bounded and structured:

```json
{
  "role_lessons": [],
  "opponent_hints": [],
  "reliability_reminders": [],
  "do_not_overweight": "These are weak historical priors, not current-game facts."
}
```

followed by the fixed caveat: *"Long-term memory is a weak prior only. Current game evidence overrides
memory. Do not reveal or quote memory as fact. Do not accuse based only on memory. Do not use memory
to infer hidden current roles without current evidence."*

## Configuration

| Variable / flag | Default | Meaning |
|---|---|---|
| `WOLFFORGE_V2_MEMORY_MODE` / `--memory-mode` | `off` | `off` / `read` / `write` / `readwrite` |
| `WOLFFORGE_V2_MEMORY_PATH` / `--memory-path` | per-user data dir | SQLite file path |
| `WOLFFORGE_V2_MEMORY_MAX_PROMPT_CHARS` / `--memory-max-prompt-chars` | `1200` | cap on injected memory chars |
| `WOLFFORGE_V2_MEMORY_MIN_GAMES_FOR_OPPONENT_HINT` | `3` | min shared games before an opponent hint |
| `WOLFFORGE_V2_MEMORY_DECAY` | `0.90` | lesson-support decay factor |

Default path: `~/Library/Application Support/persuasion-arena/wolfforge_v2_memory.sqlite` (macOS),
`$XDG_DATA_HOME/persuasion-arena/...` (Linux), `%APPDATA%/persuasion-arena/...` (Windows).

Modes:
- `off` — no memory; default V2 behavior (byte-identical prompt).
- `read` — inject previously built memory; **does not modify** the DB. Use for holdout.
- `write` — record post-game summaries; do not inject.
- `readwrite` — inject **and** record. **Development only.**

## When memory is read / written

- **Read** happens at the start of each turn (if mode is `read`/`readwrite`), bounded by
  `MAX_PROMPT_CHARS`.
- **Write** happens **only on a game-complete event** (`game_result`), never mid-game, and only in
  `write`/`readwrite`. Writes are idempotent per game and use only that game's own state (no
  cross-game leakage). If the agent process can't observe completion, derive memory from saved runs
  with a post-run hook (extension point; not a paid summarizer).

## Safety / failure behavior

- A corrupt or unwritable DB **disables memory** (mode → off) with a warning — it never crashes a game.
- Read-only mode never modifies the DB.
- No memory **contents** are logged; only metadata (mode, snapshot hash, context size, write count).

## Snapshot hash (reproducibility)

`CrossRunMemory.snapshot_hash()` = `sha256(schema_version + sorted, timestamp-free compact export)`,
stable across process restarts for identical content. Freeze it for a holdout.

## Inspect / export / prune / reset

```bash
# inspect / export the compact, hashable snapshot (no secrets, no raw text)
DATABASE_URL="" uv run python -c "from examples.wolfforge_v2_memory import CrossRunMemory; \
m=CrossRunMemory(path='./memory/wf_v2_dev_memory.sqlite', mode='read'); \
import json; print('snapshot', m.snapshot_hash()); print(json.dumps(m.export_compact(), indent=2))"

# prune/decay (development): drop low-confidence lessons deterministically
DATABASE_URL="" uv run python -c "from examples.wolfforge_v2_memory import CrossRunMemory; \
m=CrossRunMemory(path='./memory/wf_v2_dev_memory.sqlite', mode='readwrite'); \
print('removed', m.prune(min_confidence=0.1))"

# reset: just delete the file
rm ./memory/wf_v2_dev_memory.sqlite
```

## No-cost smoke commands

```bash
# terminal 1
DATABASE_URL="" uv run python -m arena.cli serve --port 8000

# memory OFF (default behavior; prints "memory: off")
DATABASE_URL="" uv run python tools/wolfforge_v2_run.py \
  --run-id wf_v2_memory_off_smoke --games 5 --fake-brain --memory-mode off \
  --server http://127.0.0.1:8000

# memory READ (model-free; prints "memory: read snapshot=<hash> path=...")
DATABASE_URL="" uv run python tools/wolfforge_v2_run.py \
  --run-id wf_v2_memory_read_smoke --games 5 --fake-brain --memory-mode read \
  --memory-path ./tmp/test_memory.sqlite --server http://127.0.0.1:8000
```

## Future paid experiments (do not run unattended)

**Build a development memory (readwrite):**
```bash
DATABASE_URL="" uv run python tools/wolfforge_v2_run.py \
  --run-id wf_v2_memory_dev --games 36 --rounds 4 --seed 120000 \
  --model openai/gpt-4o-mini --memory-mode readwrite \
  --memory-path ./memory/wf_v2_dev_memory.sqlite --server http://127.0.0.1:8000
```

**Holdout with a FROZEN snapshot (read only; do not write during holdout):**
```bash
DATABASE_URL="" uv run python tools/wolfforge_v2_run.py \
  --run-id wf_v2_memory_holdout_a --games 60 --rounds 4 --seed 220000 \
  --model openai/gpt-4o-mini --memory-mode read \
  --memory-path ./memory/frozen_snapshot.sqlite --server http://127.0.0.1:8000
```

## Experiments

### Experiment 1 — does memory help? (memory-off vs memory-read)
1. Build memory from a development block (`--memory-mode readwrite`, dev seeds).
2. Freeze the snapshot (copy the sqlite file; record `snapshot_hash()`).
3. Run **memory-off** and **memory-read** on fresh holdout seeds (read only — do **not** write).
4. Compare with `tools/wolfforge_v2_eval.py`; the report warns if modes/snapshots are mixed.

### Experiment 2 — opponent-profile memory
Only with stable `agent_id` identities; require `MIN_GAMES_FOR_OPPONENT_HINT` shared games before a
hint; report whether opponents actually repeat across train/test.

## Evaluation warnings

`tools/wolfforge_v2_eval.py` warns (in `report.md` / `summary.json`) when compared runs differ in
**memory mode** or **memory snapshot hash**, or when a **write/readwrite** mode appears during
evaluation. **Never** compare a memory-updating agent against a frozen baseline and call it a clean
prompt-policy test.

## Privacy / security risks

- Self-assessed win/loss is heuristic (believed role); it can be wrong for Robber/Drunk.
- Opponent hints are low-confidence without a stable `agent_id`; never accuse on memory alone.
- The memory file is local user data; treat it like any local cache (no secrets are stored in it).

## Does the V2 PR claim change?

No. The default V2 guarantee (stable identity + isolated per-game state) is unchanged; memory is an
optional, clearly-separated follow-up that is off by default.
