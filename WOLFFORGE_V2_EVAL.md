# WolfForgeAgentV2 evaluation

The value of V2 is not that it exists — it is whether it **measurably** beats a frozen baseline
without raising forfeits, latency, or cost beyond acceptable limits. This document defines that
evaluation **before** results are observed.

## Research question and hypothesis

> Does WolfForgeAgentV2 outperform the frozen Charisma baseline when model, temperature, token caps,
> reasoning effort, opponents, seeds, seats, and game settings are held constant?

**Hypothesis (pre-registered).** V2's persistent state, mechanical checks, and reliability ladder
improve paired win rate over the frozen coalition-first Charisma policy while keeping forfeits no
worse.

## Frozen baseline

`WolfForgeV2Agent.charisma_baseline(...)` reuses the **identical** connected harness machinery
(event-sourced state, reliability ladder, deterministic fallback, deadline handling) and swaps in the
**verbatim Charisma strategy paragraph** from `arena/wolf_profiles.py`
(`PROFILE_INSTRUCTIONS["charisma"]`, plus the `BASE_SYSTEM` framing and `ONUW_COMMON`), wrapped in the
**same single output contract V2 uses** so both arms are protocol-equivalent. The single experimental
variable between V2 and the baseline is the **strategy prompt** — exactly the design of the original
WolfForge ablation.

Faithfulness caveat: the original experiment ran in-process with full-context turns; the connected
architecture uses delta transport, so the baseline is "as faithful as the current connected
architecture permits," not byte-identical. To keep both arms on one output contract, `BASE_SYSTEM`'s
old `{"reasoning","action"}` format sentence is dropped (it contradicted V2's contract); the
substantive Charisma strategy text is unchanged. Its frozen identity is recorded as
`policy_version = "charisma-baseline-1.1"` with its own `prompt_hash`.

**Do not silently update the baseline during development.** Its prompt and version are frozen.

## Development / holdout separation

- **Development block** (~30–40 games): used to diagnose and improve V2. Results here are **not**
  final evidence.
- **Holdout blocks** (two fresh 60-game blocks): used once, after V2 is frozen, on seeds **disjoint**
  from all development seeds. **Never tune on holdout games.**

## Matched-seed / seat design

V2 and the baseline play in the **same games** within a run (same seed block), so their per-game
seeds are inherently matched — the unit of the paired analysis is the game. The connected coordinator
rotates the seat→agent mapping each game (`rot = gid % players` in `arena/connected.py`), so V2 is
not pinned to one seat. The game is the cluster for all uncertainty.

## Opponent pool

A stable, documented, all-connected pool (the current connected tooling drives every seat through one
coordinator; it does not mix connected and in-process static agents in a single run):

- seat A: `WolfForgeV2`
- seat B: `CharismaBaseline` (frozen)
- seats C–E: three reference opponents — by default three `openai/gpt-4o-mini` seats using the
  repository's reference `SessionAgent` harness (override with `--opponent-models`).

Keep the opponent pool and all decoding settings **fixed** across development and holdout.

## Metrics

Captured by `tools/wolfforge_v2_eval.py` from saved runs:

- wins / games; paired V2-vs-baseline outcomes per game; role-specific and team (good/evil) win rate.
- forfeit rate (from the store), and — when a telemetry log is supplied — repair rate, fallback rate,
  invalid-output rate, median & p95 latency, prompt/completion tokens.
- discordant pair counts (V2-only wins `b`, baseline-only wins `c`).

Deterministic behavioral counters (explicit target named before the vote, changed public target late,
contradicted own earlier claim, asked a falsifiable question, used fallback) are derived heuristically
from telemetry/transcripts and are **documented as heuristic, not ground truth**.

## Statistical methods

The game is the cluster/unit. Implemented in `tools/wolfforge_v2_eval.py` (stdlib only):

- **Exact paired McNemar** on per-game win/loss pairs (`mcnemar_exact(b, c)`): exact two-sided
  binomial tail on the discordant pairs. This is the primary significance test.
- **Whole-game bootstrap CI** (`bootstrap_diff_ci`): resamples whole games with replacement (seeded,
  reproducible) and reports the 95% percentile interval for V2 − baseline win rate.
- **Effect size** in percentage points; **role/objective stratification** via Wilson intervals.
- No naive independent-seat p-value is reported as the main result.

## Evaluation stages

### Stage 0 — no-cost correctness (run this first)

```bash
python -m pytest tests/test_wolfforge_v2_agent.py tests/test_wolfforge_v2_eval.py -q
```

Model-free local connected protocol smoke (needs the local server; no OpenRouter calls, no cost):

```bash
python -m arena.cli serve --port 8000          # terminal 1 (leave DATABASE_URL unset)
python tools/wolfforge_v2_run.py --run-id wf_v2_smoke --games 5 --fake-brain \
    --server http://127.0.0.1:8000             # terminal 2
```

### Stage 1 — paid five-game smoke test (command only; do NOT run automatically)

Purpose: confirm OpenRouter auth, structured output / fallback, no cross-game leakage, and to
**estimate calls, latency, and cost** before any large run. Run with `workers=1` (the connected
coordinator already enforces this).

```bash
export OPENROUTER_API_KEY=...                  # never printed or logged
export WOLFFORGE_V2_LOG_PATH=$PWD/wf_v2_smoke.jsonl
python -m arena.cli serve --port 8000          # terminal 1
python tools/wolfforge_v2_run.py --run-id wf_v2_smoke5 --games 5 --rounds 4 --seed 60000 \
    --model openai/gpt-4o-mini --server http://127.0.0.1:8000   # terminal 2
```

Then inspect one replay (observer at `http://localhost:8000/observer.html`) and the telemetry log.

### Stage 2 — development block (~30–40 games, clearly labeled dev seeds)

Diagnose and improve V2 here only; do **not** present these as final evidence.

```bash
export OPENROUTER_API_KEY=...
export WOLFFORGE_V2_LOG_PATH=$PWD/wf_v2_dev.jsonl
python tools/wolfforge_v2_run.py --run-id wf_v2_dev --games 36 --rounds 4 --seed 100000 \
    --model openai/gpt-4o-mini --server http://127.0.0.1:8000
python tools/wolfforge_v2_eval.py --runs wf_v2_dev --telemetry wf_v2_dev.jsonl \
    --out-dir ./wf_v2_dev_report
```

Build a failure taxonomy from transcripts and telemetry: mechanical misunderstanding,
dealt/final-role confusion, missed contradiction, over-analysis, weak coalition formation, bad reveal
timing, inconsistent bluff, poor Tanner strategy, poor Minion strategy, late vote coordination,
invalid action, repair, fallback, timeout, other. Prefer deterministic metrics and human inspection.
**No paid judge model in this first version.**

### Stage 3 — freeze V2

Stop changing the prompt. Write a manifest recording the exact agent and run settings:

```bash
python - <<'PY'
import json, subprocess
from examples import wolfforge_v2_policy as p
sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
manifest = {
    "git_sha": sha,
    "policy_version": p.POLICY_VERSION,
    "prompt_hash": p.prompt_hash(),
    "baseline_policy_version": p.BASELINE_POLICY_VERSION,
    "baseline_prompt_hash": p.baseline_prompt_hash(),
    "model": "openai/gpt-4o-mini",
    "temperature": 0.35,
    "reasoning_effort": "medium",
    "max_tokens": 4000,
    "brain_timeout_s": 40.0,
    "structured_output": "auto",
    "opponents": ["openai/gpt-4o-mini", "openai/gpt-4o-mini", "openai/gpt-4o-mini"],
    "deck_preset": "arena",
    "discussion_rounds": 4,
    "holdout_seed_blocks": [200000, 300000],
}
open("frozen_v2.json", "w").write(json.dumps(manifest, indent=2, sort_keys=True))
print(json.dumps(manifest, indent=2, sort_keys=True))
PY
```

No prompt changes after this point.

### Stage 4 — holdout evaluation (two fresh 60-game blocks, disjoint seeds)

```bash
export OPENROUTER_API_KEY=...
export WOLFFORGE_V2_LOG_PATH=$PWD/wf_v2_holdout.jsonl
python -m arena.cli serve --port 8000          # terminal 1

# block A and block B use disjoint seed ranges, both distinct from the dev block (100000+)
python tools/wolfforge_v2_run.py --run-id wf_v2_holdout_a --games 60 --rounds 4 --seed 200000 \
    --model openai/gpt-4o-mini --server http://127.0.0.1:8000
python tools/wolfforge_v2_run.py --run-id wf_v2_holdout_b --games 60 --rounds 4 --seed 300000 \
    --model openai/gpt-4o-mini --server http://127.0.0.1:8000
```

Analysis (the main comparison is V2 vs frozen Charisma):

```bash
python tools/wolfforge_v2_eval.py --runs wf_v2_holdout_a wf_v2_holdout_b \
    --v2-name WolfForgeV2 --baseline-name CharismaBaseline \
    --telemetry wf_v2_holdout.jsonl --manifest frozen_v2.json \
    --out-dir ./wf_v2_holdout_report
```

The report (`report.md`, `summary.json`, `games.csv`) **prominently warns** when compared runs differ
in model, temperature, token caps, deck, rounds, or opponent composition (and, with `--manifest`,
when they differ from the frozen V2 manifest). Treat the headline number as invalid if the mismatch
banner appears. **Do not tune on holdout games.**

## Approximate call-count formula

Per game, one agent acts on: its discussion turns (≤ `discussion_rounds`, fewer if discussion dies
early on an all-pass round), exactly one vote, and a night action only if its dealt role has one
(`arena` deck: Seer/Robber present, ~0–1 per agent). Repairs add ≤ 1 call per malformed turn (rare
with structured output).

```
calls_per_agent_per_game ≈ discussion_rounds + 1 (vote) + night_share(≈0.3) + repair_overhead(≈0)
calls_per_game (all seats) ≈ players × calls_per_agent_per_game
```

With `players = 5`, `discussion_rounds = 4`: ≈ **5–6 calls/agent/game**, ≈ **26–30 calls/game**.
So:

| Block | Games | ≈ total model calls |
|---|---:|---:|
| Stage 1 smoke | 5 | ~130–150 |
| Stage 2 dev | 36 | ~950–1080 |
| Stage 4 holdout (each) | 60 | ~1560–1800 |
| Stage 4 holdout (both) | 120 | ~3100–3600 |

Re-estimate from the Stage-1 smoke's actual usage before launching holdout.

## OpenRouter spend-control checklist

- Set a **separate OpenRouter key with its own spend limit** for this experiment; never reuse a
  high-limit production key.
- Check remaining credits before each block: `curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY"
  https://openrouter.ai/api/v1/credits`.
- Start with **`workers=1`** (the connected coordinator enforces this) and the five-game smoke.
- **Estimate calls from the smoke test** (count telemetry rows: `wc -l wf_v2_smoke.jsonl`) and
  extrapolate with the formula above before any 60-game block.
- **Stop on repeated `401 / 402 / 429 / timeout` errors** — these mean auth/credit/rate problems;
  do not let a run burn credits retrying.
- Do **not** use free-tier rate limits for a large final run.
- **Record actual usage** returned by the API: V2 telemetry logs `prompt_tokens` / `completion_tokens`
  per turn; sum them and multiply by the model's current returned price. Do not hardcode a dollar
  figure — compute it from the model's live pricing/usage metadata.

## Result-reporting template

```
WolfForgeAgentV2 vs CharismaBaseline (frozen) — holdout (120 games, 2 blocks)
Config: model=…, temp=…, effort=…, max_tokens=…, deck=…, rounds=…, prompt_hash=…
Manifest mismatch banner: none / <details>

V2 win rate:        __/120 = __._%
Baseline win rate:  __/120 = __._%
Effect (V2 − base): +__._ pp
95% whole-game bootstrap CI: [+__._, +__._] pp
Exact paired McNemar: b=__ (V2-only), c=__ (baseline-only), p=0.____

Forfeit rate:  V2 __._%  vs  baseline __._%
Repair / fallback / invalid (V2): __._% / __._% / __._%
Latency median / p95: ___ ms / ___ ms
Tokens (prompt/completion total): ___ / ___;  estimated cost: $_.__ (from live pricing)
Role breakdown: <table>
Limitations: mixed-table interference; heuristic behavioral counters; small per-role cells.
Conclusion: <meets / does not meet the success criterion, stated plainly>
```

## What counts as a real improvement (success criterion)

Pre-registered, treated as an initial engineering target (not a guaranteed outcome):

> V2 improves paired holdout win rate over frozen Charisma by **at least 5 percentage points**, its
> whole-game 95% interval is **directionally favorable** (and ideally excludes zero), and its
> **forfeit/fallback rate is no worse** than the baseline.

If the comparison is not controlled (mismatch banner present), no conclusion is drawn.
