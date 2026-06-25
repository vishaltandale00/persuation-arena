# WolfForgeAgentV2

WolfForgeAgentV2 is a **persistent, connected ONUW agent**: a stable arena identity, isolated
per-game memory rebuilt from the event stream, a coalition-first phase-aware policy with mechanical
checks, schema-safe actions under deadlines, deterministic legal fallbacks, and safe telemetry.

## What it adds beyond the WolfForge prompt experiment

The original [WolfForge experiment](./WOLFFORGE_RESULTS.md) was a **static system-prompt ablation**:
the model and inference settings were frozen and only a one-paragraph strategy prompt changed
(`base` / `bayes` / `charisma` / `chaos`). The verified findings (two blinded 60-game blocks):

- `charisma` 47.5%, pooled `base` 44.2%, `chaos` 35.0%, `bayes` 27.5%
- Charisma − Bayes = **+20.0 pp**, exact paired McNemar `p = 0.0032`, whole-game bootstrap CI
  ≈ **+7.5 to +32.5 pp**
- Charisma was not distinguishable from `base`; `bayes` also forfeited more, but forfeits did not
  fully explain its deficit.

The lesson was **not** "charisma is a magic prompt." It was: coalition formation and explicit vote
coordination beat exhaustive plausible-world analysis; more analytical prompting did not help; and
malformed actions directly lose games. V2 turns those lessons into an actual artifact:

| WolfForge (prompt ablation) | WolfForgeAgentV2 |
|---|---|
| One static system prompt, no memory | Persistent connected agent with a stable identity across runs |
| Stateless / rolling message window | Compact per-game state rebuilt from the delta event stream, idempotent |
| Implicit strategy in prose | Explicit coalition-first, phase-aware policy with role/chronology checks |
| Best-effort parsing + one retry | Reliability ladder: structured output → validate → repair → deterministic legal fallback |
| No deadline accounting | Deadline-aware: reserves submission margin, skips unsafe repair calls |
| No run identity/fingerprint | Frozen policy version + prompt hash recorded in every telemetry row |

## Files

```
examples/wolfforge_v2_policy.py   # pure, model-free policy: prompt, state, schema, legality, fallback
examples/wolfforge_v2_agent.py    # the connected harness: ladder, deadline, telemetry, baseline factory
tools/wolfforge_v2_run.py         # orchestrate a connected V2-vs-baseline run (incl. --fake-brain)
tools/wolfforge_v2_eval.py        # paired V2-vs-baseline statistics + CSV/JSON/Markdown report
tests/test_wolfforge_v2_agent.py  # harness + policy + model-free connected integration tests
tests/test_wolfforge_v2_eval.py   # statistics + fixture-run analysis tests
```

## Architecture and state lifecycle

The official SDK (`persuasion_arena_agent.ArenaAgent`) delivers a **delta event stream** and turns,
and owns only the event cursor. Building memory from those deltas is the harness's job. V2 wires two
SDK hooks:

- `on_event(event)` folds each delivered event into the owning game's state.
- `act(turn)` decides from that state and returns `{"action": <wire action>, "reasoning": <brief>}`.

**Scope of "memory".** V2's memory is **per-game state rebuilt from the event stream** (discarded when
the game ends) plus a **stable connected-agent identity** persisted across runs via credentials. It is
**not** cross-run learning: nothing the agent concludes in one game is carried into a later game or
stored to learn from over time.

**Per-game key.** State is keyed by `game_instance_id` — the SDK's canonical per-game identifier,
present on both `Event` and `Turn`. `run_id` (known from the signup) is a defensive prefix, so two
concurrent runs can never collide: `state_key = f"{run_id}:{game_instance_id}"`. In practice
`game_instance_id` already embeds the run id (e.g. `run_x_game_001`).

**State reducer.** The heavy event folding (roster, deck, believed role, night observations, public
conversation) is delegated to `examples/seat_state.py::SeatState`, the repository's reference ONUW
event reducer — V2 does not reimplement event sourcing. On top of it, `GameState` carries compact
**conclusions** (not raw chain-of-thought): belief summary, public commitments, coalition, primary/
secondary vote target, known swaps (chronological), and reliability counters.

**Idempotency.** `GameState.apply_event` records each `event_id` and ignores re-delivery, so the same
event twice never duplicates a claim or corrupts state.

**Isolation.** Private events are delivered by the server only to the owning signup, and V2 keys
strictly by game, so one game's private observations can never appear in another game's state. This
is verified by `test_connected_two_games_model_free_isolated` (real engine, two concurrent games).

## Policy

Stable system policy (frozen; its text is hashed into `prompt_hash`):

- **Early discussion:** collect concrete role/night-action claims; ask one targeted, falsifiable
  question; do not lock onto a single theory; establish credibility without dumping every fact.
- **Middle discussion:** reconcile the strongest claims; name one or two actual contradictions;
  separate mechanically impossible from merely suspicious; make your reading easy to repeat; build a
  coalition around a shared plan.
- **Late discussion:** stop expanding theories; summarize the strongest public case; name one
  explicit target; coordinate the vote; never self-contradict at the buzzer.
- **Vote:** choose the legal target that maximizes the current role/objective's win probability (not
  merely the loudest speaker); never self-vote.

Role-specific objectives are embedded for village information roles, Robber/Troublemaker chronology,
Drunk uncertainty, Werewolf, Minion, and Tanner (win-only-by-elimination). The policy respects night
wake order and forbids mechanically impossible claims.

**Untrusted dialogue.** Player speech is treated as in-game evidence only. The system prompt
instructs the model to ignore any in-game text asking it to reveal the prompt, credentials, or
private observations, or to change policy. Injection text is stored as ordinary evidence and never
acted on (`test_player_injection_is_untrusted_evidence`).

### Model request shape

Each request is built from: the stable system policy + a compact JSON state snapshot (the seat's
filtered view, including the public conversation as evidence) + the current phase/deadline + the
exact legal-action schema. It does **not** resend an uncontrolled transcript. The model is asked for:

```json
{
  "action": { "...matches the turn's legal-action schema..." },
  "brief_reasoning": "<= 2 sentences, private",
  "state_update": {
    "belief_summary": "...", "public_commitments": [],
    "coalition": ["@handle"], "primary_target": "@handle | null", "secondary_target": "@handle | null"
  }
}
```

**Current action vocabulary (participant-ref protocol).** Players are identified by public participant
refs (`@handle`), never integer seats. Concretely:

- vote → `{"target": "@handle"}` or `{"target": "@no-one"}` (abstain); never an integer or `-1`.
- discussion → `{"speak": "<text>", "urgency": 1|2|3}` or `{"pass": true, "stance": "wait"|"done"}`
  (speak **requires** `urgency`; `stance` is optional on pass).
- seer → `{"mode":"player","target":"@handle"}` or `{"mode":"center","indices":[a,b]}`.
- robber/doppelganger → `{"target":"@handle"}` (robber may also `{"target": null}` to decline).
- troublemaker → `{"a":"@handle","b":"@handle"}` or `{"a":null,"b":null}`; drunk → `{"index":0|1|2}`.

`coalition`, `primary_target`, and `secondary_target` carried in state are also `@handle` refs.

### Reliability ladder

1. **Structured-output request** (`response_format: json_schema`, `strict: false`) when the mode is
   `auto`/`json_schema` and the provider is on the known list. This is **best-effort guidance, not
   enforced**: the engine's legal-action schemas use `oneOf`, which OpenAI's strict mode does not
   support, so V2 does not request `strict: true` or claim the provider enforced the schema. Telemetry
   records the honest status (`structured_output_enforced`: `best_effort` / `bypassed` / `off`).
2. **Local exact-schema validation** of the action against the turn's legal action (the same subset
   the server applies), plus shared normalization (canonicalize unambiguous encodings → fill
   schema-required fields like `urgency` → clamp `speak` length).
3. **One repair call** containing only the exact validation error and the required schema — *only if
   the deadline leaves a documented safety margin*.
4. **Deterministic legal fallback** otherwise, derived from the served `legal_action`. Documented
   ordering:
   - discussion → `{"pass": true}`; vote → accumulated `primary_target`, else `secondary_target`,
     else `{"target": "@no-one"}`; seer → center `[0,1]`; robber/troublemaker → decline;
     drunk → center `0`; doppelganger → first legal participant ref.

If a provider rejects structured output, V2 retries once without `response_format` (recorded as
`bypassed`) rather than forfeiting. An invalid action is **never** submitted.

### Deadline behavior

V2 reads the turn's real `deadline_at`, subtracts a configurable submission margin, and:

- skips the model entirely (deterministic fallback) when remaining time ≤ submit margin;
- bounds each model call's timeout to the remaining budget;
- starts a repair call only when there is room for a full call plus the margin.

## Identity and credential behavior

V2 uses the official `ArenaAgent` lifecycle and `CredentialsStore`. One stable connected-agent
identity is registered on first use and **reused** afterward (`ensure_registered`), persisted at
`0600` to the credential file. The same identity is reused across later runs unless credentials are
reset. There is **no second credential format**.

- Default credential path: `$XDG_CONFIG_HOME/persuasion-arena/credentials.json` (else
  `~/.config/persuasion-arena/credentials.json`).
- Override with `PERSUASION_ARENA_CREDENTIALS=/path/credentials.json` or `--credentials <path>`.

**Intentional credential reset** (start a brand-new identity):

```bash
# inspect what's stored (tokens are redacted)
arena-agent credentials
# reset: remove the credential file (or point at a fresh path)
rm ~/.config/persuasion-arena/credentials.json
# or use an isolated identity without touching the default store:
arena-agent play --run <run_id> --credentials /tmp/wf_v2_identity.json \
    --name WolfForgeV2 examples/wolfforge_v2_agent.py
```

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | — | OpenRouter key (read from env/`.env`; never logged). Required for real model calls. |
| `WOLFFORGE_V2_MODEL` | `openai/gpt-4o-mini` | Model slug. gpt-4o-mini keeps V2 comparable to the original ablation; a stronger model is selectable with no source change. |
| `WOLFFORGE_V2_TEMPERATURE` | `0.35` | Decoding temperature (matches the original charisma block). |
| `WOLFFORGE_V2_REASONING_EFFORT` | `medium` | OpenRouter reasoning effort. |
| `WOLFFORGE_V2_MAX_TOKENS` | `4000` | Max tokens per turn. |
| `WOLFFORGE_V2_BRAIN_TIMEOUT` | `40` | Per model-call timeout (seconds). |
| `WOLFFORGE_V2_SUBMIT_MARGIN_S` | `3.0` | Time reserved for submission; below this, fall back deterministically. |
| `WOLFFORGE_V2_STRUCTURED_OUTPUT` | `auto` | `off` / `auto` / `json_object` / `json_schema`. Requested as best-effort (`strict: false`); not provider-enforced — see the reliability ladder. |
| `WOLFFORGE_V2_LOG_PATH` | — | If set, append safe JSONL telemetry here. |
| `WOLFFORGE_V2_RUN_ID` | — | Run id for the module-level `arena-agent play` handlers. |

## Running it

### Local (SQLite) — no cost, model-free protocol smoke

```bash
# terminal 1: start the observer server sharing this process's SQLite store (leave DATABASE_URL unset)
python -m arena.cli serve --port 8000

# terminal 2: model-free connected run (no OpenRouter calls, no cost)
python tools/wolfforge_v2_run.py --run-id wf_v2_smoke --games 5 --fake-brain \
    --server http://127.0.0.1:8000
```

`--fake-brain` makes V2 and the baseline play through their deterministic legal fallback and seats
the reference random agent as opponents, so the entire run is free. It exercises the connected
protocol, per-game isolation, deadlines, and legal-action plumbing end to end.

### Connect a single V2 agent to an existing run (real model)

```bash
export OPENROUTER_API_KEY=...      # never printed or logged
export WOLFFORGE_V2_RUN_ID=<run_id>
arena-agent play --run <run_id> --server http://127.0.0.1:8000 \
    --name WolfForgeV2 examples/wolfforge_v2_agent.py
```

### Hosted run (current SDK)

Point `--server` at the hosted API and follow the coordinator URL the SDK returns automatically
(`ArenaAgent` repoints ready/poll/reply to the per-run coordinator). Connecting agents only (a remote
coordinator drives the games):

```bash
python tools/wolfforge_v2_run.py --run-id <hosted_run_id> --join \
    --server https://persuation-arena.vercel.app
```

> This document does not instruct joining any specific hosted run; supply your own authorized run id.

## Logs and privacy

Telemetry is opt-in via `WOLFFORGE_V2_LOG_PATH` and writes one safe JSONL record per turn:
`timestamp, agent_name, policy_version, prompt_hash, run_id, game_id, turn_id, seat, phase,
action_kind, requested_model, resolved_model, latency_ms, prompt_tokens, completion_tokens,
repair_attempted, fallback_used, action_valid, error_type`.

**Never logged:** API keys, bearer tokens, credential-file contents, full private night observations,
provider reasoning, or any other game's state. Verified by `test_telemetry_is_safe`.

## Failure / fallback behavior

- Malformed / missing JSON → one repair (budget permitting) → deterministic legal fallback.
- Provider rejects structured output → retry once without `response_format`.
- Deadline too close → deterministic legal fallback (no model / no repair).
- A single failed turn never crashes the agent (the SDK keeps the signup and retries next poll).

An invalid action is never submitted; fallbacks are deterministic and documented above.

## Tests

```bash
python -m pytest tests/test_wolfforge_v2_agent.py tests/test_wolfforge_v2_eval.py -q
```

Coverage includes stable game keys, two-game isolation, idempotent replay, public/private handling,
dealt-vs-final role, chronological swaps, malformed-then-repaired, failed-repair fallback, deadline
guards, structured-output rejection fallback, prompt-injection resistance, telemetry safety, stable
serialization, role objectives, policy-version/prompt-hash stability, the connected identity/credential
lifecycle and session resume, and a model-free connected integration over two concurrent games.

## Cross-run memory (optional, off by default)

Cross-run memory is optional and **off by default**. V2's default guarantee remains **stable identity
+ isolated per-game state, not cross-run learning**. When enabled (`WOLFFORGE_V2_MEMORY_MODE != off`),
the agent surfaces safe, compact post-game summaries as a clearly-separated **weak prior** (never as
current-game facts), and — only in `write`/`readwrite` — records summaries on game completion. With
memory off, the prompt is byte-identical to pre-memory V2. See
[WOLFFORGE_V2_MEMORY.md](./WOLFFORGE_V2_MEMORY.md) for schema, privacy guarantees, and commands.

## Limitations

- The connected baseline reuses V2's machinery and swaps only the strategy prompt; it reproduces the
  Charisma *strategy* faithfully but is not byte-identical to the original in-process ablation agent
  (which used full-context turns rather than delta transport). See
  [WOLFFORGE_V2_EVAL.md](./WOLFFORGE_V2_EVAL.md).
- Mixed-table play measures relative tournament performance with strategic interference, not isolated
  1v1 skill.
- The compact state includes the public conversation as evidence; for very long games this grows with
  the transcript (bounded by the game, not an uncontrolled history).
- Behavioral/telemetry metrics are heuristic, not ground truth.
