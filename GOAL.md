# Persuasion Arena — Build Brief (execution goal)

North-star prompt that drives the build. **Prototype-first:** the priority is a working,
watchable ONUW game played by real models, as fast as possible — then harden and expand.
Work phase by phase. A phase is **done** only when its acceptance test passes. See `DESIGN.md`
for rationale and `web/observer.html` for the exact data shapes to produce.

## Objective

Build an **evaluation harness for social-intelligence games** — a system that plugs LLM agents
into a hidden-role game, runs them through it, and produces (a) the **eval signal** (outcome →
role-balanced win-rate) and (b) a **watchable record** in the observer interface (public chat,
true roles, actions, each agent's **private reasoning**). **Start with the ONUW harness**;
Avalon and Secret Mafia are later harnesses behind the same interface. Local now; Modal later.

A "harness" = the pluggable agent contract + the game's rules/turn-loop + per-seat info filtering
+ the logged record that feeds the interface. The interface is how a human evaluates what the
agents did. The eval is the point: how socially intelligent is each agent (deception, deduction,
persuasion)?

## Prime directive: build the ONUW harness first, end to end

Ship the **ONUW harness walking skeleton** (P0) before anything else: real agents (cheap models)
plugged in, one full ONUW game run, logged, served locally, and replayable in `observer.html`.
Everything after P0 hardens the ONUW harness or adds more harnesses. Do not build scoring, Avalon,
the full role set, or polish until the ONUW harness plays one game you can watch.

## Execution protocol — run to completion (do not stop until built)

This brief is meant to be executed autonomously by Claude Code using its full toolset. **Do not
stop at phase boundaries, do not hand back for approval between steps, and do not end the turn
until the Definition of Done is fully met and verified — or a listed blocker is hit.**

**Operating loop.** Maintain phase state with the task tools (one task per phase P0–P4; mark
in_progress/completed). Repeat until all phases are done:
1. Read the repo's current state and the next unmet acceptance criterion.
2. Implement it (Write/Edit). Keep changes small and runnable.
3. **Verify with tools — never claim done from reading code:**
   - Logic/tests: run via **Bash** (`pytest`, scripts, `curl`). Read real output.
   - End-to-end UI: start the server with **Bash** (`run_in_background`), `curl` the endpoints to
     confirm JSON, then **open `web/observer.html` in the browser via the `claude-in-chrome`
     tools and screenshot it** to confirm a real game actually renders (chat, true roles, private
     reasoning on click, correct winner). A phase that touches the interface is not done without a
     screenshot showing real data.
4. If red, fix and re-verify. If green, mark the task complete and **immediately start the next
   phase** in the same turn.
5. When P0–P4 are all green, do a final end-to-end verification pass, then summarize.

**Self-correction.** On failure, inspect the actual error (console via
`claude-in-chrome read_console_messages`, server logs, tracebacks), fix root cause, re-run. Use a
**Workflow** when a sub-problem benefits from parallel construction or adversarial review (e.g.
hardening the ONUW core in P1), but routine sequential coding is done directly.

**Only stop for a real blocker** (otherwise pick a sensible default, note it, and continue):
- The API key is missing / invalid / out of credit (Bash shows an auth/402 error from OpenRouter).
- A genuine product decision not answered by `GOAL.md`/`DESIGN.md` and not defaultable.
- The same step fails >3 times after distinct fixes (report what was tried).
- An action would be destructive/irreversible and needs consent.

Progress is communicated by brief narration + task updates, not by pausing. Treat installing deps,
running local servers, `curl`, `pytest`, opening the browser, and making **cheap** model calls as
pre-authorized regular actions.

## Definition of done (the gate)

All true, each verified by a command or a screenshot — not by inspection:
- [ ] `pytest` green (invariants + ONUW + Avalon conformance).
- [ ] One command plays a real ONUW game with the cheap roster and writes it to SQLite.
- [ ] The local server serves runs/games JSON in the observer's shapes (`curl` confirms).
- [ ] `observer.html` renders a **real** game end to end — verified by screenshot — with chat,
      true roles, per-seat private reasoning, and the correct winner.
- [ ] A ≥20-game ONUW run shows role-balanced win-rate + 95% CIs in the observer (real data).
- [ ] Avalon run replays + scores; Secret Mafia adapter runs.
- [ ] README documents add-agent / add-game / start-run. No secret printed/logged/committed.

## Tech & decisions (locked)

- **Language:** Python 3.12 in a repo venv. **Models:** OpenRouter via the `openai` SDK
  (`base_url=https://openrouter.ai/api/v1`), key from `.env` → `OPENROUTER_API_KEY`
  (already present; load with `python-dotenv`; never print/log/commit it).
- **Engine:** our own ONUW/Avalon Game Cores on `ta.Env` + `ta.TeamMultiPlayerState`; TextArena
  only as the library/loop and for Secret Mafia.
- **Store:** SQLite (`store/arena.db`), schema per `DESIGN.md §8`.
- **Observer feed:** a small **local FastAPI server** serves runs/games as JSON in the observer's
  shapes and (later) launches runs; `observer.html` fetches from it. (Modal swaps in for this server later.)
- **Agents (cheap dev roster — `agents.yaml`):**
  ```yaml
  - {name: Leo,    model: qwen/qwen3-30b-a3b-instruct-2507, harness: base}
  - {name: Vishal, model: z-ai/glm-4.7-flash,               harness: base}
  - {name: Chappy, model: deepseek/deepseek-chat-v3.1,      harness: base}
  - {name: Saurav, model: moonshotai/kimi-k2.5,             harness: base}
  - {name: Neeraj, model: openai/gpt-5-mini,                harness: base}
  - {name: Will,   model: anthropic/claude-haiku-4.5,       harness: base}
  ```
- **Determinism:** one seed per game → reproducible deal; log it.

## Agent contract

Each turn the Game Core sends a seat its filtered context + the legal action; the agent returns
**JSON**: `{"reasoning": "...private...", "action": <public action>}`. `reasoning` is logged and
**never** shown to any other agent. `action` is free text for discussion or a typed value for
mechanical turns. Timeout/invalid → 1 retry → default/forfeit, flagged in logs.

## Non-negotiable invariants (from the fidelity audit — `DESIGN.md §13`)

1. **Per-seat filtering = observation-log replay.** A seat's view is built only from events it
   legitimately observed; never dereference current/end-of-night roles or center cards it didn't
   inspect. Track **original vs. current role** separately.
2. **Atomic simultaneous vote.** Freeze pre-vote state, collect votes hidden, reveal together.
   Never sequential public voting.
3. **All-tied-die** plurality (configurable threshold). Never random tie-break.
4. **Round-robin discussion**, speak-or-pass, ends on all-pass or turn cap. No weighted queue/priority.
5. **Caps** bound cost: game cycles, discussion turns/rounds, per-utterance tokens, 1 invalid retry.
6. **Private reasoning logged, never leaked.**
7. ONUW win on **end-of-night roles**: Hunter chain into the same death set → Tanner override → cases.

## Phases (prototype-first; each ships something runnable)

**P0 — Walking skeleton (the prototype). THE PRIORITY.**
Scaffold (venv, deps `textarena openai python-dotenv pyyaml fastapi uvicorn`, `.gitignore` for
`.env`+`store/`, `config.py`, `agents.yaml`). OpenRouter agent returning `{reasoning, action}`.
A **minimal but rule-correct ONUW core** — roles limited to **Werewolf, Seer, Robber, Troublemaker,
Villager** + center cards; one night (wake order, snapshots, original/current role), **one round-robin
discussion** with reasoning, **atomic vote**, **all-tied-die**, end-of-night win resolution. Runner
plays one game → records to SQLite → FastAPI serves it → `observer.html` fetches and replays it.
_Verify:_ Bash runs the game and writes SQLite; Bash starts the server + `curl`s the game JSON;
`claude-in-chrome` opens `observer.html`, steps the replay, and a **screenshot** shows real chat,
true roles, per-seat private reasoning (on click), and the correct winner. **This is the prototype.**

**P1 — Full ONUW conformance.** Add Minion, Mason, Insomniac, Drunk, Hunter, Tanner, Doppelgänger
(dynamic re-wake); multi-round discussion; no-kill vote; full win edge-cases.
_Verify:_ `pytest tests/test_onuw_conformance.py tests/test_invariants.py` green (vote atomicity,
no-leak filtering, all-tied-die, Hunter chain, Tanner). Consider a Workflow to adversarially
stress the core before declaring done.

**P2 — Runs + scoring.** `batch.py` plays N games with **role/seat balancing** and mirror (CRN)
pairing; `score.py` → role-balanced win-rate per-role + overall with **95% CIs**; server exposes the
runs index + run overview. _Verify:_ Bash runs a ≥20-game ONUW run; `claude-in-chrome` screenshot of
the run overview shows per-agent/per-role win-rate + CIs from real games.

**P3 — Avalon Game Core.** Quests, team proposal/vote, assassination, Merlin/Percival/Morgana.
_Verify:_ `pytest tests/test_avalon_conformance.py` green; screenshot of a real Avalon run replay + score.

**P4 — Drivable + docs.** `cli.py run --game onuw --games 20`; observer New-run screen calls the server;
README covers add-agent / add-game / start-run. Secret Mafia adapter included as a free third game.
_Verify:_ follow the README from scratch to launch a run and watch it; Secret Mafia game replays.

## Repo layout

```
.env  agents.yaml  .gitignore
arena/ config.py  openrouter.py  agentbase.py  runner.py  batch.py  store.py  score.py  server.py
arena/games/ base.py  onuw.py  avalon.py  secret_mafia.py
tests/ test_onuw_conformance.py  test_avalon_conformance.py  test_invariants.py
web/observer.html        # switch from inline mock data to fetching the local server
store/arena.db  DESIGN.md  GOAL.md
```

## Guardrails

- **Secrets:** key from `.env` only; never print/log/commit; `.gitignore` `.env` + `store/`.
- **Cost:** cheap roster above for all dev; caps on discussion turns + per-utterance tokens; `--games`
  bounds a run; log per-run token/call counts. Smoke runs should cost cents.
- **Determinism:** one seed per game, recorded; same seed reproduces the deal.
- **Failure handling:** model timeout/invalid → 1 retry → default/forfeit, flagged.

## Out of scope for v1 (designed-for, not built now)

TrueSkill + offline ToM/deception/persuasion metrics; Modal deployment; custom-code (non-LLM) agents
+ sandboxing; auth. The local FastAPI server is the seam Modal replaces later.
```
