# Persuasion Arena — Design

A platform for evaluating the **social intelligence** of AI agents by having them play
text-based multiplayer social-deduction games (One Night Ultimate Werewolf, Avalon,
Secret Mafia, …) against each other, and ranking them by outcome.

Status: **design locked except where marked.** No production code yet.

---

## 1. Purpose & scope

- **What we measure:** social intelligence — deception, deduction, persuasion, theory of
  mind — via head-to-head play with a concrete win/loss outcome.
- **Games:** text-based, communication-driven, hidden-role/role-asymmetric. Day-one:
  **ONUW** and **Avalon** (authored), plus TextArena's **Secret Mafia** for free.
- **Agents:** internal/trusted, a mix of API models and custom-code scaffolds. Easy to add
  (via git). Plug-and-play across all games through one shared surface.
- **Scale:** run *many* games per matchup so the win-rate distribution is tight enough to
  rank agents. A match is the unit of parallelism.

## 2. Locked decisions

1. Build on **TextArena** as a game-logic library / reference — not as the agent contract.
2. **Bespoke Game Core per game** owns all game-specific rules; agents talk one generic surface.
3. **No liveness, no streaming, no wall-clock.** Logical turns, bounded by **caps**.
4. **Round-robin discussion**, speak-or-pass, randomized seat order, ends on all-pass or turn cap.
   No weighted queue / priority / directed-act machinery (rejected as harness-imposed bias).
5. **Incremental (delta) context per turn** + a small always-present **pinned header**; agents
   are stateful; **resync** available on cold start. Env holds full authoritative context.
6. **Per-seat information filtering = write-once observation-log replay.** Never leak unobserved state.
7. **Hybrid action format:** typed JSON for mechanical actions (votes, night actions); free text for talk.
8. **Scoring v1 = per-role + overall win-rate**, with confidence intervals and fresh deals per
   game. Full logs retained for later social-intelligence metrics.
9. **Deployment: self-host (docker-compose) first, then private Modal.** Orchestrator is deployment-agnostic.
10. Trust model: internal/trusted agents → process/resource isolation, no hard sandboxing.

## 3. Architecture

```
Scheduler ──enqueue matches──▶ Queue ──▶ Match Workers (run a Game Core + the round-robin loop)
                                              │  delta context  ▲  action (HTTP)
                                              ▼                 │
                                         Agent Services (one container per agent; model or custom code)
                                              │  full event log + private reasoning
                                              ▼
                                    Postgres (matches, events, observations, votes, ratings)
                                              │
                                   Scoring/Rating  ──▶  Leaderboard + Observer UI
```

- **Match Worker** — stateless; runs one game to completion; calls agent services per turn.
- **Agent Service** — implements the agent surface (§5); scaled independently.
- **Deployment-agnostic interfaces:** `MatchRunner` (local process → Modal function) and
  `AgentTransport` (local HTTP → Modal endpoint) are the only things that change between phases.

## 4. Game Core interface (bespoke per game)

```
reset(num_players, seed, roles)        -> initial authoritative state
current_actor()                        -> seat to act next (or simultaneous set)
legal_actions(seat)                    -> typed action schema for this turn
observe(seat)                          -> { delta_events, pinned_header }   # info-filtered
apply(seat, action)                    -> validate + transition (+ append observation events)
is_terminal() / outcome()              -> winners (the score signal)
caps                                   -> game/phase/utterance limits
```

Owns: phase machine, wake/turn order, info-filtering, win resolution, caps. The filtered view
for a seat is a **pure replay of that seat's append-only observation log** — the renderer never
dereferences current/end-of-night roles or center cards the seat did not legitimately inspect.

## 5. Agent surface (generic; one implementation plays every game)

```
game_start(role, rules)
act(delta_context, legal_actions) -> action     # discussion: {speak:text}|{pass:true}; else typed JSON
game_end(result)
resync() -> full filtered replay                 # for cold start / lost session
```

- Per-call **logical deadline**; timeout/invalid → 1 retry → default/forfeit.
- **Delta transport:** `act` receives only new events since the agent's last turn + the pinned
  header (private role facts, frozen night-observation log, deck multiset, win condition). The
  pinned header is **never compressed or dropped**; only the day-transcript body may be trimmed
  losslessly. Caps bound **outbound** message length only, never **inbound** re-supplied history.

## 6. Communication model

Discussion is **round-robin** over a per-game randomized seat order. On its turn an agent emits a
free-text utterance or **PASS**. Phase ends when a full round is all-passes or the turn cap is hit.
All social acts (address, rebut, accuse) live in **content**; the harness imposes only order + cap.
Any structure needed for *scoring* (who accused whom, influence) is derived **offline** from the
transcript — never imposed on the interaction. Accepted, symmetric tradeoff: no instant rebuttal;
an accused agent answers on its next turn.

> Mechanical phases are unaffected: ONUW night = role-gated sequential; voting = simultaneous-atomic.

## 7. Determinism & caps

Fixed RNG/deal per seed → exact replay. Caps: **game** (max phase cycles), **phase** (max
utterances), **utterance** (max length/tokens), **invalid-move** (1 retry → default). Cost per game
is bounded and known up front → we can budget "N games for $X."

## 8. Data model / logging (also drives the Observer UI)

- `matches(id, game, seed, started_at, outcome, win_team)`
- `match_players(match_id, seat, agent, dealt_role, end_role, team, won)`
- `events(match_id, seq, phase, kind, actor, target, visibility, payload)`  — utterances, actions, system, votes
- `observations(match_id, seat, wake_step, observer, target, value)`  — the write-once private log
- `reasoning(match_id, seat, phase, text)`  — private chain-of-thought (logged, never leaked in-game)
- `ratings(agent, game, role, games, wins, mu, sigma)`

Everything the Observer UI shows (true roles, actions, private reasoning, per-phase synthesis) is
a projection of these tables — so the mock doubles as a logging spec.

## 9. Evaluation & scoring

- **v1:** per-role + overall win-rate with 95% CIs. Each game gets a fresh deal while seats rotate
  across agents; **sequential stopping** when CIs separate.
- **ONUW default deck:** `arena` pressure preset: Minion + core information/swap roles + Drunk +
  Tanner at 5p, then Insomniac/Hunter at larger tables. `classic` and `tanner` remain selectable
  per run for lower-chaos or Tanner-focused experiments.
- Distinguishing close agents (~5-pt gap) needs ~1.5k games independently, far fewer paired — hence
  ONUW (seconds/game) is the statistical workhorse; Avalon is fewer, higher-signal games.
- **Later (offline, no harness change):** TrueSkill; theory-of-mind/deception/persuasion metrics
  derived from transcripts + ground-truth roles + the private reasoning channel.
- _Deeper ranking methodology (attribution in team games, Bradley-Terry/Plackett-Luce) — TBD._

## 10. Observability — the Observer UI

A spectator/broadcast view (mock: `web/observer.html`). Omniscient: shows the public chat, every
agent's **true role + action**, click-through to **private reasoning**, a **game-state board**, and a
per-phase **omniscient synthesis**. The observer sees truth; the agents see only their filtered views
— the gap is the whole drama.

## 11. Adding a game / adding an agent (git)

- **Add an agent (Tier 1, model):** PR a YAML manifest `{provider, model, system_prompt}` → runs on a
  shared model-agent image.
- **Add an agent (Tier 2, code):** PR a repo with a `Dockerfile` implementing the agent surface; an
  SDK/template handles the HTTP boilerplate (write only `act`). CI builds → registry → schedulable.
- **Add a game:** PR a Game Core class + metadata (player range, teams, win condition, social-skill
  tags) → engine image rebuilds → new `game` id is schedulable.

## 12. Deployment phases

- **Phase 1 (self-host):** one docker-compose — Postgres + Redis queue + N match workers + agent
  containers. Thousands of ONUW games on one box.
- **Phase 2 (Modal):** swap `MatchRunner`/`AgentTransport` to Modal backends. Same code, elastic.

## 13. ONUW reference Game Core — conformance checklist

From the fidelity audit (the bespoke vs general comparison):

- Night = event/priority queue with **wake-time snapshots**; **original vs current role** tracked.
- **Doppelgänger** = copy-then-perform with dynamic re-wakes (Mason@step4, Insomniac@end).
- **Directed recognition edges** (Minion→wolves, never back; Doppelgänger-WW→wolves, never back).
- **Center cards** are first-class indexed entities; targetable; observations stamped per slot.
- Per-action **typed result channel**: Robber/Seer/Insomniac/lone-wolf write a value; Drunk/Troublemaker
  write only a bare success token (no role value).
- **Vote = atomic simultaneous** (freeze pre-vote state, buffer hidden, reveal together). **Do not**
  inherit TextArena's sequential public voting.
- **All-tied-die** plurality (configurable threshold). **Do not** inherit TextArena's random tie-break.
- Win on **end-of-night roles**: Hunter chain into the same death set → Tanner override → village/wolf cases.

## 14. Roadmap

1. ONUW Game Core + conformance tests; round-robin loop; delta transport; SQLite store.
2. Two reference agents (a model agent + a scripted/no-API agent) on the shared surface.
3. Match runner → outcome → per-role win-rate + CIs + leaderboard.
4. Avalon Game Core. Tournament scheduler with CRN pairing.
5. Self-host docker-compose. Observer UI wired to real logs.
6. Modal backends. Offline social-intelligence metrics.

## 15. Open items

- Deeper evaluation/ranking methodology for team games (attribution, multiple comparisons).
- Exact caps per game (ONUW proposal: 1 cycle, ~2 discussion rounds, ~80-word utterances).
- Whether the private reasoning channel is mandatory (recommended yes) for offline ToM scoring.
