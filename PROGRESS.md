# Build progress (loop state)

## ✅ GOAL.md COMPLETE — all phases P0–P4 done and verified (2026-06-23).

Definition of Done (all green): 24 pytest pass; all 3 games (onuw, avalon, secret_mafia) registered,
run, and render in the observer (screenshot-verified, no console errors); ≥20-game ONUW run shows
per-agent win-rate + 95% CIs; Avalon + Secret Mafia replay + score; New-run launch works
(POST /api/run); README covers add-agent/add-game/start-run; no secret leaked (.env gitignored).
Server: `python -m arena.cli serve` → http://localhost:8000/observer.html.

Executing `GOAL.md` via `/loop until GOAL.md is complete`. This file tracks where the build is
so any iteration (even after a context reset) can resume.

## Done
- **P0 — ONUW harness walking skeleton ✅ (verified by screenshot).**
  - venv (`.venv`, python3.14), deps installed (openai, dotenv, pyyaml, fastapi, uvicorn, playwright).
  - `.env` has `OPENROUTER_API_KEY` (verified working with a cheap call).
  - `agents.yaml` cheap roster (qwen/glm/deepseek/kimi/gpt-5-mini/haiku).
  - `arena/`: config, openrouter agent ({reasoning,action} JSON), games/base, games/onuw
    (minimal rule-correct: WW/Seer/Robber/Troublemaker/Villager + center; night wake order +
    snapshots + original/current role; round-robin discussion; atomic vote; all-tied-die;
    end-of-night win), store (SQLite), runner, server (FastAPI).
  - `web/observer.html` now fetches from the server (`/api/runs`, `/api/runs/{id}`,
    `/api/runs/{id}/games/{gid}`) instead of inline mock data.
  - One real game stored: `run_8841` game 1. Headless verify: `tools/verify_observer.py` (Playwright)
    — green, no console errors, real chat + reasoning rendered.

## How to run
- Server: `. .venv/bin/activate && uvicorn arena.server:app --port 8000`
- Play a game: `python -c "from arena.runner import play_run; play_run('onuw', n_games=1, seed_base=<seed>)"`
- Verify UI: `python tools/verify_observer.py` (needs server up)
- Observer: http://localhost:8000/observer.html

- **P1 — full ONUW conformance + tests ✅ (19 pytest green).**
  - Full role set in `arena/games/onuw.py`: Werewolf, Minion, Mason, Seer, Robber, Troublemaker,
    Drunk, Insomniac, Hunter, Tanner, Villager, Doppelgänger (copy + perform). Canonical wake order.
  - `deal_override` for deterministic tests; no-kill vote (NO_KILL=-1); Hunter chain + Tanner override
    in pure `arena/games/base.compute_winners`; `tally_votes` all-tied-die + no-kill plurality.
  - Tests: `tests/scripted.py` (ScriptedDefault, PolicyAgent — no API), `tests/test_invariants.py`,
    `tests/test_onuw_conformance.py`. Run: `python -m pytest tests/ -q`.
  - Default live deck now includes a Minion (evil) — observer renders it fine.

- **P2 — batch + role-balanced scoring ✅ (screenshot-verified).** 20-game run `run_9000` renders
  in the observer with per-agent overall win-rate + 95% CI bars and per-faction breakdown. Leaderboard:
  Vishal 60% [38-76], Neeraj 55%, Chappy/Saurav 45%, Leo 25% (n=20 each). Tooling:
  `tools/verify_overview.py <run_id>`.

- **P3 — Avalon core: CODE + TESTS DONE (24 pytest green); awaiting 6-game batch to verify.**
  - `arena/games/avalon.py`: 5p (Merlin, Percival, Loyal Servant, Morgana, Assassin); knowledge
    (Merlin sees evil, evil see each other, Percival sees Merlin+Morgana); 5 quests (sizes 2,3,2,3,3),
    leader proposal + team approve/reject vote + quest cards (good auto-success, evil may fail),
    re-proposal cap 3; 3 successes → Assassination (Assassin names Merlin); 3 fails → evil win.
  - Registered in `batch.GAME_CORES`. `tests/test_avalon_conformance.py` green. team_of extended to
    Avalon evil roles in `base.py`.
  - **In flight:** `run_5000` (6 avalon games) running in bg (pid 20481, log /tmp/arena_avalon.log);
    watcher armed → wakes loop on completion. Verify: screenshot Avalon replay + score in observer.
- **P4 — nearly done; awaiting Secret Mafia real run for final screenshot.**
  - `arena/cli.py` ✅ (run/score/runs/agents/serve).
  - New-run wiring ✅: server `POST /api/run` launches a run in a bg thread; observer "Launch run"
    button POSTs game/games/agents. Verified end-to-end (POST → run_200367 appears "running").
    `MODEL_OPTS` in the observer set to real OpenRouter slugs.
  - Secret Mafia ✅ (`arena/games/secret_mafia.py`, registered in batch). Scripted smoke passes; a real
    2-game run (run_200367) launched via the New-run endpoint and is finishing in bg (watcher armed,
    server log /tmp/arena_server.log). Observer has a generic state panel for non-onuw/avalon games;
    deadSet generalized for multi-round vote phases.
  - README.md ✅ (add-agent / add-game / start-run / scoring / tests / limitations).
  - **REMAINING:** when run_200367 completes, screenshot Secret Mafia replay; then final DoD pass +
    STOP the loop (GOAL.md complete). All 24 pytest green; no secret printed/logged/committed.

## (historical) plans below

## (historical) P3 plan
`arena/games/avalon.py` (5p: Merlin, Percival, 1 Loyal Servant, Morgana, Assassin). Phases: 5 quests
(team proposal -> approve/reject vote -> if approved, quest success/fail cards), 3 quest wins for good
triggers Assassination (Assassin names Merlin; correct = evil win). Reuse the round-robin discussion +
{reasoning,action} agent contract + transcript shape. Register in batch.GAME_CORES and runner. Observer
already has stateAvalon rendering (quest track, proposal, team vote, assassination) — feed it the same
phase/event shapes. Tests: `tests/test_avalon_conformance.py` (team approve/reject, quest pass/fail,
assassination win logic). Then a small real Avalon batch (background) + screenshot.

## (historical) P2 detail
- **P2 — batch + scoring: CODE DONE, awaiting 20-game batch to verify.**
  - `arena/batch.py`: concurrent games via ThreadPoolExecutor (workers=8); seats rotate per game;
    workers only play+return, main thread writes SQLite (no concurrent-write locks). discussion_rounds=1
    for bulk speed.
  - `arena/score.py`: Wilson 95% CI; `score_run` → per-agent overall + per-faction (good/evil) win-rate.
  - `arena/store.player_rows`; server `/api/runs/{id}` now returns `scores`; observer run overview
    renders overall win-rate + 95% CI bars and a per-faction breakdown.
  - **In flight:** `run_9000` (20 games, seed_base 9000) running in background (pid 75668, log
    /tmp/arena_batch.log). Completion watcher armed (bg task) → wakes the loop when done.
  - **TО VERIFY when batch done:** restart server, screenshot the run overview showing win-rate+CIs.
  - Early signal: werewolves winning most games (cheap models coordinate poorly as village) — realistic.

## (superseded) P2 plan
`arena/batch.py`: play N games with role/seat balancing (seats already rotate per game) + mirror/CRN.
`arena/score.py`: role-balanced win-rate per-role + overall with 95% CIs (Wilson interval). Expose in
the run overview (server already returns per-agent wins; add per-role + CI fields). 20-game run → show
win-rate + CIs in the observer.

**SPEED — must fix before the 20-game run.** A real game is ~7 min (10 sequential discussion calls +
slow models). 20×7 = too long. Fix: run GAMES CONCURRENTLY in batch.py via ThreadPoolExecutor
(games are independent; OpenAI client is thread-safe) — wall time ≈ games/workers × per-game. Also
parallelize the simultaneous vote (5 concurrent calls) and consider discussion_rounds=1 for bulk.
Target ~6-8 concurrent games.

## Known issues / notes
- **Speed:** ~7 min/real game (cheap models, sequential calls). Before P2's 20-game runs, parallelize
  the simultaneous vote (5 concurrent calls) and consider async; discussion stays sequential.
- **gpt-5-mini** occasionally returns empty/fails JSON (reasoning models burn the token budget on
  hidden reasoning). It defaults gracefully. Consider raising max_tokens for that model or detecting it.
- Seats rotate per game (seat i = roster[(i+rot)%n]); transcript players carry name+model so the
  observer shows the right agent per seat.
- Tests must NOT call the API — use a scripted/deterministic agent fixture.
