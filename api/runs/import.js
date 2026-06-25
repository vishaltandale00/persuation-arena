// POST /api/runs/import — upload a finished LOCAL run (from a laptop's SQLite store) to the prod
//   Neon-backed leaderboard. This is the ONLY prod write path for `arena push`: the FastAPI
//   /api/ingest is .vercelignore'd and DEAD in prod, so do NOT rely on it.
//
// Scope: hackathon DISASTER PROTECTION, not anti-cheat. Auth reuses the agent's registration bearer
//   token (the `pa_live_` token from POST /api/agents/register, matched by sha256 against
//   agents.token_hash). It FAILS CLOSED: a missing or unknown token is rejected (401/403) — no token,
//   no write. (Registration is open, so this is a trusted-contributor gate, not an anti-cheat one.)
//
// Payload (mirrors the legacy /api/ingest shape, batched):
//   { run: <run header>, games: [{ gid, transcript, agents: [{name,model,agent_id?,signup_id?}] }] }
// The server derives game_players FROM transcript.players joined to agents[] by seat, EXACTLY like
// store.save_game (keys: seat, dealt, end, team, won, calls, forfeits).
//
// The @neondatabase/serverless HTTP driver cannot do interactive RETURNING-then-branch transactions,
// so every write is UNCONDITIONALLY idempotent inside one non-interactive sql.transaction([...]):
//   1. runs upsert (monotonic status CASE, copied from api/runs/index.js),
//   2. games   INSERT ... ON CONFLICT (run_id,gid) DO NOTHING,
//   3. game_players INSERT ... ON CONFLICT (run_id,gid,seat) DO NOTHING.
//
// COMPUTE-ON-WRITE: after that transaction commits, IF any games were actually inserted
//   (games_inserted > 0) we call _rating.recompute() to rebuild the prod rating tables
//   (role_difficulty / rating_events / ratings) from the now-current game_players — server-side,
//   over the Vercel function's existing DATABASE_URL (no client creds, no shrink-guard: the server
//   only ever ADDS games). The recompute is a FULL replay, so it is correct after any insert.
//   The games are already committed and durable BEFORE we recompute, so a recompute failure is
//   non-fatal: we still return ok with a `recompute` note (a later push/retry recomputes). When
//   games_inserted === 0 the board is unchanged, so we skip recompute entirely.
import { send, readBody, bearer, utcnow, stmt, tx, agentFromToken } from '../_db.js';
import { recompute } from '../_rating.js';

// Terminal run statuses that the monotonic upsert must never regress — mirrors the
// `runs.status IN ('done','partial')` guard in api/runs/index.js. Kept local so this
// endpoint does not depend on the (separate) stopped/cancelled run-state work.
const TERMINAL_RUN_STATUSES = ['done', 'partial'];

const GAME_LABELS = {
  onuw: 'One Night Ultimate Werewolf',
  avalon: 'The Resistance: Avalon',
  secret_mafia: 'Secret Mafia',
};

// ---- pure helpers (no DB / no I/O) — exported so a standalone node assertion can test them ----

/** SQLite/Python truthy -> 0/1 (save_game writes `1 if p['won'] else 0`). */
function won01(v) {
  return v ? 1 : 0;
}
function int0(v) {
  const n = parseInt(v, 10);
  return Number.isFinite(n) ? n : 0;
}

/**
 * Build the parameter arrays for one game's writes from the import payload, DERIVING game_players
 * rows from transcript.players joined to agents[] by seat — exactly like store.save_game.
 * Returns { gameParams, playerParams: [...] } (no DB access). Throws on malformed input.
 */
export function gameToSqlParams(runId, game) {
  const gid = game.gid;
  if (gid == null) throw new Error('game missing gid');
  const t = game.transcript;
  if (!t || typeof t !== 'object') throw new Error(`game ${gid} missing transcript`);
  const agents = game.agents || [];

  const gameParams = [
    runId,
    gid,
    t.seed,
    t.winner_team,
    (t.outcome && t.outcome.text) ?? null,
    JSON.stringify(t),
  ];

  const playerParams = [];
  for (const p of t.players || []) {
    const agent = agents[p.seat];
    if (!agent) throw new Error(`game ${gid} seat ${p.seat} has no agent`);
    playerParams.push([
      runId,
      gid,
      p.seat,
      agent.name,
      agent.model,
      p.dealt,
      p.end,
      p.team,
      won01(p.won),
      int0(p.calls),
      int0(p.forfeits),
      agent.agent_id ?? null,
      agent.signup_id ?? null,
    ]);
  }
  return { gameParams, playerParams };
}

/** Build the runs-upsert parameter array from the run header (monotonic status CASE applied in SQL). */
export function runHeaderToSqlParams(run, owner, now) {
  const game = run.game || 'onuw';
  const label = run.label || GAME_LABELS[game] || game;
  const status = run.status || 'done';
  const nGames = run.n_games ?? run.nGames ?? null;
  const players = run.players ?? null;
  const seedBase = run.seed_base ?? run.seed ?? null;
  const created = run.created || now.slice(0, 16).replace('T', ' ');
  const agentsJson = JSON.stringify(run.agents || []);
  const submitter = run.submitter || owner;
  const createdUtc = run.created_utc || now;
  const deckPreset = run.deck_preset ?? run.deckPreset ?? null;
  return [
    run.id, game, label, status, nGames, players, seedBase, created,
    agentsJson, submitter, createdUtc, deckPreset, TERMINAL_RUN_STATUSES,
  ];
}

// SQL text reused below + by tests; the runs upsert is verbatim from api/runs/index.js (status
// parameterized instead of the hardcoded 'queued' literal there).
const RUNS_UPSERT = `INSERT INTO runs (id,game,label,status,n_games,players,seed_base,created,agents_json,submitter,created_utc,deck_preset)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
     ON CONFLICT (id) DO UPDATE SET
       game=excluded.game, label=excluded.label,
       status=CASE WHEN runs.status = ANY($13::text[]) THEN runs.status ELSE excluded.status END,
       n_games=excluded.n_games, players=excluded.players, seed_base=excluded.seed_base,
       created=excluded.created, agents_json=excluded.agents_json,
       submitter=excluded.submitter, created_utc=excluded.created_utc,
       deck_preset=excluded.deck_preset`;

// RETURNING gid lets us count ACTUAL first-writer inserts: a conflict (DO NOTHING) yields no row,
// so results[i].length tells inserted-vs-skipped exactly, independent of the driver's rowCount mode.
const GAMES_INSERT = `INSERT INTO games (run_id,gid,seed,winner_team,line,transcript_json)
     VALUES ($1,$2,$3,$4,$5,$6)
     ON CONFLICT (run_id,gid) DO NOTHING
     RETURNING gid`;

const PLAYERS_INSERT = `INSERT INTO game_players
       (run_id,gid,seat,agent,model,dealt_role,end_role,team,won,calls,forfeits,agent_id,signup_id)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
     ON CONFLICT (run_id,gid,seat) DO NOTHING`;

export { RUNS_UPSERT, GAMES_INSERT, PLAYERS_INSERT };

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return send(res, 204, {});
  if (req.method !== 'POST') return send(res, 405, { error: 'method not allowed' });

  // Auth: the caller must present a registered agent's bearer token (the `pa_live_` token from
  // POST /api/agents/register, matched by sha256 against agents.token_hash). Fail closed.
  const agent = await agentFromToken(req);
  if (!agent) {
    return send(res, bearer(req) ? 403 : 401, { error: 'unknown or missing agent token; register an identity first' });
  }

  const body = await readBody(req);
  const run = body.run;
  const games = body.games || [];
  if (!run || !run.id) return send(res, 400, { error: 'missing run header' });
  const runId = String(run.id).trim();
  if (!runId) return send(res, 400, { error: 'empty run id' });

  const now = utcnow();

  // Build the full non-interactive transaction: runs upsert, then one games + N players inserts
  // per game. Idempotency is unconditional (ON CONFLICT DO NOTHING + monotonic status CASE), so a
  // re-push is a server-side no-op and never duplicates or clobbers rows.
  const submitter = agent.display_name || agent.id;
  const queries = [stmt(RUNS_UPSERT, runHeaderToSqlParams(run, submitter, now))];
  const gameResultIdx = []; // transaction-result index of each game's RETURNING-gid insert
  let playersWritten = 0;
  try {
    for (const game of games) {
      const { gameParams, playerParams } = gameToSqlParams(runId, game);
      gameResultIdx.push(queries.length);
      queries.push(stmt(GAMES_INSERT, gameParams));
      for (const pp of playerParams) queries.push(stmt(PLAYERS_INSERT, pp));
      playersWritten += playerParams.length;
    }
  } catch (e) {
    return send(res, 400, { error: String(e.message || e) });
  }

  let results;
  try {
    results = await tx(queries);
  } catch (e) {
    return send(res, 500, { error: 'import failed', detail: String(e.message || e) });
  }

  // Each games insert RETURNs gid only when it actually inserted (DO NOTHING -> no row). With the
  // default (non-fullResults) driver mode, each transaction result IS the rows array, so a length of
  // 1 means inserted, 0 means skipped (already on the board). This makes a re-push a visible no-op.
  let gamesInserted = 0;
  for (const idx of gameResultIdx) {
    const r = results[idx];
    const rows = Array.isArray(r) ? r : (r && Array.isArray(r.rows) ? r.rows : []);
    if (rows.length > 0) gamesInserted += 1;
  }

  // COMPUTE-ON-WRITE: the games above are now committed and durable. Only rebuild the rating tables
  // when new games actually landed (games_inserted > 0); a re-push that inserted nothing leaves the
  // board unchanged, so skip the (full-replay) recompute. recompute() reads the now-current
  // game_players over this function's own DATABASE_URL and atomically swaps in the snapshot. On any
  // recompute error the inserted games stay durable, so we return ok with a `recompute_failed` note
  // (HTTP 200) rather than 500 — a later push or retry will recompute.
  let recomputeNote = gamesInserted > 0 ? 'pending' : 'skipped';
  let recomputeSummary = null;
  if (gamesInserted > 0) {
    try {
      recomputeSummary = await recompute();
      recomputeNote = 'ok';
    } catch (e) {
      recomputeNote = 'recompute_failed';
      recomputeSummary = { error: String(e.message || e) };
    }
  }

  return send(res, 200, {
    ok: true,
    run_id: runId,
    games_inserted: gamesInserted,
    games_skipped: games.length - gamesInserted,
    players_written: playersWritten,
    recompute: recomputeNote,
    rating: recomputeSummary,
  });
}
