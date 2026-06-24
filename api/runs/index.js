// GET /api/runs  — the observer's run list. Returns a JSON ARRAY (not wrapped) of run summaries.
//   Port of arena/server.py api_runs (-> store.list_runs). Each run gets a teamSplit derived from
//   the distinct winning team per recorded game, and a deckPreset that defaults to the ONUW preset.
// POST /api/runs — queue a central job for a laptop worker, or (body.connected) create an open
//   connected run agents can sign up for. The observer never POSTs here; this mirrors
//   server.api_submit_run -> _queue_run / _create_connected_run for CLI/registry parity.
import { q, send, readBody, utcnow, newId } from '../_db.js';
import { DEFAULT_DECK_PRESET, normalizeDeckPreset } from '../_read.js';

const GAME_LABELS = {
  onuw: 'One Night Ultimate Werewolf',
  avalon: 'The Resistance: Avalon',
  secret_mafia: 'Secret Mafia',
};

// arena/games/*.py: (MIN_PLAYERS, MAX_PLAYERS, TITLE) per game core.
const GAME_CORES = {
  onuw: { min: 5, max: 7, title: 'One Night Ultimate Werewolf' },
  avalon: { min: 5, max: 7, title: 'The Resistance: Avalon' },
  secret_mafia: { min: 5, max: 7, title: 'Secret Mafia' },
};

// Port of arena/server.py _job_owner.
function jobOwner(payload) {
  const owner = String(payload.owner || payload.submitter || 'default').trim();
  return owner || 'default';
}

// Port of arena/server.py _roster_from_payload (without the SETTINGS.roster() fallback, which is
// server-config-only; an empty roster simply yields a 0-player table that fails the bounds check,
// matching the Python error path for connected/queued runs submitted without agents).
function rosterFromPayload(payload) {
  const agents = payload.agents || [];
  return agents
    .map((a) => ({
      name: String(a.name || '').trim(),
      model: String(a.model || '').trim(),
      harness: String(a.harness || 'base').trim() || 'base',
    }))
    .filter((a) => a.name && a.model);
}

// Port of arena/server.py _deck_preset_from_payload: null unless ONUW; normalize or 400 on unknown.
function deckPresetFromPayload(game, payload) {
  if (game !== 'onuw') return null;
  return normalizeDeckPreset(payload.deck_preset || payload.deckPreset);
}

// Port of arena/server.py _queue_run + store.enqueue_job: writes the visible queued run row and its
// job row (the worker claims the job and plays the run).
async function queueRun(payload, owner, res) {
  const game = payload.game || 'onuw';
  if (!(game in GAME_LABELS)) return send(res, 400, { error: `unknown game: ${game}` });
  const games = Math.max(1, parseInt(payload.games ?? payload.n_games ?? 6, 10) || 0);
  const rounds = Math.max(1, parseInt(payload.rounds ?? 5, 10) || 0);
  const seed = parseInt(payload.seed || Math.floor((Date.now()) % 1000000), 10);
  const runId = String(payload.run_id || `run_${seed}_${newId('').slice(0, 6)}`).trim();
  const agents = rosterFromPayload(payload);
  const core = GAME_CORES[game];
  const nPlayers = agents.length; // the roster IS the table — no fixed player count
  if (!(core.min <= nPlayers && nPlayers <= core.max)) {
    return send(res, 400, { error: `${core.title} supports ${core.min}–${core.max} players, got ${nPlayers}` });
  }
  let deckPreset;
  try {
    deckPreset = deckPresetFromPayload(game, payload);
  } catch (e) {
    return send(res, 400, { error: e.message });
  }

  const jobId = `job_${newId('').slice(0, 12)}`;
  const now = utcnow();
  const created = now.slice(0, 16).replace('T', ' ');
  const agentsJson = JSON.stringify(agents);

  // store.enqueue_job first upserts the visible run row (store.save_run, MONOTONIC on done/partial)...
  await q(
    `INSERT INTO runs (id,game,label,status,n_games,players,seed_base,created,agents_json,submitter,created_utc,deck_preset)
     VALUES ($1,$2,$3,'queued',$4,$5,$6,$7,$8,$9,$10,$11)
     ON CONFLICT (id) DO UPDATE SET
       game=excluded.game, label=excluded.label,
       status=CASE WHEN runs.status IN ('done','partial') THEN runs.status ELSE excluded.status END,
       n_games=excluded.n_games, players=excluded.players, seed_base=excluded.seed_base,
       created=excluded.created, agents_json=excluded.agents_json,
       submitter=excluded.submitter, created_utc=excluded.created_utc,
       deck_preset=excluded.deck_preset`,
    [runId, game, GAME_LABELS[game], games, nPlayers, seed, created, agentsJson, owner, now, deckPreset],
  );

  // ...then upserts the job row (ON CONFLICT (run_id) resets the lease so a re-queue re-runs).
  await q(
    `INSERT INTO jobs (id,run_id,owner,status,game,n_games,seed_base,rounds,players,
       agents_json,created_utc,updated_utc,lease_expires_utc,heartbeat_utc,worker_id,last_error,deck_preset)
     VALUES ($1,$2,$3,'queued',$4,$5,$6,$7,$8,$9,$10,$10,NULL,NULL,NULL,NULL,$11)
     ON CONFLICT (run_id) DO UPDATE SET
       owner=excluded.owner, status=excluded.status, game=excluded.game,
       n_games=excluded.n_games, seed_base=excluded.seed_base, rounds=excluded.rounds,
       players=excluded.players, agents_json=excluded.agents_json, updated_utc=excluded.updated_utc,
       deck_preset=excluded.deck_preset,
       lease_expires_utc=NULL, heartbeat_utc=NULL, worker_id=NULL, last_error=NULL`,
    [jobId, runId, owner, game, games, seed, rounds, nPlayers, agentsJson, now, deckPreset],
  );

  return send(res, 200, {
    run_id: runId, job_id: jobId, status: 'queued', owner,
    game, games, rounds, deck_preset: deckPreset,
  });
}

// Port of arena/server.py _create_connected_run + store.create_connected_run: a concrete open run
// that connected agents sign up for (no job row). The Modal coordinator spawn is a no-op here.
async function createConnectedRun(payload, res) {
  const game = payload.game || 'onuw';
  if (!(game in GAME_LABELS)) return send(res, 400, { error: `unknown game: ${game}` });
  const core = GAME_CORES[game];
  const players = parseInt(payload.players || core.min, 10);
  if (!(core.min <= players && players <= core.max)) {
    return send(res, 400, { error: `${core.title} supports ${core.min}–${core.max} players, got ${players}` });
  }
  const games = Math.max(1, parseInt(payload.games ?? payload.n_games ?? 6, 10) || 0);
  const seed = parseInt(payload.seed || Math.floor((Date.now()) % 1000000), 10);
  const runId = String(payload.run_id || `run_${seed}_${newId('').slice(0, 6)}`).trim();
  let deckPreset;
  try {
    deckPreset = deckPresetFromPayload(game, payload);
  } catch (e) {
    return send(res, 400, { error: e.message });
  }
  const submitter = payload.owner || payload.submitter || 'connected';

  const now = utcnow();
  const created = now.slice(0, 16).replace('T', ' ');
  // store.create_connected_run -> store.save_run with status 'open' and an empty roster.
  await q(
    `INSERT INTO runs (id,game,label,status,n_games,players,seed_base,created,agents_json,submitter,created_utc,deck_preset)
     VALUES ($1,$2,$3,'open',$4,$5,$6,$7,'[]',$8,$9,$10)
     ON CONFLICT (id) DO UPDATE SET
       game=excluded.game, label=excluded.label,
       status=CASE WHEN runs.status IN ('done','partial') THEN runs.status ELSE excluded.status END,
       n_games=excluded.n_games, players=excluded.players, seed_base=excluded.seed_base,
       created=excluded.created, agents_json=excluded.agents_json,
       submitter=excluded.submitter, created_utc=excluded.created_utc,
       deck_preset=excluded.deck_preset`,
    [runId, game, GAME_LABELS[game], games, players, seed, created, submitter, now, deckPreset],
  );

  return send(res, 200, {
    run_id: runId, status: 'open', game, games, players, deck_preset: deckPreset,
  });
}

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return send(res, 204, {});

  if (req.method === 'GET') {
    // store.list_runs: newest first; team_split from distinct winning team per recorded game.
    const runs = await q(
      'SELECT * FROM runs ORDER BY created_utc DESC NULLS LAST, created DESC',
    );
    const out = [];
    for (const r of runs) {
      const wins = await q(
        `SELECT team, COUNT(*)::int AS n FROM (
           SELECT DISTINCT run_id, gid, winner_team AS team FROM games WHERE run_id = $1
         ) sub GROUP BY team`,
        [r.id],
      );
      const split = { good: 0, evil: 0 };
      for (const w of wins) split[w.team] = w.n;
      out.push({
        id: r.id,
        game: r.game,
        label: r.label,
        status: r.status,
        nGames: r.n_games == null ? r.n_games : Number(r.n_games),
        players: r.players == null ? r.players : Number(r.players),
        seed: r.seed_base == null ? r.seed_base : Number(r.seed_base),
        when: r.created,
        teamSplit: split,
        deckPreset: r.deck_preset || (r.game === 'onuw' ? DEFAULT_DECK_PRESET : null),
      });
    }
    return send(res, 200, out);
  }

  if (req.method === 'POST') {
    const body = await readBody(req);
    if (body.connected) return createConnectedRun(body, res);
    return queueRun(body, jobOwner(body), res);
  }

  return send(res, 405, { error: 'method not allowed' });
}
