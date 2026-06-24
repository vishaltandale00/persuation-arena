// POST /api/run — the observer's "Start run" button.
//
// Port of arena/server.py api_launch -> _queue_run -> store.enqueue_job. Because DATABASE_URL is set
// on the hosted deploy, api_launch always takes the QUEUE path: it inserts a job row into Neon's
// `jobs` table (status="queued") plus the visible `runs` row, and a Modal worker claims the job later.
// We only enqueue here — no game is played in this handler.
import { q, utcnow, readBody, send } from './_db.js';
import { normalizeDeckPreset, DEFAULT_DECK_PRESET } from './_read.js';
import crypto from 'node:crypto';

// arena/server.py GAME_LABELS — the set of known games and their titles.
const GAME_LABELS = {
  onuw: 'One Night Ultimate Werewolf',
  avalon: 'The Resistance: Avalon',
  secret_mafia: 'Secret Mafia',
};

// arena/games/*.py: every game core declares MIN_PLAYERS, MAX_PLAYERS = 5, 7 and a TITLE that
// matches GAME_LABELS. Mirrors GAME_CORES[game].{MIN_PLAYERS,MAX_PLAYERS,TITLE} used by _queue_run.
const GAME_CORES = {
  onuw: { MIN_PLAYERS: 5, MAX_PLAYERS: 7, TITLE: GAME_LABELS.onuw },
  avalon: { MIN_PLAYERS: 5, MAX_PLAYERS: 7, TITLE: GAME_LABELS.avalon },
  secret_mafia: { MIN_PLAYERS: 5, MAX_PLAYERS: 7, TITLE: GAME_LABELS.secret_mafia },
};

// Port of server._job_owner: (owner | submitter | "default").strip() || "default".
function jobOwner(payload) {
  const owner = String(payload.owner || payload.submitter || 'default').trim();
  return owner || 'default';
}

// Port of server._roster_from_payload: one agent per seat, keeping only entries that have BOTH a
// non-empty name and model (stripped); harness defaults to "base". The roster IS the table, so its
// length is the player count. There is no agents.yaml fallback on the hosted serverless runtime —
// the observer always supplies agents, and an empty roster fails the player-count check below.
function rosterFromPayload(payload) {
  const agents = payload.agents || [];
  return agents
    .map((a) => ({
      name: String((a && a.name) || '').trim(),
      model: String((a && a.model) || '').trim(),
      harness: String((a && a.harness) || 'base').trim() || 'base',
    }))
    .filter((a) => a.name && a.model);
}

// Port of server._deck_preset_from_payload: null unless onuw, else the normalized preset.
// normalizeDeckPreset throws a plain Error on an unknown preset; we catch it and re-throw as
// HttpError(400), mirroring Python's `except ValueError: raise HTTPException(400, ...)`.
function deckPresetFromPayload(game, payload) {
  if (game !== 'onuw') return null;
  try {
    return normalizeDeckPreset(payload.deck_preset || payload.deckPreset);
  } catch (e) {
    throw new HttpError(400, String((e && e.message) || e));
  }
}

// Port of Python int(payload.get(key, fallbacks...)) with max(1, .) where applicable.
function intOr(value, fallback) {
  const n = parseInt(value, 10);
  return Number.isNaN(n) ? fallback : n;
}

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return send(res, 204, {});
  if (req.method !== 'POST') return send(res, 405, { error: 'method not allowed' });

  try {
    const payload = await readBody(req);
    const owner = jobOwner(payload);

    // --- _queue_run -----------------------------------------------------------------------------
    const game = payload.game || 'onuw';
    if (!(game in GAME_LABELS)) throw new HttpError(400, `unknown game: ${game}`);

    // games: int(payload.get("games", payload.get("n_games", 6))), floored at 1.
    const gamesRaw = payload.games != null ? payload.games
      : (payload.n_games != null ? payload.n_games : 6);
    const games = Math.max(1, intOr(gamesRaw, 6));
    const rounds = Math.max(1, intOr(payload.rounds != null ? payload.rounds : 5, 5));

    // seed: int(payload.get("seed") or (time.time()*1000) % 1000000). A falsy/absent seed (None, 0,
    // "") falls through to the time-derived value; Date.now() is integer millis so % matches int().
    const seedRaw = payload.seed;
    const seed = seedRaw ? intOr(seedRaw, Date.now() % 1000000) : Date.now() % 1000000;

    const runId = String(payload.run_id || `run_${seed}_${crypto.randomBytes(3).toString('hex')}`).trim();

    const agents = rosterFromPayload(payload);
    const core = GAME_CORES[game];
    const nPlayers = agents.length; // the roster IS the table — no fixed player count
    if (!(core.MIN_PLAYERS <= nPlayers && nPlayers <= core.MAX_PLAYERS)) {
      throw new HttpError(400,
        `${core.TITLE} supports ${core.MIN_PLAYERS}–${core.MAX_PLAYERS} players, got ${nPlayers}`);
    }

    const deckPreset = deckPresetFromPayload(game, payload); // throws HttpError(400) on unknown preset

    // --- store.enqueue_job ----------------------------------------------------------------------
    const jobId = 'job_' + crypto.randomBytes(6).toString('hex'); // uuid4().hex[:12]
    const now = utcnow();
    const agentsJson = JSON.stringify(agents); // jobs.agents_json / runs.agents_json are TEXT columns
    const created = now.slice(0, 16).replace('T', ' ');

    // Visible run row (mirror store.save_run; status NOT regressed once done/partial).
    await q(
      `INSERT INTO runs (id,game,label,status,n_games,players,seed_base,created,agents_json,submitter,created_utc,deck_preset)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
       ON CONFLICT (id) DO UPDATE SET
         game=excluded.game, label=excluded.label,
         status=CASE WHEN runs.status IN ('done','partial') THEN runs.status ELSE excluded.status END,
         n_games=excluded.n_games, players=excluded.players, seed_base=excluded.seed_base,
         created=excluded.created, agents_json=excluded.agents_json,
         submitter=excluded.submitter, created_utc=excluded.created_utc,
         deck_preset=excluded.deck_preset`,
      [runId, game, GAME_LABELS[game], 'queued', games, nPlayers, seed, created,
        agentsJson, owner, now, deckPreset],
    );

    // Queued job row (mirror store.enqueue_job INSERT exactly: 17 columns, status="queued").
    await q(
      `INSERT INTO jobs (id,run_id,owner,status,game,n_games,seed_base,rounds,players,
         agents_json,created_utc,updated_utc,lease_expires_utc,heartbeat_utc,worker_id,last_error,deck_preset)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17)
       ON CONFLICT (run_id) DO UPDATE SET
         owner=excluded.owner, status=excluded.status, game=excluded.game,
         n_games=excluded.n_games, seed_base=excluded.seed_base, rounds=excluded.rounds,
         players=excluded.players, agents_json=excluded.agents_json, updated_utc=excluded.updated_utc,
         deck_preset=excluded.deck_preset,
         lease_expires_utc=NULL, heartbeat_utc=NULL, worker_id=NULL, last_error=NULL`,
      [jobId, runId, owner, 'queued', game, games, seed, rounds, nPlayers,
        agentsJson, now, now, null, null, null, null, deckPreset],
    );

    return send(res, 200, {
      run_id: runId, job_id: jobId, status: 'queued', owner,
      game, games, rounds, deck_preset: deckPreset,
    });
  } catch (e) {
    if (e instanceof HttpError) return send(res, e.status, { error: e.message });
    return send(res, 500, { error: String((e && e.message) || e) });
  }
}
