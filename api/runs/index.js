// GET /api/runs  — the observer's run list. Returns a JSON ARRAY (not wrapped) of run summaries.
//   Port of arena/server.py api_runs (-> store.list_runs). Each run gets a teamSplit derived from
//   the distinct winning team per recorded game, and a deckPreset that defaults to the ONUW preset.
// POST /api/runs — create an open connected-agent run (body.connected). Static (fixed-model) runs
//   are maintainer-local only and are rejected here. Mirrors server.api_submit_run.
import { q, send, readBody, utcnow, newId } from '../_db.js';
import { DEFAULT_DECK_PRESET, normalizeDeckPreset } from '../_read.js';
import { runKind, aggregateIndexRow } from '../_shards.js';

// store.child_run_ids: the shard child run rows for a parent, in shard order (empty for a non-parent).
// Returns the rows (id + status + winner-team split) the index parent branch needs to roll up.
async function childRunsForIndex(parentId) {
  const children = await q(
    'SELECT id, status FROM runs WHERE parent_run_id = $1 ORDER BY shard_index',
    [parentId],
  );
  for (const c of children) {
    // team_split from the distinct winning team per recorded game (same derivation as a normal row).
    const wins = await q(
      `SELECT team, COUNT(*)::int AS n FROM (
         SELECT DISTINCT run_id, gid, winner_team AS team FROM games WHERE run_id = $1
       ) sub GROUP BY team`,
      [c.id],
    );
    const split = { good: 0, evil: 0 };
    for (const w of wins) split[w.team] = w.n;
    c.team_split = split;
  }
  return children;
}

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

const REASONING_EFFORTS = new Set(['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max']);

function envInt(name, fallback) {
  const n = parseInt(process.env[name] ?? fallback, 10);
  return Number.isFinite(n) ? n : fallback;
}

function envFloat(name, fallback) {
  const n = parseFloat(process.env[name] ?? fallback);
  return Number.isFinite(n) ? n : fallback;
}

function envOptionalFloat(name) {
  const raw = process.env[name];
  if (raw === undefined || String(raw).trim() === '') return null;
  const n = parseFloat(raw);
  return Number.isFinite(n) ? n : null;
}

function payloadGet(payload, ...names) {
  for (const name of names) {
    if (payload[name] !== undefined && payload[name] !== null) return payload[name];
  }
  return undefined;
}

function positiveInt(value, name) {
  const n = parseInt(value, 10);
  if (!Number.isFinite(n) || n <= 0) throw new Error(`${name} must be positive`);
  return n;
}

function nonnegativeInt(value, name) {
  const n = parseInt(value, 10);
  if (!Number.isFinite(n) || n < 0) throw new Error(`${name} must be nonnegative`);
  return n;
}

function nonnegativeFloat(value, name) {
  const n = parseFloat(value);
  if (!Number.isFinite(n) || n < 0) throw new Error(`${name} must be nonnegative`);
  return n;
}

function priorMessageTurns(value) {
  if (value === undefined) return envInt('ARENA_PRIOR_MESSAGE_TURNS', -1);
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (normalized === '' || normalized === 'all' || normalized === 'infinite' || normalized === 'inf') {
      return -1;
    }
  }
  const n = parseInt(value, 10);
  if (!Number.isFinite(n) || n < -1) {
    throw new Error('prior_message_turns must be -1 for all history, or nonnegative');
  }
  return n;
}

function runConfigFromPayload(payload, rounds) {
  const effort = String(payloadGet(payload, 'reasoning_effort', 'reasoningEffort')
    ?? process.env.ARENA_REASONING_EFFORT ?? 'medium').trim().toLowerCase();
  if (!REASONING_EFFORTS.has(effort)) {
    throw new Error(`reasoning_effort must be one of ${Array.from(REASONING_EFFORTS).sort().join(', ')}`);
  }
  const maxTokens = payloadGet(payload, 'max_tokens_per_turn', 'maxTokensPerTurn', 'max_tokens', 'maxTokens');
  const retries = payloadGet(payload, 'retries');
  const temperature = payloadGet(payload, 'temperature');
  const priorTurns = payloadGet(payload, 'prior_message_turns', 'priorMessageTurns');
  return {
    discussion_rounds: rounds,
    reasoning_effort: effort,
    max_tokens_per_turn: maxTokens === undefined
      ? envInt('ARENA_MAX_TOKENS_PER_TURN', 4000)
      : positiveInt(maxTokens, 'max_tokens_per_turn'),
    temperature: temperature === undefined
      ? envOptionalFloat('ARENA_TEMPERATURE')
      : nonnegativeFloat(temperature, 'temperature'),
    retries: retries === undefined
      ? envInt('ARENA_RETRIES', 1)
      : nonnegativeInt(retries, 'retries'),
    prior_message_turns: priorMessageTurns(priorTurns),
  };
}

function runConfigOverrides(payload) {
  return {
    temperature: payloadGet(payload, 'temperature') !== undefined,
    prior_message_turns: payloadGet(payload, 'prior_message_turns', 'priorMessageTurns') !== undefined,
  };
}

// Port of arena/server.py _deck_preset_from_payload: null unless ONUW; normalize or 400 on unknown.
function deckPresetFromPayload(game, payload) {
  if (game !== 'onuw') return null;
  return normalizeDeckPreset(payload.deck_preset || payload.deckPreset);
}

// Port of arena/server.py _create_connected_run + store.create_connected_run: a concrete open run
// that connected agents sign up for (no job row). After the row is written we best-effort POST the
// Modal spawn endpoint (ARENA_SPAWN_URL / ARENA_SPAWN_TOKEN) to launch the per-run coordinator.
async function createConnectedRun(payload, res) {
  // Sharded creation is not wired on this Vercel surface yet — the read/signup paths render and gate
  // parent/child rows, but creating a sharded run (parent + K children + K Modal spawns under the
  // diagonal spawn model) is host-launcher-only for now (arena.sharded.create_sharded_run). Reject
  // shards>1 rather than silently create a single normal run (codex). [follow-up: wire JS creation]
  const shards = parseInt(payload.shards ?? payload.num_shards ?? 1, 10) || 1;
  if (shards > 1) {
    return send(res, 400, {
      error: 'sharded runs (shards>1) are not supported via the API yet; use the host-run launcher',
    });
  }
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
  let rounds;
  let runConfig;
  try {
    rounds = positiveInt(payload.rounds ?? 5, 'rounds'); // discussion rounds the coordinator honors
    runConfig = runConfigFromPayload(payload, rounds);
  } catch (e) {
    return send(res, 400, { error: e.message });
  }

  const now = utcnow();
  const created = now.slice(0, 16).replace('T', ' ');
  const metadataJson = JSON.stringify({ run_config: runConfig, run_config_overrides: runConfigOverrides(payload) });
  // store.create_connected_run -> store.save_run with status 'open' and an empty roster.
  await q(
    `INSERT INTO runs (id,game,label,status,n_games,players,seed_base,created,agents_json,submitter,created_utc,deck_preset,metadata_json)
     VALUES ($1,$2,$3,'open',$4,$5,$6,$7,'[]',$8,$9,$10,$11)
     ON CONFLICT (id) DO UPDATE SET
       game=excluded.game, label=excluded.label,
       status=CASE WHEN runs.status IN ('done','partial') THEN runs.status ELSE excluded.status END,
       n_games=excluded.n_games, players=excluded.players, seed_base=excluded.seed_base,
       created=excluded.created, agents_json=excluded.agents_json,
       submitter=excluded.submitter, created_utc=excluded.created_utc,
       deck_preset=excluded.deck_preset, metadata_json=excluded.metadata_json`,
    [runId, game, GAME_LABELS[game], games, players, seed, created, submitter, now, deckPreset, metadataJson],
  );

  // Fire the per-run coordinator. Best-effort: the row already exists as 'open', so on any failure the
  // run just waits coordinator-less (agents poll coordinator_url=null) instead of failing creation.
  const spawnUrl = process.env.ARENA_SPAWN_URL;
  const spawnTok = process.env.ARENA_SPAWN_TOKEN;
  if (spawnUrl && spawnTok) {
    try {
      const ac = new AbortController();
      const timer = setTimeout(() => ac.abort(), 5000);
      const r = await fetch(spawnUrl, {
        method: 'POST',
        signal: ac.signal,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          token: spawnTok,
          rounds,
          run_config: {
            id: runId, game, label: GAME_LABELS[game], status: 'open',
            n_games: games, players, seed_base: seed, submitter,
            deck_preset: deckPreset, metadata: JSON.parse(metadataJson),
          },
        }),
      });
      clearTimeout(timer);
      if (!r.ok) console.error(`[spawn] non-ok ${r.status} for ${runId}`);
    } catch (e) {
      console.error(`[spawn] failed for ${runId}:`, e);
    }
  }

  return send(res, 200, {
    run_id: runId, status: 'open', game, games, players, rounds, deck_preset: deckPreset,
    run_config: runConfig,
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
      // SPEC D7 / INV-4: child shards are invisible in the observer — the parent renders as one
      // normal run. (Parents and plain runs are listed; only run_kind='child' is hidden.)
      const kind = runKind(r);
      if (kind === 'child') continue;

      let status = r.status;
      let split;
      if (kind === 'parent') {
        // SPEC D7 / §6.9(a): the parent's OWN row never gets child progress written back, so roll
        // status up and aggregate the children's team-split (the parent's own status is stale 'open'
        // and its split is 0-0).
        const children = await childRunsForIndex(r.id);
        const agg = aggregateIndexRow(r, children);
        status = agg.status;
        split = agg.teamSplit;
      } else {
        // team_split from the distinct winning team per recorded game.
        const wins = await q(
          `SELECT team, COUNT(*)::int AS n FROM (
             SELECT DISTINCT run_id, gid, winner_team AS team FROM games WHERE run_id = $1
           ) sub GROUP BY team`,
          [r.id],
        );
        split = { good: 0, evil: 0 };
        for (const w of wins) split[w.team] = w.n;
      }

      out.push({
        id: r.id,
        game: r.game,
        label: r.label,
        status,
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
    if (!body.connected) {
      return send(res, 400, { error: 'static runs are maintainer-local only; use `arena serve` / `arena run`' });
    }
    return createConnectedRun(body, res);
  }

  return send(res, 405, { error: 'method not allowed' });
}
