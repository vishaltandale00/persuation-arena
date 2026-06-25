// JS port of arena/rating.py — the objective-handicap Elo recompute for the Vercel/Neon prod board.
//
// computeRatings({gamePlayers, runMeta, agents}) is a PURE function (no DB) that mirrors
// arena.rating.recompute() EXACTLY: same constants, formulas, identity keying, canonical replay
// order, and floating-point operation order, so a JS recompute reproduces the Python snapshot
// byte-for-byte (see tests/fixtures/rating_golden.json + tests/rating_parity.test.mjs).
//
// recompute(db) is the impure wrapper: it reads the three inputs from Neon (matching store's
// SELECTs), calls computeRatings, and atomically REPLACES role_difficulty / rating_events / ratings
// in one non-interactive transaction (mirror of store.replace_ratings).
import { q, stmt, tx } from './_db.js';

// --- tunables (mirror arena/rating.py lines 28-37) ------------------------------------------------
export const K_BASE = 0.08;          // base skill step, in logits
export const RD0 = 0.6;              // initial rating deviation (logits)
export const RD_MIN = 0.1;          // rating-deviation floor
export const K_RD_CAP = 4.0;         // cap on the provisional step multiplier (rd/RD_MIN)
export const PROVISIONAL_GAMES = 30; // below this many games a competitor is provisional
export const BETA_ALPHA = 2.0;       // Beta prior for role-difficulty base rates
export const MIN_ROLE_TRIALS = 30;   // a (bucket,role) d_r cell needs this many trials to be trusted
export const ELO_BASE = 1500.0;
export const ELO_SCALE = 173.0;      // logit -> Elo scale
const _EPS = 1e-9;

// --- numerics (numerically-stable branched forms, exact mirror of Python) ------------------------
function _sigmoid(x) {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1.0 / (1.0 + z);
  }
  const z = Math.exp(x);
  return z / (1.0 + z);
}

function _logit(p) {
  p = Math.min(1.0 - _EPS, Math.max(_EPS, p));
  return Math.log(p / (1.0 - p));
}

export function objectiveGroup(dealtRole, team) {
  if (dealtRole === 'Tanner') return 'tanner';
  return team === 'evil' ? 'werewolf' : 'village';
}

function _resists(myGroup, otherGroup) {
  if (myGroup === 'village') return otherGroup === 'werewolf';
  if (myGroup === 'werewolf') return otherGroup === 'village';
  return true; // Tanner is resisted by the whole table
}

function _bucket(game, players, deck) {
  return `${game}|${players}|${deck != null ? deck : '-'}`;
}

function _rdFor(games) {
  return Math.max(RD_MIN, RD0 / Math.sqrt(1 + games));
}

// is_no_contest (arena/games/base.py): no evil seat AND no winner. Mirror of _read.js isNoContest.
export function isNoContest(seats) {
  const noEvil = !seats.some((s) => s.team === 'evil');
  const noWinner = !seats.some((s) => (Number(s.won) || 0) > 0);
  return noEvil && noWinner;
}

// --- role difficulty d_r (compute_difficulty) ----------------------------------------------------
function computeDifficulty(rows, runMeta, stamp) {
  const agg = new Map(); // bucket -> Map(role -> [won_sum, n])

  function bump(bucket, role, won) {
    let roles = agg.get(bucket);
    if (!roles) { roles = new Map(); agg.set(bucket, roles); }
    let cell = roles.get(role);
    if (!cell) { cell = [0, 0]; roles.set(role, cell); }
    cell[0] += won;
    cell[1] += 1;
  }

  // group into games so no-contest games drop out of the base rates too.
  const byGame = new Map();
  for (const r of rows) {
    const key = `${r.run_id}\x00${r.gid}`;
    let g = byGame.get(key);
    if (!g) { g = []; byGame.set(key, g); }
    g.push(r);
  }
  for (const seats of byGame.values()) {
    if (isNoContest(seats)) continue;
    for (const r of seats) {
      const rm = runMeta[r.run_id] || {};
      const role = r.dealt_role || '?';
      const won = Number(r.won) || 0;
      bump(_bucket(rm.game, rm.players, rm.deck_preset), role, won);
      bump(`${rm.game}`, role, won);
      bump('*', role, won);
    }
  }

  const diff = new Map(); // bucket -> Map(role -> {n, d_r})
  const storage = [];
  for (const [bucket, roles] of agg.entries()) {
    let dRoles = diff.get(bucket);
    if (!dRoles) { dRoles = new Map(); diff.set(bucket, dRoles); }
    for (const [role, cell] of roles.entries()) {
      const w = cell[0];
      const n = cell[1];
      const base = (w + BETA_ALPHA) / (n + 2 * BETA_ALPHA);
      const dR = -_logit(base);
      dRoles.set(role, { n, d_r: dR });
      storage.push({ bucket, role, w, n, base_rate: base, d_r: dR, updated_utc: stamp });
    }
  }
  storage.sort((a, b) => (a.bucket < b.bucket ? -1 : a.bucket > b.bucket ? 1
    : a.role < b.role ? -1 : a.role > b.role ? 1 : 0));

  function lookup(game, players, deck, role) {
    for (const b of [_bucket(game, players, deck), `${game}`, '*']) {
      const roles = diff.get(b);
      const cell = roles && roles.get(role);
      if (cell && cell.n >= MIN_ROLE_TRIALS) return cell.d_r;
    }
    const glob = diff.get('*');
    const cell = glob && glob.get(role);
    return cell ? cell.d_r : 0.0;
  }

  return { storage, lookup };
}

// --- identity keying (_identity) -----------------------------------------------------------------
// roster_harness: Map keyed `${run_id}\x00${agent_name}` -> harness.
function _identity(row, rosterHarness) {
  const agentId = row.agent_id;
  if (agentId) {
    return { ident: agentId, agentId, disp: row.agent || agentId, dmodel: null, dharness: null };
  }
  const model = row.model || '?';
  if (model === 'connected-agent') {
    const name = row.agent || `seat-${row.seat != null ? row.seat : '?'}`;
    return {
      ident: `legacy-connected:${row.run_id}:${name}`, agentId: null, disp: name,
      dmodel: model, dharness: 'connected-legacy',
    };
  }
  const harnessKey = `${row.run_id}\x00${row.agent}`;
  const harness = rosterHarness.has(harnessKey) ? rosterHarness.get(harnessKey) : 'base';
  return { ident: `static:${model}:${harness}`, agentId: null, disp: model, dmodel: model, dharness: harness };
}

/**
 * PURE recompute. Mirrors arena.rating.recompute() (lines 149-260).
 * @param gamePlayers list of game_players rows (run_id,gid,seat,agent,agent_id,model,dealt_role,team,won,calls,forfeits)
 * @param runMeta     object run_id -> {game,players,deck_preset,created_utc,agents:[...]}
 * @param agents      list of agents-table rows (id,display_name,declared_model,declared_harness,...)
 * @returns {ratings, ratingEvents, roleDifficulty, summary}
 */
export function computeRatings({ gamePlayers, runMeta, agents }) {
  const rows = gamePlayers || [];
  const meta = runMeta || {};
  const agentsById = new Map();
  for (const a of agents || []) agentsById.set(a.id, a);

  // static roster harness lookup: (run_id, agent_name) -> harness
  const rosterHarness = new Map();
  for (const rid of Object.keys(meta)) {
    const rm = meta[rid] || {};
    for (const a of rm.agents || []) {
      if (a && a.name) {
        rosterHarness.set(`${rid}\x00${a.name}`, a.harness || 'base');
      }
    }
  }

  // data-derived stamp = max(created_utc) across runs (lexicographic == chronological for ISO 'Z').
  const runIds = Object.keys(meta);
  let stamp = '';
  if (runIds.length) {
    for (const rid of runIds) {
      const c = (meta[rid] || {}).created_utc || '';
      if (c > stamp) stamp = c;
    }
  }

  const { storage: difficultyRows, lookup: drLookup } = computeDifficulty(rows, meta, stamp);

  // group rows into games
  const games = new Map(); // `${run_id}\x00${gid}` -> {rid, gid, rows:[]}
  for (const r of rows) {
    const key = `${r.run_id}\x00${r.gid}`;
    let g = games.get(key);
    if (!g) { g = { rid: r.run_id, gid: r.gid, rows: [] }; games.set(key, g); }
    g.rows.push(r);
  }

  // canonical order: (run.created_utc, run_id, gid). Strings compared lexicographically, gid numeric.
  const orderedKeys = [...games.values()].sort((a, b) => {
    const ca = (meta[a.rid] || {}).created_utc || '';
    const cb = (meta[b.rid] || {}).created_utc || '';
    if (ca < cb) return -1;
    if (ca > cb) return 1;
    if (a.rid < b.rid) return -1;
    if (a.rid > b.rid) return 1;
    return a.gid - b.gid;
  });

  const skills = new Map();
  const gcount = new Map();
  const wins = new Map();
  const callsSum = new Map();
  const forfeitsSum = new Map();
  const competitorMeta = new Map(); // ident -> {agent_id, display_name, declared_model, declared_harness}
  const events = [];
  let ratedGames = 0;

  const getNum = (m, k) => (m.has(k) ? m.get(k) : 0);

  for (const game of orderedKeys) {
    const seats = game.rows.slice().sort((a, b) => a.seat - b.seat);
    if (isNoContest(seats)) continue;
    ratedGames += 1;
    const rid = game.rid;
    const gid = game.gid;
    const rm = meta[rid] || {};

    // PASS A: resolve identities + groups once (pre-game snapshot)
    const info = [];
    for (const r of seats) {
      const { ident, agentId, disp, dmodel, dharness } = _identity(r, rosterHarness);
      const group = objectiveGroup(r.dealt_role, r.team);
      info.push({ row: r, ident, group });
      if (!competitorMeta.has(ident)) {
        competitorMeta.set(ident, {
          agent_id: agentId, display_name: disp, declared_model: dmodel, declared_harness: dharness,
        });
      }
      if (agentId && agentsById.has(agentId)) {
        const a = agentsById.get(agentId);
        const m = competitorMeta.get(ident);
        m.display_name = a.display_name || disp;
        m.declared_model = a.declared_model != null ? a.declared_model : null;
        m.declared_harness = a.declared_harness != null ? a.declared_harness : null;
      }
    }

    // PASS B: compute all deltas from PRE-game skills only
    const deltas = [];
    for (let idx = 0; idx < info.length; idx++) {
      const this_ = info[idx];
      const r = this_.row;
      const ident = this_.ident;
      const group = this_.group;
      const sI = getNum(skills, ident);
      const rd = _rdFor(getNum(gcount, ident));
      const resisting = info.filter((o, j) => j !== idx && _resists(group, o.group));
      let rho = 0.0;
      if (resisting.length) {
        let sum = 0.0;
        for (const o of resisting) sum += getNum(skills, o.ident);
        rho = sum / resisting.length;
      }
      const dR = drLookup(rm.game, rm.players, rm.deck_preset, r.dealt_role || '?');
      const p = _sigmoid(sI - dR - rho);
      const calls = Number(r.calls) || 0;
      const forf = Number(r.forfeits) || 0;
      const damp = Math.max(0.0, 1.0 - (calls ? forf / calls : 0.0));
      const kEff = K_BASE * Math.min(K_RD_CAP, rd / RD_MIN) * damp;
      const won = Number(r.won) || 0;
      const delta = kEff * (won - p);
      deltas.push(delta);
      events.push({
        id: `re_${rid}_${gid}_${r.seat}`, identity_key: ident, run_id: rid, gid, seat: r.seat,
        dealt_role: r.dealt_role != null ? r.dealt_role : null, objective_group: group, won,
        pre_skill: sI, post_skill: sI + delta, delta, expected: p, d_r: dR, resistance: rho,
        k: kEff, damped: damp, created_utc: stamp,
      });
    }

    // PASS C: apply all deltas AFTER
    for (let idx = 0; idx < info.length; idx++) {
      const this_ = info[idx];
      const r = this_.row;
      const ident = this_.ident;
      skills.set(ident, getNum(skills, ident) + deltas[idx]);
      gcount.set(ident, getNum(gcount, ident) + 1);
      wins.set(ident, getNum(wins, ident) + (Number(r.won) || 0));
      callsSum.set(ident, getNum(callsSum, ident) + (Number(r.calls) || 0));
      forfeitsSum.set(ident, getNum(forfeitsSum, ident) + (Number(r.forfeits) || 0));
    }
  }

  // BUILD ratings, ordered by sorted(skills.keys())
  const ratings = [];
  const identKeys = [...skills.keys()].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  for (const ident of identKeys) {
    const g = getNum(gcount, ident);
    const rd = _rdFor(g);
    const s = skills.get(ident);
    const c = getNum(callsSum, ident);
    const f = getNum(forfeitsSum, ident);
    const m = competitorMeta.get(ident) || {};
    ratings.push({
      identity_key: ident,
      agent_id: m.agent_id != null ? m.agent_id : null,
      display_name: m.display_name != null ? m.display_name : null,
      declared_model: m.declared_model != null ? m.declared_model : null,
      declared_harness: m.declared_harness != null ? m.declared_harness : null,
      skill: s,
      rd,
      elo: ELO_BASE + ELO_SCALE * s,
      games: g,
      wins: getNum(wins, ident),
      forfeit_rate: c ? round4(f / c) : 0.0,
      provisional: g < PROVISIONAL_GAMES ? 1 : 0,
      updated_utc: stamp,
    });
  }

  // events sort before write: (created_utc, run_id, gid, seat); created_utc is one constant stamp.
  events.sort((a, b) => {
    if (a.created_utc < b.created_utc) return -1;
    if (a.created_utc > b.created_utc) return 1;
    if (a.run_id < b.run_id) return -1;
    if (a.run_id > b.run_id) return 1;
    if (a.gid !== b.gid) return a.gid - b.gid;
    return a.seat - b.seat;
  });

  return {
    ratings,
    ratingEvents: events,
    roleDifficulty: difficultyRows,
    summary: {
      games: ratedGames, competitors: ratings.length,
      events: events.length, difficulty_cells: difficultyRows.length,
    },
  };
}

// Python round(x, 4) is banker's rounding; for forfeit_rate ratios in [0,1] plain 4-decimal rounding
// reproduces it. (Matches the fixture's exact forfeit_rate values.)
function round4(x) {
  return Math.round(x * 1e4) / 1e4;
}

// --- impure wrapper: read Neon, compute, atomically replace the three tables ----------------------

/** Read inputs matching store.all_game_player_rows / run_meta_map / list_agents. */
export async function readRatingInputs() {
  const gamePlayers = await q(
    'SELECT run_id, gid, seat, agent, agent_id, signup_id, model, dealt_role, end_role, ' +
    'team, won, calls, forfeits FROM game_players',
  );
  const runRows = await q(
    'SELECT id, game, players, deck_preset, created_utc, agents_json FROM runs',
  );
  const runMeta = {};
  for (const r of runRows) {
    let agentsList = [];
    try { agentsList = r.agents_json ? JSON.parse(r.agents_json) : []; }
    catch { agentsList = []; }
    runMeta[r.id] = {
      game: r.game, players: r.players, deck_preset: r.deck_preset,
      created_utc: r.created_utc, agents: agentsList,
    };
  }
  const agents = await q(
    'SELECT * FROM agents ORDER BY last_seen_utc DESC NULLS LAST, created_utc DESC',
  );
  return { gamePlayers, runMeta, agents };
}

/**
 * Recompute ratings from Neon and atomically swap in the snapshot — mirror of
 * arena.rating.recompute() + store.replace_ratings(). Destructive: DELETEs all three rating tables
 * then re-inserts, in ONE non-interactive transaction.
 */
export async function recompute() {
  const inputs = await readRatingInputs();
  const { ratings, ratingEvents, roleDifficulty, summary } = computeRatings(inputs);

  const queries = [
    stmt('DELETE FROM role_difficulty'),
    stmt('DELETE FROM rating_events'),
    stmt('DELETE FROM ratings'),
  ];
  for (const d of roleDifficulty) {
    queries.push(stmt(
      'INSERT INTO role_difficulty (bucket,role,w,n,base_rate,d_r,updated_utc) ' +
      'VALUES ($1,$2,$3,$4,$5,$6,$7)',
      [d.bucket, d.role, d.w, d.n, d.base_rate, d.d_r, d.updated_utc],
    ));
  }
  for (const e of ratingEvents) {
    queries.push(stmt(
      'INSERT INTO rating_events (id,identity_key,run_id,gid,seat,dealt_role,objective_group,' +
      'won,pre_skill,post_skill,delta,expected,d_r,resistance,k,damped,created_utc) ' +
      'VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17)',
      [e.id, e.identity_key, e.run_id, e.gid, e.seat, e.dealt_role, e.objective_group, e.won,
        e.pre_skill, e.post_skill, e.delta, e.expected, e.d_r, e.resistance, e.k, e.damped,
        e.created_utc],
    ));
  }
  for (const r of ratings) {
    queries.push(stmt(
      'INSERT INTO ratings (identity_key,agent_id,display_name,declared_model,declared_harness,' +
      'skill,rd,elo,games,wins,forfeit_rate,provisional,updated_utc) ' +
      'VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)',
      [r.identity_key, r.agent_id, r.display_name, r.declared_model, r.declared_harness, r.skill,
        r.rd, r.elo, r.games, r.wins, r.forfeit_rate, r.provisional, r.updated_utc],
    ));
  }

  await tx(queries);
  return summary;
}
