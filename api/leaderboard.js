// GET /api/leaderboard - read-only JS port of arena.rating.leaderboard().
// Rating math stays in Python; this endpoint only projects the precomputed Neon snapshot.
import { q, send } from './_db.js';
import { wilson } from './_read.js';

const ELO_SCALE = 173.0;

function roundHalfEven(x, digits) {
  if (!Number.isFinite(x)) return x;
  const factor = 10 ** digits;
  const y = x * factor;
  const sign = Math.sign(y) || 1;
  const abs = Math.abs(y);
  const floor = Math.floor(abs);
  const frac = abs - floor;
  let rounded;
  if (Math.abs(frac - 0.5) < 1e-9) {
    rounded = floor % 2 === 0 ? floor : floor + 1;
  } else {
    rounded = Math.round(abs);
  }
  return (sign * rounded) / factor;
}

const r3 = (x) => roundHalfEven(x, 3);
const r1 = (x) => roundHalfEven(x, 1);

function cell(w, n) {
  if (n === 0) return { w: 0, n: 0, rate: 0, lo: 0, hi: 0 };
  const { lo, hi, p } = wilson(w, n);
  return { w, n, rate: r3(p), lo: r3(lo), hi: r3(hi) };
}

function publicCompetitor(row) {
  const elo = Number(row.elo);
  const rd = Number(row.rd);
  return {
    identity_key: row.identity_key,
    agent_id: row.agent_id,
    display_name: row.display_name,
    declared_model: row.declared_model,
    declared_harness: row.declared_harness,
    elo: r1(elo),
    rd: r1(ELO_SCALE * rd),
    conservative: r1(elo - 2 * ELO_SCALE * rd),
    games: Number(row.games),
    wins: Number(row.wins),
    provisional: Boolean(Number(row.provisional)),
    forfeit_rate: Number(row.forfeit_rate),
  };
}

function aggregate(events, diffGlobal) {
  const overall = [0, 0];
  const byObjective = {};
  const byRole = {};

  for (const event of events) {
    const won = Number(event.won) || 0;
    overall[0] += won;
    overall[1] += 1;

    const objective = event.objective_group || 'village';
    const objectiveCell = byObjective[objective] ||= [0, 0];
    objectiveCell[0] += won;
    objectiveCell[1] += 1;

    const role = event.dealt_role || '?';
    const roleCell = byRole[role] ||= [0, 0];
    roleCell[0] += won;
    roleCell[1] += 1;
  }

  const overallOut = cell(overall[0], overall[1]);
  if (overallOut.n !== events.length) throw new Error('leaderboard invariant failed: event count mismatch');

  const byObjectiveOut = {};
  for (const key of Object.keys(byObjective)) {
    byObjectiveOut[key] = cell(byObjective[key][0], byObjective[key][1]);
  }
  for (const key of ['village', 'werewolf', 'tanner']) {
    byObjectiveOut[key] ||= cell(0, 0);
  }
  const forcedObjectiveTotal = ['village', 'werewolf', 'tanner']
    .reduce((total, key) => total + byObjectiveOut[key].n, 0);
  if (forcedObjectiveTotal !== overallOut.n) {
    throw new Error('leaderboard invariant failed: objective count mismatch');
  }

  const byRoleOut = {};
  for (const role of Object.keys(byRole).sort()) {
    const [w, n] = byRole[role];
    const out = cell(w, n);
    const diff = diffGlobal[role] || {};
    const baseRate = diff.base_rate == null ? null : Number(diff.base_rate);
    const dR = diff.d_r == null ? null : Number(diff.d_r);
    const dROut = dR == null ? null : r3(dR);
    out.base = baseRate == null ? null : r3(baseRate);
    out.d_r = dROut;
    out.vs_spread = baseRate == null ? null : r3(out.rate - baseRate);
    out.hard = dROut != null && dROut > 0;
    byRoleOut[role] = out;
  }

  return { overall: overallOut, by_objective: byObjectiveOut, by_role: byRoleOut };
}

export function assembleLeaderboard(ratings, events, difficulties) {
  const eventsByIdentity = {};
  for (const event of events) (eventsByIdentity[event.identity_key] ||= []).push(event);

  const diffGlobal = {};
  for (const row of difficulties) diffGlobal[row.role] = row;

  return ratings.map((row) => {
    const breakdown = aggregate(eventsByIdentity[row.identity_key] || [], diffGlobal);
    return { ...publicCompetitor(row), ...breakdown };
  });
}

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return send(res, 204, {});

  if (req.method === 'GET') {
    const ratings = await q('SELECT * FROM ratings ORDER BY (elo - 2*173.0*rd) DESC, games DESC');
    const events = await q('SELECT * FROM rating_events');
    const difficulties = await q("SELECT role, base_rate, d_r FROM role_difficulty WHERE bucket = '*'");

    const competitors = assembleLeaderboard(ratings, events, difficulties);

    return send(res, 200, { competitors });
  }

  return send(res, 405, { error: 'method not allowed' });
}
