// GET /api/runs/open[?game=onuw] — open runs an agent can still join (not full, still in a lobby state).
import { q, send, OPEN_RUN_STATUSES, utcAfter } from '../_db.js';

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return send(res, 204, {});
  const game = (req.query && req.query.game) || null;
  const runs = game
    ? await q('SELECT * FROM runs WHERE game = $1 ORDER BY created_utc', [game])
    : await q('SELECT * FROM runs ORDER BY created_utc');
  const out = [];
  for (const r of runs) {
    if (!OPEN_RUN_STATUSES.includes(r.status)) continue;
    const signed = (await q(
      `SELECT COUNT(*)::int AS n FROM run_signups WHERE run_id = $1
       AND status IN ('waiting','ready_required','ready','active')`, [r.id]))[0].n;
    if (signed >= Number(r.players)) continue;
    out.push({
      run_id: r.id, game: r.game, status: r.status,
      players_needed: Number(r.players), players_signed_up: signed,
      games: Number(r.n_games), waiting_expires_at: utcAfter(600),
    });
  }
  return send(res, 200, { runs: out });
}
