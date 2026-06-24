// GET /api/runs/{run_id}/games/{gid}
// Port of arena/server.py api_game (-> store.get_game). Returns the stored game transcript
// object verbatim. `transcript_json` is a TEXT column holding JSON, so it must be JSON.parse'd
// before returning. 404 {error:"game not found"} when the row is missing.
import { q, send } from '../../../_db.js';

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return send(res, 204, {});

  const runId = req.query.run_id;
  const gid = parseInt(req.query.gid, 10); // gid INTEGER in Neon

  const rows = await q(
    'SELECT transcript_json FROM games WHERE run_id = $1 AND gid = $2',
    [runId, gid],
  );
  const row = rows[0];
  if (!row) return send(res, 404, { error: 'game not found' });

  return send(res, 200, JSON.parse(row.transcript_json));
}
