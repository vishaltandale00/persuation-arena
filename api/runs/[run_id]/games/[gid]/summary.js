// POST /api/runs/{run_id}/games/{gid}/summary
// Generates a concise observer-facing game summary and persists it inside transcript_json.summary.
import { q, send } from '../../../../_db.js';
import { summarizeGame } from '../../../../_game_summary.js';
import { parentGameSource, runKind, sourceRunIds } from '../../../../_shards.js';

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return send(res, 204, {});
  if (req.method !== 'POST') return send(res, 405, { error: 'method not allowed' });

  const runId = req.query.run_id;
  const gid = parseInt(req.query.gid, 10);
  const run = (await q('SELECT id, run_kind FROM runs WHERE id = $1', [runId]))[0];
  if (!run) return send(res, 404, { error: 'run not found' });

  let sourceIds = [runId];
  let mappedSource = null;
  if (runKind(run) === 'parent') {
    const childIds = (await q(
      'SELECT id FROM runs WHERE parent_run_id = $1 ORDER BY shard_index', [runId],
    )).map((c) => c.id);
    sourceIds = sourceRunIds(run, childIds);
    const childRuns = [];
    for (const childId of sourceIds) {
      const games = await q('SELECT gid FROM games WHERE run_id = $1 ORDER BY gid', [childId]);
      childRuns.push({ id: childId, games });
    }
    mappedSource = parentGameSource(childRuns, gid);
  }

  let row = null;
  let sourceRunId = runId;
  let sourceGid = gid;
  if (mappedSource) {
    sourceIds = [mappedSource.runId];
    sourceGid = mappedSource.gid;
  }
  for (const srcId of sourceIds) {
    const rows = await q(
      'SELECT transcript_json FROM games WHERE run_id = $1 AND gid = $2',
      [srcId, sourceGid],
    );
    if (rows[0]) {
      row = rows[0];
      sourceRunId = srcId;
      break;
    }
  }
  if (!row) return send(res, 404, { error: 'game not found' });

  const transcript = JSON.parse(row.transcript_json);
  if (transcript.summary && transcript.summary.text) {
    return send(res, 200, { ok: true, cached: true, summary: transcript.summary });
  }

  let summary;
  try {
    summary = await summarizeGame(transcript);
  } catch (e) {
    return send(res, 502, { error: 'summary generation failed', detail: String(e.message || e) });
  }
  transcript.summary = summary;
  await q(
    'UPDATE games SET transcript_json = $1 WHERE run_id = $2 AND gid = $3',
    [JSON.stringify(transcript), sourceRunId, sourceGid],
  );
  return send(res, 200, { ok: true, cached: false, summary });
}
