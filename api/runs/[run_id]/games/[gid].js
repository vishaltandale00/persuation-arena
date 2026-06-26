// GET /api/runs/{run_id}/games/{gid}
// Port of arena/server.py api_game (-> store.get_game). Returns the stored game transcript
// object verbatim. `transcript_json` is a TEXT column holding JSON, so it must be JSON.parse'd
// before returning. 404 {error:"game not found"} when the row is missing.
import { q, send } from '../../../_db.js';
import { runKind, sourceRunIds } from '../../../_shards.js';

function gameInstanceId(runId, gid) {
  return `${runId}_game_${String(gid).padStart(3, '0')}`;
}

function enrichTranscriptTurnReasoning(transcript, events, runId, gid) {
  const enriched = JSON.parse(JSON.stringify(transcript));
  const iid = gameInstanceId(runId, gid);
  const queues = new Map();
  for (const event of events) {
    if (event.game_instance_id !== iid || event.type !== 'model_turn_completed') continue;
    let payload = {};
    try { payload = JSON.parse(event.payload_json || '{}'); } catch (_) { payload = {}; }
    if (payload.seat == null || !event.phase) continue;
    const key = `${String(event.phase).toLowerCase()}:${Number(payload.seat)}`;
    if (!queues.has(key)) queues.set(key, []);
    queues.get(key).push(payload);
  }

  function normalizedText(value) {
    return String(value || '').split(/\s+/).filter(Boolean).join(' ');
  }

  function actionMatchesEvent(payload, replayEvent) {
    const action = payload.action;
    if (replayEvent.t === 'say') {
      return action && typeof action === 'object'
        && normalizedText(action.speak) === normalizedText(replayEvent.text);
    }
    if (replayEvent.t === 'pass') {
      if (!action || typeof action !== 'object' || action.pass !== true) return false;
      if (replayEvent.stance) return String(action.stance || 'done') === String(replayEvent.stance);
      return true;
    }
    if (replayEvent.t === 'vote') {
      return Number(action) === Number(replayEvent.tgt);
    }
    if (replayEvent.t === 'act') {
      return payload.ok === true;
    }
    return false;
  }

  for (const phase of enriched.phases || []) {
    const phaseKey = String(phase.name || phase.kind || '').toLowerCase();
    for (const event of phase.events || []) {
      if (event.pid == null || !['act', 'say', 'pass', 'vote'].includes(event.t)) continue;
      const queue = queues.get(`${phaseKey}:${Number(event.pid)}`) || [];
      if (!queue.length) continue;
      const matchIdx = queue.findIndex((candidate) => actionMatchesEvent(candidate, event));
      if (matchIdx < 0) continue;
      const skipped = queue.slice(0, matchIdx);
      if (skipped.length) {
        event.hidden_model_turns_before = skipped.map((candidate) => {
          const out = {};
          for (const key of [
            'action',
            'reasoning',
            'provider_reasoning',
            'provider_reasoning_details',
            'raw',
            'action_kind',
            'model',
            'ms',
          ]) {
            if (candidate[key] != null) out[key] = candidate[key];
          }
          return out;
        });
      }
      const payload = queue[matchIdx];
      queue.splice(0, matchIdx + 1);
      if (payload.reasoning) event.declared_reasoning = payload.reasoning;
      if (payload.provider_reasoning != null) event.provider_reasoning = payload.provider_reasoning;
      if (payload.provider_reasoning_details != null) event.provider_reasoning_details = payload.provider_reasoning_details;
      if (payload.raw) event.raw_model_output = payload.raw;
      if (payload.action_kind) event.action_kind = payload.action_kind;
      if (payload.model) event.model = payload.model;
    }
  }
  return enriched;
}

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return send(res, 204, {});

  const runId = req.query.run_id;
  const gid = parseInt(req.query.gid, 10); // gid INTEGER in Neon

  // store.get_run: load the run row to know its kind. 404 if the run itself is missing.
  const run = (await q('SELECT id, run_kind FROM runs WHERE id = $1', [runId]))[0];
  if (!run) return send(res, 404, { error: 'run not found' });

  // SPEC D7: a sharded parent's game lives on whichever child holds that GLOBAL gid (gids are
  // disjoint across shards, D8); resolve it there. A normal/child run resolves against itself.
  let sourceIds = [runId];
  if (runKind(run) === 'parent') {
    const childIds = (await q(
      'SELECT id FROM runs WHERE parent_run_id = $1 ORDER BY shard_index', [runId],
    )).map((c) => c.id);
    sourceIds = sourceRunIds(run, childIds);
  }

  for (const srcId of sourceIds) {
    const rows = await q(
      'SELECT transcript_json FROM games WHERE run_id = $1 AND gid = $2',
      [srcId, gid],
    );
    const row = rows[0];
    if (!row) continue;
    const transcript = JSON.parse(row.transcript_json);
    const events = await q(
      'SELECT * FROM run_events WHERE run_id = $1 ORDER BY seq ASC LIMIT 1000',
      [srcId],
    );
    return send(res, 200, enrichTranscriptTurnReasoning(transcript, events, srcId, gid));
  }
  return send(res, 404, { error: 'game not found' });
}
