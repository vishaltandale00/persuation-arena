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
    try { payload = JSON.parse(event.payload_json || '{}'); } catch { payload = {}; }
    if (payload.seat == null || !event.phase) continue;
    const key = `${String(event.phase).toLowerCase()}:${Number(payload.seat)}`;
    if (!queues.has(key)) queues.set(key, []);
    queues.get(key).push(payload);
  }
  const callLogs = transcript.agentCallLog || transcript.callLog || {};
  const callQueues = new Map();
  if (callLogs && typeof callLogs === 'object') {
    for (const [seat, calls] of Object.entries(callLogs)) {
      if (!Array.isArray(calls)) continue;
      for (const call of calls) {
        if (!call || typeof call !== 'object' || !call.action_kind) continue;
        const key = `${Number(seat)}:${String(call.action_kind)}`;
        if (!callQueues.has(key)) callQueues.set(key, []);
        callQueues.get(key).push(call);
      }
    }
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
      return true;
    }
    return false;
  }

  function matchingCallLog(payload, seat) {
    if (!payload.action_kind) return {};
    const queue = callQueues.get(`${Number(seat)}:${String(payload.action_kind)}`) || [];
    if (!queue.length) return {};
    for (let idx = 0; idx < queue.length; idx += 1) {
      const candidate = queue[idx];
      const sameRaw = payload.raw != null && candidate.raw === payload.raw;
      const sameOk = payload.ok == null || Boolean(candidate.ok !== false) === Boolean(payload.ok);
      const sameMs = Number.isFinite(Number(candidate.ms)) && Number.isFinite(Number(payload.ms))
        && Math.abs(Number(candidate.ms) - Number(payload.ms)) < 0.01;
      if (sameOk && (sameMs || sameRaw)) return queue.splice(idx, 1)[0];
    }
    return queue.shift() || {};
  }

  function attachTurnTelemetry(event, payload, call = {}) {
    const ok = payload.ok ?? call.ok;
    if (ok != null) {
      event.model_call_ok = Boolean(ok);
      if (!ok) event.defaulted = true;
    }
    const validationError = payload.validation_error || call.validation_error;
    if (validationError) event.validation_error = validationError;
    const raw = payload.raw || call.raw;
    if (raw) event.raw_model_output = raw;
    if (payload.action_kind || call.action_kind) event.action_kind = payload.action_kind || call.action_kind;
    if (payload.model || call.model) event.model = payload.model || call.model;
    if (call.structured_output) event.structured_output = call.structured_output;
    if (call.finish_reason != null) event.finish_reason = call.finish_reason;
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
            'ok',
            'validation_error',
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
      const call = matchingCallLog(payload, Number(event.pid));
      if (payload.reasoning) event.declared_reasoning = payload.reasoning;
      if (payload.provider_reasoning != null) event.provider_reasoning = payload.provider_reasoning;
      if (payload.provider_reasoning_details != null) event.provider_reasoning_details = payload.provider_reasoning_details;
      attachTurnTelemetry(event, payload, call);
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
