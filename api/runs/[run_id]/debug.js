// GET /api/runs/{run_id}/debug — observer debug snapshot.
// Port of arena/server.py:api_run_debug (~488), backed by store.list_run_signups (~860),
// store.list_run_turn_debug (~1070), store.list_run_events (~953).
// Response shape must match server.py:496-539 exactly (keys, casing, null-vs-absent, types).
import { q, getRun, utcnow, send } from '../../_db.js';
import { apiEvent } from '../../_read.js';

// Port of store._expire_signup_if_needed: a waiting/ready_required signup whose deadline has
// passed reads back as 'expired' (with updated_utc bumped). Read-only here — the Python
// coordinator owns the matching DB write; we only reproduce the returned-dict transform so the
// response shape matches. waiting_expires_utc / ready_deadline_utc are string-comparable ISO.
function expireSignupIfNeeded(s, now) {
  let status = s.status;
  if (status === 'waiting' && s.waiting_expires_utc && s.waiting_expires_utc < now) status = 'expired';
  if (status === 'ready_required' && s.ready_deadline_utc && s.ready_deadline_utc < now) status = 'expired';
  if (status !== s.status) return { ...s, status, updated_utc: now };
  return s;
}

// Port of store.list_run_signups: signups joined to their agent, ordered seated-first then by
// seat then signup time, each run through the expiry check.
async function listRunSignups(runId) {
  const rows = await q(
    `SELECT s.*, a.display_name, a.status agent_status, a.last_seen_utc
       FROM run_signups s JOIN agents a ON a.id = s.agent_id
       WHERE s.run_id = $1
       ORDER BY s.seat IS NULL, s.seat, s.created_utc`,
    [runId],
  );
  const now = utcnow();
  return rows.map((r) => expireSignupIfNeeded(r, now));
}

// Port of store.list_run_turn_debug: latest turns (LEFT JOIN their reply), newest first, capped at
// 100. action_json is a TEXT column holding JSON → parse to reply_action only when present
// (mirrors `if d.get("action_json")`). observation_json/legal_action_json are parsed by the Python
// helper but unused by this endpoint, so we skip them.
async function listRunTurnDebug(runId, maxTurns = 100) {
  return await q(
    `SELECT t.*, r.action_json, r.reasoning, r.client_ms, r.accepted,
            r.created_utc reply_created_utc
       FROM turns t LEFT JOIN turn_replies r ON r.turn_id = t.id
       WHERE t.run_id = $1 ORDER BY t.created_utc DESC LIMIT ${Number(maxTurns) | 0}`,
    [runId],
  );
}

// Port of store.list_run_events: newest `max_events` rows returned oldest-first, with payload_json
// (TEXT) parsed to `payload` and id renamed to `event_id` so apiEvent can shape them.
async function listRunEvents(runId, maxEvents = 200) {
  const rows = await q(
    `SELECT * FROM run_events WHERE run_id = $1 ORDER BY seq DESC LIMIT ${Number(maxEvents) | 0}`,
    [runId],
  );
  const out = [];
  for (const r of rows.reverse()) {
    const d = { ...r };
    d.payload = JSON.parse(d.payload_json);
    delete d.payload_json;
    d.event_id = d.id;
    delete d.id;
    out.push(d);
  }
  return out;
}

// Parse a turn-reply action (TEXT column) the way list_run_turn_debug does: only when truthy.
function replyAction(actionJson) {
  if (!actionJson) return null;
  return JSON.parse(actionJson);
}

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return send(res, 204, {});
  if (req.method !== 'GET') return send(res, 405, { detail: 'method not allowed' });

  const runId = req.query.run_id;
  if (!runId) return send(res, 400, { detail: 'run_id required' });

  const r = await getRun(runId);
  if (!r) return send(res, 404, { detail: 'run not found' });

  const signups = await listRunSignups(runId);
  const turns = await listRunTurnDebug(runId);
  const events = await listRunEvents(runId, 300);
  const recentEvents = events.map((e) => apiEvent(e));

  return send(res, 200, {
    run_id: runId,
    status: r.status,
    connected: signups.length > 0,
    signups: signups.map((s) => ({
      signup_id: s.id,
      agent_id: s.agent_id,
      display_name: s.display_name,
      agent_status: s.agent_status ?? null,
      status: s.status,
      seat: s.seat ?? null,
      created_utc: s.created_utc ?? null,
      updated_utc: s.updated_utc ?? null,
      waiting_expires_at: s.waiting_expires_utc ?? null,
      ready_deadline_at: s.ready_deadline_utc ?? null,
      last_poll_utc: s.last_poll_utc ?? null,
      last_event_id: s.last_event_id ?? null,
    })),
    pending_turns: turns
      .filter((t) => t.status === 'pending')
      .map((t) => ({
        turn_id: t.id,
        signup_id: t.signup_id,
        seat: t.seat,
        phase: t.phase,
        action_kind: t.action_kind,
        deadline_at: t.deadline_utc,
        claimed_at: t.claimed_utc ?? null,
        status: t.status,
      })),
    turns: turns.map((t) => ({
      turn_id: t.id,
      signup_id: t.signup_id,
      game_instance_id: t.game_instance_id,
      seat: t.seat,
      phase: t.phase,
      action_kind: t.action_kind,
      status: t.status,
      deadline_at: t.deadline_utc,
      accepted: t.accepted ?? null,
      reply_action: replyAction(t.action_json),
      reasoning: t.reasoning ?? null,
    })),
    recent_events: recentEvents,
    forfeits: recentEvents.filter((e) => e.type === 'forfeit'),
  });
}
