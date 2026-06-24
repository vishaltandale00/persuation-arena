// POST /api/runs/:run_id/signups — join a run's lobby. Inserts the signup, then advances the lobby
// (seat assignment + roster + activation, all atomic). Returns the run's coordinator URL so the agent
// can re-point poll/reply at the per-run Modal container once it's active.
import {
  q, send, readBody, agentFromToken, signupResponse, advanceLobby, getRun,
  OPEN_RUN_STATUSES, PROTOCOL_VERSION, newId, utcnow, utcAfter,
  publicRef, validateUniquePublicNames,
} from '../../_db.js';

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return send(res, 204, {});
  if (req.method !== 'POST') return send(res, 405, { error: 'method not allowed' });
  const runId = req.query.run_id;
  const agent = await agentFromToken(req);
  if (!agent) return send(res, 403, { error: 'invalid bearer token' });
  const body = await readBody(req);
  if ((body.protocol_version || PROTOCOL_VERSION) !== PROTOCOL_VERSION)
    return send(res, 400, { error: 'unsupported protocol_version' });

  const run = await getRun(runId);
  if (!run) return send(res, 404, { error: 'run not found' });

  let s = (await q(
    `SELECT * FROM run_signups WHERE run_id=$1 AND agent_id=$2
     AND status NOT IN ('completed','rejected','expired','cancelled')`, [runId, agent.id]))[0];

  if (!s) {
    const activeRows = await q(
      `SELECT a.display_name FROM run_signups s JOIN agents a ON a.id = s.agent_id
       WHERE s.run_id=$1 AND s.status IN ('waiting','ready_required','ready','active')`,
      [runId],
    );
    if (activeRows.length >= Number(run.players)) return send(res, 409, { error: 'run_full' });
    if (!OPEN_RUN_STATUSES.includes(run.status)) return send(res, 409, { error: 'run_not_open' });
    const identityErr = validateUniquePublicNames([...activeRows.map((row) => row.display_name), agent.display_name]);
    if (identityErr) return send(res, 409, { error: identityErr });
    const id = newId('signup_');
    const now = utcnow();
    const candidateHandle = publicRef(agent.display_name).slice(1);
    const inserted = await q(
      `WITH run_lock AS (
         SELECT pg_advisory_xact_lock(hashtext($2), 0)
       ),
       active AS (
         SELECT a.display_name,
                btrim(lower(regexp_replace(a.display_name, '[^A-Za-z0-9]+', '-', 'g')), '-') AS public_handle
         FROM run_lock, run_signups s JOIN agents a ON a.id = s.agent_id
         WHERE s.run_id=$2 AND s.status IN ('waiting','ready_required','ready','active')
       )
       INSERT INTO run_signups (id,run_id,agent_id,status,seat,created_utc,updated_utc,
         waiting_expires_utc,ready_deadline_utc,last_poll_utc,last_event_id,max_concurrent_turns)
       SELECT $1,$2,$3,'waiting',NULL,$4,$4,$5,NULL,NULL,NULL,$6
       FROM run_lock
       WHERE (SELECT COUNT(*) FROM active) < $7
         AND NOT EXISTS (SELECT 1 FROM active WHERE public_handle = $8)
       RETURNING *`,
      [id, runId, agent.id, now, utcAfter(600), Number(body.max_concurrent_turns || 1),
        Number(run.players), candidateHandle],
    );
    if (!inserted[0]) {
      const afterRows = await q(
        `SELECT a.display_name FROM run_signups s JOIN agents a ON a.id = s.agent_id
         WHERE s.run_id=$1 AND s.status IN ('waiting','ready_required','ready','active')`,
        [runId],
      );
      if (afterRows.length >= Number(run.players)) return send(res, 409, { error: 'run_full' });
      const afterIdentityErr = validateUniquePublicNames([...afterRows.map((row) => row.display_name), agent.display_name]);
      return send(res, 409, { error: afterIdentityErr || 'signup_conflict' });
    }
    await q('UPDATE agents SET last_seen_utc=$1 WHERE id=$2', [now, agent.id]);
    s = (await q('SELECT * FROM run_signups WHERE id=$1', [id]))[0];
  }

  await advanceLobby(runId);
  s = (await q('SELECT * FROM run_signups WHERE id=$1', [s.id]))[0];
  const r2 = await getRun(runId);
  return send(res, 200, signupResponse(s, r2.coordinator_url));
}
