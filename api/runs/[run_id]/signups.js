// POST /api/runs/:run_id/signups — join a run's lobby. Inserts the signup, then advances the lobby
// (seat assignment + roster + activation, all atomic). Returns the run's coordinator URL so the agent
// can re-point poll/reply at the per-run Modal container once it's active.
import {
  q, send, readBody, agentFromToken, signupResponse, advanceLobby, getRun,
  OPEN_RUN_STATUSES, ACTIVE_SIGNUP_IN, TERMINAL_SIGNUP_IN, PROTOCOL_VERSION, newId, utcnow, utcAfter,
  publicRef, validateUniquePublicNames,
} from '../../_db.js';
import { signupGateError } from '../../_shards.js';

// FINDINGS #2/#3: a Postgres unique-constraint violation surfaces as SQLSTATE 23505. The Neon
// serverless driver exposes it on err.code.
function isUniqueViolation(err) {
  return !!err && err.code === '23505';
}

// FINDING P2 (codex round-10): run_signups has TWO unique constraints — the explicit-seat partial
// index uq_run_signups_run_roster (run_id, roster_index) AND the table's UNIQUE(run_id, agent_id)
// (run_signups_run_id_agent_id_key). BOTH surface as 23505, so the catch MUST discriminate. ONLY the
// seat index is a seat error -> 400 invalid_seat; a (run_id, agent_id) collision is a racing/retried
// NORMAL signup by the same agent and must reload the existing signup, NOT mislabel it 'invalid_seat'.
// Mirrors store._is_seat_index_violation. Postgres exposes the violated constraint name on
// err.constraint; fall back to the message text containing 'roster'/'uq_run_signups_run_roster'.
function isSeatIndexViolation(err) {
  const name = String((err && err.constraint) || '').toLowerCase();
  if (name) return name === 'uq_run_signups_run_roster' || name.includes('roster');
  const msg = String((err && err.message) || err || '').toLowerCase();
  return msg.includes('uq_run_signups_run_roster') || msg.includes('roster_index');
}

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

  // store.create_signup run_kind/join_token gate (INV-4 / SPEC D7). A shard parent is never joinable
  // (presentational umbrella); a child is joinable ONLY with the matching per-parent join_token; a
  // normal run ignores the token (byte-identical to pre-shard behavior, INV-2). server.api_signup_run
  // maps run_not_joinable -> HTTP 403.
  const gateErr = signupGateError(run, body.join_token);
  if (gateErr) return send(res, 403, { error: gateErr });

  // Optional explicit seat (roster index): the orchestrator's deterministic-seat request (SPEC
  // D5/V-7), mirroring store.create_signup(seat=). Stored as roster_index and honored by advanceLobby
  // only when EVERY seated signup carries one; otherwise arrival-order (INV-2). Absent for normal
  // signups so the request/INSERT is byte-identical to before.
  const hasSeat = body.seat !== undefined && body.seat !== null;
  // FINDING #2/#3 (codex round-6): only an actual JS number may be a seat. A boolean (true/false) must
  // NOT coerce via Number() to 1/0, and a string ("x") / float (1.7) must be rejected — mirror
  // store.create_signup's "bool is not an int, must be integer" rule. Do NOT Number()-coerce a
  // non-number type.
  const rosterIndex = hasSeat ? body.seat : null;
  // FINDING #3: bounds-check an explicit seat against run.players BEFORE insert (mirrors
  // store.create_signup). advanceLobby only seats r.seat < players, so an out-of-range / non-integer
  // seat would leave this signup unseated forever and wedge the shard. No seat (normal runs) is
  // unaffected (INV-2).
  if (hasSeat &&
      (typeof rosterIndex !== 'number' || !Number.isInteger(rosterIndex) ||
       rosterIndex < 0 || rosterIndex >= Number(run.players))) {
    return send(res, 400, { error: 'invalid_seat' });
  }
  // FINDING #3 (codex round-6): an explicit seat must be UNIQUE among active signups in this run.
  // Without this, two agents could claim the same seat and advanceLobby would assign duplicate seats,
  // corrupting the deterministic identity->seat contract (SPEC D5/V-7). No seat is unaffected (INV-2).
  if (hasSeat) {
    const seatTaken = await q(
      `SELECT 1 FROM run_signups WHERE run_id=$1 AND agent_id<>$2 AND roster_index=$3
       AND status IN (${ACTIVE_SIGNUP_IN})`,
      [runId, agent.id, rosterIndex],
    );
    if (seatTaken.length) return send(res, 400, { error: 'invalid_seat' });
  }

  let s = (await q(
    `SELECT * FROM run_signups WHERE run_id=$1 AND agent_id=$2
     AND status NOT IN (${TERMINAL_SIGNUP_IN})`, [runId, agent.id]))[0];

  if (!s) {
    const activeRows = await q(
      `SELECT a.display_name FROM run_signups s JOIN agents a ON a.id = s.agent_id
       WHERE s.run_id=$1 AND s.status IN (${ACTIVE_SIGNUP_IN})`,
      [runId],
    );
    if (activeRows.length >= Number(run.players)) return send(res, 409, { error: 'run_full' });
    if (!OPEN_RUN_STATUSES.includes(run.status)) return send(res, 409, { error: 'run_not_open' });
    const identityErr = validateUniquePublicNames([...activeRows.map((row) => row.display_name), agent.display_name]);
    if (identityErr) return send(res, 409, { error: identityErr });
    const id = newId('signup_');
    const now = utcnow();
    const candidateHandle = publicRef(agent.display_name).slice(1);
    // FINDINGS #2/#3: the duplicate-seat pre-read above is a NON-ATOMIC fast path (TOCTOU) — two
    // concurrent signups can both pass it, then both INSERT the same roster_index. The partial unique
    // index uq_run_signups_run_roster (run_id, roster_index) WHERE roster_index IS NOT NULL makes the
    // INSERT the authoritative arbiter; catch its violation (PG SQLSTATE 23505) and 400 invalid_seat so
    // a racing duplicate seat is rejected atomically. No-seat (NULL) signups are excluded by the partial
    // index, so this never fires for normal runs (INV-2).
    let inserted;
    try {
      inserted = await q(
      `WITH run_lock AS (
         SELECT pg_advisory_xact_lock(hashtext($2), 0)
       ),
       active AS (
         SELECT a.display_name,
                btrim(lower(regexp_replace(a.display_name, '[^A-Za-z0-9]+', '-', 'g')), '-') AS public_handle
         FROM run_lock, run_signups s JOIN agents a ON a.id = s.agent_id
         WHERE s.run_id=$2 AND s.status IN (${ACTIVE_SIGNUP_IN})
       )
       INSERT INTO run_signups (id,run_id,agent_id,status,seat,created_utc,updated_utc,
         waiting_expires_utc,ready_deadline_utc,last_poll_utc,last_event_id,max_concurrent_turns,
         roster_index)
       SELECT $1,$2,$3,'waiting',NULL,$4,$4,$5,NULL,NULL,NULL,$6,$9
       FROM run_lock
       WHERE (SELECT COUNT(*) FROM active) < $7
         AND NOT EXISTS (SELECT 1 FROM active WHERE public_handle = $8)
       RETURNING *`,
      [id, runId, agent.id, now, utcAfter(600), Number(body.max_concurrent_turns || 1),
        Number(run.players), candidateHandle, rosterIndex],
      );
    } catch (err) {
      if (isUniqueViolation(err)) {
        // ONLY the explicit-seat index is a seat error (FINDINGS #2/#3). A (run_id, agent_id)
        // collision means a signup for this agent already exists (a racing/retried NORMAL signup, or
        // this agent's prior — possibly terminal — row); reload and return it (idempotent path,
        // FINDING P2 / round-9 Python parity), NOT 'invalid_seat'.
        if (isSeatIndexViolation(err)) return send(res, 400, { error: 'invalid_seat' });
        const raced = (await q(
          'SELECT * FROM run_signups WHERE run_id=$1 AND agent_id=$2', [runId, agent.id]))[0];
        if (raced) {
          await advanceLobby(runId);
          const reloaded = (await q('SELECT * FROM run_signups WHERE id=$1', [raced.id]))[0];
          const r2 = await getRun(runId);
          return send(res, 200, signupResponse(reloaded, r2.coordinator_url));
        }
      }
      throw err;
    }
    if (!inserted[0]) {
      const afterRows = await q(
        `SELECT a.display_name FROM run_signups s JOIN agents a ON a.id = s.agent_id
         WHERE s.run_id=$1 AND s.status IN (${ACTIVE_SIGNUP_IN})`,
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
