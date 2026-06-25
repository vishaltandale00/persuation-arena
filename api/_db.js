// Shared helpers for the JS registry (the stateless lobby: register / discover / signup / seat
// assignment / ready / activation / status). All state lives in Neon; the Python per-run coordinator
// only reads it and plays the game. Race conditions (concurrent signup/ready) are handled with single
// atomic SQL statements, so no transactions/locking are needed across serverless invocations.
import { neon } from '@neondatabase/serverless';
import crypto from 'node:crypto';

const sql = neon(process.env.DATABASE_URL);

/** Parameterized query ($1,$2,...) -> array of row objects. */
export async function q(text, params = []) {
  return await sql.query(text, params);
}

export const PROTOCOL_VERSION = 'arena-agent-v1';
export const OPEN_RUN_STATUSES = ['open', 'waiting', 'ready_required'];

// Timestamps: match Python's _utcnow (ISO, microsecond precision, trailing Z). JS gives millis, so we
// pad to 6 digits — string-comparable and same shape as the Python writer/reader.
const micros = (d) => d.toISOString().replace(/(\.\d{3})Z$/, '$1000Z');
export const utcnow = () => micros(new Date());
export const utcAfter = (seconds) => micros(new Date(Date.now() + seconds * 1000));

export const sha256hex = (s) => crypto.createHash('sha256').update(s, 'utf8').digest('hex');
export const newId = (prefix) => prefix + crypto.randomBytes(8).toString('hex'); // ~uuid4().hex[:16]
export const issueToken = () => 'pa_live_' + crypto.randomBytes(24).toString('base64url');
export const NO_ONE_REF = '@no-one';
export const MAX_PUBLIC_NAME_LENGTH = 64;

export function normalizePublicName(value) {
  return String(value || '').trim().split(/\s+/).filter(Boolean).join(' ');
}

export function publicHandle(name) {
  return normalizePublicName(name)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

export function publicRef(name) {
  const handle = publicHandle(name);
  return handle ? `@${handle}` : '';
}

export function publicParticipant(name) {
  const clean = normalizePublicName(name);
  return { name: clean, ref: publicRef(clean) };
}

export function validatePublicName(value) {
  const name = normalizePublicName(value);
  if (!name) return { error: 'display_name required' };
  if (name.length > MAX_PUBLIC_NAME_LENGTH)
    return { error: `display_name must be ${MAX_PUBLIC_NAME_LENGTH} characters or fewer` };
  if (!publicHandle(name)) return { error: 'display_name must contain at least one ASCII letter or number' };
  if (publicRef(name).toLocaleLowerCase() === NO_ONE_REF)
    return { error: `${NO_ONE_REF} is reserved for abstention votes` };
  return { name };
}

export function validateUniquePublicNames(names) {
  const seenNames = new Map();
  const seenRefs = new Map();
  for (const raw of names) {
    const checked = validatePublicName(raw);
    if (checked.error) return checked.error;
    const name = checked.name;
    const nameKey = name.toLocaleLowerCase();
    const refKey = publicRef(name).toLocaleLowerCase();
    if (seenNames.has(nameKey)) return `duplicate public participant name: ${name}`;
    if (seenRefs.has(refKey)) return `ambiguous public participant names: ${seenRefs.get(refKey)} and ${name}`;
    seenNames.set(nameKey, name);
    seenRefs.set(refKey, name);
  }
  return null;
}

export function bearer(req) {
  const a = req.headers['authorization'] || req.headers['Authorization'] || '';
  const [scheme, token] = String(a).split(' ');
  return scheme && scheme.toLowerCase() === 'bearer' && token ? token.trim() : null;
}

export async function readBody(req) {
  if (req.body && typeof req.body === 'object') return req.body;
  if (typeof req.body === 'string' && req.body) { try { return JSON.parse(req.body); } catch { return {}; } }
  let data = '';
  for await (const chunk of req) data += chunk;
  try { return data ? JSON.parse(data) : {}; } catch { return {}; }
}

export function send(res, status, body) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'authorization,content-type');
  res.setHeader('content-type', 'application/json');
  res.statusCode = status;
  res.end(JSON.stringify(body));
}

export async function agentFromToken(req) {
  const token = bearer(req);
  if (!token) return null;
  const rows = await q('SELECT * FROM agents WHERE token_hash = $1', [sha256hex(token)]);
  return rows[0] || null;
}

const SIGNUP_MESSAGES = {
  waiting: 'Waiting for more agents.',
  ready_required: 'Seat assigned. Ready required before gameplay.',
  ready: 'Ready. Waiting for the run to start.',
  active: 'Run is active.',
  completed: 'Run completed.',
  expired: 'Signup expired.',
  cancelled: 'Run cancelled.',
  rejected: 'Signup rejected.',
};

export function signupResponse(s, coordinatorUrl) {
  const status = s.status;
  const pollAfter = status === 'waiting' ? 5000
    : (status === 'ready_required' || status === 'ready') ? 1000 : 250;
  return {
    signup_id: s.id, run_id: s.run_id, agent_id: s.agent_id, status,
    seat: s.seat ?? null, message: SIGNUP_MESSAGES[status] || status,
    poll_after_ms: pollAfter, heartbeat_after_ms: 15000,
    waiting_expires_at: s.waiting_expires_utc ?? null,
    ready_deadline_at: s.ready_deadline_utc ?? null,
    coordinator_url: coordinatorUrl ?? null,
  };
}

/**
 * Race-free lobby advance over Neon, run on every signup/ready/status hit:
 *  1. assign seats (0..players-1 by signup order) + waiting->ready_required, once the lobby is full,
 *  2. refresh the run's connected roster (agents_json),
 *  3. promote ready->active (and run->running) once every seated agent is ready.
 * Every step is a single atomic statement gated on a COUNT subquery, so concurrent invocations can't
 * double-assign or miss the promotion.
 */
export async function advanceLobby(runId) {
  const now = utcnow();
  const readyDeadline = utcAfter(60);

  // 1. seat assignment when full (atomic; idempotent — same order => same seats)
  await q(
    `WITH ranked AS (
       SELECT id, (ROW_NUMBER() OVER (ORDER BY created_utc, id) - 1) AS seat
       FROM run_signups
       WHERE run_id = $1 AND status IN ('waiting','ready_required','ready','active')
     )
     UPDATE run_signups s
     SET seat = r.seat,
         status = CASE WHEN s.status = 'waiting' THEN 'ready_required' ELSE s.status END,
         ready_deadline_utc = COALESCE(s.ready_deadline_utc, $2),
         updated_utc = $3
     FROM ranked r
     WHERE s.id = r.id
       AND r.seat < (SELECT players FROM runs WHERE id = $1)
       AND (SELECT COUNT(*) FROM run_signups WHERE run_id = $1
              AND status IN ('waiting','ready_required','ready','active'))
           >= (SELECT players FROM runs WHERE id = $1)`,
    [runId, readyDeadline, now]
  );

  // run open->waiting/ready_required to mirror the seating state
  await q(
    `UPDATE runs SET status = CASE
        WHEN (SELECT COUNT(*) FROM run_signups WHERE run_id = $1 AND seat IS NOT NULL
                AND status IN ('ready_required','ready','active')) >= players THEN 'ready_required'
        ELSE 'waiting' END
     WHERE id = $1 AND status IN ('open','waiting','ready_required')`,
    [runId]
  );

  // 2. refresh connected roster so the observer shows the seated agents
  await q(
    `UPDATE runs SET agents_json = COALESCE((
        SELECT json_agg(json_build_object(
                 'name', a.display_name, 'model', 'connected-agent', 'harness', 'connected',
                 'agent_id', s.agent_id, 'signup_id', s.id) ORDER BY s.seat)::text
        FROM run_signups s JOIN agents a ON a.id = s.agent_id
        WHERE s.run_id = $1 AND s.seat IS NOT NULL
          AND s.status IN ('ready_required','ready','active')
     ), agents_json)
     WHERE id = $1`,
    [runId]
  );

  // 3. activation: promote ready->active once enough seated agents are ready (atomic, race-free)
  const promoted = await q(
    `UPDATE run_signups SET status = 'active', updated_utc = $2
     WHERE run_id = $1 AND status IN ('ready','ready_required')
       AND (SELECT COUNT(*) FROM run_signups WHERE run_id = $1 AND status IN ('ready','active'))
           >= (SELECT players FROM runs WHERE id = $1)
     RETURNING id`,
    [runId, now]
  );
  if (promoted.length > 0) {
    await q(`UPDATE runs SET status = 'running' WHERE id = $1 AND status != 'done'`, [runId]);
  }
}

export async function getRun(runId) {
  return (await q('SELECT * FROM runs WHERE id = $1', [runId]))[0] || null;
}
