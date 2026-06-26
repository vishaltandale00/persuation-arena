// POST /api/signups/:signup_id/ready — agent declares it's ready. Sets ready_required->ready, then
// advances the lobby (which promotes the run to active once every seated agent is ready — atomic, so
// concurrent ready calls can't miss the promotion).
import {
  q, send, readBody, agentFromToken, signupResponse, advanceLobby, getRun, PROTOCOL_VERSION, utcnow,
  READY_TRANSITION_SIGNUP_IN,
} from '../../_db.js';

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return send(res, 204, {});
  if (req.method !== 'POST') return send(res, 405, { error: 'method not allowed' });
  const signupId = req.query.signup_id;
  const agent = await agentFromToken(req);
  if (!agent) return send(res, 403, { error: 'invalid bearer token' });
  const body = await readBody(req);
  if ((body.protocol_version || PROTOCOL_VERSION) !== PROTOCOL_VERSION)
    return send(res, 400, { error: 'unsupported protocol_version' });
  let s = (await q('SELECT * FROM run_signups WHERE id=$1 AND agent_id=$2', [signupId, agent.id]))[0];
  if (!s) return send(res, 404, { error: 'signup not found' });
  await q(`UPDATE run_signups SET status='ready', updated_utc=$2
           WHERE id=$1 AND status IN (${READY_TRANSITION_SIGNUP_IN})`, [signupId, utcnow()]);
  await advanceLobby(s.run_id);
  s = (await q('SELECT * FROM run_signups WHERE id=$1', [signupId]))[0];
  const run = await getRun(s.run_id);
  return send(res, 200, { ok: true, ...signupResponse(s, run.coordinator_url) });
}
