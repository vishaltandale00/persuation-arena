// GET /api/signups/:signup_id — the agent's status poll. Advances the lobby (so seats/activation
// progress as agents poll) and returns the current signup + the run's coordinator URL.
import { q, send, agentFromToken, signupResponse, advanceLobby, getRun } from '../_db.js';

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return send(res, 204, {});
  const signupId = req.query.signup_id;
  const agent = await agentFromToken(req);
  if (!agent) return send(res, 403, { error: 'invalid bearer token' });
  let s = (await q('SELECT * FROM run_signups WHERE id=$1 AND agent_id=$2', [signupId, agent.id]))[0];
  if (!s) return send(res, 404, { error: 'signup not found' });
  await advanceLobby(s.run_id);
  s = (await q('SELECT * FROM run_signups WHERE id=$1', [signupId]))[0];
  const run = await getRun(s.run_id);
  return send(res, 200, signupResponse(s, run.coordinator_url));
}
