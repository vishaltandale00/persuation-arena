// POST /api/agents/register — issue an agent identity + bearer token (the token hash matches the
// Python sha256 so the per-run coordinator can authenticate poll/reply).
import { q, send, readBody, PROTOCOL_VERSION, issueToken, sha256hex, newId, utcnow } from '../_db.js';

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return send(res, 204, {});
  if (req.method !== 'POST') return send(res, 405, { error: 'method not allowed' });
  const body = await readBody(req);
  const name = (body.display_name || body.name || '').trim();
  if (!name) return send(res, 400, { error: 'display_name required' });
  const protocol = (body.protocol_version || PROTOCOL_VERSION).trim();
  if (protocol !== PROTOCOL_VERSION) return send(res, 400, { error: 'unsupported protocol_version' });
  const token = issueToken();
  const id = newId('agent_');
  const now = utcnow();
  const declaredModel = String(body.model || body.declared_model || '').trim() || null;
  const declaredHarness = String(body.harness || body.declared_harness || '').trim() || null;
  await q(
    `INSERT INTO agents (id,display_name,token_hash,protocol_version,sdk_version,created_utc,last_seen_utc,status,declared_model,declared_harness)
     VALUES ($1,$2,$3,$4,$5,$6,$6,'idle',$7,$8)`,
    [id, name, sha256hex(token), protocol, body.sdk_version || null, now, declaredModel, declaredHarness]
  );
  return send(res, 200, { agent_id: id, agent_token: token, protocol_version: PROTOCOL_VERSION });
}
