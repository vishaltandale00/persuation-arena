// POST /api/keys/test — port of arena/server.py api_keys_test / _test_openrouter.
// Live-validates the server-side OpenRouter key (env secret) against OpenRouter's /key endpoint.
// This is the hosted (Vercel) deployment, so account metadata (usage/limit/label) is always
// redacted — the endpoint is public — matching Python's _hosted() branch.
import { send } from '../_db.js';
import { OPENROUTER_BASE_URL } from '../_read.js';

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return send(res, 204, {});

  // config.has_api_key(): truthy only when the trimmed env key is non-empty.
  const key = (process.env.OPENROUTER_API_KEY || '').trim();
  if (!key) return send(res, 200, { ok: false, error: 'no key configured' });

  try {
    const r = await fetch(OPENROUTER_BASE_URL + '/key', {
      headers: { Authorization: `Bearer ${key}` },
    });
    if (r.status === 200) {
      // Hosted: redact usage/limit/label.
      return send(res, 200, { ok: true, configured: true, source: 'server_env' });
    }
    if (r.status === 401 || r.status === 403) {
      return send(res, 200, { ok: false, error: 'invalid key' });
    }
    return send(res, 200, { ok: false, error: `status ${r.status}` });
  } catch (e) {
    // Mirror Python's type(e).__name__ — the exception class name.
    return send(res, 200, { ok: false, error: e && e.name ? e.name : 'Error' });
  }
}
