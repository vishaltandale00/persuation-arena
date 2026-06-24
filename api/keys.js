// GET /api/keys  — key/provider status for the new-run picker.
// POST /api/keys — browser-submitted key save.
//
// Port of arena/server.py api_keys (~621) and api_keys_set (~645). This is a HOSTED (Vercel)
// deployment, so _browser_key_management_enabled() is always False here: server_key_management is
// always false and POST is rejected with 403 (browser-submitted keys are disabled on hosted
// deployments; only the server-side OPENROUTER_API_KEY env secret is used).
import { send } from './_db.js';
import { OPENROUTER_BASE_URL, rosterModels } from './_read.js';

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return send(res, 204, {});

  if (req.method === 'GET') {
    // configured := has_api_key(): truthy, non-empty (after strip) OPENROUTER_API_KEY.
    const configured = !!(process.env.OPENROUTER_API_KEY || '').trim();
    return send(res, 200, {
      provider: 'openrouter',
      base_url: OPENROUTER_BASE_URL,
      configured,
      models: rosterModels(),
      // Hosted: _browser_key_management_enabled() === false, always.
      server_key_management: false,
      // _hosted() && configured -> "server_env"; hosted is always true here.
      key_source: configured ? 'server_env' : null,
    });
  }

  if (req.method === 'POST') {
    // Mirror api_keys_set: browser-submitted key save is disabled on hosted deployments.
    return send(res, 403, { error: 'browser-submitted keys are disabled on hosted deployments' });
  }

  return send(res, 405, { error: 'method not allowed' });
}
