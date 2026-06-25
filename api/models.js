// GET /api/models — OpenRouter model catalog for the static run picker.
// The helper uses the public /models endpoint and falls back to known roster models.
import { send } from './_db.js';
import { fetchModels, ROSTER_FALLBACK } from './_read.js';

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return send(res, 204, {});
  if (req.method !== 'GET') return send(res, 405, { error: 'method not allowed' });

  const models = await fetchModels();
  return send(res, 200, { models: models.length ? models : ROSTER_FALLBACK });
}
