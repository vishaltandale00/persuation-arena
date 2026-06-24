// GET /api/models — full OpenRouter model catalog for the new-run picker.
// Port of arena/server.py api_models (~630) + _fetch_models (~572). Falls back to the
// known roster models if OpenRouter is unreachable so the picker is never empty.
import { send } from './_db.js';
import { fetchModels, ROSTER_FALLBACK } from './_read.js';

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return send(res, 204, {});
  let models = await fetchModels();
  if (!models.length) models = ROSTER_FALLBACK;
  return send(res, 200, { models });
}
