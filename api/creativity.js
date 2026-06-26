// GET /api/creativity - read-only projection of the precomputed creativity snapshot.
//
// Creativity rows are derived from public say-event transcripts by tools/creativity_signal.py.
// The source of truth remains games.transcript_json; this route only serves the latest uploaded
// metric version to the observer.
import { q, send } from './_db.js';
import { roundHalfEven } from './_round.js';

const r3 = (x) => (x == null ? null : roundHalfEven(Number(x), 3));

function boolish(value) {
  return Boolean(Number(value || 0));
}

function intish(value) {
  return Number(value || 0);
}

function parsePhrases(value) {
  if (!value) return [];
  if (Array.isArray(value)) return value.map(String).filter(Boolean);
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed.map(String).filter(Boolean) : [];
  } catch {
    return [];
  }
}

function publicRole(row) {
  return {
    raw_distance: r3(row.raw_distance),
    role_z: r3(row.role_z),
    valid_utterances: intish(row.valid_utterances),
    valid_pairs: intish(row.valid_pairs),
    sampled_pairs: intish(row.sampled_pairs),
    excluded_pairs: intish(row.excluded_pairs),
    provisional: boolish(row.provisional),
    insufficient: boolish(row.insufficient),
  };
}

function publicSample(row) {
  return {
    role: row.role,
    utterance_a: row.utterance_a,
    utterance_b: row.utterance_b,
    embedding_similarity: r3(row.embedding_similarity),
    judge_similarity: r3(row.judge_similarity),
    distance: r3(row.distance),
    coherence_a: row.coherence_a,
    coherence_b: row.coherence_b,
    reason: row.reason,
    divergence_phrases: parsePhrases(row.divergence_phrases_json),
  };
}

export function assembleCreativity(rows, version = null, updatedUtc = null, sampleRows = []) {
  const byIdentity = {};
  for (const row of rows) {
    const entry = (byIdentity[row.identity_key] ||= {
      version: row.version,
      identity_key: row.identity_key,
      agent_id: row.agent_id,
      display_name: row.display_name,
      declared_model: row.declared_model,
      declared_harness: row.declared_harness,
      by_role: {},
      samples: [],
    });

    if (row.role === '*') {
      Object.assign(entry, {
        raw_distance: r3(row.raw_distance),
        overall_z: r3(row.overall_z),
        valid_utterances: intish(row.valid_utterances),
        valid_pairs: intish(row.valid_pairs),
        sampled_pairs: intish(row.sampled_pairs),
        excluded_pairs: intish(row.excluded_pairs),
        eligible_roles: intish(row.eligible_roles),
        provisional: boolish(row.provisional),
        insufficient: boolish(row.insufficient),
        updated_utc: row.updated_utc,
      });
    } else {
      entry.by_role[row.role] = publicRole(row);
    }
  }

  for (const row of sampleRows) {
    const entry = byIdentity[row.identity_key];
    if (entry) entry.samples.push(publicSample(row));
  }

  const competitors = Object.values(byIdentity).sort((a, b) => {
    const az = a.overall_z ?? -9999;
    const bz = b.overall_z ?? -9999;
    if (bz !== az) return bz - az;
    const ar = a.raw_distance ?? -9999;
    const br = b.raw_distance ?? -9999;
    if (br !== ar) return br - ar;
    return (b.valid_pairs || 0) - (a.valid_pairs || 0);
  });

  return { version, updated_utc: updatedUtc, competitors };
}

async function creativityTableExists() {
  const rows = await q("SELECT to_regclass('public.creativity_scores') AS table_name");
  return Boolean(rows[0]?.table_name);
}

async function creativityJudgmentsTableExists() {
  const rows = await q("SELECT to_regclass('public.creativity_judgments') AS table_name");
  return Boolean(rows[0]?.table_name);
}

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') return send(res, 204, {});

  if (req.method === 'GET') {
    try {
      if (!(await creativityTableExists())) {
        return send(res, 200, { version: null, updated_utc: null, competitors: [] });
      }

      const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
      let version = url.searchParams.get('version');
      let updatedUtc = null;
      if (!version) {
        const latest = await q(
          'SELECT version, MAX(updated_utc) AS updated_utc FROM creativity_scores ' +
            'GROUP BY version ORDER BY updated_utc DESC LIMIT 1',
        );
        if (!latest.length) {
          return send(res, 200, { version: null, updated_utc: null, competitors: [] });
        }
        version = latest[0].version;
        updatedUtc = latest[0].updated_utc;
      } else {
        const stamp = await q('SELECT MAX(updated_utc) AS updated_utc FROM creativity_scores WHERE version = $1', [
          version,
        ]);
        updatedUtc = stamp[0]?.updated_utc || null;
      }

      const rows = await q('SELECT * FROM creativity_scores WHERE version = $1', [version]);
      let samples = [];
      if (await creativityJudgmentsTableExists()) {
        const sampleRows = await q(
          'SELECT identity_key, role, utterance_a, utterance_b, embedding_similarity, judge_similarity, distance, ' +
            'coherence_a, coherence_b, reason, divergence_phrases_json ' +
            'FROM creativity_judgments WHERE version = $1 ' +
            "ORDER BY identity_key, CASE WHEN coherence_a = 'valid' AND coherence_b = 'valid' THEN 0 ELSE 1 END, " +
            "CASE WHEN tokens_a >= 4 AND tokens_b >= 4 AND lower(trim(utterance_a)) <> 'none' " +
            "AND lower(trim(utterance_b)) <> 'none' THEN 0 ELSE 1 END, " +
            'distance DESC NULLS LAST, judgment_key',
          [version],
        );
        const counts = new Map();
        samples = sampleRows.filter(row => {
          const count = counts.get(row.identity_key) || 0;
          if (count >= 4) return false;
          counts.set(row.identity_key, count + 1);
          return true;
        });
      }
      return send(res, 200, assembleCreativity(rows, version, updatedUtc, samples));
    } catch (err) {
      if (err?.code === '42P01') return send(res, 200, { version: null, updated_utc: null, competitors: [] });
      throw err;
    }
  }

  return send(res, 405, { error: 'method not allowed' });
}
