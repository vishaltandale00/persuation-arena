import { test } from 'node:test';
import assert from 'node:assert/strict';

process.env.DATABASE_URL ||= 'postgres://u:p@localhost/db';
const { assembleCreativity } = await import('./creativity.js');

test('assembleCreativity groups overall and role rows by identity', () => {
  const rows = [
    {
      version: 'creativity_v1',
      identity_key: 'static:model-a:base',
      display_name: 'model-a',
      declared_model: 'model-a',
      declared_harness: 'base',
      role: '*',
      raw_distance: 0.72,
      overall_z: 1.25,
      valid_utterances: 20,
      valid_pairs: 12,
      sampled_pairs: 14,
      excluded_pairs: 2,
      eligible_roles: 2,
      provisional: 0,
      insufficient: 0,
      updated_utc: '2026-06-26T00:00:00.000000Z',
    },
    {
      version: 'creativity_v1',
      identity_key: 'static:model-a:base',
      display_name: 'model-a',
      declared_model: 'model-a',
      declared_harness: 'base',
      role: 'Villager',
      raw_distance: 0.7,
      role_z: 1.1,
      valid_utterances: 10,
      valid_pairs: 6,
      sampled_pairs: 8,
      excluded_pairs: 1,
      provisional: 0,
      insufficient: 0,
    },
    {
      version: 'creativity_v1',
      identity_key: 'static:model-b:base',
      display_name: 'model-b',
      declared_model: 'model-b',
      declared_harness: 'base',
      role: '*',
      raw_distance: 0.5,
      overall_z: -0.25,
      valid_utterances: 8,
      valid_pairs: 4,
      sampled_pairs: 6,
      excluded_pairs: 2,
      eligible_roles: 1,
      provisional: 1,
      insufficient: 0,
      updated_utc: '2026-06-26T00:00:00.000000Z',
    },
  ];

  const samples = [
    {
      identity_key: 'static:model-a:base',
      role: 'Villager',
      utterance_a: 'I am pushing a Seer claim now.',
      utterance_b: 'That sounds like a late deflection.',
      embedding_similarity: 0.21,
      judge_similarity: 0.18,
      distance: 0.82,
      coherence_a: 'valid',
      coherence_b: 'valid',
      reason: 'Different persuasive moves.',
      divergence_phrases_json: '["Seer claim","late deflection"]',
    },
  ];

  const out = assembleCreativity(rows, 'creativity_v1', '2026-06-26T00:00:00.000000Z', samples);

  assert.equal(out.version, 'creativity_v1');
  assert.equal(out.competitors.length, 2);
  assert.equal(out.competitors[0].identity_key, 'static:model-a:base');
  assert.equal(out.competitors[0].overall_z, 1.25);
  assert.equal(out.competitors[0].by_role.Villager.raw_distance, 0.7);
  assert.equal(out.competitors[0].samples.length, 1);
  assert.equal(out.competitors[0].samples[0].distance, 0.82);
  assert.deepEqual(out.competitors[0].samples[0].divergence_phrases, ['Seer claim', 'late deflection']);
  assert.equal(out.competitors[1].provisional, true);
});
