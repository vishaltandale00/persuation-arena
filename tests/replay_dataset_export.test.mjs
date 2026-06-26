import assert from 'node:assert/strict';
import { unzipSync, strFromU8 } from 'fflate';

import {
  buildReplayDataset,
  zipReplayDataset,
} from '../tools/replay_dataset_export.mjs';

function j(value) {
  return JSON.stringify(value);
}

const rows = {
  runs: [
    {
      id: 'run_connected',
      game: 'onuw',
      status: 'done',
      n_games: 1,
      players: 2,
      seed_base: 100,
      deck_preset: 'arena',
      created_utc: '2026-06-24T00:00:00.000Z',
    },
    {
      id: 'run_static',
      game: 'onuw',
      status: 'done',
      n_games: 1,
      players: 2,
      seed_base: 200,
      deck_preset: 'arena',
      created_utc: '2026-06-24T01:00:00.000Z',
    },
  ],
  games: [
    {
      run_id: 'run_connected',
      gid: 1,
      seed: 101,
      winner_team: 'good',
      line: 'Village wins.',
      transcript_json: j({ seed: 101, players: [{ seat: 0 }, { seat: 1 }] }),
    },
    {
      run_id: 'run_static',
      gid: 1,
      seed: 201,
      winner_team: 'evil',
      line: 'Werewolves win.',
      transcript_json: j({
        seed: 201,
        players: [{ seat: 0, name: 'Static A' }, { seat: 1, name: 'Static B' }],
        phases: [
          {
            name: 'Night',
            kind: 'night',
            events: [
              { t: 'sys', text: 'PRIVATE NIGHT SETUP SHOULD NOT EXPORT' },
              { t: 'act', pid: 1, text: 'Werewolf sees the center.' },
            ],
          },
          {
            name: 'Discussion',
            kind: 'talk',
            events: [
              { t: 'sys', text: 'Discussion starts.' },
              { t: 'say', pid: 1, text: 'Static claim from transcript.', urgency: 2 },
              { t: 'pass', pid: 0, stance: 'done' },
            ],
          },
        ],
      }),
    },
  ],
  gamePlayers: [
    {
      run_id: 'run_connected',
      gid: 1,
      seat: 0,
      agent: 'Alpha',
      model: 'connected-agent',
      dealt_role: 'Seer',
      end_role: 'Seer',
      team: 'good',
      won: 1,
      calls: 2,
      forfeits: 0,
      agent_id: 'agent_alpha',
      signup_id: 'signup_alpha',
    },
    {
      run_id: 'run_connected',
      gid: 1,
      seat: 1,
      agent: 'Beta',
      model: 'connected-agent',
      dealt_role: 'Werewolf',
      end_role: 'Werewolf',
      team: 'evil',
      won: 0,
      calls: 2,
      forfeits: 1,
      agent_id: 'agent_beta',
      signup_id: 'signup_beta',
    },
    {
      run_id: 'run_static',
      gid: 1,
      seat: 0,
      agent: 'Static A',
      model: 'static',
      dealt_role: 'Robber',
      end_role: 'Robber',
      team: 'good',
      won: 0,
      calls: 0,
      forfeits: 0,
    },
    {
      run_id: 'run_static',
      gid: 1,
      seat: 1,
      agent: 'Static B',
      model: 'static',
      dealt_role: 'Werewolf',
      end_role: 'Werewolf',
      team: 'evil',
      won: 1,
      calls: 0,
      forfeits: 0,
    },
  ],
  runSignups: [
    { id: 'signup_alpha', run_id: 'run_connected', agent_id: 'agent_alpha', seat: 0, display_name: 'Alpha' },
    { id: 'signup_beta', run_id: 'run_connected', agent_id: 'agent_beta', seat: 1, display_name: 'Beta' },
  ],
  agents: [
    { id: 'agent_alpha', display_name: 'Alpha', declared_model: 'alpha-model', declared_harness: 'connected' },
    { id: 'agent_beta', display_name: 'Beta', declared_model: 'beta-model', declared_harness: 'connected' },
  ],
  runEvents: [
    {
      id: 'evt_setup',
      run_id: 'run_connected',
      game_instance_id: 'run_connected_game_001',
      seq: 1,
      visibility: 'public',
      phase: 'setup',
      type: 'game_setup',
      payload_json: j({ n: 2, roster: { 0: 'Alpha', 1: 'Beta' }, deck: ['Seer', 'Werewolf', 'Robber'], win_condition: 'test' }),
    },
    {
      id: 'evt_alpha_role',
      run_id: 'run_connected',
      game_instance_id: 'run_connected_game_001',
      seq: 2,
      visibility: 'private',
      target_signup_id: 'signup_alpha',
      phase: 'setup',
      type: 'role_info',
      payload_json: j({ seat: 0, role: 'Seer', text: 'ALPHA_PRIVATE_ROLE' }),
    },
    {
      id: 'evt_beta_role',
      run_id: 'run_connected',
      game_instance_id: 'run_connected_game_001',
      seq: 3,
      visibility: 'private',
      target_signup_id: 'signup_beta',
      phase: 'setup',
      type: 'role_info',
      payload_json: j({ seat: 1, role: 'Werewolf', text: 'BETA_PRIVATE_ROLE' }),
    },
    {
      id: 'evt_discussion',
      run_id: 'run_connected',
      game_instance_id: 'run_connected_game_001',
      seq: 4,
      visibility: 'public',
      phase: 'discussion',
      type: 'phase_started',
      payload_json: j({ phase: 'discussion', text: 'Day discussion begins.' }),
    },
    {
      id: 'evt_beta_speech',
      run_id: 'run_connected',
      game_instance_id: 'run_connected_game_001',
      seq: 5,
      visibility: 'public',
      phase: 'discussion',
      type: 'speech',
      payload_json: j({ actor_seat: 1, text: 'I am just a villager.', urgency: 1 }),
    },
    {
      id: 'evt_alpha_turn',
      run_id: 'run_connected',
      game_instance_id: 'run_connected_game_001',
      seq: 6,
      visibility: 'private',
      target_signup_id: 'signup_alpha',
      phase: 'discussion',
      type: 'private_observation',
      payload_json: j({ seat: 0, turn_id: 'turn_alpha', action_kind: 'onuw.discussion.speak_or_pass' }),
    },
    {
      id: 'evt_beta_turn',
      run_id: 'run_connected',
      game_instance_id: 'run_connected_game_001',
      seq: 7,
      visibility: 'private',
      target_signup_id: 'signup_beta',
      phase: 'discussion',
      type: 'private_observation',
      payload_json: j({ seat: 1, turn_id: 'turn_beta', action_kind: 'onuw.discussion.speak_or_pass' }),
    },
    {
      id: 'evt_alpha_result',
      run_id: 'run_connected',
      game_instance_id: 'run_connected_game_001',
      seq: 8,
      visibility: 'private',
      target_signup_id: 'signup_alpha',
      phase: 'discussion',
      type: 'action_result',
      payload_json: j({ seat: 0, turn_id: 'turn_alpha', accepted: true }),
    },
    {
      id: 'evt_alpha_speech',
      run_id: 'run_connected',
      game_instance_id: 'run_connected_game_001',
      seq: 9,
      visibility: 'public',
      phase: 'discussion',
      type: 'speech',
      payload_json: j({ actor_seat: 0, text: 'I saw evil.', urgency: 3 }),
    },
    {
      id: 'evt_static_telemetry',
      run_id: 'run_connected',
      game_instance_id: 'run_connected_game_001',
      seq: 10,
      visibility: 'public',
      phase: 'discussion',
      type: 'model_turn_completed',
      payload_json: j({ seat: 1, reasoning: 'BETA_INTERNAL_REASONING_SHOULD_NOT_BE_PUBLIC', action: { pass: true } }),
    },
  ],
  turns: [
    {
      id: 'turn_alpha',
      run_id: 'run_connected',
      signup_id: 'signup_alpha',
      game_instance_id: 'run_connected_game_001',
      seat: 0,
      phase: 'discussion',
      action_kind: 'onuw.discussion.speak_or_pass',
      observation_json: j({ format: 'text', text: 'Alpha prompt at decision time.' }),
      legal_action_json: j({ type: 'object', choices: ['speak', 'pass'] }),
      status: 'replied',
      created_utc: '2026-06-24T00:00:01.000Z',
    },
    {
      id: 'turn_beta',
      run_id: 'run_connected',
      signup_id: 'signup_beta',
      game_instance_id: 'run_connected_game_001',
      seat: 1,
      phase: 'discussion',
      action_kind: 'onuw.discussion.speak_or_pass',
      observation_json: j({ format: 'text', text: 'Beta prompt at decision time.' }),
      legal_action_json: j({ type: 'object', choices: ['speak', 'pass'] }),
      status: 'replied',
      created_utc: '2026-06-24T00:00:02.000Z',
    },
  ],
  turnReplies: [
    {
      turn_id: 'turn_alpha',
      signup_id: 'signup_alpha',
      action_json: j({ speak: 'I saw evil.' }),
      reasoning: 'ALPHA_DECLARED_REASONING',
      client_ms: 111,
      accepted: 1,
      created_utc: '2026-06-24T00:00:03.000Z',
    },
    {
      turn_id: 'turn_beta',
      signup_id: 'signup_beta',
      action_json: j({ pass: true, stance: 'done' }),
      reasoning: 'BETA_DECLARED_REASONING_SHOULD_NOT_LEAK_TO_ALPHA',
      client_ms: 222,
      accepted: 1,
      created_utc: '2026-06-24T00:00:04.000Z',
    },
  ],
};

const dataset = buildReplayDataset(rows, {
  createdUtc: '2026-06-25T00:00:00.000Z',
  runIds: ['run_connected', 'run_static'],
});

assert.deepEqual(dataset.manifest.counts, {
  runs: 2,
  games: 2,
  perspectives: 2,
  skipped_games: 1,
  skipped_perspectives: 0,
});
assert.equal(dataset.manifest.skips[0].reason, 'missing_turn_trace');
assert.equal(dataset.manifest.skips[0].run_id, 'run_static');

const game = dataset.files['games/run_connected__g001.json'];
assert.ok(game);
assert.deepEqual(
  game.public_events.map((e) => e.type),
  ['game_setup', 'phase_started', 'speech', 'speech'],
);
assert.equal(game.public_events[2].action.speak, 'I am just a villager.');
assert.equal(JSON.stringify(game).includes('BETA_INTERNAL_REASONING_SHOULD_NOT_BE_PUBLIC'), false);

const staticGame = dataset.files['games/run_static__g001.json'];
assert.equal(staticGame.trace_quality, 'transcript_only');
assert.deepEqual(
  staticGame.public_events.map((e) => e.type),
  ['discussion_notice', 'speech', 'pass'],
);
assert.equal(staticGame.public_events[1].source, 'transcript_json');
assert.equal(staticGame.public_events[1].action.speak, 'Static claim from transcript.');
assert.equal(JSON.stringify(staticGame).includes('PRIVATE NIGHT SETUP SHOULD NOT EXPORT'), false);

const alpha = dataset.files['perspectives/run_connected__g001__seat0.json'];
assert.ok(alpha);
assert.equal(alpha.game_ref, '../games/run_connected__g001.json');
assert.deepEqual(alpha.private_events.map((e) => e.seq), [2, 6, 8]);
assert.equal(alpha.target_decisions.length, 1);
assert.equal(alpha.target_decisions[0].event_context.public_events_before_seq, 6);
assert.equal(alpha.target_decisions[0].event_context.private_events_before_seq, 6);
assert.equal(alpha.target_decisions[0].event_context.cursor_semantics, 'exclusive_seq');
assert.equal(alpha.target_decisions[0].output.declared_reasoning, 'ALPHA_DECLARED_REASONING');
assert.deepEqual(alpha.target_decisions[0].output.action, { speak: 'I saw evil.' });
assert.equal(alpha.target_decisions[0].output.accepted, true);
assert.equal(alpha.target_decisions[0].output.defaulted, false);
assert.equal(alpha.labels.dealt_role, 'Seer');
assert.equal(alpha.labels.winner_team, 'good');

const alphaJson = JSON.stringify(alpha);
assert.equal(alphaJson.includes('BETA_PRIVATE_ROLE'), false);
assert.equal(alphaJson.includes('BETA_DECLARED_REASONING_SHOULD_NOT_LEAK_TO_ALPHA'), false);
assert.equal(alphaJson.includes('BETA_INTERNAL_REASONING_SHOULD_NOT_BE_PUBLIC'), false);
assert.equal(alphaJson.includes('I am just a villager.'), false);
assert.equal(Object.hasOwn(alpha, 'public_events'), false);
assert.equal(Object.hasOwn(alpha, 'public_events_before_turn'), false);
assert.equal(Object.hasOwn(alpha, 'public_conversation_before_turn'), false);
assert.equal(alpha.target_decisions.some((d) => Object.hasOwn(d, 'in_game_message')), false);

const zipBuffer = zipReplayDataset(dataset.files);
const unzipped = unzipSync(zipBuffer);
assert.ok(unzipped['manifest.json']);
assert.ok(unzipped['games/run_connected__g001.json']);
assert.ok(unzipped['perspectives/run_connected__g001__seat0.json']);
const manifestFromZip = JSON.parse(strFromU8(unzipped['manifest.json']));
assert.deepEqual(manifestFromZip.counts, dataset.manifest.counts);

console.log('PASS  replay dataset export boundaries, cursors, skips, and zip layout');
