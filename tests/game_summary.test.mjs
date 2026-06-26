// Offline assertions for api/_game_summary.js pure helpers.
process.env.DATABASE_URL ||= 'postgres://u:p@localhost/db';

import assert from 'node:assert/strict';
import { gameSummaryInput } from '../api/_game_summary.js';

const transcript = {
  seed: 9001,
  winner_team: 'good',
  outcome: { text: 'Village wins by eliminating the werewolf.' },
  players: [
    { seat: 0, name: 'Alice', dealt: 'Seer', end: 'Seer', team: 'good', won: true, calls: 3 },
    { seat: 1, name: 'Bob', dealt: 'Werewolf', end: 'Werewolf', team: 'evil', won: false },
  ],
  phases: [
    {
      name: 'Discussion',
      events: [
        { t: 'say', pid: 0, text: 'I checked Bob and found a wolf.' },
        { t: 'pass', pid: 1, stance: 'done', declared_reasoning: 'private text must not matter' },
      ],
    },
    {
      name: 'Vote',
      events: [
        { t: 'vote', pid: 0, tgt: 1, text: 'Bob is the wolf.' },
        { t: 'result', text: 'Village wins by eliminating Bob.', team: 'good' },
      ],
    },
  ],
};

const shaped = gameSummaryInput(transcript);
assert.equal(shaped.seed, 9001);
assert.equal(shaped.winner_team, 'good');
assert.deepEqual(shaped.players[1], {
  seat: 1,
  name: 'Bob',
  dealt: 'Werewolf',
  end: 'Werewolf',
  team: 'evil',
  won: false,
  calls: 0,
  forfeits: 0,
});
assert.deepEqual(shaped.phases[0].events, [
  'Alice: I checked Bob and found a wolf.',
  'Bob: passed (done)',
]);
assert.ok(!JSON.stringify(shaped).includes('private text must not matter'));

console.log('game_summary.test.mjs: assertions passed');
