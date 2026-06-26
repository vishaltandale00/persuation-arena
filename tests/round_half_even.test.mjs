// Guards api/_round.js roundHalfEven() — the single rounding fn shared by leaderboard.js, _read.js,
// and _rating.js so JS-ported numbers match Python's banker's round() (the golden's source of truth).
//
// Pins the BUG cases: exact-half values where Python round() and the old Math.round (half-UP) that
// _read.js/_rating.js used diverge. roundHalfEven must reproduce Python AND differ from half-up.
// (End-to-end fixture coverage lives in rating_parity / leaderboard_parity; this pins the primitive.)
import assert from 'node:assert/strict';
import { roundHalfEven } from '../api/_round.js';

// [input, digits, Python round(input, digits)] — verified against CPython.
const PY_BANKERS = [
  [0.03125, 4, 0.0312],   // forfeit_rate 1/32 (half-up -> 0.0313)
  [0.0625, 3, 0.062],     // a Wilson CI bound (half-up -> 0.063)
  [0.0312, 4, 0.0312],    // non-boundary: unchanged
  [0.999, 3, 0.999],
  [0.1236, 3, 0.124],
  [0.1234, 3, 0.123],
  [2.5, 0, 2],
  [3.5, 0, 4],
  [0.5, 0, 0],
  [1.5, 0, 2],
  [-0.0625, 3, -0.062],   // sign-symmetric
];

for (const [x, d, expected] of PY_BANKERS) {
  assert.equal(roundHalfEven(x, d), expected, `roundHalfEven(${x}, ${d}) should be ${expected} (Python round)`);
}

// The whole point: roundHalfEven must NOT be the naive half-up that produced wrong last digits.
const halfUp = (x, d) => Math.round(x * 10 ** d) / 10 ** d;
assert.equal(halfUp(0.03125, 4), 0.0313);              // the bug
assert.notEqual(roundHalfEven(0.03125, 4), halfUp(0.03125, 4));
assert.equal(halfUp(0.0625, 3), 0.063);                // the bug
assert.notEqual(roundHalfEven(0.0625, 3), halfUp(0.0625, 3));

// Non-finite passthrough (publicCompetitor can feed NaN/Infinity defensively).
assert.ok(Number.isNaN(roundHalfEven(NaN, 3)));
assert.equal(roundHalfEven(Infinity, 3), Infinity);

console.log('\nPASS  round-half-even (banker\'s rounding matches Python on the boundary cases; differs from half-up)');
