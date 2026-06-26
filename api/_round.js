// Banker's rounding (round-half-to-even) — matches Python's built-in round(), which ALL the rating
// and scoring math in arena/ uses. JS Math.round() is half-UP and diverges on exact .5 boundaries
// (e.g. 1/32 = 0.03125 -> Python round() 0.0312 vs Math.round() 0.0313; a Wilson bound 0.0625 ->
// 0.062 vs 0.063), so every JS port of a Python-rounded value MUST round through here to stay
// byte-identical to the Python golden. Guarded by tests/round_half_even.test.mjs.
export function roundHalfEven(x, digits) {
  if (!Number.isFinite(x)) return x;
  const factor = 10 ** digits;
  const y = x * factor;
  const sign = Math.sign(y) || 1;
  const abs = Math.abs(y);
  const floor = Math.floor(abs);
  const frac = abs - floor;
  let rounded;
  if (Math.abs(frac - 0.5) < 1e-9) {
    rounded = floor % 2 === 0 ? floor : floor + 1;
  } else {
    rounded = Math.round(abs);
  }
  return (sign * rounded) / factor;
}
