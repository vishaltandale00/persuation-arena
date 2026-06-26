export const OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1';
export const GAME_SUMMARY_MODEL = process.env.ARENA_GAME_SUMMARY_MODEL || 'openai/gpt-5.5';
export const GAME_SUMMARY_REASONING_EFFORT = process.env.ARENA_GAME_SUMMARY_REASONING_EFFORT || 'medium';

function cleanText(value, max = 1200) {
  return String(value ?? '').replace(/\s+/g, ' ').trim().slice(0, max);
}

function phaseDigest(transcript) {
  const names = playersBySeat(transcript);
  const out = [];
  for (const phase of transcript.phases || []) {
    const events = [];
    for (const event of phase.events || []) {
      if (event.t === 'sys' || event.t === 'round') continue;
      if (event.t === 'result') {
        events.push(`Result: ${cleanText(event.text, 500)}`);
      } else if (event.t === 'say') {
        events.push(`${names.get(event.pid) || `Seat ${event.pid}`}: ${cleanText(event.text, 500)}`);
      } else if (event.t === 'pass') {
        events.push(`${names.get(event.pid) || `Seat ${event.pid}`}: passed${event.stance ? ` (${event.stance})` : ''}`);
      } else if (event.t === 'act') {
        events.push(`${names.get(event.pid) || `Seat ${event.pid}`}: ${cleanText(event.text, 500)}`);
      } else if (event.t === 'vote') {
        const target = Number(event.tgt) < 0 ? 'no one' : (names.get(event.tgt) || `Seat ${event.tgt}`);
        events.push(`${names.get(event.pid) || `Seat ${event.pid}`} voted for ${target}: ${cleanText(event.text, 500)}`);
      }
      if (events.length >= 80) break;
    }
    out.push({ phase: phase.name || phase.kind || 'Phase', events });
  }
  return out;
}

function playersBySeat(transcript) {
  const names = new Map();
  for (const player of transcript.players || []) {
    names.set(player.seat, player.name || `Seat ${player.seat}`);
  }
  return names;
}

export function gameSummaryInput(transcript) {
  return {
    meta: transcript.meta || null,
    seed: transcript.seed ?? null,
    winner_team: transcript.winner_team ?? null,
    outcome: transcript.outcome || null,
    players: (transcript.players || []).map((p) => ({
      seat: p.seat,
      name: p.name || `Seat ${p.seat}`,
      dealt: p.dealt,
      end: p.end,
      team: p.team,
      won: Boolean(p.won),
      calls: p.calls ?? 0,
      forfeits: p.forfeits ?? 0,
    })),
    phases: phaseDigest(transcript),
  };
}

function normalizeSummary(value, model, generatedAt) {
  const src = value && typeof value === 'object' ? value : { summary: String(value || '') };
  const highlights = Array.isArray(src.highlights) ? src.highlights.map((x) => cleanText(x, 280)).filter(Boolean).slice(0, 5) : [];
  const turningPoints = Array.isArray(src.turning_points) ? src.turning_points.map((x) => cleanText(x, 280)).filter(Boolean).slice(0, 5) : [];
  return {
    model,
    reasoning_effort: GAME_SUMMARY_REASONING_EFFORT,
    generated_at: generatedAt,
    text: cleanText(src.summary || src.text || '', 2000),
    result: cleanText(src.result || '', 500),
    highlights,
    turning_points: turningPoints,
  };
}

function parseSummary(content) {
  try {
    return JSON.parse(content);
  } catch {
    const match = String(content || '').match(/\{[\s\S]*\}/);
    if (!match) return { summary: content };
    try { return JSON.parse(match[0]); } catch { return { summary: content }; }
  }
}

async function completion(messages, { structured = true, reasoning = true } = {}) {
  const key = process.env.OPENROUTER_API_KEY;
  if (!key) throw new Error('OPENROUTER_API_KEY is required to generate game summaries');
  const body = {
    model: GAME_SUMMARY_MODEL,
    messages,
    max_tokens: 700,
    temperature: 0.2,
  };
  if (structured) body.response_format = { type: 'json_object' };
  if (reasoning) body.reasoning = { effort: GAME_SUMMARY_REASONING_EFFORT, exclude: true };
  const response = await fetch(`${OPENROUTER_BASE_URL}/chat/completions`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${key}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => '');
    throw new Error(`OpenRouter summary request failed: HTTP ${response.status}${detail ? ` ${detail.slice(0, 300)}` : ''}`);
  }
  return await response.json();
}

export async function summarizeGame(transcript) {
  const input = gameSummaryInput(transcript);
  const messages = [
    {
      role: 'system',
      content: [
        'You summarize completed hidden-role game results for an observer UI.',
        'Use only the supplied public actions, votes, final roles, and outcome.',
        'Do not mention private reasoning, provider traces, hidden model logs, or raw JSON.',
        'Return JSON with keys: summary, result, highlights, turning_points.',
      ].join(' '),
    },
    { role: 'user', content: JSON.stringify(input) },
  ];
  let data;
  try {
    data = await completion(messages, { structured: true, reasoning: true });
  } catch (err) {
    const text = String(err && err.message ? err.message : err).toLowerCase();
    if (!text.includes('response_format') && !text.includes('reasoning') && !text.includes('unsupported')) throw err;
    data = await completion(messages, { structured: false, reasoning: false });
  }
  const content = data?.choices?.[0]?.message?.content || '';
  const generatedAt = new Date().toISOString().replace(/(\.\d{3})Z$/, '$1000Z');
  return normalizeSummary(parseSummary(content), GAME_SUMMARY_MODEL, generatedAt);
}
