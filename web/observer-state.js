export function roleTeam(transcript, role) {
  const card = (transcript.cardsInPlay || []).find(candidate => candidate[0] === role);
  return card ? card[1] : "good";
}

export function onuwStartingCenter(transcript) {
  const remaining = (transcript.cardsInPlay || []).map(card => [card[0], card[1]]);
  for (const player of transcript.players || []) {
    const dealtRole = player.dealt || player.end;
    const index = remaining.findIndex(card => card[0] === dealtRole);
    if (index >= 0) remaining.splice(index, 1);
  }
  return remaining;
}
