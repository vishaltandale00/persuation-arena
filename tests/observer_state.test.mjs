import assert from "node:assert/strict";

import { onuwStartingCenter, roleTeam } from "../web/observer-state.js";

const transcript = {
  cardsInPlay: [
    ["Werewolf", "evil"],
    ["Werewolf", "evil"],
    ["Minion", "evil"],
    ["Seer", "good"],
    ["Tanner", "evil"],
    ["Robber", "good"],
    ["Mason", "good"],
    ["Mason", "good"],
  ],
  players: [
    { dealt: "Mason", end: "Mason" },
    { dealt: "Werewolf", end: "Werewolf" },
    { dealt: "Minion", end: "Minion" },
    { dealt: "Mason", end: "Robber" },
    { dealt: "Robber", end: "Mason" },
  ],
};

assert.deepEqual(
  onuwStartingCenter(transcript).map(card => card[0]),
  ["Werewolf", "Seer", "Tanner"],
  "subtracts one card per dealt player while preserving duplicate roles",
);
assert.equal(roleTeam(transcript, "Werewolf"), "evil");
assert.equal(roleTeam(transcript, "Mason"), "good");

const cardsBefore = structuredClone(transcript.cardsInPlay);
onuwStartingCenter(transcript);
assert.deepEqual(transcript.cardsInPlay, cardsBefore, "does not mutate transcript cards");

console.log("observer state tests passed");
