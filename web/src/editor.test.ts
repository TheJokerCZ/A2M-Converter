import test from "node:test";
import assert from "node:assert/strict";
import {
  EditorHistory,
  addNote,
  deleteNotes,
  mergeInstrument,
  moveNotes,
  resizeNote,
  setNotesVelocity,
  toExportPayload,
  type EditableNote,
} from "./editor.ts";

const sampleNotes = (): EditableNote[] => [
  { id: 1, instrument: "acoustic_piano", pitch: 60, start: 0.5, end: 1.0, velocity: 85 },
  { id: 2, instrument: "electric_piano", pitch: 64, start: 1.0, end: 1.5, velocity: 92 },
  { id: 3, instrument: "drums", pitch: 36, start: 0.5, end: 0.55, velocity: 118 },
];

test("toExportPayload zeros onset_delay so edited audio-aligned notes are never double-shifted", () => {
  const notes = sampleNotes();
  const payload = toExportPayload(notes, {
    bpm: 120,
    beats_per_bar: 4,
    first_downbeat: 0.5,
    onset_delay: 0.025,
    beat_subdivision: 4,
    steady: true,
    beats: [0.5, 1.0, 1.5],
  });
  assert.equal(payload.beatGrid?.onset_delay, 0);
  assert.equal(payload.beatGrid?.bpm, 120);
  assert.deepEqual(payload.notes[0], {
    instrument: "acoustic_piano",
    pitch: 60,
    start: 0.5,
    end: 1.0,
    velocity: 85,
  });
});

test("EditorHistory supports undo, redo, and 50-snapshot cap", () => {
  const history = new EditorHistory(50);
  let cur = sampleNotes();
  assert.equal(history.canUndo, false);
  assert.equal(history.canRedo, false);

  for (let i = 0; i < 60; i++) {
    history.push(cur);
    cur = setNotesVelocity(cur, new Set([1]), 50 + (i % 60));
  }
  assert.equal(history.undoDepth, 50);
  assert.equal(history.canUndo, true);

  const undone = history.undo(cur);
  assert.ok(undone !== null);
  assert.equal(history.canRedo, true);
  const redone = history.redo(undone!);
  assert.deepEqual(redone, cur);
});

test("mergeInstrument consolidates notes and preserves pitch, timing, and velocity", () => {
  const merged = mergeInstrument(sampleNotes(), "electric_piano", "acoustic_piano");
  const pianoNotes = merged.filter((n) => n.instrument === "acoustic_piano");
  assert.equal(pianoNotes.length, 2);
  assert.equal(pianoNotes[1].pitch, 64);
  assert.equal(pianoNotes[1].velocity, 92);
  assert.equal(merged.some((n) => n.instrument === "electric_piano"), false);
});

test("moveNotes, resizeNote, deleteNotes, addNote, and setNotesVelocity clamp cleanly", () => {
  const moved = moveNotes(sampleNotes(), new Set([1]), -2.0, -80);
  assert.equal(moved[0].start, 0);
  assert.equal(moved[0].end, 0.5);
  assert.equal(moved[0].pitch, 21);

  const resized = resizeNote(sampleNotes(), 1, 0.501);
  assert.ok(resized[0].end - resized[0].start >= 0.02);

  const deleted = deleteNotes(sampleNotes(), new Set([2]));
  assert.deepEqual(
    deleted.map((n) => n.id),
    [1, 3],
  );

  const { notes: withAdded, added } = addNote(sampleNotes(), {
    instrument: "acoustic_piano",
    pitch: 72,
    start: 2.0,
    duration: 0.25,
    velocity: 200,
  });
  assert.equal(withAdded.length, 4);
  assert.equal(added.velocity, 127);
  assert.equal(added.id, 4);

  const velClamped = setNotesVelocity(sampleNotes(), new Set([1]), -10);
  assert.equal(velClamped[0].velocity, 1);
});
