import type { BeatGrid } from "./pianoroll";

/** Canonical note representation inside the interactive editor (audio-aligned time). */
export interface EditableNote {
  id: number;
  instrument: string;
  pitch: number;
  start: number;
  end: number;
  velocity: number;
}

const MIN_PITCH = 21;
const MAX_PITCH = 108;
const MIN_DURATION_S = 0.02;

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function cloneNotes(notes: readonly EditableNote[]): EditableNote[] {
  return notes.map((n) => ({ ...n }));
}

export class EditorHistory {
  private readonly maxDepth: number;
  private past: EditableNote[][] = [];
  private future: EditableNote[][] = [];

  constructor(maxDepth = 50) {
    this.maxDepth = Math.max(1, maxDepth);
  }

  get canUndo(): boolean {
    return this.past.length > 0;
  }

  get canRedo(): boolean {
    return this.future.length > 0;
  }

  get undoDepth(): number {
    return this.past.length;
  }

  clear(): void {
    this.past = [];
    this.future = [];
  }

  push(snapshot: readonly EditableNote[]): void {
    this.past.push(cloneNotes(snapshot));
    if (this.past.length > this.maxDepth) {
      this.past.shift();
    }
    this.future = [];
  }

  undo(current: readonly EditableNote[]): EditableNote[] | null {
    const prev = this.past.pop();
    if (!prev) return null;
    this.future.push(cloneNotes(current));
    return cloneNotes(prev);
  }

  redo(current: readonly EditableNote[]): EditableNote[] | null {
    const next = this.future.pop();
    if (!next) return null;
    this.past.push(cloneNotes(current));
    return cloneNotes(next);
  }
}

export function deleteNotes(
  notes: readonly EditableNote[],
  ids: ReadonlySet<number>,
): EditableNote[] {
  return notes.filter((n) => !ids.has(n.id)).map((n) => ({ ...n }));
}

export function moveNotes(
  notes: readonly EditableNote[],
  ids: ReadonlySet<number>,
  deltaSeconds: number,
  deltaPitch: number,
): EditableNote[] {
  return notes.map((n) => {
    if (!ids.has(n.id)) return { ...n };
    const dur = Math.max(MIN_DURATION_S, n.end - n.start);
    const start = Math.max(0, n.start + deltaSeconds);
    const pitch = clamp(Math.round(n.pitch + deltaPitch), MIN_PITCH, MAX_PITCH);
    return { ...n, start, end: start + dur, pitch };
  });
}

export function resizeNote(
  notes: readonly EditableNote[],
  id: number,
  newEndSeconds: number,
): EditableNote[] {
  return notes.map((n) => {
    if (n.id !== id) return { ...n };
    return { ...n, end: Math.max(n.start + MIN_DURATION_S, newEndSeconds) };
  });
}

export function addNote(
  notes: readonly EditableNote[],
  spec: {
    instrument: string;
    pitch: number;
    start: number;
    duration?: number;
    velocity?: number;
  },
): { notes: EditableNote[]; added: EditableNote } {
  let maxId = 0;
  for (const n of notes) if (n.id > maxId) maxId = n.id;
  const start = Math.max(0, spec.start);
  const dur = Math.max(MIN_DURATION_S, spec.duration ?? 0.25);
  const added: EditableNote = {
    id: maxId + 1,
    instrument: spec.instrument,
    pitch: clamp(Math.round(spec.pitch), MIN_PITCH, MAX_PITCH),
    start,
    end: start + dur,
    velocity: clamp(Math.round(spec.velocity ?? 100), 1, 127),
  };
  return { notes: [...cloneNotes(notes), added], added };
}

export function setNotesVelocity(
  notes: readonly EditableNote[],
  ids: ReadonlySet<number>,
  velocity: number,
): EditableNote[] {
  const vel = clamp(Math.round(velocity), 1, 127);
  return notes.map((n) => (ids.has(n.id) ? { ...n, velocity: vel } : { ...n }));
}

export function mergeInstrument(
  notes: readonly EditableNote[],
  sourceInstrument: string,
  targetInstrument: string,
): EditableNote[] {
  if (sourceInstrument === targetInstrument) return cloneNotes(notes);
  return notes.map((n) =>
    n.instrument === sourceInstrument ? { ...n, instrument: targetInstrument } : { ...n },
  );
}

/**
 * Convert canonical audio-aligned `EditableNote[]` into backend `/export` /
 * `/export/fl-bundle` payload. Because editable notes have ALREADY been shifted
 * by `-onset_delay`, `beatGrid.onset_delay` is explicitly zeroed so the backend
 * never double-subtracts the model lag.
 */
export function toExportPayload(
  notes: readonly EditableNote[],
  beatGrid: BeatGrid | null,
): {
  notes: Array<{
    instrument: string;
    pitch: number;
    start: number;
    end: number;
    velocity: number;
  }>;
  beatGrid: BeatGrid | null;
} {
  return {
    notes: notes.map((n) => ({
      instrument: n.instrument,
      pitch: n.pitch,
      start: n.start,
      end: n.end,
      velocity: clamp(Math.round(n.velocity), 1, 127),
    })),
    beatGrid: beatGrid ? { ...beatGrid, onset_delay: 0 } : null,
  };
}
