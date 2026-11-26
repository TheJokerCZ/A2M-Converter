import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type RefObject,
  type SetStateAction,
} from "react";
import clsx from "clsx";
import type { AudioEngine } from "../audio";
import { PianoRoll, KEY_WIDTH } from "../pianoroll";
import { Button } from "./Button";
import { label } from "../instruments";
import {
  addNote,
  deleteNotes,
  moveNotes,
  resizeNote,
  setNotesVelocity,
  type EditableNote,
} from "../editor";

export type RollMode = "scrub" | "edit";

export function PianoRollCanvas(props: {
  rollRef: RefObject<PianoRoll | null>;
  audio: AudioEngine;
  setUserScrolled: Dispatch<SetStateAction<boolean>>;
  editable?: boolean;
  notes?: EditableNote[];
  instruments?: string[];
  selectedIds?: Set<number>;
  onSelectIds?: (ids: Set<number>) => void;
  /** Commit an edit (pushes previous snapshot onto Undo stack). */
  onCommitNotes?: (next: EditableNote[], pushHistory?: boolean) => void;
  /** Preview notes live during drag without pushing a history entry per pixel. */
  onPreviewNotes?: (next: EditableNote[]) => void;
  canUndo?: boolean;
  canRedo?: boolean;
  onUndo?: () => void;
  onRedo?: () => void;
}) {
  const {
    rollRef,
    audio,
    setUserScrolled,
    editable = false,
    notes = [],
    instruments = [],
    selectedIds = new Set<number>(),
    onSelectIds,
    onCommitNotes,
    onPreviewNotes,
    canUndo = false,
    canRedo = false,
    onUndo,
    onRedo,
  } = props;

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [mode, setMode] = useState<RollMode>("scrub");
  const [activeInstrument, setActiveInstrument] = useState<string>("");

  // Keep refs to latest props so canvas event handlers never go stale without re-mounting.
  const modeRef = useRef<RollMode>(mode);
  modeRef.current = editable ? mode : "scrub";
  const notesRef = useRef<EditableNote[]>(notes);
  notesRef.current = notes;
  const selectedRef = useRef<Set<number>>(selectedIds);
  selectedRef.current = selectedIds;
  const activeInstRef = useRef<string>(activeInstrument);
  activeInstRef.current = activeInstrument || instruments[0] || "acoustic_piano";
  const callbacksRef = useRef({
    onSelectIds,
    onCommitNotes,
    onPreviewNotes,
    onUndo,
    onRedo,
  });
  callbacksRef.current = { onSelectIds, onCommitNotes, onPreviewNotes, onUndo, onRedo };

  // Default active instrument to the first detected instrument whenever instruments change.
  useEffect(() => {
    if (instruments.length > 0 && !instruments.includes(activeInstrument)) {
      setActiveInstrument(instruments[0]);
    }
  }, [instruments, activeInstrument]);

  // Keep canvas selection highlight synced with React state.
  useEffect(() => {
    rollRef.current?.setSelectedIds(selectedIds);
  }, [rollRef, selectedIds]);

  // Reset to scrub mode if editing becomes unavailable (new run started).
  useEffect(() => {
    if (!editable) setMode("scrub");
  }, [editable]);

  // Average velocity of currently selected notes for the velocity slider.
  const selectedVelocity = useMemo(() => {
    if (selectedIds.size === 0) return 100;
    let sum = 0;
    let count = 0;
    for (const n of notes) {
      if (selectedIds.has(n.id)) {
        sum += n.velocity;
        count++;
      }
    }
    return count > 0 ? Math.round(sum / count) : 100;
  }, [notes, selectedIds]);

  // Keyboard shortcuts for Edit mode (Delete, Pitch nudge, Undo/Redo).
  useEffect(() => {
    if (!editable) return;
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;

      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.key.toLowerCase() === "z" && !e.shiftKey) {
        e.preventDefault();
        callbacksRef.current.onUndo?.();
        return;
      }
      if ((mod && e.key.toLowerCase() === "y") || (mod && e.shiftKey && e.key.toLowerCase() === "z")) {
        e.preventDefault();
        callbacksRef.current.onRedo?.();
        return;
      }

      if (modeRef.current !== "edit") return;
      const sel = selectedRef.current;
      if (sel.size === 0) return;

      if (e.key === "Delete" || e.key === "Backspace") {
        e.preventDefault();
        const next = deleteNotes(notesRef.current, sel);
        callbacksRef.current.onCommitNotes?.(next, true);
        callbacksRef.current.onSelectIds?.(new Set());
      } else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        e.preventDefault();
        const step = (e.shiftKey ? 12 : 1) * (e.key === "ArrowUp" ? 1 : -1);
        const next = moveNotes(notesRef.current, sel, 0, step);
        callbacksRef.current.onCommitNotes?.(next, true);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [editable]);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const roll = new PianoRoll(canvas);
    rollRef.current = roll;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const factor = Math.exp(-e.deltaY * 0.01);
      if (e.altKey) {
        roll.zoomPitch(factor, e.clientY - rect.top);
        setUserScrolled(true);
        return;
      }
      if (e.ctrlKey || e.metaKey) {
        roll.zoomTime(factor, e.clientX - rect.left);
        setUserScrolled(true);
        return;
      }
      if (e.shiftKey) {
        const d = e.deltaX !== 0 ? e.deltaX : e.deltaY;
        if (d !== 0) roll.scrollBy(d / roll.pxPerSec);
      } else {
        if (e.deltaX !== 0) roll.scrollBy(e.deltaX / roll.pxPerSec);
        if (e.deltaY !== 0) roll.scrollPitchBy(-e.deltaY);
      }
      setUserScrolled(true);
    };

    let scrub: { wasPlaying: boolean } | null = null;
    const scrubTo = (clientX: number) => {
      const rect = canvas.getBoundingClientRect();
      const t =
        clientX < rect.left ? 0 : Math.max(0, roll.xToSeconds(clientX - rect.left));
      audio.scrubTo(t);
      roll.setPlayhead(t);
    };

    let keyDrag: { x: number; y: number } | null = null;
    let editDrag:
      | {
          kind: "move";
          originNotes: EditableNote[];
          ids: Set<number>;
          startSec: number;
          startPitch: number;
          moved: boolean;
        }
      | {
          kind: "resize";
          originNotes: EditableNote[];
          id: number;
          moved: boolean;
        }
      | {
          kind: "marquee";
          x0: number;
          y0: number;
          shift: boolean;
          baseSelected: Set<number>;
        }
      | null = null;

    const onMouseDown = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      if (x < KEY_WIDTH) {
        keyDrag = { x, y };
        e.preventDefault();
        return;
      }

      if (modeRef.current === "edit") {
        const hit = roll.hitTestNote(x, y);
        if (hit) {
          let nextSel = new Set(selectedRef.current);
          if (e.shiftKey) {
            if (nextSel.has(hit.id)) nextSel.delete(hit.id);
            else nextSel.add(hit.id);
          } else if (!nextSel.has(hit.id)) {
            nextSel = new Set([hit.id]);
          }
          callbacksRef.current.onSelectIds?.(nextSel);

          if (hit.edge === "right") {
            editDrag = {
              kind: "resize",
              originNotes: notesRef.current.map((n) => ({ ...n })),
              id: hit.id,
              moved: false,
            };
          } else if (nextSel.has(hit.id)) {
            editDrag = {
              kind: "move",
              originNotes: notesRef.current.map((n) => ({ ...n })),
              ids: nextSel,
              startSec: roll.xToSeconds(x),
              startPitch: roll.yToPitch(y),
              moved: false,
            };
          }
        } else {
          const base = e.shiftKey ? new Set(selectedRef.current) : new Set<number>();
          if (!e.shiftKey) callbacksRef.current.onSelectIds?.(new Set());
          editDrag = { kind: "marquee", x0: x, y0: y, shift: e.shiftKey, baseSelected: base };
          roll.setMarquee({ x0: x, y0: y, x1: x, y1: y });
        }
        e.preventDefault();
        return;
      }

      scrub = { wasPlaying: audio.state === "started" };
      if (scrub.wasPlaying) audio.pause();
      scrubTo(e.clientX);
      e.preventDefault();
    };

    const onDoubleClick = (e: MouseEvent) => {
      if (modeRef.current !== "edit") return;
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      if (x < KEY_WIDTH) return;
      if (roll.hitTestNote(x, y)) return;

      const start = roll.xToSeconds(x);
      const pitch = roll.yToPitch(y);
      const { notes: next, added } = addNote(notesRef.current, {
        instrument: activeInstRef.current,
        pitch,
        start,
        duration: 0.25,
        velocity: 100,
      });
      callbacksRef.current.onCommitNotes?.(next, true);
      callbacksRef.current.onSelectIds?.(new Set([added.id]));
    };

    const onMouseMove = (e: MouseEvent) => {
      if (scrub) {
        scrubTo(e.clientX);
        return;
      }
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      if (keyDrag) {
        const dx = x - keyDrag.x;
        const dy = y - keyDrag.y;
        if (dx !== 0) roll.zoomPitch(Math.exp(dx * 0.01), y);
        if (dy !== 0) roll.scrollPitchBy(dy);
        keyDrag = { x, y };
        setUserScrolled(true);
        return;
      }

      if (editDrag) {
        if (editDrag.kind === "move") {
          const dt = roll.xToSeconds(x) - editDrag.startSec;
          const dp = roll.yToPitch(y) - editDrag.startPitch;
          if (Math.abs(dt) > 0.005 || dp !== 0) {
            editDrag.moved = true;
            const preview = moveNotes(editDrag.originNotes, editDrag.ids, dt, dp);
            callbacksRef.current.onPreviewNotes?.(preview);
          }
        } else if (editDrag.kind === "resize") {
          editDrag.moved = true;
          const preview = resizeNote(editDrag.originNotes, editDrag.id, roll.xToSeconds(x));
          callbacksRef.current.onPreviewNotes?.(preview);
        } else if (editDrag.kind === "marquee") {
          roll.setMarquee({ x0: editDrag.x0, y0: editDrag.y0, x1: x, y1: y });
          const inside = roll.notesInRect(editDrag.x0, editDrag.y0, x, y);
          const merged = new Set(editDrag.baseSelected);
          for (const id of inside) merged.add(id);
          callbacksRef.current.onSelectIds?.(merged);
        }
        return;
      }

      if (x < KEY_WIDTH) {
        canvas.style.cursor = "ew-resize";
      } else if (modeRef.current === "edit") {
        const hit = roll.hitTestNote(x, y);
        canvas.style.cursor =
          hit?.edge === "right" ? "ew-resize" : hit ? "grab" : "crosshair";
      } else {
        canvas.style.cursor = "default";
      }
    };

    const onMouseUp = () => {
      if (scrub) {
        audio.seek(audio.seconds);
        if (scrub.wasPlaying) audio.play();
        scrub = null;
      }
      if (editDrag) {
        if (editDrag.kind === "marquee") {
          roll.setMarquee(null);
        } else if (editDrag.moved) {
          // Commit the already-previewed notes while pushing originNotes onto Undo stack.
          callbacksRef.current.onCommitNotes?.(notesRef.current, true);
        }
        editDrag = null;
      }
      keyDrag = null;
    };

    let pan: { x: number; y: number } | null = null;
    let pinch: { dx: number; dy: number } | null = null;
    const pinchSpan = (t: TouchList) => ({
      dx: Math.abs(t[0].clientX - t[1].clientX),
      dy: Math.abs(t[0].clientY - t[1].clientY),
    });

    const onTouchStart = (e: TouchEvent) => {
      if (e.touches.length === 2) {
        pinch = pinchSpan(e.touches);
        pan = null;
        e.preventDefault();
      } else if (e.touches.length === 1) {
        pan = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      }
    };

    const onTouchMove = (e: TouchEvent) => {
      const rect = canvas.getBoundingClientRect();
      if (pinch && e.touches.length === 2) {
        e.preventDefault();
        const span = pinchSpan(e.touches);
        const cx = (e.touches[0].clientX + e.touches[1].clientX) / 2 - rect.left;
        const cy = (e.touches[0].clientY + e.touches[1].clientY) / 2 - rect.top;
        const MIN = 12;
        if (pinch.dx > MIN && span.dx > MIN) roll.zoomTime(span.dx / pinch.dx, cx);
        if (pinch.dy > MIN && span.dy > MIN) roll.zoomPitch(span.dy / pinch.dy, cy);
        pinch = span;
        setUserScrolled(true);
      } else if (pan && e.touches.length === 1) {
        e.preventDefault();
        const t = e.touches[0];
        roll.scrollBy(-(t.clientX - pan.x) / roll.pxPerSec);
        roll.scrollPitchBy(t.clientY - pan.y);
        pan = { x: t.clientX, y: t.clientY };
        setUserScrolled(true);
      }
    };

    const onTouchEnd = (e: TouchEvent) => {
      if (e.touches.length < 2) pinch = null;
      if (e.touches.length === 0) pan = null;
    };

    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("touchstart", onTouchStart, { passive: false });
    canvas.addEventListener("touchmove", onTouchMove, { passive: false });
    canvas.addEventListener("touchend", onTouchEnd, { passive: false });
    canvas.addEventListener("mousedown", onMouseDown);
    canvas.addEventListener("dblclick", onDoubleClick);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    return () => {
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("touchstart", onTouchStart);
      canvas.removeEventListener("touchmove", onTouchMove);
      canvas.removeEventListener("touchend", onTouchEnd);
      canvas.removeEventListener("mousedown", onMouseDown);
      canvas.removeEventListener("dblclick", onDoubleClick);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, [audio, rollRef, setUserScrolled]);

  const seg = (active: boolean) =>
    clsx(
      "rounded px-2.5 py-1 text-xs font-medium",
      active ? "bg-surface-2 text-content shadow-pop" : "text-muted hover:text-content",
    );

  return (
    <section className="relative col-start-1 overflow-hidden rounded-card border border-line bg-[linear-gradient(180deg,rgba(255,255,255,0.02),transparent_60px),#0a0b0e] p-0 shadow-canvas animate-rise [animation-delay:0.12s]">
      {editable && (
        <div className="flex flex-wrap items-center gap-2 border-b border-line bg-surface/90 px-3 py-2 text-xs text-muted">
          <div className="flex gap-1 rounded-md border border-line bg-bg p-0.5">
            <Button
              type="button"
              kind="ghost"
              className={seg(mode === "scrub")}
              onClick={() => setMode("scrub")}
            >
              Scrub
            </Button>
            <Button
              type="button"
              kind="ghost"
              className={seg(mode === "edit")}
              title="Click/box-select notes, drag to move or resize edge, double-click to add"
              onClick={() => setMode("edit")}
            >
              Edit notes
            </Button>
          </div>

          {mode === "edit" && instruments.length > 0 && (
            <label className="inline-flex items-center gap-1.5">
              <span className="text-faint">Add as:</span>
              <select
                aria-label="Instrument for new notes"
                className="rounded border border-line-strong bg-surface-2 px-1.5 py-0.5 text-xs text-content outline-none focus:border-accent"
                value={activeInstrument || instruments[0]}
                onChange={(e) => setActiveInstrument(e.target.value)}
              >
                {instruments.map((inst) => (
                  <option key={inst} value={inst}>
                    {label(inst)}
                  </option>
                ))}
              </select>
            </label>
          )}

          {mode === "edit" && selectedIds.size > 0 && (
            <div className="flex flex-wrap items-center gap-2 border-l border-line pl-2">
              <span className="font-mono text-[11px] text-content">
                {selectedIds.size} selected
              </span>
              <label className="inline-flex items-center gap-1.5">
                <span className="text-faint">Vel</span>
                <input
                  type="range"
                  min={1}
                  max={127}
                  value={selectedVelocity}
                  onChange={(e) => {
                    const vel = Number(e.target.value);
                    onPreviewNotes?.(setNotesVelocity(notes, selectedIds, vel));
                  }}
                  onPointerUp={() => onCommitNotes?.(notesRef.current, true)}
                  className="w-20 accent-accent"
                />
                <span className="w-6 font-mono text-[11px] tabular-nums text-content">
                  {selectedVelocity}
                </span>
              </label>
              <Button
                type="button"
                pad="px-1.5 py-0.5 text-[11px]"
                title="Transpose selected down 1 octave (Shift+Down)"
                onClick={() => onCommitNotes?.(moveNotes(notes, selectedIds, 0, -12), true)}
              >
                -12
              </Button>
              <Button
                type="button"
                pad="px-1.5 py-0.5 text-[11px]"
                title="Transpose selected up 1 octave (Shift+Up)"
                onClick={() => onCommitNotes?.(moveNotes(notes, selectedIds, 0, 12), true)}
              >
                +12
              </Button>
              <Button
                type="button"
                pad="px-2 py-0.5 text-[11px]"
                className="border-red/40 text-red hover:bg-red/10"
                title="Delete selected notes (Del)"
                onClick={() => {
                  onCommitNotes?.(deleteNotes(notes, selectedIds), true);
                  onSelectIds?.(new Set());
                }}
              >
                Delete
              </Button>
            </div>
          )}

          <div className="ml-auto flex items-center gap-1.5">
            {mode === "edit" && selectedIds.size === 0 && (
              <span className="hidden text-[11px] text-faint lg:inline">
                Drag to box-select • Double-click to add • Del to delete
              </span>
            )}
            <Button
              type="button"
              pad="px-2 py-0.5 text-[11px]"
              disabled={!canUndo}
              title="Undo (Ctrl+Z)"
              onClick={() => onUndo?.()}
            >
              Undo
            </Button>
            <Button
              type="button"
              pad="px-2 py-0.5 text-[11px]"
              disabled={!canRedo}
              title="Redo (Ctrl+Y)"
              onClick={() => onRedo?.()}
            >
              Redo
            </Button>
          </div>
        </div>
      )}
      <canvas className="block h-[420px] w-full" width={1200} height={400} ref={canvasRef} />
    </section>
  );
}
