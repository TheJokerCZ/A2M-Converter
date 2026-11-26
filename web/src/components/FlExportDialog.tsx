import { useEffect, useState } from "react";
import clsx from "clsx";
import { Button } from "./Button";
import { IconDownload } from "./icons";
import type { TranscriptionResult } from "../hooks/useTranscription";
import type { EditableNote } from "../editor";
import {
  FL_PRESET_DEFAULTS,
  correctedGrid,
  exportFlBundle,
  exportMidi,
  followsBeats,
  isRetimed,
  saveBlob,
  type FlBundleSettings,
  type NoteCleanup,
  type TimingOverrides,
} from "../export";

const STORAGE_KEY = "flStudioExportSettings";

const GRIDS: { value: number; label: string }[] = [
  { value: 1, label: "1/4" },
  { value: 2, label: "1/8" },
  { value: 3, label: "1/8 triplets" },
  { value: 4, label: "1/16" },
  { value: 6, label: "1/16 triplets" },
  { value: 8, label: "1/32" },
];
const FALLBACK_GRID = 4;

const SELECT =
  "rounded-md border border-line-strong bg-surface-2 px-2 py-1 text-[13px] text-content outline-none focus:border-accent disabled:opacity-50";

function loadSettings(): FlBundleSettings {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) return { ...FL_PRESET_DEFAULTS, ...JSON.parse(saved) };
  } catch {
    // ignore storage errors
  }
  return FL_PRESET_DEFAULTS;
}

function saveSettings(s: FlBundleSettings) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
  } catch {
    // ignore storage errors
  }
}

export function FlExportDialog(props: {
  result: TranscriptionResult;
  notes: EditableNote[];
  mutedInstruments: ReadonlySet<string>;
  timing: TimingOverrides;
  songName: string;
  onSwitchToFixedTempo: () => void;
  onClose: () => void;
}) {
  const {
    result,
    notes,
    mutedInstruments,
    timing,
    songName,
    onSwitchToFixedTempo,
    onClose,
  } = props;
  const [settings, setSettings] = useState<FlBundleSettings>(loadSettings);
  const [busy, setBusy] = useState(false);

  const grid = correctedGrid(result.beatGrid, timing);
  const autoGrid = isRetimed(result.beatGrid, timing)
    ? null
    : (result.beatGrid?.beat_subdivision ?? null);
  const canQuantize = grid !== null;
  const quantize = settings.quantize && canQuantize;
  const hasStems = result.stems.length > 0 && result.sessionId !== null;
  const variableTempoWithStems =
    settings.format === "bundle" &&
    hasStems &&
    settings.include_stems &&
    followsBeats(result.beatGrid, timing);

  const update = (patch: Partial<FlBundleSettings>) => {
    const next = { ...settings, ...patch };
    setSettings(next);
    saveSettings(next);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  async function handleExport() {
    const cleanup: NoteCleanup = {
      quantize,
      subdivision: settings.subdivision ?? (autoGrid === null ? FALLBACK_GRID : null),
      strength: settings.strength,
      fill_gaps: settings.fill_gaps,
      trim_leading_bars: settings.trim_leading_bars,
      use_velocities: settings.use_velocities,
      velocity_scale: settings.velocity_scale,
      min_note_duration_ms: settings.min_note_duration_ms,
      min_velocity: settings.min_velocity,
      deduplicate_ms: settings.deduplicate_ms,
      split_drums: settings.split_drums,
      excluded_instruments: settings.exclude_muted ? Array.from(mutedInstruments) : [],
    };
    setBusy(true);
    try {
      if (settings.format === "bundle") {
        const zip = await exportFlBundle(notes, result.beatGrid, timing, cleanup, {
          sessionId: result.sessionId,
          includeStems: hasStems && settings.include_stems,
          normalizeDrumLanesToC5: settings.normalize_drum_lanes_to_c5,
          songName,
        });
        saveBlob(zip, `${songName}_fl_studio.zip`);
      } else {
        const midi = await exportMidi(notes, result.beatGrid, timing, cleanup, true);
        saveBlob(midi, `${songName}.mid`);
      }
      onClose();
    } catch (e) {
      alert("Export failed: " + (e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const seg = (active: boolean) =>
    clsx(
      "flex-1 rounded-md px-3 py-1.5 text-xs font-medium",
      active ? "bg-surface-2 text-content shadow-pop" : "text-muted hover:text-content",
    );

  return (
    <div
      className="fixed inset-0 z-[150] flex items-center justify-center bg-[rgba(11,12,16,0.75)] p-6 backdrop-blur-sm"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Export for FL Studio"
        className="flex max-h-[90vh] w-full max-w-lg flex-col overflow-hidden rounded-card border border-line-strong bg-surface shadow-overlay"
      >
        <div className="border-b border-line px-5 py-4">
          <h2 className="m-0 text-base font-semibold text-content">Export for FL Studio</h2>
          <p className="m-0 mt-1 text-xs text-faint">
            {grid
              ? `${grid.steady === false ? `Variable ~${grid.bpm.toFixed(0)}` : grid.bpm.toFixed(1)} BPM${grid.beats_per_bar ? `, ${grid.beats_per_bar}/4` : ""} • ${notes.length} notes`
              : `120 BPM placeholder • ${notes.length} notes`}
          </p>
        </div>

        <div className="flex flex-1 flex-col gap-5 overflow-y-auto px-5 py-4 text-sm text-muted">
          {/* Format */}
          <section>
            <h3 className="m-0 mb-2 text-[13px] font-semibold text-content">Format</h3>
            <div className="flex gap-1 rounded-lg border border-line bg-bg p-1">
              <Button
                type="button"
                kind="ghost"
                className={seg(settings.format === "bundle")}
                onClick={() => update({ format: "bundle" })}
              >
                FL Studio Bundle (.zip)
              </Button>
              <Button
                type="button"
                kind="ghost"
                className={seg(settings.format === "midi")}
                onClick={() => update({ format: "midi" })}
              >
                Single Multi-Track (.mid)
              </Button>
            </div>
            <p className="m-0 mt-1.5 text-xs text-faint">
              {settings.format === "bundle"
                ? "Includes 00_full_mix.mid, individual per-instrument MIDIs, split drum lanes, optional aligned WAV stems, and import guide."
                : "One Type 1 MIDI file with named tracks, real velocities, and your cleanup settings."}
            </p>
          </section>

          {variableTempoWithStems && (
            <div className="flex items-center justify-between gap-3 rounded-lg border border-amber-400/40 bg-amber-400/10 px-3.5 py-2.5 text-xs text-amber-200">
              <span>
                Variable tempo map is active — WAV stems align at Bar 1 but won't follow tempo drift
                automatically in FL Studio.
              </span>
              <Button
                type="button"
                pad="px-2.5 py-1 text-xs shrink-0"
                onClick={onSwitchToFixedTempo}
              >
                Use fixed tempo
              </Button>
            </div>
          )}

          {/* Timing & Quantization */}
          <section className="flex flex-col gap-2.5">
            <h3 className="m-0 text-[13px] font-semibold text-content">Timing &amp; Grid</h3>
            <label className="flex items-center justify-between gap-3">
              <span>Quantize notes onto beat grid</span>
              <input
                type="checkbox"
                className="accent-accent"
                checked={quantize}
                disabled={!canQuantize}
                onChange={(e) => update({ quantize: e.target.checked })}
              />
            </label>
            {quantize && (
              <>
                <label className="flex items-center justify-between gap-3">
                  <span>Grid resolution</span>
                  <select
                    className={SELECT}
                    value={settings.subdivision ?? ""}
                    onChange={(e) =>
                      update({ subdivision: e.target.value ? Number(e.target.value) : null })
                    }
                  >
                    {autoGrid !== null && <option value="">Auto (1/{autoGrid * 4})</option>}
                    {GRIDS.map((g) => (
                      <option key={g.value} value={g.value}>
                        {g.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="flex items-center justify-between gap-3">
                  <span>Quantize strength ({Math.round(settings.strength * 100)}%)</span>
                  <input
                    type="range"
                    min={0.1}
                    max={1}
                    step={0.05}
                    value={settings.strength}
                    onChange={(e) => update({ strength: Number(e.target.value) })}
                    className="w-32 accent-accent"
                  />
                </label>
              </>
            )}
            <label className="flex items-center justify-between gap-3">
              <span>Trim leading empty bars</span>
              <input
                type="checkbox"
                className="accent-accent"
                checked={settings.trim_leading_bars}
                onChange={(e) => update({ trim_leading_bars: e.target.checked })}
              />
            </label>
          </section>

          {/* Dynamics & Cleanup */}
          <section className="flex flex-col gap-2.5">
            <h3 className="m-0 text-[13px] font-semibold text-content">Dynamics &amp; Cleanup</h3>
            <label className="flex items-center justify-between gap-3">
              <span>Use audio-derived MIDI velocities</span>
              <input
                type="checkbox"
                className="accent-accent"
                checked={settings.use_velocities}
                onChange={(e) => update({ use_velocities: e.target.checked })}
              />
            </label>
            {settings.use_velocities && (
              <label className="flex items-center justify-between gap-3">
                <span>Dynamic contrast ({settings.velocity_scale.toFixed(1)}×)</span>
                <input
                  type="range"
                  min={0.5}
                  max={1.6}
                  step={0.1}
                  value={settings.velocity_scale}
                  onChange={(e) => update({ velocity_scale: Number(e.target.value) })}
                  className="w-32 accent-accent"
                />
              </label>
            )}
            <label className="flex items-center justify-between gap-3">
              <span>Min note length (drops micro-glitches)</span>
              <select
                className={SELECT}
                value={settings.min_note_duration_ms}
                onChange={(e) => update({ min_note_duration_ms: Number(e.target.value) })}
              >
                <option value={0}>Off (0 ms)</option>
                <option value={25}>25 ms</option>
                <option value={35}>35 ms (Recommended)</option>
                <option value={50}>50 ms</option>
              </select>
            </label>
            <label className="flex items-center justify-between gap-3">
              <span>Min velocity floor ({settings.min_velocity})</span>
              <input
                type="range"
                min={1}
                max={50}
                step={1}
                value={settings.min_velocity}
                onChange={(e) => update({ min_velocity: Number(e.target.value) })}
                className="w-32 accent-accent"
              />
            </label>
            <label className="flex items-center justify-between gap-3">
              <span>Deduplicate double-hits</span>
              <select
                className={SELECT}
                value={settings.deduplicate_ms}
                onChange={(e) => update({ deduplicate_ms: Number(e.target.value) })}
              >
                <option value={0}>Off</option>
                <option value={25}>25 ms (Recommended)</option>
                <option value={40}>40 ms</option>
              </select>
            </label>
          </section>

          {/* FL Studio Layout */}
          <section className="flex flex-col gap-2.5">
            <h3 className="m-0 text-[13px] font-semibold text-content">FL Studio Layout</h3>
            <label className="flex items-center justify-between gap-3">
              <span>Split drums into 6 lanes (Kick / Snare / Hat / Toms / Cymbals / Perc)</span>
              <input
                type="checkbox"
                className="accent-accent"
                checked={settings.split_drums}
                onChange={(e) => update({ split_drums: e.target.checked })}
              />
            </label>
            {settings.format === "bundle" && settings.split_drums && (
              <label className="flex items-center justify-between gap-3">
                <span>Normalize Kick/Snare/Hi-Hat MIDIs to C5 (FL Sampler)</span>
                <input
                  type="checkbox"
                  className="accent-accent"
                  checked={settings.normalize_drum_lanes_to_c5}
                  onChange={(e) => update({ normalize_drum_lanes_to_c5: e.target.checked })}
                />
              </label>
            )}
            <label className="flex items-center justify-between gap-3">
              <span>
                Exclude muted instruments
                {mutedInstruments.size > 0 ? ` (${mutedInstruments.size} muted)` : ""}
              </span>
              <input
                type="checkbox"
                className="accent-accent"
                checked={settings.exclude_muted}
                onChange={(e) => update({ exclude_muted: e.target.checked })}
              />
            </label>
            {settings.format === "bundle" && (
              <label className={clsx("flex items-center justify-between gap-3", !hasStems && "opacity-45")}>
                <span>Include bar-1-aligned WAV stems ({result.stems.length} available)</span>
                <input
                  type="checkbox"
                  className="accent-accent"
                  disabled={!hasStems}
                  checked={hasStems && settings.include_stems}
                  onChange={(e) => update({ include_stems: e.target.checked })}
                />
              </label>
            )}
          </section>
        </div>

        <div className="flex items-center justify-end gap-2.5 border-t border-line px-5 py-3.5">
          <Button type="button" onClick={onClose}>
            Cancel
          </Button>
          <Button
            type="button"
            kind="primary"
            className="inline-flex items-center gap-2"
            disabled={busy}
            onClick={handleExport}
          >
            <IconDownload />
            {busy
              ? "Building export…"
              : settings.format === "bundle"
                ? "Download FL Studio Bundle (.zip)"
                : "Download MIDI (.mid)"}
          </Button>
        </div>
      </div>
    </div>
  );
}
