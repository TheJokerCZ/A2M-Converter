import clsx from "clsx";
import { Button } from "./Button";
import {
  STEMS_FOR_MODE,
  STEM_LABELS,
  type ModelSize,
  type SeparationMode,
  type ServerCapabilities,
  type StudioConfig,
} from "../capabilities";

const MODEL_META: Record<ModelSize, { label: string; hint: string }> = {
  small: { label: "Small (103M)", hint: "Fastest" },
  medium: { label: "Medium (307M)", hint: "Balanced" },
  large: { label: "Large (1.4B)", hint: "Best quality • bf16/fp16 on 8 GB GPU" },
};

export function StudioSettingsPanel(props: {
  capabilities: ServerCapabilities | null;
  config: StudioConfig;
  onChange: (next: StudioConfig) => void;
}) {
  const { capabilities, config, onChange } = props;
  const sepAvailable = capabilities?.separation_available ?? false;
  const modeStems = STEMS_FOR_MODE[config.separation];

  function setSeparation(mode: SeparationMode) {
    const nextStems = STEMS_FOR_MODE[mode];
    // Preserve previous stem toggles where names overlap; default newly added stems on.
    const prevSet = new Set(config.stems);
    const merged =
      config.separation === "none"
        ? [...nextStems]
        : nextStems.filter((s) => prevSet.has(s) || !STEMS_FOR_MODE[config.separation].includes(s));
    onChange({
      ...config,
      separation: mode,
      stems: merged.length > 0 ? merged : [...nextStems],
    });
  }

  function toggleStem(stem: string) {
    const has = config.stems.includes(stem);
    if (has && config.stems.length <= 1) return; // keep at least one stem active
    const next = has
      ? config.stems.filter((s) => s !== stem)
      : modeStems.filter((s) => s === stem || config.stems.includes(s));
    onChange({ ...config, stems: next });
  }

  const seg = (active: boolean) =>
    clsx(
      "flex-1 rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
      active ? "bg-surface-2 text-content shadow-pop" : "text-muted hover:text-content",
    );

  return (
    <section className="card col-span-full flex flex-col gap-4 px-5 pb-5 pt-4 animate-rise [animation-delay:0.18s]">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="m-0 text-lg font-semibold">Studio Pipeline (FL Studio MIDI)</h2>
          <p className="m-0 mt-0.5 text-xs text-faint">
            Separates stems before transcribing and estimates real per-note MIDI velocity from the audio.
          </p>
        </div>
        {capabilities && (
          <span className="rounded-md border border-line bg-bg px-2.5 py-1 font-mono text-[11px] text-muted">
            {capabilities.gpu_name
              ? `${capabilities.gpu_name}${capabilities.vram_mb ? ` • ${(capabilities.vram_mb / 1024).toFixed(0)} GB` : ""}`
              : capabilities.device.toUpperCase()}
          </span>
        )}
      </div>

      {capabilities && !capabilities.hf_auth_configured && (
        <div className="rounded-lg border border-amber-400/40 bg-amber-400/10 px-3.5 py-2.5 text-xs leading-relaxed text-amber-200">
          <strong className="font-semibold">HuggingFace login needed for model weights:</strong>{" "}
          Accept the CC BY-NC 4.0 license on{" "}
          <a
            href="https://huggingface.co/MuScriptor"
            target="_blank"
            rel="noreferrer"
            className="underline"
          >
            huggingface.co/MuScriptor
          </a>{" "}
          and set <code className="font-mono">HF_TOKEN</code> or run{" "}
          <code className="font-mono">hf auth login</code> before transcribing.
        </div>
      )}

      {/* Separation mode */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-muted">
            Stem separation
          </span>
          {!sepAvailable && capabilities !== null && (
            <span className="text-xs text-faint">
              Install <code className="font-mono">demucs-infer</code> to enable stem separation
            </span>
          )}
        </div>
        <div className="flex gap-1 rounded-lg border border-line bg-bg p-1">
          <Button
            type="button"
            kind="ghost"
            className={clsx(seg(config.separation === "stems_4"), !sepAvailable && "opacity-40")}
            disabled={!sepAvailable}
            title="htdemucs_ft (Drums, Bass, Vocals, Other) — highest separation clarity"
            onClick={() => setSeparation("stems_4")}
          >
            4 Stems (Recommended)
          </Button>
          <Button
            type="button"
            kind="ghost"
            className={clsx(seg(config.separation === "stems_6"), !sepAvailable && "opacity-40")}
            disabled={!sepAvailable}
            title="htdemucs_6s (adds dedicated Guitar & Piano stems)"
            onClick={() => setSeparation("stems_6")}
          >
            6 Stems (+ Guitar / Piano)
          </Button>
          <Button
            type="button"
            kind="ghost"
            className={seg(config.separation === "none")}
            title="Single-pass full mix transcription (fastest)"
            onClick={() => setSeparation("none")}
          >
            Full Mix (Fast)
          </Button>
        </div>

        {config.separation !== "none" && (
          <div className="mt-1 flex flex-wrap items-center gap-2">
            {modeStems.map((stem) => {
              const on = config.stems.includes(stem);
              return (
                <button
                  key={stem}
                  type="button"
                  onClick={() => toggleStem(stem)}
                  aria-pressed={on}
                  className={clsx(
                    "cursor-pointer rounded-md border px-2.5 py-1 text-xs font-medium transition-colors",
                    on
                      ? "border-accent-glow bg-accent-soft text-content"
                      : "border-line bg-bg text-faint hover:text-muted",
                  )}
                >
                  {STEM_LABELS[stem] ?? stem}
                </button>
              );
            })}
            <button
              type="button"
              onClick={() =>
                onChange({
                  ...config,
                  stems: modeStems.filter((s) => s !== "vocals"),
                })
              }
              className="ml-auto cursor-pointer text-xs text-muted underline underline-offset-4 hover:text-content"
            >
              Instrumental only (skip vocals)
            </button>
          </div>
        )}
      </div>

      {/* Model & Decoding */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-semibold uppercase tracking-wider text-muted">
            MuScriptor model
          </span>
          {capabilities?.custom_model_locked ? (
            <div className="rounded-lg border border-line bg-bg px-3 py-2 font-mono text-xs text-muted">
              Custom checkpoint: {capabilities.default_model}
            </div>
          ) : (
            <div className="flex gap-1 rounded-lg border border-line bg-bg p-1">
              {(capabilities?.available_models ?? (["small", "medium", "large"] as ModelSize[])).map(
                (sz) => (
                  <Button
                    key={sz}
                    type="button"
                    kind="ghost"
                    className={seg(config.modelSize === sz)}
                    title={MODEL_META[sz].hint}
                    onClick={() => onChange({ ...config, modelSize: sz })}
                  >
                    {sz === "small" ? "Small" : sz === "medium" ? "Medium" : "Large 1.4B"}
                  </Button>
                ),
              )}
            </div>
          )}
          <span className="text-[11px] text-faint">{MODEL_META[config.modelSize].hint}</span>
        </div>

        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-semibold uppercase tracking-wider text-muted">
            Decoding quality
          </span>
          <div className="flex gap-1 rounded-lg border border-line bg-bg p-1">
            <Button
              type="button"
              kind="ghost"
              className={seg(config.beamSize === 1)}
              onClick={() => onChange({ ...config, beamSize: 1 })}
            >
              Standard (Fast)
            </Button>
            <Button
              type="button"
              kind="ghost"
              className={seg(config.beamSize === 2)}
              title="Beam width 2 — slightly slower, can improve dense chords"
              onClick={() => onChange({ ...config, beamSize: 2 })}
            >
              High (Beam 2)
            </Button>
          </div>
          <span className="text-[11px] text-faint">
            {config.beamSize === 2
              ? "Explores top-2 token paths (~1.8× slower)"
              : "Greedy decoding with prelude forcing"}
          </span>
        </div>
      </div>
    </section>
  );
}
