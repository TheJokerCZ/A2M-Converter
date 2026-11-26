export type SeparationMode = "none" | "stems_4" | "stems_6";
export type ModelSize = "small" | "medium" | "large";

export interface ServerCapabilities {
  device: string;
  gpu_name: string | null;
  vram_mb: number | null;
  loaded_model: string | null;
  default_model: string;
  custom_model_locked: boolean;
  available_models: ModelSize[];
  hf_auth_configured: boolean;
  separation_available: boolean;
  separation_modes: SeparationMode[];
}

export interface StudioConfig {
  separation: SeparationMode;
  stems: string[];
  modelSize: ModelSize;
  beamSize: 1 | 2;
}

export const STEMS_FOR_MODE: Record<SeparationMode, string[]> = {
  none: [],
  stems_4: ["drums", "bass", "vocals", "other"],
  stems_6: ["drums", "bass", "vocals", "guitar", "piano", "other"],
};

export const STEM_LABELS: Record<string, string> = {
  drums: "Drums",
  bass: "Bass",
  vocals: "Vocals",
  guitar: "Guitar",
  piano: "Piano",
  other: "Other (Synths / Keys / Strings)",
};

export const DEFAULT_STUDIO_CONFIG: StudioConfig = {
  separation: "stems_4",
  stems: [...STEMS_FOR_MODE.stems_4],
  modelSize: "medium",
  beamSize: 1,
};

export async function fetchCapabilities(): Promise<ServerCapabilities | null> {
  try {
    const resp = await fetch("/capabilities");
    if (!resp.ok) return null;
    return (await resp.json()) as ServerCapabilities;
  } catch {
    return null;
  }
}
