// ============================================================
// Command Center — Shared Types
// 4-Layer Architecture: Voice → AI+RAG → Blob → Widgets
// ============================================================

// --- Layer 1: Voice I/O (PersonaPlex) ---

export type PersonaPlexState =
  | "idle"
  | "connecting"
  | "listening"
  | "processing"
  | "speaking"
  | "error"
  | "disconnected";

export interface Transcript {
  id: string;
  role: "user" | "assistant";
  text: string;
  timestamp: number;
  isFinal: boolean;
}

export interface PersonaPlexConfig {
  serverUrl: string;
  model: string;
  voice: string;
  hotkey: string;
  alwaysOn: boolean;
}

// --- Spot / SpotWalk ---

export type SpotState =
  | "idle"
  | "listening"
  | "speaking"
  | "processing"
  | "success"
  | "error";

export type SpotSize = "small" | "medium" | "large";

export interface SpotParticleConfig {
  speed: number;    // 0.0 - 2.0
  density: number;  // particle count
  color: string;    // hex or gradient
  chaos: number;    // 0.0 - 1.0
}

export interface SpotWalkState {
  state: SpotState;
  size: SpotSize;
  particleConfig: SpotParticleConfig;
}

// --- Layer 2: AI + RAG ---

export type RAGDomain =
  | "industrial"
  | "supply"
  | "people"
  | "tasks"
  | "alerts";

export interface RAGQuery {
  domain: RAGDomain;
  query: string;
  context?: Record<string, unknown>;
}

export interface RAGResult {
  domain: RAGDomain;
  rawData: Record<string, unknown>;
  timestamp: number;
  error?: string;
}

export interface OrchestratorOutput {
  intent: string[];
  domains: RAGDomain[];
  spawnRag: RAGQuery[];
  skipDomains: RAGDomain[];
}

// --- Layer 3: Blob (Layout Executor) ---

export type WidgetSize = "hero" | "expanded" | "normal" | "compact" | "hidden";

export type WidgetHeightHint = "short" | "medium" | "tall" | "x-tall";

export type WidgetPosition =
  | "top-center"
  | "middle-left"
  | "middle-right"
  | "bottom"
  | null;

export type TransitionType =
  | "slide-in-from-top"
  | "slide-in-from-left"
  | "expand"
  | "shrink"
  | "fade-out";

export interface WidgetInstruction {
  scenario: string;                              // scenario slug from widget DB (e.g. "kpi", "alerts")
  fixture: string;                               // fixture variant slug (e.g. "kpi_live-standard")
  size: WidgetSize;
  heightHint?: WidgetHeightHint;                 // row-span hint: short=1, medium=2, tall=3, x-tall=4
  position: WidgetPosition;
  relevance: number;
  data_override: Record<string, unknown> | null; // real data from RAG, overrides fixture demoData
  description?: string;                          // user-facing explanation of what this widget shows
}

export interface LayoutJSON {
  heading?: string | null;
  widgets: WidgetInstruction[];
  transitions: Record<string, TransitionType>;
}

// --- Layer 4: Widget Templates ---

export type WidgetDomain =
  | "industrial"
  | "supply"
  | "people"
  | "projects"
  | "messaging"
  | "reporting";

export interface WidgetContract {
  id: string;
  domain: WidgetDomain;
  requiredCapabilities: string[];
  minSize: { width: number; height: number };
  maxSize: { width: number; height: number };
  priority: number;
}

// --- Event System ---

// --- Widget Interaction ---

export interface LayoutSnapshot {
  id: string;
  timestamp: number;
  heading: string;
  layout: LayoutJSON;
  label?: string;
}

export type CommandCenterEvent =
  | { type: "PERSONAPLEX_STATE_CHANGE"; state: PersonaPlexState }
  | { type: "TRANSCRIPT_UPDATE"; transcript: Transcript }
  | { type: "SPOT_STATE_CHANGE"; state: SpotWalkState }
  | { type: "RAG_RESULT"; result: RAGResult }
  | { type: "LAYOUT_UPDATE"; layout: LayoutJSON }
  | { type: "VOICE_INPUT_START" }
  | { type: "VOICE_INPUT_STOP" }
  | { type: "TEXT_INPUT_SUBMIT"; text: string }
  | { type: "WIDGET_PIN"; widgetKey: string }
  | { type: "WIDGET_DISMISS"; widgetKey: string }
  | { type: "WIDGET_RESIZE"; widgetKey: string; size: WidgetSize }
  | { type: "WIDGET_FOCUS"; scenario: string; label: string }
  | { type: "WIDGET_DRILL_DOWN"; scenario: string; label: string; context: string }
  | { type: "WIDGET_SNAPSHOT" };
