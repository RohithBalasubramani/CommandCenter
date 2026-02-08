// ============================================================
// Command Center — Configuration
// ============================================================

import { PersonaPlexConfig } from "@/types";

/**
 * Feature flags (per README blueprint):
 *   SPOTVOX_MODE:              "4layer" | "legacy" — controls pipeline mode
 *   ENABLE_BLOB:               true/false — whether Blob layout engine is active
 *   ENABLE_LEDGER:             true/false — whether Ledger Panel is shown
 *   ENABLE_RAG:                true/false — whether RAG pipeline runs
 *   RAG_INDUSTRIAL_ENABLED:    true/false — per-domain RAG toggle
 *   RAG_SUPPLY_ENABLED:        true/false
 *   RAG_PEOPLE_ENABLED:        true/false
 *   RAG_TASKS_ENABLED:         true/false
 *   RAG_ALERTS_ENABLED:        true/false
 *   SPOTVOX_ALWAYS_ON:         true/false — always-listening mode
 *   BLOB_TRANSITION_DURATION:  number (ms) — widget transition duration
 */
export const config = {
  api: {
    baseUrl: process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8100",
  },

  // Feature flags — enforced in code
  flags: {
    spotvoxMode: (process.env.NEXT_PUBLIC_SPOTVOX_MODE || "4layer") as "4layer" | "legacy",
    enableBlob: process.env.NEXT_PUBLIC_ENABLE_BLOB !== "false",
    enableLedger: process.env.NEXT_PUBLIC_ENABLE_LEDGER === "true",
    enableRag: process.env.NEXT_PUBLIC_ENABLE_RAG !== "false",
    ragDomains: {
      industrial: process.env.NEXT_PUBLIC_RAG_INDUSTRIAL_ENABLED !== "false",
      supply: process.env.NEXT_PUBLIC_RAG_SUPPLY_ENABLED !== "false",
      people: process.env.NEXT_PUBLIC_RAG_PEOPLE_ENABLED !== "false",
      tasks: process.env.NEXT_PUBLIC_RAG_TASKS_ENABLED !== "false",
      alerts: process.env.NEXT_PUBLIC_RAG_ALERTS_ENABLED !== "false",
    },
  },

  personaPlex: {
    serverUrl:
      process.env.NEXT_PUBLIC_PERSONAPLEX_SERVER_URL ||
      "http://localhost:8998",
    model: process.env.NEXT_PUBLIC_PERSONAPLEX_MODEL || "personaplex-7b-v1",
    voice: process.env.NEXT_PUBLIC_PERSONAPLEX_VOICE || "NATF2",
    hotkey: process.env.NEXT_PUBLIC_SPOTVOX_HOTKEY || "ctrl+space",
    alwaysOn: process.env.NEXT_PUBLIC_SPOTVOX_ALWAYS_ON === "true",
  } satisfies PersonaPlexConfig,

  stt: {
    serverUrl:
      process.env.NEXT_PUBLIC_STT_SERVER_URL || "http://localhost:8890",
  },

  tts: {
    serverUrl:
      process.env.NEXT_PUBLIC_TTS_SERVER_URL || "http://localhost:8880",
  },

  spot: {
    enabled: process.env.NEXT_PUBLIC_SPOT_ENABLED !== "false",
  },

  layer2: {
    orchestratorModel:
      process.env.NEXT_PUBLIC_AI_ORCHESTRATOR_MODEL || "phi3",
  },

  blob: {
    transitionDuration: Number(
      process.env.NEXT_PUBLIC_BLOB_TRANSITION_DURATION || "300"
    ),
  },
} as const;
