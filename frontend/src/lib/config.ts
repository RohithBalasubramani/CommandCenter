// ============================================================
// Command Center — Configuration
// ============================================================

import { PersonaPlexConfig } from "@/types";

export const config = {
  api: {
    baseUrl: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8100",
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
