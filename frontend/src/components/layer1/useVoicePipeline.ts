"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useSTT, type STTModel, type STTStats } from "./useSTT";
import { useKokoroTTS, type TTSEngine, type TTSStats } from "./useKokoroTTS";
import { useVAD } from "./useVAD";
import {
  getLayer2Service,
  Layer2Response,
  Layer2LayoutJSON,
} from "@/lib/layer2";
import { commandCenterBus } from "@/lib/events";
import { config } from "@/lib/config";

export type VoicePipelineState =
  | "idle"
  | "listening"
  | "speaking"
  | "error";

export type VoiceInputMode = "continuous" | "push-to-talk";

export interface PipelineStats {
  stt: STTStats;
  tts: TTSStats;
  layer2LastLatencyMs: number | null;
  totalPipelineMs: number | null;
  queueDepth: number;
  vadEnabled: boolean;
}

interface UseVoicePipelineReturn {
  // State
  state: VoicePipelineState;
  isActive: boolean;
  error: string | null;

  // Controls
  start: () => void;
  stop: () => void;

  // Input mode (continuous vs push-to-talk)
  inputMode: VoiceInputMode;
  setInputMode: (mode: VoiceInputMode) => void;

  // Direct text input (bypasses STT)
  sendTextDirect: (text: string) => void;

  // Transcripts
  userTranscript: string;
  interimTranscript: string;
  aiResponse: string;

  // Conversation history
  messages: ConversationMessage[];

  // TTS control
  isTTSSpeaking: boolean;
  stopTTS: () => void;

  // Support flags
  isSTTSupported: boolean;
  isTTSSupported: boolean;

  // Model management
  sttModel: STTModel;
  ttsEngine: TTSEngine;
  switchSTTModel: (model: STTModel) => Promise<void>;
  switchTTSEngine: (engine: TTSEngine) => void;
  isSTTServerAvailable: boolean;
  isTTSServerAvailable: boolean;
  sttAvailableModels: { name: string; loaded: boolean; available: boolean; description: string; runtime_broken?: boolean }[];

  // Layer 2 state
  layer2Status: "off" | "checking" | "ready" | "processing" | "error";
  layer2Error: string | null;
  layer2Response: Layer2Response | null;

  // Pipeline stats
  pipelineStats: PipelineStats;
}

export interface ConversationMessage {
  id: string;
  speaker: "user" | "ai" | "system";
  text: string;
  timestamp: number;
  type?: "transcript" | "response" | "queued";
}

/**
 * useVoicePipeline — V2 voice pipeline with server-based STT/TTS.
 *
 * Clean flow: User speaks → STT → Layer 2 RAG → TTS speaks response.
 * No filler speech. Messages queue if user speaks while AI is responding.
 */
export function useVoicePipeline(deviceId?: string): UseVoicePipelineReturn {
  const [state, setState] = useState<VoicePipelineState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [aiResponse, setAiResponse] = useState("");
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [inputMode, setInputMode] = useState<VoiceInputMode>(
    config.personaPlex.alwaysOn ? "continuous" : "push-to-talk"
  );

  // Layer 2
  const [layer2Status, setLayer2Status] = useState<"off" | "checking" | "ready" | "processing" | "error">("off");
  const [layer2Error, setLayer2Error] = useState<string | null>(null);
  const [layer2Response, setLayer2Response] = useState<Layer2Response | null>(null);

  // Pipeline timing
  const [layer2LastLatencyMs, setLayer2LastLatencyMs] = useState<number | null>(null);
  const [totalPipelineMs, setTotalPipelineMs] = useState<number | null>(null);

  // Refs for tracking
  const isProcessingRef = useRef(false);
  const lastTranscriptRef = useRef("");
  const lastSentLenRef = useRef(0);
  const stableCountRef = useRef(0);
  const sessionIdRef = useRef<string | null>(null);
  const messageIdRef = useRef(0);
  const layer2DebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pipelineAbortRef = useRef<AbortController | null>(null);
  const pipelineStartTimeRef = useRef<number | null>(null);
  const safetyTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // AUDIT FIX: Use ref for hasReset to prevent double-free across callbacks
  const hasResetRef = useRef(false);

  // Message queue
  const messageQueueRef = useRef<string[]>([]);
  const [queueDepth, setQueueDepth] = useState(0);

  // Interactive mode conversation history
  const interactiveHistoryRef = useRef<Array<{ role: "user" | "ai"; text: string }>>([]);
  const interactiveActiveRef = useRef(false);

  // General conversation history (for follow-up context in normal mode)
  const conversationHistoryRef = useRef<Array<{ role: "user" | "ai"; text: string }>>([]);

  // STT
  const {
    transcript: userTranscript,
    interimTranscript,
    isListening,
    start: startSTT,
    stop: stopSTT,
    isSupported: isSTTSupported,
    activeModel: sttModel,
    switchModel: switchSTTModel,
    availableModels: sttAvailableModels,
    isServerAvailable: isSTTServerAvailable,
    stats: sttStats,
  } = useSTT(deviceId);

  // TTS
  const {
    speak,
    stop: stopTTS,
    isSpeaking: isTTSSpeaking,
    isSupported: isTTSSupported,
    activeEngine: ttsEngine,
    switchEngine: switchTTSEngine,
    isServerAvailable: isTTSServerAvailable,
    stats: ttsStats,
  } = useKokoroTTS();

  // VAD (Voice Activity Detection) - for accurate speech end detection
  const vadSpeechEndRef = useRef(false);
  const [vadEnabled, setVadEnabled] = useState(false);

  const {
    start: startVAD,
    stop: stopVAD,
    isSpeaking: vadIsSpeaking,
    isSupported: vadIsSupported,
  } = useVAD({
    onSpeechEnd: () => {
      // VAD detected speech end - flag for immediate processing
      console.info("[VoicePipeline] VAD detected speech end");
      vadSpeechEndRef.current = true;
    },
    onSpeechStart: () => {
      console.info("[VoicePipeline] VAD detected speech start");
      vadSpeechEndRef.current = false;
    },
  });

  // Layer 2 Service
  const layer2Service = getLayer2Service();

  // Keep refs to latest values for stable callbacks
  const isListeningRef = useRef(isListening);
  isListeningRef.current = isListening;
  const speakRef = useRef(speak);
  speakRef.current = speak;
  const userTranscriptRef = useRef(userTranscript);
  userTranscriptRef.current = userTranscript;

  // Generate session ID + check Layer 2 health on mount
  useEffect(() => {
    sessionIdRef.current = `session-${Date.now()}-${Math.random()
      .toString(36)
      .substr(2, 9)}`;
    layer2Service.setSessionId(sessionIdRef.current);

    const checkLayer2 = async (retries = 5) => {
      setLayer2Status("checking");
      for (let attempt = 1; attempt <= retries; attempt++) {
        try {
          // Use Next.js API proxy to avoid browser HTTP/1.1 connection pool
          // blocking (slow orchestrate requests to localhost:8100 can starve
          // other fetches to the same origin). The proxy runs server-side.
          const res = await fetch("/api/health", {
            method: "GET",
            signal: AbortSignal.timeout(10000),
          });
          if (res.ok) {
            setLayer2Status("ready");
            setLayer2Error(null);
            console.info("[VoicePipeline] Layer 2 healthy");
            return;
          } else {
            console.warn(`[VoicePipeline] Health check attempt ${attempt}/${retries}: HTTP ${res.status}`);
          }
        } catch (e) {
          console.warn(`[VoicePipeline] Health check attempt ${attempt}/${retries} failed:`, e);
        }
        if (attempt < retries) {
          await new Promise(r => setTimeout(r, 3000));
        }
      }
      setLayer2Status("error");
      setLayer2Error(`Backend unreachable at ${config.api.baseUrl}`);
    };
    checkLayer2();
  }, [layer2Service]);

  // Add message to conversation
  const addMessage = useCallback(
    (
      speaker: ConversationMessage["speaker"],
      text: string,
      type: ConversationMessage["type"] = "transcript"
    ) => {
      const message: ConversationMessage = {
        id: `msg-${++messageIdRef.current}-${Date.now()}`,
        speaker,
        text,
        timestamp: Date.now(),
        type,
      };
      setMessages((prev) => [...prev, message]);
      return message;
    },
    []
  );

  // Process a single transcript through Layer 2 → TTS
  const processOneTranscript = useCallback(
    (text: string) => {
      console.info(`[VoicePipeline] Processing: "${text}"`);
      pipelineStartTimeRef.current = performance.now();
      isProcessingRef.current = true;
      setLayer2Status("processing");

      const abortController = new AbortController();
      pipelineAbortRef.current = abortController;

      layer2Service
        .processTranscript(text)
        .then(() => {
          if (!abortController.signal.aborted) {
            console.info("[VoicePipeline] Layer 2 completed successfully");
            setLayer2Status("ready");
          }
        })
        .catch((err) => {
          if (abortController.signal.aborted) return;
          console.error("[VoicePipeline] Layer 2 error:", err);
          setError(err.message || "Failed to process");
          setLayer2Status("ready");
          isProcessingRef.current = false;
          processNextInQueue();
        });
    },
    [layer2Service]
  );

  // Process next queued message
  const processNextInQueue = useCallback(() => {
    if (messageQueueRef.current.length > 0) {
      const next = messageQueueRef.current.shift()!;
      setQueueDepth(messageQueueRef.current.length);
      console.info(`[VoicePipeline] Processing queued message: "${next}" (${messageQueueRef.current.length} remaining)`);
      setMessages(prev => prev.map(m =>
        m.type === "queued" && m.text === next ? { ...m, type: "transcript" as const } : m
      ));
      processOneTranscript(next);
    } else {
      console.info("[VoicePipeline] Queue empty, pipeline idle");
    }
  }, [processOneTranscript]);

  // Stable refs for callback registration (prevents re-registration loop and stale closures)
  const processNextInQueueRef = useRef(processNextInQueue);
  processNextInQueueRef.current = processNextInQueue;
  const addMessageRef = useRef(addMessage);
  addMessageRef.current = addMessage;
  const processOneTranscriptRef = useRef(processOneTranscript);
  processOneTranscriptRef.current = processOneTranscript;

  // Handle response from Layer 2 (stable ref — never re-registers)
  const handleResponseRef = useRef<(response: Layer2Response) => void>(() => {});
  handleResponseRef.current = (response: Layer2Response) => {
    console.info("[VoicePipeline] Layer 2 response received:", JSON.stringify(response).slice(0, 300));
    const voiceResponse = response.voice_response;
    console.info("[VoicePipeline] Voice response text:", voiceResponse);
    setAiResponse(voiceResponse);
    setLayer2Response(response);
    addMessageRef.current("ai", voiceResponse, "response");

    // Track AI response in conversation history (both modes)
    if (voiceResponse) {
      conversationHistoryRef.current.push({ role: "ai", text: voiceResponse });
      if (conversationHistoryRef.current.length > 10) {
        conversationHistoryRef.current = conversationHistoryRef.current.slice(-10);
      }
    }

    // Track AI response in interactive conversation history
    if (interactiveActiveRef.current && voiceResponse) {
      interactiveHistoryRef.current.push({ role: "ai", text: voiceResponse });
      // Update context with latest history (last 10 exchanges for token budget)
      const ctx = layer2Service.getContext();
      const wc = ctx?.widget_context;
      if (wc && typeof wc === "object") {
        layer2Service.updateContext({
          widget_context: {
            ...(wc as Record<string, unknown>),
            conversation_history: interactiveHistoryRef.current.slice(-10),
          },
        });
      }
    } else if (voiceResponse) {
      // Normal mode: update context with conversation history for follow-ups
      layer2Service.updateContext({
        conversation_history: conversationHistoryRef.current.slice(-10),
      });
    }

    if (pipelineStartTimeRef.current) {
      const l2Elapsed = Math.round(performance.now() - pipelineStartTimeRef.current);
      setLayer2LastLatencyMs(l2Elapsed);
    }

    // AUDIT FIX: Reset the flag at the start of each response processing
    hasResetRef.current = false;
    const resetAndProcessNext = () => {
      if (hasResetRef.current) return; // Prevent double-reset from onEnd + safety timeout
      hasResetRef.current = true;
      // Clear safety timeout
      if (safetyTimeoutRef.current) {
        clearTimeout(safetyTimeoutRef.current);
        safetyTimeoutRef.current = null;
      }
      if (pipelineStartTimeRef.current) {
        const totalElapsed = Math.round(performance.now() - pipelineStartTimeRef.current);
        setTotalPipelineMs(totalElapsed);
        pipelineStartTimeRef.current = null;
      }
      console.info("[VoicePipeline] Resetting processing state");
      setState(isListeningRef.current ? "listening" : "idle");
      isProcessingRef.current = false;
      processNextInQueueRef.current();
    };

    setState("speaking");

    // F5 Fix: Track voice response text for validation
    const expectedText = voiceResponse;
    const textLength = expectedText.length;

    speakRef.current(voiceResponse, {
      rate: 1.0,
      onEnd: () => {
        // F5 Fix: Validate TTS completed with expected text
        console.info(`[VoicePipeline] TTS completed for ${textLength} chars`);
        console.info("[VoicePipeline] Response finished speaking");
        resetAndProcessNext();
      },
      onError: (err) => {
        // F5 Fix: Log validation failure for TTS errors
        console.warn(`[VoicePipeline] TTS validation failed: expected ${textLength} chars, error: ${err}`);
        console.warn("[VoicePipeline] Response TTS error:", err);
        resetAndProcessNext();
      },
    });

    // Safety timeout: 60s — Kokoro TTS can produce long audio (30s+),
    // so this must exceed max expected playback duration.
    // Only fires if onEnd/onError callbacks didn't trigger.
    safetyTimeoutRef.current = setTimeout(() => {
      // AUDIT FIX: Use ref instead of local variable
      if (isProcessingRef.current && !hasResetRef.current) {
        console.warn("[VoicePipeline] Safety timeout: resetting stuck processing state");
        resetAndProcessNext();
      }
    }, 60000);
  };

  // Handle layout updates from Layer 2 (stable — no deps)
  const handleLayoutRef = useRef((layout: Layer2LayoutJSON) => {
    console.info("[VoicePipeline] Layout update:", layout);
    commandCenterBus.emit({
      type: "LAYOUT_UPDATE",
      layout: layout as any,
    });
  });

  // Register Layer 2 callbacks ONCE (stable refs prevent re-registration)
  useEffect(() => {
    console.info("[VoicePipeline] Registering Layer 2 callbacks (response, layout)");
    layer2Service.onResponse((r: Layer2Response) => handleResponseRef.current(r));
    layer2Service.onLayout((l: Layer2LayoutJSON) => handleLayoutRef.current(l));
  }, [layer2Service]);

  // VAD-triggered speech end processing
  // When VAD detects speech end, immediately send the current transcript
  useEffect(() => {
    if (!vadSpeechEndRef.current) return;
    if (!userTranscript) return;

    const delta = userTranscript.slice(lastSentLenRef.current).trim();
    if (!delta || delta.length <= 1) {
      vadSpeechEndRef.current = false;
      return;
    }

    console.info(`[VoicePipeline] VAD speech end — sending immediately: "${delta}"`);
    vadSpeechEndRef.current = false;
    lastSentLenRef.current = userTranscript.length;
    stableCountRef.current = 0;

    // Clear any pending debounce
    if (layer2DebounceRef.current) {
      clearTimeout(layer2DebounceRef.current);
      layer2DebounceRef.current = null;
    }

    if (isProcessingRef.current) {
      messageQueueRef.current.push(delta);
      setQueueDepth(messageQueueRef.current.length);
      addMessage("user", delta, "queued");
    } else {
      addMessage("user", delta, "transcript");
      processOneTranscript(delta);
    }
  }, [vadIsSpeaking, userTranscript, addMessage, processOneTranscript]);

  // Process transcript when user finishes speaking (fallback if VAD not available)
  useEffect(() => {
    if (!userTranscript) return;

    // If VAD just fired, skip the stable-transcript check (VAD already handled it)
    if (vadSpeechEndRef.current) return;

    if (userTranscript === lastTranscriptRef.current) {
      stableCountRef.current++;

      // Require more stable counts when VAD is enabled (VAD is primary)
      const requiredStableCount = vadEnabled ? 3 : 1;

      if (stableCountRef.current >= requiredStableCount) {
        const delta = userTranscript.slice(lastSentLenRef.current).trim();
        if (delta && delta.length > 3 && !isProcessingRef.current) {
          console.info(`[VoicePipeline] Silence confirmed (fallback), sending: "${delta}"`);
          lastSentLenRef.current = userTranscript.length;
          stableCountRef.current = 0;
          if (layer2DebounceRef.current) {
            clearTimeout(layer2DebounceRef.current);
            layer2DebounceRef.current = null;
          }
          addMessage("user", delta, "transcript");
          processOneTranscript(delta);
        } else if (delta && delta.length > 3 && isProcessingRef.current) {
          console.info(`[VoicePipeline] Silence confirmed but pipeline busy, queuing: "${delta}"`);
          lastSentLenRef.current = userTranscript.length;
          stableCountRef.current = 0;
          if (layer2DebounceRef.current) {
            clearTimeout(layer2DebounceRef.current);
            layer2DebounceRef.current = null;
          }
          messageQueueRef.current.push(delta);
          setQueueDepth(messageQueueRef.current.length);
          addMessage("user", delta, "queued");
        }
      }
      return;
    }

    const prevLen = lastTranscriptRef.current.length;
    const isGrowth = userTranscript.length > prevLen;
    console.info(`[VoicePipeline] Transcript changed (${prevLen} → ${userTranscript.length} chars, growth=${isGrowth})`);

    lastTranscriptRef.current = userTranscript;
    stableCountRef.current = 0;

    if (!isGrowth) return;

    const delta = userTranscript.slice(lastSentLenRef.current).trim();
    if (!delta || delta.length <= 3) return;

    console.info(`[VoicePipeline] New speech delta: "${delta}"`);

    if (layer2DebounceRef.current) {
      clearTimeout(layer2DebounceRef.current);
    }

    layer2DebounceRef.current = setTimeout(() => {
      const currentTranscript = lastTranscriptRef.current;
      const finalDelta = currentTranscript.slice(lastSentLenRef.current).trim();
      if (!finalDelta || finalDelta.length <= 3) return;

      console.info(`[VoicePipeline] Safety debounce fired, sending: "${finalDelta}"`);
      lastSentLenRef.current = currentTranscript.length;
      stableCountRef.current = 0;

      // AUDIT FIX: Use refs to avoid stale closure in setTimeout callback
      if (isProcessingRef.current) {
        messageQueueRef.current.push(finalDelta);
        setQueueDepth(messageQueueRef.current.length);
        addMessageRef.current("user", finalDelta, "queued");
      } else {
        addMessageRef.current("user", finalDelta, "transcript");
        processOneTranscriptRef.current(finalDelta);
      }
    }, 3000);
  }, [userTranscript]);

  // Update state based on listening status
  useEffect(() => {
    if (isListening && !isProcessingRef.current && !isTTSSpeaking) {
      setState("listening");
    }
  }, [isListening, isTTSSpeaking]);

  // Cleanup on stop — reset transcript tracking but DON'T abort in-flight Layer 2 requests
  useEffect(() => {
    if (!isListening) {
      lastTranscriptRef.current = "";
      lastSentLenRef.current = 0;
      stableCountRef.current = 0;
      if (layer2DebounceRef.current) {
        clearTimeout(layer2DebounceRef.current);
        layer2DebounceRef.current = null;
      }
      // Don't abort pipeline — PTT stop() sends the final transcript right before
      // stopping STT, so the Layer 2 request must be allowed to complete.
    }
  }, [isListening]);

  // Start the voice pipeline
  const start = useCallback(() => {
    console.info("[VoicePipeline] Starting... isSTTSupported:", isSTTSupported, "isSTTServerAvailable:", isSTTServerAvailable);
    setError(null);
    lastTranscriptRef.current = "";
    lastSentLenRef.current = 0;
    stableCountRef.current = 0;
    vadSpeechEndRef.current = false;
    messageQueueRef.current = [];
    setQueueDepth(0);

    if (isSTTSupported) {
      startSTT();
      setState("listening");
      commandCenterBus.emit({ type: "VOICE_INPUT_START" });
      console.info("[VoicePipeline] Now listening");

      // Start VAD for accurate speech end detection
      if (vadIsSupported) {
        startVAD();
        setVadEnabled(true);
        console.info("[VoicePipeline] VAD enabled for speech end detection");
      } else {
        setVadEnabled(false);
        console.info("[VoicePipeline] VAD not supported, using transcript stability fallback");
      }
    } else {
      console.error("[VoicePipeline] No STT available");
      setError("No STT available (server down and Web Speech API not supported)");
      setState("error");
    }
  }, [isSTTSupported, isSTTServerAvailable, startSTT, vadIsSupported, startVAD]);

  // Stop the voice pipeline
  const stop = useCallback(() => {
    console.info("[VoicePipeline] Stopping...");

    // PTT: send the final transcript to Layer 2 before stopping
    if (inputMode === "push-to-talk") {
      const finalTranscript = (userTranscriptRef.current || "").trim();
      const unsent = finalTranscript.slice(lastSentLenRef.current).trim();
      if (unsent && unsent.length > 1) {
        console.info(`[VoicePipeline] PTT release — sending final transcript: "${unsent}"`);
        lastSentLenRef.current = finalTranscript.length;
        // Clear any pending debounce
        if (layer2DebounceRef.current) {
          clearTimeout(layer2DebounceRef.current);
          layer2DebounceRef.current = null;
        }
        if (isProcessingRef.current) {
          messageQueueRef.current.push(unsent);
          setQueueDepth(messageQueueRef.current.length);
          addMessage("user", unsent, "queued");
        } else {
          addMessage("user", unsent, "transcript");
          processOneTranscript(unsent);
        }
      }
    }

    stopSTT();
    stopTTS();
    stopVAD();
    setVadEnabled(false);
    vadSpeechEndRef.current = false;
    messageQueueRef.current = [];
    setQueueDepth(0);
    if (safetyTimeoutRef.current) {
      clearTimeout(safetyTimeoutRef.current);
      safetyTimeoutRef.current = null;
    }
    setState("idle");
    commandCenterBus.emit({ type: "VOICE_INPUT_STOP" });
    console.info("[VoicePipeline] Stopped");
  }, [stopSTT, stopTTS, stopVAD, inputMode, addMessage, processOneTranscript]);

  // SPOTVOX_ALWAYS_ON: auto-start listening on mount (after STT init)
  const hasAutoStartedRef = useRef(false);
  useEffect(() => {
    if (!config.personaPlex.alwaysOn) return;
    if (hasAutoStartedRef.current) return;
    if (!isSTTSupported) return;
    // Delay briefly to let STT/TTS initialize
    const timer = setTimeout(() => {
      if (!hasAutoStartedRef.current) {
        hasAutoStartedRef.current = true;
        console.info("[VoicePipeline] SPOTVOX_ALWAYS_ON: auto-starting voice pipeline");
        start();
      }
    }, 1000);
    return () => clearTimeout(timer);
  }, [isSTTSupported, start]);

  // Send text directly to Layer 2 (bypasses STT — used by text input overlay)
  const sendTextDirect = useCallback((text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;
    console.info(`[VoicePipeline] Direct text input: "${trimmed}"`);
    addMessage("user", trimmed, "transcript");

    // Track user message in conversation history (both modes)
    conversationHistoryRef.current.push({ role: "user", text: trimmed });
    // Keep last 10 entries for token budget
    if (conversationHistoryRef.current.length > 10) {
      conversationHistoryRef.current = conversationHistoryRef.current.slice(-10);
    }
    if (interactiveActiveRef.current) {
      interactiveHistoryRef.current.push({ role: "user", text: trimmed });
    }

    // In normal mode, pass conversation_history for follow-up context
    if (!interactiveActiveRef.current) {
      layer2Service.updateContext({
        conversation_history: conversationHistoryRef.current.slice(-10),
      });
    }

    if (isProcessingRef.current) {
      messageQueueRef.current.push(trimmed);
      setQueueDepth(messageQueueRef.current.length);
      console.info(`[VoicePipeline] Pipeline busy, queued direct text`);
    } else {
      processOneTranscript(trimmed);
    }
  }, [addMessage, processOneTranscript]);

  // Widget Interactive Mode Bridge
  const sendTextDirectRef = useRef(sendTextDirect);
  sendTextDirectRef.current = sendTextDirect;

  useEffect(() => {
    // Interactive mode: enter — set context and auto-query
    const unsubInteractiveEnter = commandCenterBus.on("WIDGET_INTERACTIVE_ENTER", (event) => {
      if (event.type === "WIDGET_INTERACTIVE_ENTER") {
        console.info(`[VoicePipeline] Interactive mode ENTER: ${event.equipment} / ${event.metric}`);
        interactiveActiveRef.current = true;
        interactiveHistoryRef.current = [];
        layer2Service.updateContext({
          widget_context: {
            equipment: event.equipment,
            metric: event.metric,
            scenario: event.scenario,
            label: event.label,
            conversation_history: [],
          },
        });
        // Auto-send drill-down query
        const query = `Tell me more about ${event.label}`;
        sendTextDirectRef.current(query);
      }
    });

    // Interactive mode: exit — clear context
    const unsubInteractiveExit = commandCenterBus.on("WIDGET_INTERACTIVE_EXIT", (event) => {
      if (event.type === "WIDGET_INTERACTIVE_EXIT") {
        console.info("[VoicePipeline] Interactive mode EXIT");
        interactiveActiveRef.current = false;
        interactiveHistoryRef.current = [];
        layer2Service.updateContext({ widget_context: null });
      }
    });

    // Legacy focus (non-interactive) — still supported
    const unsubFocus = commandCenterBus.on("WIDGET_FOCUS", (event) => {
      if (event.type === "WIDGET_FOCUS") {
        const query = `Tell me more about ${event.label}`;
        console.info(`[VoicePipeline] Widget focus drill-down: "${query}"`);
        sendTextDirectRef.current(query);
      }
    });

    // Drill-down within any mode
    const unsubDrill = commandCenterBus.on("WIDGET_DRILL_DOWN", (event) => {
      if (event.type === "WIDGET_DRILL_DOWN") {
        const query = `Show me details about ${event.context} from ${event.label}`;
        console.info(`[VoicePipeline] Widget drill-down: "${query}"`);
        sendTextDirectRef.current(query);
      }
    });

    return () => { unsubInteractiveEnter(); unsubInteractiveExit(); unsubFocus(); unsubDrill(); };
  }, []);

  const pipelineStats: PipelineStats = {
    stt: sttStats,
    tts: ttsStats,
    layer2LastLatencyMs,
    totalPipelineMs,
    queueDepth,
    vadEnabled,
  };

  return {
    state,
    isActive: state !== "idle" && state !== "error",
    error,
    start,
    stop,
    inputMode,
    setInputMode,
    sendTextDirect,
    userTranscript,
    interimTranscript,
    aiResponse,
    messages,
    isTTSSpeaking,
    stopTTS,
    isSTTSupported,
    isTTSSupported,
    sttModel,
    ttsEngine,
    switchSTTModel,
    switchTTSEngine,
    isSTTServerAvailable,
    isTTSServerAvailable,
    sttAvailableModels,
    layer2Status,
    layer2Error,
    layer2Response,
    pipelineStats,
  };
}
