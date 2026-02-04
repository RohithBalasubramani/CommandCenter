"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useSTT, type STTModel, type STTStats } from "./useSTT";
import { useKokoroTTS, type TTSEngine, type TTSStats } from "./useKokoroTTS";
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
  const [inputMode, setInputMode] = useState<VoiceInputMode>("push-to-talk");

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

  // Message queue
  const messageQueueRef = useRef<string[]>([]);
  const [queueDepth, setQueueDepth] = useState(0);

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

    const checkLayer2 = async () => {
      setLayer2Status("checking");
      try {
        const res = await fetch(`${config.api.baseUrl}/api/layer2/rag/industrial/health/`, {
          method: "GET",
          signal: AbortSignal.timeout(5000),
        });
        if (res.ok) {
          setLayer2Status("ready");
          console.info("[VoicePipeline] Layer 2 healthy");
        } else {
          setLayer2Status("error");
          setLayer2Error(`Backend returned ${res.status}`);
        }
      } catch {
        setLayer2Status("error");
        setLayer2Error(`Backend unreachable at ${config.api.baseUrl}`);
      }
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

  // Stable refs for callback registration (prevents re-registration loop)
  const processNextInQueueRef = useRef(processNextInQueue);
  processNextInQueueRef.current = processNextInQueue;
  const addMessageRef = useRef(addMessage);
  addMessageRef.current = addMessage;

  // Handle response from Layer 2 (stable ref — never re-registers)
  const handleResponseRef = useRef<(response: Layer2Response) => void>(() => {});
  handleResponseRef.current = (response: Layer2Response) => {
    console.info("[VoicePipeline] Layer 2 response received:", JSON.stringify(response).slice(0, 300));
    const voiceResponse = response.voice_response;
    console.info("[VoicePipeline] Voice response text:", voiceResponse);
    setAiResponse(voiceResponse);
    setLayer2Response(response);
    addMessageRef.current("ai", voiceResponse, "response");

    if (pipelineStartTimeRef.current) {
      const l2Elapsed = Math.round(performance.now() - pipelineStartTimeRef.current);
      setLayer2LastLatencyMs(l2Elapsed);
    }

    let hasReset = false;
    const resetAndProcessNext = () => {
      if (hasReset) return; // Prevent double-reset from onEnd + safety timeout
      hasReset = true;
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
    speakRef.current(voiceResponse, {
      rate: 1.0,
      onEnd: () => {
        console.info("[VoicePipeline] Response finished speaking");
        resetAndProcessNext();
      },
      onError: (err) => {
        console.warn("[VoicePipeline] Response TTS error:", err);
        resetAndProcessNext();
      },
    });

    // Safety timeout: 60s — Kokoro TTS can produce long audio (30s+),
    // so this must exceed max expected playback duration.
    // Only fires if onEnd/onError callbacks didn't trigger.
    safetyTimeoutRef.current = setTimeout(() => {
      if (isProcessingRef.current && !hasReset) {
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

  // Process transcript when user finishes speaking
  useEffect(() => {
    if (!userTranscript) return;

    if (userTranscript === lastTranscriptRef.current) {
      stableCountRef.current++;

      if (stableCountRef.current >= 1) {
        const delta = userTranscript.slice(lastSentLenRef.current).trim();
        if (delta && delta.length > 3 && !isProcessingRef.current) {
          console.info(`[VoicePipeline] Silence confirmed, sending: "${delta}"`);
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

      if (isProcessingRef.current) {
        messageQueueRef.current.push(finalDelta);
        setQueueDepth(messageQueueRef.current.length);
        addMessage("user", finalDelta, "queued");
      } else {
        addMessage("user", finalDelta, "transcript");
        processOneTranscript(finalDelta);
      }
    }, 3000);
  }, [userTranscript, addMessage, processOneTranscript]);

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
    messageQueueRef.current = [];
    setQueueDepth(0);

    if (isSTTSupported) {
      startSTT();
      setState("listening");
      commandCenterBus.emit({ type: "VOICE_INPUT_START" });
      console.info("[VoicePipeline] Now listening");
    } else {
      console.error("[VoicePipeline] No STT available");
      setError("No STT available (server down and Web Speech API not supported)");
      setState("error");
    }
  }, [isSTTSupported, isSTTServerAvailable, startSTT]);

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
    messageQueueRef.current = [];
    setQueueDepth(0);
    if (safetyTimeoutRef.current) {
      clearTimeout(safetyTimeoutRef.current);
      safetyTimeoutRef.current = null;
    }
    setState("idle");
    commandCenterBus.emit({ type: "VOICE_INPUT_STOP" });
    console.info("[VoicePipeline] Stopped");
  }, [stopSTT, stopTTS, inputMode, addMessage, processOneTranscript]);

  // Send text directly to Layer 2 (bypasses STT — used by text input overlay)
  const sendTextDirect = useCallback((text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;
    console.info(`[VoicePipeline] Direct text input: "${trimmed}"`);
    addMessage("user", trimmed, "transcript");
    if (isProcessingRef.current) {
      messageQueueRef.current.push(trimmed);
      setQueueDepth(messageQueueRef.current.length);
      console.info(`[VoicePipeline] Pipeline busy, queued direct text`);
    } else {
      processOneTranscript(trimmed);
    }
  }, [addMessage, processOneTranscript]);

  // Widget Focus → Voice Drill-Down Bridge
  // When a user clicks "Focus" on a widget, auto-send a drill-down query to Layer 2
  const sendTextDirectRef = useRef(sendTextDirect);
  sendTextDirectRef.current = sendTextDirect;

  useEffect(() => {
    const unsubFocus = commandCenterBus.on("WIDGET_FOCUS", (event) => {
      if (event.type === "WIDGET_FOCUS") {
        const query = `Tell me more about ${event.label}`;
        console.info(`[VoicePipeline] Widget focus drill-down: "${query}"`);
        sendTextDirectRef.current(query);
      }
    });
    const unsubDrill = commandCenterBus.on("WIDGET_DRILL_DOWN", (event) => {
      if (event.type === "WIDGET_DRILL_DOWN") {
        const query = `Show me details about ${event.context} from ${event.label}`;
        console.info(`[VoicePipeline] Widget drill-down: "${query}"`);
        sendTextDirectRef.current(query);
      }
    });
    return () => { unsubFocus(); unsubDrill(); };
  }, []);

  const pipelineStats: PipelineStats = {
    stt: sttStats,
    tts: ttsStats,
    layer2LastLatencyMs,
    totalPipelineMs,
    queueDepth,
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
