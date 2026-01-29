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

export type VoicePipelineState =
  | "idle"
  | "listening"
  | "speaking"
  | "error";

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
  sttAvailableModels: { name: string; loaded: boolean; available: boolean; description: string }[];

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
export function useVoicePipeline(): UseVoicePipelineReturn {
  const [state, setState] = useState<VoicePipelineState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [aiResponse, setAiResponse] = useState("");
  const [messages, setMessages] = useState<ConversationMessage[]>([]);

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
  const lastSentLenRef = useRef(0); // tracks how much of the cumulative transcript we already sent
  const stableCountRef = useRef(0); // counts consecutive identical transcripts (silence detection)
  const sessionIdRef = useRef<string | null>(null);
  const messageIdRef = useRef(0);
  const layer2DebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pipelineAbortRef = useRef<AbortController | null>(null);
  const pipelineStartTimeRef = useRef<number | null>(null);

  // Message queue: holds transcripts waiting to be processed
  const messageQueueRef = useRef<string[]>([]);
  const [queueDepth, setQueueDepth] = useState(0);

  // STT (server-based with fallback)
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
  } = useSTT();

  // TTS (Kokoro/Piper with browser fallback)
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

  // Generate session ID + check Layer 2 health on mount
  useEffect(() => {
    sessionIdRef.current = `session-${Date.now()}-${Math.random()
      .toString(36)
      .substr(2, 9)}`;
    layer2Service.setSessionId(sessionIdRef.current);

    const checkLayer2 = async () => {
      setLayer2Status("checking");
      try {
        const res = await fetch("http://localhost:8100/api/layer2/rag/industrial/health/", {
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
        setLayer2Error("Backend unreachable at localhost:8100");
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
          // On error, reset and process next in queue
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

      // Update the queued message to transcript type
      setMessages(prev => prev.map(m =>
        m.type === "queued" && m.text === next ? { ...m, type: "transcript" as const } : m
      ));

      processOneTranscript(next);
    } else {
      console.info("[VoicePipeline] Queue empty, pipeline idle");
    }
  }, [processOneTranscript]);

  // Handle response from Layer 2
  const handleResponse = useCallback(
    (response: Layer2Response) => {
      console.info("[VoicePipeline] Layer 2 response received:", JSON.stringify(response).slice(0, 300));
      const voiceResponse = response.voice_response;
      console.info("[VoicePipeline] Voice response text:", voiceResponse);
      setAiResponse(voiceResponse);
      setLayer2Response(response);
      addMessage("ai", voiceResponse, "response");

      // Track Layer 2 latency
      if (pipelineStartTimeRef.current) {
        const l2Elapsed = Math.round(performance.now() - pipelineStartTimeRef.current);
        setLayer2LastLatencyMs(l2Elapsed);
      }

      const resetAndProcessNext = () => {
        // Track total pipeline time (from speech to end of TTS)
        if (pipelineStartTimeRef.current) {
          const totalElapsed = Math.round(performance.now() - pipelineStartTimeRef.current);
          setTotalPipelineMs(totalElapsed);
          pipelineStartTimeRef.current = null;
        }
        console.info("[VoicePipeline] Resetting processing state");
        setState(isListening ? "listening" : "idle");
        isProcessingRef.current = false;
        // Process next queued message
        processNextInQueue();
      };

      // Speak the response
      setState("speaking");
      speak(voiceResponse, {
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

      // Safety net: if TTS never calls onEnd/onError within 30s, force reset
      setTimeout(() => {
        if (isProcessingRef.current) {
          console.warn("[VoicePipeline] Safety timeout: resetting stuck processing state");
          resetAndProcessNext();
        }
      }, 30000);
    },
    [speak, addMessage, isListening, processNextInQueue]
  );

  // Handle layout updates from Layer 2
  const handleLayout = useCallback((layout: Layer2LayoutJSON) => {
    console.info("[VoicePipeline] Layout update:", layout);
    commandCenterBus.emit({
      type: "LAYOUT_UPDATE",
      layout: layout as any,
    });
  }, []);

  // Set up Layer 2 callbacks (no filler — clean flow)
  useEffect(() => {
    console.info("[VoicePipeline] Registering Layer 2 callbacks (response, layout)");
    layer2Service.onResponse(handleResponse);
    layer2Service.onLayout(handleLayout);
  }, [layer2Service, handleResponse, handleLayout]);

  // Process transcript when user finishes speaking.
  // STT is cumulative — it re-transcribes all audio each cycle (every 3s).
  // We extract only the NEW text (delta) since the last send to Layer 2.
  //
  // Silence detection: we wait until the transcript is STABLE across two
  // consecutive STT cycles (meaning the user stopped speaking), then send.
  // This prevents cutting off mid-sentence during natural pauses.
  useEffect(() => {
    if (!userTranscript) {
      return;
    }

    // Same transcript as last time = user is silent, STT returned same text
    if (userTranscript === lastTranscriptRef.current) {
      stableCountRef.current++;
      console.info(`[VoicePipeline] Transcript stable (count=${stableCountRef.current})`);

      // After 1 stable cycle (user stopped talking for ~3s),
      // check if there's unsent text and fire it off
      if (stableCountRef.current >= 1) {
        const delta = userTranscript.slice(lastSentLenRef.current).trim();
        if (delta && delta.length > 3 && !isProcessingRef.current) {
          console.info(`[VoicePipeline] Silence confirmed, sending: "${delta}"`);
          lastSentLenRef.current = userTranscript.length;
          stableCountRef.current = 0;
          // Clear any pending debounce
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

    // Transcript changed — user is speaking
    const prevLen = lastTranscriptRef.current.length;
    const isGrowth = userTranscript.length > prevLen;
    console.info(`[VoicePipeline] Transcript changed (${prevLen} → ${userTranscript.length} chars, growth=${isGrowth})`);

    lastTranscriptRef.current = userTranscript;
    stableCountRef.current = 0; // Reset stability counter — user is still talking

    // Only process if transcript actually grew (new speech detected)
    if (!isGrowth) {
      console.info("[VoicePipeline] Transcript didn't grow, just refined — skipping");
      return;
    }

    // Extract only the NEW portion since the last text we sent
    const delta = userTranscript.slice(lastSentLenRef.current).trim();
    if (!delta || delta.length <= 3) {
      console.info(`[VoicePipeline] Delta too short ("${delta}"), waiting for more speech`);
      return;
    }

    console.info(`[VoicePipeline] New speech delta: "${delta}"`);

    // Safety debounce: 4s after last transcript change (fallback if stability
    // detection doesn't trigger, e.g. STT stops sending). This is longer than
    // the 3s STT cycle so stability detection normally fires first.
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
        console.info(`[VoicePipeline] Pipeline busy, queuing: "${finalDelta}"`);
        messageQueueRef.current.push(finalDelta);
        setQueueDepth(messageQueueRef.current.length);
        addMessage("user", finalDelta, "queued");
      } else {
        addMessage("user", finalDelta, "transcript");
        processOneTranscript(finalDelta);
      }
    }, 4000);
  }, [userTranscript, addMessage, processOneTranscript]);

  // Update state based on listening status
  useEffect(() => {
    if (isListening && !isProcessingRef.current && !isTTSSpeaking) {
      setState("listening");
    }
  }, [isListening, isTTSSpeaking]);

  // Cleanup on stop
  useEffect(() => {
    if (!isListening) {
      lastTranscriptRef.current = "";
      lastSentLenRef.current = 0;
      stableCountRef.current = 0;
      if (layer2DebounceRef.current) {
        clearTimeout(layer2DebounceRef.current);
        layer2DebounceRef.current = null;
      }
      if (pipelineAbortRef.current) {
        pipelineAbortRef.current.abort();
        pipelineAbortRef.current = null;
      }
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
    stopSTT();
    stopTTS();
    isProcessingRef.current = false;
    messageQueueRef.current = [];
    setQueueDepth(0);
    setState("idle");
    commandCenterBus.emit({ type: "VOICE_INPUT_STOP" });
    console.info("[VoicePipeline] Stopped");
  }, [stopSTT, stopTTS]);

  // Aggregate pipeline stats
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
