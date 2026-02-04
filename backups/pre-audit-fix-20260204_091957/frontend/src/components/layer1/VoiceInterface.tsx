"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { usePersonaPlex } from "./usePersonaPlex";
import { useSpeechRecognition } from "./useSpeechRecognition";
import { PersonaPlexState } from "@/types";
import ConversationTranscript, { TranscriptMessage } from "./ConversationTranscript";
import AudioVisualizer from "./AudioVisualizer";
import ConnectionStatus from "./ConnectionStatus";
import DeviceSelector from "./DeviceSelector";
import MetricsDashboard from "./MetricsDashboard";
import { getLayer2Service, getFiller, type Layer2Response } from "@/lib/layer2";
import { config } from "@/lib/config";

const stateLabels: Record<PersonaPlexState, string> = {
  idle: "Press Ctrl+Space or click Speak to start",
  connecting: "Connecting to PersonaPlex-7B...",
  listening: "Listening — full-duplex active",
  processing: "Processing...",
  speaking: "PersonaPlex speaking...",
  error: "PersonaPlex unavailable — check server",
  disconnected: "Disconnected — click Speak to reconnect",
};

// Pipeline layer status types
interface LayerStatus {
  name: string;
  status: "off" | "loading" | "ready" | "active" | "error";
  detail?: string;
}

/**
 * VoiceInterface — Unified Voice Pipeline UI.
 *
 * Pipeline flow:
 *   User speaks → PersonaPlex captures audio (voice I/O layer)
 *                → SpeechRecognition transcribes (STT)
 *                → Layer 2 processes transcript (AI + RAG)
 *                → PersonaPlex speaks the response (text injection → voice output)
 *
 * PersonaPlex is the ONLY voice layer — both input and output.
 * While Layer 2 processes, PersonaPlex naturally fills with conversation.
 * When the RAG response arrives, it's injected into PersonaPlex as forced
 * text tokens, making PersonaPlex speak the answer in its own voice.
 */
export default function VoiceInterface() {
  const {
    state,
    isListening,
    startListening: startPersonaPlex,
    stopListening: stopPersonaPlex,
    error,
    selectedDeviceId,
    setSelectedDeviceId,
    metricsSnapshot,
    latencyStats,
    connectionMetrics,
    inputAnalyser,
    outputAnalyser,
    conversationMessages: aiMessages,
    sendText: sendTextToPersonaPlex,
    sendControl,
    onQueueDrained,
  } = usePersonaPlex();

  // Speech recognition for user transcripts
  const {
    transcript: userTranscript,
    interimTranscript,
    start: startSpeechRecognition,
    stop: stopSpeechRecognition,
    isSupported: speechRecognitionSupported,
  } = useSpeechRecognition();

  const [dashboardVisible, setDashboardVisible] = useState(false);
  const [selectedInputDevice, setSelectedInputDevice] = useState<string | null>(selectedDeviceId);
  const [selectedOutputDevice, setSelectedOutputDevice] = useState<string | null>(null);

  // Merged conversation messages (user + AI)
  const [allMessages, setAllMessages] = useState<TranscriptMessage[]>([]);
  const userMessageIdRef = useRef(0);
  const lastUserTranscriptRef = useRef("");

  // Layer 2 state
  const [layer2Status, setLayer2Status] = useState<"off" | "checking" | "ready" | "processing" | "error">("off");
  const [layer2Error, setLayer2Error] = useState<string | null>(null);
  const [layer2Response, setLayer2Response] = useState<Layer2Response | null>(null);
  const [sttNoSpeechCount, setSttNoSpeechCount] = useState(0);
  const layer2ServiceRef = useRef(getLayer2Service());
  const lastLayer2TranscriptRef = useRef("");
  const layer2DebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pipelineAbortRef = useRef<AbortController | null>(null);
  const pipelineActiveRef = useRef(false);

  // Check Layer 2 health on mount
  useEffect(() => {
    const checkLayer2 = async () => {
      setLayer2Status("checking");
      try {
        const res = await fetch(`${config.api.baseUrl}/api/layer2/rag/industrial/health/`, {
          method: "GET",
          signal: AbortSignal.timeout(5000),
        });
        if (res.ok) {
          const data = await res.json();
          console.info("[Layer2] Health check:", data);
          setLayer2Status("ready");
        } else {
          console.warn("[Layer2] Health check failed:", res.status);
          setLayer2Status("error");
          setLayer2Error(`Backend returned ${res.status}`);
        }
      } catch (e) {
        console.warn("[Layer2] Health check failed:", e);
        setLayer2Status("error");
        setLayer2Error(`Backend unreachable at ${config.api.baseUrl}`);
      }
    };
    checkLayer2();
    // Set session ID
    layer2ServiceRef.current.setSessionId(`session-${Date.now()}`);
  }, []);

  // Track SpeechRecognition no-speech errors
  useEffect(() => {
    const handleSpeechError = (e: Event) => {
      const detail = (e as CustomEvent)?.detail;
      if (detail === "no-speech") {
        setSttNoSpeechCount((c) => c + 1);
      }
    };
    window.addEventListener("speechrecognition-error", handleSpeechError);
    return () => window.removeEventListener("speechrecognition-error", handleSpeechError);
  }, []);

  // Start both PersonaPlex and speech recognition
  const startListening = useCallback(() => {
    startPersonaPlex();
    if (speechRecognitionSupported) {
      startSpeechRecognition();
    }
    setSttNoSpeechCount(0);
  }, [startPersonaPlex, startSpeechRecognition, speechRecognitionSupported]);

  // Stop both PersonaPlex and speech recognition
  const stopListening = useCallback(() => {
    stopPersonaPlex();
    if (speechRecognitionSupported) {
      stopSpeechRecognition();
    }
  }, [stopPersonaPlex, stopSpeechRecognition, speechRecognitionSupported]);

  // Merge AI messages into allMessages
  useEffect(() => {
    setAllMessages((prev) => {
      const userMsgs = prev.filter((m) => m.speaker === "user");
      const systemMsgs = prev.filter((m) => (m as any).type === "layer2");
      const newMessages = [...userMsgs, ...systemMsgs];

      for (const aiMsg of aiMessages) {
        const existing = newMessages.findIndex((m) => m.id === aiMsg.id);
        if (existing >= 0) {
          newMessages[existing] = aiMsg;
        } else {
          newMessages.push(aiMsg);
        }
      }

      newMessages.sort((a, b) => a.timestamp - b.timestamp);
      return newMessages;
    });
  }, [aiMessages]);

  // Add user transcript to messages AND send to Layer 2
  useEffect(() => {
    if (userTranscript && userTranscript !== lastUserTranscriptRef.current) {
      const newText = userTranscript.slice(lastUserTranscriptRef.current.length).trim();

      if (newText) {
        const userMessage: TranscriptMessage = {
          id: `user-${++userMessageIdRef.current}-${Date.now()}`,
          speaker: "user",
          text: newText,
          timestamp: Date.now(),
        };

        setAllMessages((prev) => {
          const sorted = [...prev, userMessage].sort((a, b) => a.timestamp - b.timestamp);
          return sorted;
        });

        // Send to Layer 2 for RAG processing (debounced).
        // Suppression is always ON — PersonaPlex only speaks injected text.
        if (layer2Status === "ready" || layer2Status === "processing") {
          if (newText !== lastLayer2TranscriptRef.current && newText.length > 3) {
            lastLayer2TranscriptRef.current = newText;

            // Debounce: wait 1.5s after last transcript change before firing pipeline.
            // This prevents overlapping queries from rapid STT updates.
            if (layer2DebounceRef.current) {
              clearTimeout(layer2DebounceRef.current);
            }

            const capturedText = newText;
            layer2DebounceRef.current = setTimeout(() => {
              // Cancel any in-flight pipeline
              if (pipelineAbortRef.current) {
                pipelineAbortRef.current.abort();
                console.info("[Pipeline] Cancelled previous pipeline");
              }
              onQueueDrained.current = null;
              pipelineActiveRef.current = true;

              const abortController = new AbortController();
              pipelineAbortRef.current = abortController;

              setLayer2Status("processing");
              console.info("[Layer2] Sending transcript:", capturedText);

              // Step 1: Inject filler so PersonaPlex speaks it while RAG processes
              getFiller("query", ["industrial"])
                .then((filler) => {
                  if (abortController.signal.aborted) return;
                  if (filler) {
                    console.info("[Pipeline] Injecting filler into voice:", filler);
                    sendTextToPersonaPlex(filler);
                  }
                })
                .catch(() => {});

              // Step 2: Process through RAG pipeline
              layer2ServiceRef.current
                .processTranscript(capturedText)
                .then((response) => {
                  if (abortController.signal.aborted) return;
                  console.info("[Layer2] Response received:", response);
                  setLayer2Response(response);
                  setLayer2Status("ready");

                  if (response.voice_response) {
                    const domains = response.rag_results
                      .filter((r: { success: boolean }) => r.success)
                      .map((r: { domain: string }) => r.domain)
                      .join(", ");

                    // Step 3: Wait for filler queue to drain on server,
                    // then pause 1s for natural breathing room, then inject RAG.
                    const injectRAG = () => {
                      if (abortController.signal.aborted) return;
                      setTimeout(() => {
                        if (abortController.signal.aborted) return;
                        console.info("[Pipeline] Injecting RAG response into PersonaPlex voice");
                        sendTextToPersonaPlex(response.voice_response);
                        pipelineActiveRef.current = false;
                      }, 1000);
                    };

                    // Register one-shot callback for when server queue drains
                    onQueueDrained.current = injectRAG;

                    // Fallback: if drain notification never arrives (no filler,
                    // or filler was empty), inject after 4s max
                    setTimeout(() => {
                      if (abortController.signal.aborted) return;
                      if (onQueueDrained.current === injectRAG) {
                        console.info("[Pipeline] Fallback: injecting RAG after timeout");
                        onQueueDrained.current = null;
                        injectRAG();
                      }
                    }, 4000);

                    // Show in transcript with routing info
                    const ragMessage: TranscriptMessage = {
                      id: `layer2-${Date.now()}`,
                      speaker: "ai",
                      text: `[${domains || "RAG"}] ${response.voice_response}`,
                      timestamp: Date.now(),
                    };
                    setAllMessages((prev) =>
                      [...prev, ragMessage].sort((a, b) => a.timestamp - b.timestamp)
                    );
                  } else {
                    pipelineActiveRef.current = false;
                  }
                })
                .catch((err) => {
                  if (abortController.signal.aborted) return;
                  console.error("[Layer2] Processing failed:", err);
                  setLayer2Error(err.message);
                  setLayer2Status("ready");
                  pipelineActiveRef.current = false;
                });
            }, 1500);
          }
        }
      }

      lastUserTranscriptRef.current = userTranscript;
    }
  }, [userTranscript, layer2Status, sendTextToPersonaPlex, onQueueDrained]);

  // Reset user transcript tracking when session ends
  useEffect(() => {
    if (!isListening) {
      lastUserTranscriptRef.current = "";
      lastLayer2TranscriptRef.current = "";
      if (layer2DebounceRef.current) {
        clearTimeout(layer2DebounceRef.current);
        layer2DebounceRef.current = null;
      }
      if (pipelineAbortRef.current) {
        pipelineAbortRef.current.abort();
        pipelineAbortRef.current = null;
      }
      pipelineActiveRef.current = false;
    }
  }, [isListening]);

  // Sync device selection
  useEffect(() => {
    if (selectedDeviceId !== selectedInputDevice) {
      setSelectedInputDevice(selectedDeviceId);
    }
  }, [selectedDeviceId, selectedInputDevice]);

  const handleInputDeviceChange = (deviceId: string) => {
    setSelectedInputDevice(deviceId);
    setSelectedDeviceId(deviceId);
  };

  const handleOutputDeviceChange = (deviceId: string) => {
    setSelectedOutputDevice(deviceId);
  };

  const handleExportMetrics = () => {
    if (!metricsSnapshot) return;
    const json = JSON.stringify(metricsSnapshot, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `commandcenter-metrics-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Keyboard shortcut for metrics dashboard (Ctrl/Cmd+M)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "m" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        setDashboardVisible((prev) => !prev);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const isUserSpeaking = state === "listening" && !!interimTranscript;
  const isAISpeaking = state === "speaking";

  // Build pipeline status
  const layers: LayerStatus[] = [
    {
      name: "Voice I/O",
      status:
        state === "error" || state === "disconnected"
          ? "error"
          : state === "connecting"
            ? "loading"
            : isListening
              ? "active"
              : connectionMetrics.wsState === "open"
                ? "ready"
                : "off",
      detail:
        state === "listening"
          ? "PersonaPlex"
          : state === "connecting"
            ? "Connecting..."
            : state === "error"
              ? "Down"
              : "PersonaPlex-7B",
    },
    {
      name: "STT",
      status:
        !speechRecognitionSupported
          ? "error"
          : sttNoSpeechCount > 3
            ? "error"
            : isListening
              ? interimTranscript
                ? "active"
                : "ready"
              : "off",
      detail:
        !speechRecognitionSupported
          ? "Not supported"
          : sttNoSpeechCount > 3
            ? "No mic (SSH?)"
            : interimTranscript
              ? "Transcribing"
              : "Web Speech",
    },
    {
      name: "AI/RAG",
      status:
        layer2Status === "error"
          ? "error"
          : layer2Status === "checking"
            ? "loading"
            : layer2Status === "processing"
              ? "active"
              : layer2Status === "ready"
                ? "ready"
                : "off",
      detail:
        layer2Status === "processing"
          ? "Querying RAG..."
          : layer2Status === "error"
            ? layer2Error || "Down"
            : layer2Status === "ready"
              ? "Ollama + ChromaDB"
              : undefined,
    },
  ];

  const getStatusDot = (status: LayerStatus["status"]) => {
    switch (status) {
      case "active":
        return "bg-green-400 animate-pulse";
      case "ready":
        return "bg-green-400";
      case "loading":
        return "bg-yellow-400 animate-pulse";
      case "error":
        return "bg-red-400";
      case "off":
      default:
        return "bg-white/20";
    }
  };

  return (
    <div className="flex flex-col h-full bg-[var(--cc-background)]">
      {/* Pipeline Status Bar */}
      <div className="px-3 pt-3 flex gap-2">
        {/* Connection Status */}
        <div className="flex-1">
          <ConnectionStatus metrics={connectionMetrics} />
        </div>
      </div>

      {/* Layer Status Chips */}
      <div className="px-3 pt-2 flex gap-2 flex-wrap">
        {layers.map((layer) => (
          <div
            key={layer.name}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs border ${
              layer.status === "error"
                ? "border-red-500/30 bg-red-500/10 text-red-300"
                : layer.status === "active"
                  ? "border-green-500/30 bg-green-500/10 text-green-300"
                  : layer.status === "loading"
                    ? "border-yellow-500/30 bg-yellow-500/10 text-yellow-300"
                    : layer.status === "ready"
                      ? "border-white/20 bg-white/5 text-white/70"
                      : "border-white/10 bg-white/5 text-white/30"
            }`}
          >
            <div className={`w-1.5 h-1.5 rounded-full ${getStatusDot(layer.status)}`} />
            <span className="font-medium">{layer.name}</span>
            {layer.detail && (
              <span className="text-[10px] opacity-70">{layer.detail}</span>
            )}
          </div>
        ))}
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex gap-3 px-3 py-3 min-h-0">
        {/* Left Column: Conversation Transcript (Primary) */}
        <div className="flex-1 min-w-0">
          {allMessages.length === 0 && !isListening ? (
            <div className="flex flex-col items-center justify-center h-full bg-black/20 backdrop-blur-sm rounded-lg border border-white/10 p-6">
              <p className="text-white/70 text-sm mb-3">
                {stateLabels[state]}
              </p>

              {/* Pipeline readiness checklist */}
              <div className="w-full max-w-sm space-y-2 mb-4">
                <div className="text-white/50 text-xs font-medium mb-2">Pipeline Status:</div>
                {layers.map((layer) => (
                  <div key={layer.name} className="flex items-center gap-2 text-xs">
                    <div className={`w-2 h-2 rounded-full ${getStatusDot(layer.status)}`} />
                    <span className={layer.status === "error" ? "text-red-400" : "text-white/60"}>
                      {layer.name}
                    </span>
                    <span className="text-white/30 ml-auto">
                      {layer.status === "active"
                        ? "Active"
                        : layer.status === "ready"
                          ? "Ready"
                          : layer.status === "loading"
                            ? "Loading..."
                            : layer.status === "error"
                              ? layer.detail || "Error"
                              : "Off"}
                    </span>
                  </div>
                ))}
              </div>

              {state === "error" && error && (
                <div className="max-w-md text-center">
                  <p className="text-red-400 text-xs font-mono mb-2">{error}</p>
                  <p className="text-white/50 text-xs">
                    Check terminal session logs for details.
                  </p>
                </div>
              )}
              {state === "disconnected" && (
                <p className="text-white/50 text-xs">
                  Click Speak or press Ctrl+Space to reconnect.
                </p>
              )}
            </div>
          ) : (
            <ConversationTranscript
              messages={allMessages}
              isUserSpeaking={isUserSpeaking}
              isAISpeaking={isAISpeaking}
              interimText={interimTranscript}
            />
          )}
        </div>

        {/* Right Column: Metrics & Visualization */}
        <div className="flex flex-col gap-3 w-80">
          {/* Audio Visualizers */}
          <AudioVisualizer
            analyser={inputAnalyser}
            type="input"
            isActive={isListening}
          />
          <AudioVisualizer
            analyser={outputAnalyser}
            type="output"
            isActive={state === "listening" || state === "speaking"}
          />

          {/* Layer 2 RAG Results Panel */}
          {layer2Response && (
            <div className="bg-purple-500/10 backdrop-blur-sm rounded-lg border border-purple-500/20 p-3">
              <div className="text-xs text-purple-300 font-medium mb-2">
                Pipeline Routing
              </div>
              {layer2Response.intent && (
                <div className="text-xs text-white/60 mb-1">
                  Intent: <span className="text-purple-300">{layer2Response.intent.type}</span>
                  {layer2Response.intent.confidence > 0 && (
                    <span className="text-white/40"> ({Math.round(layer2Response.intent.confidence * 100)}%)</span>
                  )}
                </div>
              )}
              {layer2Response.rag_results.map((r, i) => (
                <div key={i} className="text-xs text-white/50 mt-1 flex items-center gap-1.5">
                  <div className={`w-1.5 h-1.5 rounded-full ${r.success ? "bg-green-400" : "bg-red-400"}`} />
                  <span className={r.success ? "text-green-300" : "text-red-300"}>
                    {r.domain}
                  </span>
                  <span className="text-white/30 ml-auto">{r.execution_time_ms}ms</span>
                  {r.error && <span className="text-red-400 text-[10px]">({r.error})</span>}
                </div>
              ))}
              <div className="text-xs text-white/30 mt-2 pt-1.5 border-t border-white/10 flex justify-between">
                <span>Routed to PersonaPlex voice</span>
                <span>{layer2Response.processing_time_ms}ms</span>
              </div>
            </div>
          )}

          {/* Session Stats Card */}
          {metricsSnapshot && (
            <div className="bg-black/20 backdrop-blur-sm rounded-lg border border-white/10 p-3">
              <div className="text-xs text-white/60 mb-2">Session Stats</div>
              <div className="space-y-1.5">
                <div className="flex justify-between items-center">
                  <span className="text-xs text-white/50">Uptime</span>
                  <span className="text-xs text-white/80 font-medium">
                    {Math.floor(metricsSnapshot.session.uptime / 60)}m{" "}
                    {Math.floor(metricsSnapshot.session.uptime % 60)}s
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs text-white/50">Avg Latency</span>
                  <span
                    className={`text-xs font-medium ${
                      metricsSnapshot.session.averageLatency < 200
                        ? "text-green-400"
                        : metricsSnapshot.session.averageLatency < 500
                          ? "text-yellow-400"
                          : "text-red-400"
                    }`}
                  >
                    {metricsSnapshot.session.averageLatency.toFixed(0)}ms
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs text-white/50">Audio Chunks</span>
                  <span className="text-xs text-white/80 font-medium">
                    {metricsSnapshot.session.totalChunksProcessed}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs text-white/50">Errors</span>
                  <span
                    className={`text-xs font-medium ${
                      metricsSnapshot.session.totalErrors === 0
                        ? "text-green-400"
                        : "text-red-400"
                    }`}
                  >
                    {metricsSnapshot.session.totalErrors}
                  </span>
                </div>
              </div>
              <button
                onClick={() => setDashboardVisible(true)}
                className="w-full mt-3 px-3 py-1.5 text-xs font-medium text-white/80 hover:text-white bg-white/10 hover:bg-white/20 rounded transition-colors"
              >
                View Full Metrics (Ctrl+M)
              </button>
            </div>
          )}

          {/* STT Warning for SSH tunnel */}
          {sttNoSpeechCount > 3 && isListening && (
            <div className="bg-yellow-500/10 rounded-lg border border-yellow-500/20 p-3">
              <div className="text-xs text-yellow-300 font-medium mb-1">
                Mic not detected by STT
              </div>
              <div className="text-[10px] text-yellow-200/60">
                Speech Recognition cannot hear audio (common over SSH tunnels).
                PersonaPlex voice I/O still works — AI audio is flowing.
                User transcript won't appear but AI responses will.
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Bottom Controls */}
      <div className="px-3 pb-3 space-y-3">
        {/* Device Selector */}
        <DeviceSelector
          selectedInputId={selectedInputDevice}
          selectedOutputId={selectedOutputDevice}
          onInputChange={handleInputDeviceChange}
          onOutputChange={handleOutputDeviceChange}
          disabled={isListening}
        />

        {/* Voice Control Bar */}
        <div className="bg-black/20 backdrop-blur-sm rounded-lg border border-white/10 px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            {/* Waveform visualization (visible when listening) */}
            {isListening && (
              <div className="flex items-center gap-[2px] h-5">
                {Array.from({ length: 5 }).map((_, i) => (
                  <div
                    key={i}
                    className="waveform-bar w-[3px] bg-blue-400 rounded-full animate-pulse"
                    style={{
                      animationDelay: `${i * 0.1}s`,
                      height: `${8 + i * 4}px`,
                    }}
                  />
                ))}
              </div>
            )}

            {/* Status text */}
            <span className="text-sm text-white/60">
              {stateLabels[state]}
            </span>

            {/* Show interim transcript while speaking */}
            {interimTranscript && (
              <span className="text-sm text-blue-400/70 italic ml-2 max-w-xs truncate">
                &quot;{interimTranscript}&quot;
              </span>
            )}

            {/* Layer 2 processing indicator */}
            {layer2Status === "processing" && (
              <span className="text-sm text-purple-400/70 ml-2 animate-pulse">
                RAG processing...
              </span>
            )}
          </div>

          {/* Push-to-talk button */}
          <div className="flex items-center gap-3">
            <span className="text-xs text-white/40">Ctrl+Space</span>
            <button
              onClick={isListening ? stopListening : startListening}
              disabled={state === "connecting" || state === "processing"}
              className={`
                px-6 py-2 rounded-lg text-sm font-medium transition-all
                ${
                  isListening
                    ? "bg-red-500 text-white hover:bg-red-600"
                    : state === "connecting" || state === "processing"
                      ? "bg-white/10 text-white/40 cursor-wait"
                      : state === "error" || state === "disconnected"
                        ? "bg-yellow-500 text-black hover:bg-yellow-400"
                        : "bg-blue-500 text-white hover:bg-blue-600"
                }
              `}
            >
              {isListening
                ? "Stop"
                : state === "connecting"
                  ? "Connecting..."
                  : state === "processing"
                    ? "Processing..."
                    : state === "error" || state === "disconnected"
                      ? "Reconnect"
                      : "Speak"}
            </button>
          </div>
        </div>

        {/* Error banner */}
        {error && (
          <div className="px-4 py-2 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-xs space-y-1">
            <div className="font-mono">{error}</div>
            {error.includes("wss://") && (
              <div className="text-red-300/80 text-[10px]">
                <span className="font-semibold">SSL Tip:</span> Visit{" "}
                <a
                  href="https://localhost:8998"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline hover:text-red-200"
                >
                  https://localhost:8998
                </a>{" "}
                in your browser and accept the certificate, then try again.
              </div>
            )}
            <div className="text-red-300/60 text-[10px]">
              Check browser DevTools console (F12) for detailed logs.
            </div>
          </div>
        )}

        {/* Layer 2 error (non-blocking) */}
        {layer2Error && !error && (
          <div className="px-4 py-2 bg-yellow-500/10 border border-yellow-500/30 rounded-lg text-yellow-400 text-xs">
            <span className="font-medium">Layer 2:</span> {layer2Error}
          </div>
        )}
      </div>

      {/* Metrics Dashboard Overlay */}
      {metricsSnapshot && (
        <MetricsDashboard
          snapshot={metricsSnapshot}
          latencyStats={latencyStats}
          isVisible={dashboardVisible}
          onClose={() => setDashboardVisible(false)}
          onExport={handleExportMetrics}
        />
      )}
    </div>
  );
}
