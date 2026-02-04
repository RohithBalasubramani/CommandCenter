"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { useVoicePipeline, ConversationMessage, type PipelineStats } from "./useVoicePipeline";
import type { STTModel } from "./useSTT";
import type { TTSEngine } from "./useKokoroTTS";
import DeviceSelector from "./DeviceSelector";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const stateLabels: Record<string, string> = {
  idle: "Press Shift+Space or click Speak to start",
  listening: "Listening — speak now",
  speaking: "AI is responding...",
  error: "Error — click Speak to retry",
};

const stateColors: Record<string, string> = {
  idle: "text-white/60",
  listening: "text-blue-400",
  speaking: "text-purple-400",
  error: "text-red-400",
};

interface LayerStatus {
  name: string;
  status: "off" | "loading" | "ready" | "active" | "error";
  detail?: string;
}

// ---------------------------------------------------------------------------
// Stat helpers
// ---------------------------------------------------------------------------

function fmtMs(v: number | null): string {
  if (v === null) return "--";
  return `${v}ms`;
}

function fmtSec(v: number | null): string {
  if (v === null || v === 0) return "--";
  return `${v}s`;
}

function fmtNum(v: number): string {
  return v.toString();
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/**
 * VoiceInterfaceV2 — V2 voice pipeline UI with server-based STT/TTS.
 *
 * Clean flow: Human speaks → AI responds. No filler speech.
 * Messages queue if user speaks while AI is still responding.
 */
export default function VoiceInterfaceV2() {
  // Restore persisted device selection
  const [selectedInputDevice, setSelectedInputDevice] = useState<string | null>(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("cc-input-device") || null;
    }
    return null;
  });
  const [selectedOutputDevice, setSelectedOutputDevice] = useState<string | null>(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("cc-output-device") || null;
    }
    return null;
  });

  const {
    state,
    isActive,
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
  } = useVoicePipeline(selectedInputDevice ?? undefined);

  const [settingsOpen, setSettingsOpen] = useState(false);
  const [showInterim, setShowInterim] = useState(true);

  // Session stats
  const [sessionStart] = useState(Date.now());
  const [queryCount, setQueryCount] = useState(0);
  const [errorCount, setErrorCount] = useState(0);
  const [uptimeStr, setUptimeStr] = useState("0m 0s");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Log state transitions
  useEffect(() => {
    console.info(`[VoiceUI] State: ${state} | active=${isActive} | STT=${sttModel}(server=${isSTTServerAvailable}) | TTS=${ttsEngine}(server=${isTTSServerAvailable}) | Layer2=${layer2Status}`);
  }, [state, isActive, sttModel, isSTTServerAvailable, ttsEngine, isTTSServerAvailable, layer2Status]);

  // Log messages
  useEffect(() => {
    if (messages.length > 0) {
      const last = messages[messages.length - 1];
      console.info(`[VoiceUI] New message: [${last.speaker}/${last.type}] "${last.text.slice(0, 80)}"`);
    }
  }, [messages]);

  // Log transcript changes
  useEffect(() => {
    if (userTranscript) {
      console.info(`[VoiceUI] userTranscript: "${userTranscript}"`);
    }
  }, [userTranscript]);

  useEffect(() => {
    if (interimTranscript) {
      console.debug(`[VoiceUI] interimTranscript: "${interimTranscript}"`);
    }
  }, [interimTranscript]);

  // Track query count
  useEffect(() => {
    const responses = messages.filter((m) => m.type === "response");
    setQueryCount(responses.length);
  }, [messages]);

  // Track errors
  useEffect(() => {
    if (error) {
      console.error(`[VoiceUI] Error: ${error}`);
      setErrorCount((c) => c + 1);
    }
  }, [error]);

  // Uptime ticker
  useEffect(() => {
    if (!isActive) return;
    const interval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - sessionStart) / 1000);
      const m = Math.floor(elapsed / 60);
      const s = elapsed % 60;
      setUptimeStr(`${m}m ${s}s`);
    }, 1000);
    return () => clearInterval(interval);
  }, [isActive, sessionStart]);

  // Auto-scroll conversation
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, interimTranscript]);

  // Keyboard: Shift+Space toggle, Ctrl+M settings
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === "Space" && e.shiftKey) {
        e.preventDefault();
        if (isActive) stop();
        else start();
      }
      if (e.key === "m" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        setSettingsOpen((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isActive, start, stop]);

  // ---------------------------------------------------------------------------
  // Pipeline status chips
  // ---------------------------------------------------------------------------

  const sttModelLabel = sttModel === "web-speech" ? "Web Speech" : sttModel === "parakeet" ? "Parakeet" : "Whisper";
  const ttsEngineLabel = ttsEngine === "browser" ? "Browser" : ttsEngine === "kokoro" ? "Kokoro" : "Piper";

  const layers: LayerStatus[] = [
    {
      name: "STT",
      status:
        !isSTTSupported
          ? "error"
          : isActive
            ? interimTranscript
              ? "active"
              : "ready"
            : isSTTServerAvailable
              ? "ready"
              : "off",
      detail:
        !isSTTSupported
          ? "Not available"
          : interimTranscript
            ? "Transcribing"
            : sttModelLabel,
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
    {
      name: "TTS",
      status:
        !isTTSSupported
          ? "error"
          : isTTSSpeaking
            ? "active"
            : isTTSServerAvailable
              ? "ready"
              : "off",
      detail:
        isTTSSpeaking
          ? "Speaking..."
          : ttsEngineLabel,
    },
  ];

  const getStatusDot = (status: LayerStatus["status"]) => {
    switch (status) {
      case "active": return "bg-green-400 animate-pulse";
      case "ready": return "bg-green-400";
      case "loading": return "bg-yellow-400 animate-pulse";
      case "error": return "bg-red-400";
      default: return "bg-white/20";
    }
  };

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  const formatTime = (timestamp: number) =>
    new Date(timestamp).toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });

  const getMessageStyle = (msg: ConversationMessage) => {
    if (msg.speaker === "user" && msg.type === "queued") return "bg-white/10 border-white/20";
    if (msg.speaker === "user") return "bg-blue-600/30 border-blue-500/40";
    return "bg-purple-600/30 border-purple-500/40";
  };

  const getMessageLabel = (msg: ConversationMessage) => {
    if (msg.speaker === "user" && msg.type === "queued") return "You (queued)";
    if (msg.speaker === "user") return "You";
    return "AI Assistant";
  };

  const getMessageLabelColor = (msg: ConversationMessage) => {
    if (msg.speaker === "user" && msg.type === "queued") return "text-white/40";
    if (msg.speaker === "user") return "text-blue-300";
    return "text-purple-300";
  };

  // ---------------------------------------------------------------------------
  // Stat row component
  // ---------------------------------------------------------------------------

  const StatRow = ({ label, value, color }: { label: string; value: string; color?: string }) => (
    <div className="flex justify-between items-center">
      <span className="text-xs text-white/50">{label}</span>
      <span className={`text-xs font-medium ${color || "text-white/80"}`}>{value}</span>
    </div>
  );

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="flex flex-col h-full bg-[var(--cc-background)]">
      {/* Pipeline Status Chips */}
      <div className="px-3 pt-3 flex items-center gap-2 flex-wrap">
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

        {/* Queue depth indicator */}
        {pipelineStats.queueDepth > 0 && (
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs border border-yellow-500/30 bg-yellow-500/10 text-yellow-300">
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            {pipelineStats.queueDepth} queued
          </div>
        )}

        {/* Settings toggle */}
        <button
          onClick={() => setSettingsOpen((prev) => !prev)}
          className="ml-auto flex items-center gap-1 px-2.5 py-1 rounded-full text-xs border border-white/10 bg-white/5 text-white/50 hover:text-white/80 hover:bg-white/10 transition-colors"
        >
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
          Settings (Ctrl+M)
        </button>
      </div>

      {/* Settings Panel (collapsible) */}
      {settingsOpen && (
        <div className="mx-3 mt-2 bg-black/30 backdrop-blur-sm rounded-lg border border-white/10 p-4 space-y-4">
          <div className="text-sm text-white/80 font-medium">Pipeline Settings</div>

          <div className="grid grid-cols-2 gap-4">
            {/* STT Model */}
            <div>
              <label className="text-xs text-white/50 block mb-1.5">Speech-to-Text</label>
              <select
                value={sttModel}
                onChange={(e) => switchSTTModel(e.target.value as STTModel)}
                className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-1.5 text-sm text-white/90 focus:outline-none focus:border-blue-500/50"
              >
                <option value="parakeet" disabled={!sttAvailableModels.some(m => m.name === "parakeet" && m.available && !m.runtime_broken)}>
                  Parakeet (NVIDIA NeMo) {sttAvailableModels.some(m => m.name === "parakeet" && m.runtime_broken) ? "— runtime error" : !isSTTServerAvailable ? "— server offline" : ""}
                </option>
                <option value="whisper" disabled={!sttAvailableModels.some(m => m.name === "whisper" && m.available && !m.runtime_broken)}>
                  Whisper (faster-whisper) {sttAvailableModels.some(m => m.name === "whisper" && m.runtime_broken) ? "— runtime error" : !isSTTServerAvailable ? "— server offline" : ""}
                </option>
                <option value="web-speech">Web Speech API (browser)</option>
              </select>
              <div className="text-[10px] text-white/30 mt-1">
                {isSTTServerAvailable
                  ? `Server: localhost:8890 (${sttModelLabel} active)`
                  : "Server offline — using browser fallback"}
              </div>
            </div>

            {/* TTS Engine */}
            <div>
              <label className="text-xs text-white/50 block mb-1.5">Text-to-Speech</label>
              <select
                value={ttsEngine}
                onChange={(e) => switchTTSEngine(e.target.value as TTSEngine)}
                className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-1.5 text-sm text-white/90 focus:outline-none focus:border-blue-500/50"
              >
                <option value="kokoro">Kokoro (neural, GPU)</option>
                <option value="piper">Piper (lightweight, CPU)</option>
                <option value="browser">Browser (Web Speech API)</option>
              </select>
              <div className="text-[10px] text-white/30 mt-1">
                {isTTSServerAvailable
                  ? `Server: localhost:8880 (${ttsEngineLabel})`
                  : "Server offline — using browser fallback"}
              </div>
            </div>
          </div>

          {/* Interim transcript toggle */}
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="showInterim"
              checked={showInterim}
              onChange={(e) => setShowInterim(e.target.checked)}
              className="rounded border-white/20 bg-white/10"
            />
            <label htmlFor="showInterim" className="text-xs text-white/60">
              Show real-time interim transcripts
            </label>
          </div>
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex-1 flex gap-3 px-3 py-3 min-h-0">
        {/* Left: Conversation Transcript */}
        <div className="flex-1 min-w-0 overflow-y-auto bg-black/20 backdrop-blur-sm rounded-lg border border-white/10 p-3">
          {messages.length === 0 && !interimTranscript ? (
            <div className="flex flex-col items-center justify-center h-full">
              <p className="text-white/70 text-sm mb-3">{stateLabels[state]}</p>

              {/* Pipeline readiness */}
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
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex ${msg.speaker === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div className={`max-w-[75%] ${getMessageStyle(msg)} border rounded-lg px-3 py-2`}>
                    <div className="flex items-center justify-between gap-3 mb-1">
                      <span className={`text-xs font-medium ${getMessageLabelColor(msg)}`}>
                        {getMessageLabel(msg)}
                      </span>
                      <span className="text-xs text-white/40">{formatTime(msg.timestamp)}</span>
                    </div>
                    <div
                      className={`text-sm text-white/90 leading-relaxed ${
                        msg.type === "queued" ? "italic text-white/50" : ""
                      }`}
                    >
                      {msg.text}
                    </div>
                  </div>
                </div>
              ))}

              {/* Interim transcript */}
              {showInterim && interimTranscript && (
                <div className="flex justify-end">
                  <div className="max-w-[75%] bg-blue-600/20 border-blue-500/30 border border-dashed rounded-lg px-3 py-2">
                    <div className="flex items-center gap-3 mb-1">
                      <span className="text-xs font-medium text-blue-300/70">You (speaking...)</span>
                    </div>
                    <div className="text-sm text-white/60 leading-relaxed italic">
                      {interimTranscript}
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Right Column: Stats & Diagnostics */}
        <div className="flex flex-col gap-3 w-80 overflow-y-auto">

          {/* Session Overview */}
          <div className="bg-black/20 backdrop-blur-sm rounded-lg border border-white/10 p-3">
            <div className="text-xs text-white/60 mb-2">Session</div>
            <div className="space-y-1.5">
              <StatRow label="Uptime" value={isActive ? uptimeStr : "--"} />
              <StatRow label="Queries" value={fmtNum(queryCount)} />
              <StatRow label="Errors" value={fmtNum(errorCount)} color={errorCount === 0 ? "text-green-400" : "text-red-400"} />
              {pipelineStats.queueDepth > 0 && (
                <StatRow label="Queue" value={`${pipelineStats.queueDepth} pending`} color="text-yellow-400" />
              )}
            </div>
          </div>

          {/* STT Performance */}
          <div className="bg-black/20 backdrop-blur-sm rounded-lg border border-white/10 p-3">
            <div className="text-xs text-white/60 mb-2">STT Performance</div>
            <div className="space-y-1.5">
              <StatRow label="Model" value={sttModelLabel} />
              <StatRow label="Server" value={isSTTServerAvailable ? "Connected" : "Offline"} color={isSTTServerAvailable ? "text-green-400" : "text-yellow-400"} />
              <StatRow label="Requests" value={fmtNum(pipelineStats.stt.requestCount)} />
              <StatRow label="Errors" value={fmtNum(pipelineStats.stt.errorCount)} color={pipelineStats.stt.errorCount === 0 ? "text-green-400" : "text-red-400"} />
              <StatRow label="Last latency" value={fmtMs(pipelineStats.stt.lastLatencyMs)} />
              <StatRow label="Avg latency" value={fmtMs(pipelineStats.stt.avgLatencyMs)} />
              <StatRow label="Words" value={fmtNum(pipelineStats.stt.wordsTranscribed)} />
              <StatRow label="Audio processed" value={fmtSec(pipelineStats.stt.totalAudioSeconds)} />
            </div>
          </div>

          {/* TTS Performance */}
          <div className="bg-black/20 backdrop-blur-sm rounded-lg border border-white/10 p-3">
            <div className="text-xs text-white/60 mb-2">TTS Performance</div>
            <div className="space-y-1.5">
              <StatRow label="Engine" value={ttsEngineLabel} />
              <StatRow label="Server" value={isTTSServerAvailable ? "Connected" : "Offline"} color={isTTSServerAvailable ? "text-green-400" : "text-yellow-400"} />
              <StatRow label="Requests" value={fmtNum(pipelineStats.tts.requestCount)} />
              <StatRow label="Errors" value={fmtNum(pipelineStats.tts.errorCount)} color={pipelineStats.tts.errorCount === 0 ? "text-green-400" : "text-red-400"} />
              <StatRow label="Last latency" value={fmtMs(pipelineStats.tts.lastLatencyMs)} />
            </div>
          </div>

          {/* Layer 2 / RAG */}
          <div className="bg-purple-500/10 backdrop-blur-sm rounded-lg border border-purple-500/20 p-3">
            <div className="text-xs text-purple-300 font-medium mb-2">AI / RAG</div>
            <div className="space-y-1.5">
              <StatRow label="Status" value={layer2Status} color={
                layer2Status === "ready" ? "text-green-400" :
                layer2Status === "processing" ? "text-yellow-400" :
                layer2Status === "error" ? "text-red-400" : "text-white/60"
              } />
              <StatRow label="Last latency" value={fmtMs(pipelineStats.layer2LastLatencyMs)} />
              {layer2Response?.intent && (
                <>
                  <StatRow label="Intent" value={layer2Response.intent.type} color="text-purple-300" />
                  {layer2Response.intent.confidence > 0 && (
                    <StatRow label="Confidence" value={`${Math.round(layer2Response.intent.confidence * 100)}%`} />
                  )}
                </>
              )}
              {layer2Response?.rag_results && layer2Response.rag_results.length > 0 && (
                <>
                  <div className="text-[10px] text-white/40 mt-1">RAG Domains:</div>
                  {layer2Response.rag_results.map((r: any, i: number) => (
                    <div key={i} className="flex items-center gap-1.5 text-xs">
                      <div className={`w-1.5 h-1.5 rounded-full ${r.success ? "bg-green-400" : "bg-red-400"}`} />
                      <span className={r.success ? "text-green-300" : "text-red-300"}>{r.domain}</span>
                      <span className="text-white/30 ml-auto">{r.execution_time_ms}ms</span>
                    </div>
                  ))}
                </>
              )}
              {layer2Response?.processing_time_ms && (
                <StatRow label="Total processing" value={`${layer2Response.processing_time_ms}ms`} />
              )}
            </div>
          </div>

          {/* Pipeline End-to-End */}
          <div className="bg-black/20 backdrop-blur-sm rounded-lg border border-white/10 p-3">
            <div className="text-xs text-white/60 mb-2">Pipeline E2E</div>
            <div className="space-y-1.5">
              <StatRow label="Total latency" value={fmtMs(pipelineStats.totalPipelineMs)} />
              <StatRow label="Pipeline" value={`${sttModelLabel} → RAG → ${ttsEngineLabel}`} />
            </div>
          </div>

          {/* Audio Waveform (CSS-based) */}
          {isActive && (
            <div className="bg-black/20 backdrop-blur-sm rounded-lg border border-white/10 p-3">
              <div className="text-xs text-white/60 mb-2">
                {isTTSSpeaking ? "AI Speaking" : state === "listening" ? "Listening" : "Ready"}
              </div>
              <div className="flex items-end justify-center gap-[3px] h-10">
                {Array.from({ length: 16 }).map((_, i) => (
                  <div
                    key={i}
                    className={`w-[3px] rounded-full transition-all duration-150 ${
                      isTTSSpeaking
                        ? "bg-purple-400 animate-pulse"
                        : state === "listening"
                          ? "bg-blue-400 animate-pulse"
                          : "bg-white/20"
                    }`}
                    style={{
                      height: isActive
                        ? `${6 + Math.sin(i * 0.7) * 12 + Math.random() * 8}px`
                        : "4px",
                      animationDelay: `${i * 0.05}s`,
                    }}
                  />
                ))}
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
          onInputChange={(id) => {
            setSelectedInputDevice(id);
            if (id) localStorage.setItem("cc-input-device", id);
            else localStorage.removeItem("cc-input-device");
          }}
          onOutputChange={(id) => {
            setSelectedOutputDevice(id);
            if (id) localStorage.setItem("cc-output-device", id);
            else localStorage.removeItem("cc-output-device");
          }}
          disabled={isActive}
        />

        {/* Voice Control Bar */}
        <div className="bg-black/20 backdrop-blur-sm rounded-lg border border-white/10 px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            {/* Waveform when listening */}
            {state === "listening" && (
              <div className="flex items-center gap-[2px] h-5">
                {Array.from({ length: 5 }).map((_, i) => (
                  <div
                    key={i}
                    className="w-[3px] bg-blue-400 rounded-full animate-pulse"
                    style={{
                      animationDelay: `${i * 0.1}s`,
                      height: `${8 + i * 4}px`,
                    }}
                  />
                ))}
              </div>
            )}

            <span className="text-sm text-white/60">{stateLabels[state]}</span>

            {interimTranscript && (
              <span className="text-sm text-blue-400/70 italic ml-2 max-w-xs truncate">
                &quot;{interimTranscript}&quot;
              </span>
            )}

            {layer2Status === "processing" && (
              <span className="text-sm text-purple-400/70 ml-2 animate-pulse">
                RAG processing...
              </span>
            )}

            {pipelineStats.queueDepth > 0 && (
              <span className="text-sm text-yellow-400/70 ml-2">
                ({pipelineStats.queueDepth} queued)
              </span>
            )}
          </div>

          <div className="flex items-center gap-3">
            <span className="text-xs text-white/40">Shift+Space</span>

            {isTTSSpeaking && (
              <button
                onClick={stopTTS}
                className="px-3 py-1.5 rounded-lg text-xs font-medium bg-yellow-500/20 text-yellow-400 hover:bg-yellow-500/30 transition-colors"
              >
                Stop Speaking
              </button>
            )}

            <button
              onClick={isActive ? stop : start}
              className={`
                px-6 py-2 rounded-lg text-sm font-medium transition-all
                ${
                  isActive
                    ? "bg-red-500 text-white hover:bg-red-600"
                    : state === "error"
                      ? "bg-yellow-500 text-black hover:bg-yellow-400"
                      : "bg-blue-500 text-white hover:bg-blue-600"
                }
              `}
            >
              {isActive
                ? "Stop"
                : state === "error"
                  ? "Retry"
                  : "Speak"}
            </button>
          </div>
        </div>

        {/* Error banner */}
        {error && (
          <div className="px-4 py-2 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-xs space-y-1">
            <div className="font-mono">{error}</div>
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

        {/* Pipeline info bar */}
        <div className="px-4 py-2 bg-blue-500/10 border border-blue-500/30 rounded-lg text-blue-400 text-xs">
          <strong>V2 Pipeline:</strong> {sttModelLabel} STT → AI Orchestrator → {ttsEngineLabel} TTS
        </div>
      </div>
    </div>
  );
}
