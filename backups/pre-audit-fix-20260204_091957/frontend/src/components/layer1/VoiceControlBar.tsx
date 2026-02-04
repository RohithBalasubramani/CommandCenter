"use client";

import { useEffect, useRef } from "react";
import { useVoicePipeline } from "./useVoicePipeline";
import { commandCenterBus } from "@/lib/events";

/**
 * VoiceControlBar — Compact floating voice controls for dashboard view.
 *
 * Default: push-to-talk (hold Shift+Space to speak, release to stop).
 * Toggle to continuous mode with the mode button.
 */

export default function VoiceControlBar() {
  const {
    state,
    isActive,
    error,
    start,
    stop,
    inputMode,
    setInputMode,
    sendTextDirect,
    interimTranscript,
    isTTSSpeaking,
    stopTTS,
    layer2Status,
    pipelineStats,
    layer2Error,
    isSTTServerAvailable,
    isTTSServerAvailable,
    messages,
    userTranscript,
  } = useVoicePipeline();

  // Prevent Shift+Space keydown repeat from re-triggering start
  const shiftSpaceHeldRef = useRef(false);

  // Listen for TEXT_INPUT_SUBMIT from the text input overlay
  useEffect(() => {
    const unsub = commandCenterBus.on("TEXT_INPUT_SUBMIT", (event) => {
      if (event.type === "TEXT_INPUT_SUBMIT") {
        sendTextDirect(event.text);
      }
    });
    return unsub;
  }, [sendTextDirect]);

  // Emit TRANSCRIPT_UPDATE events so TranscriptPanel and transcript viewer can subscribe
  useEffect(() => {
    if (messages.length === 0) return;
    const last = messages[messages.length - 1];
    commandCenterBus.emit({
      type: "TRANSCRIPT_UPDATE",
      transcript: {
        id: last.id,
        role: last.speaker === "user" ? "user" : "assistant",
        text: last.text,
        timestamp: last.timestamp,
        isFinal: last.type !== "queued",
      },
    });
  }, [messages]);

  // Keyboard: Shift+Space
  // Push-to-talk (default): hold to speak, release to stop — ignores key repeat
  // Continuous: toggle on/off
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === "Space" && e.shiftKey) {
        e.preventDefault();
        if (inputMode === "push-to-talk") {
          if (!shiftSpaceHeldRef.current) {
            shiftSpaceHeldRef.current = true;
            if (!isActive) start();
          }
        } else {
          if (isActive) stop();
          else start();
        }
      }
    };
    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        shiftSpaceHeldRef.current = false;
        if (inputMode === "push-to-talk" && isActive) {
          e.preventDefault();
          stop();
        }
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [isActive, start, stop, inputMode]);

  // Derive visual state
  const isListening = state === "listening";
  const isSpeaking = state === "speaking";
  const isProcessing = layer2Status === "processing";
  const isError = state === "error";

  // Status text
  const statusText = isListening
    ? userTranscript || interimTranscript || "Listening..."
    : isSpeaking
      ? "Speaking..."
      : isProcessing
        ? "Thinking..."
        : isError
          ? error || "Error"
          : inputMode === "push-to-talk"
            ? "Hold Shift+Space to speak"
            : "Shift+Space or click mic";

  return (
    <div className="bg-black/50 backdrop-blur-xl rounded-2xl border border-white/10 px-5 py-3 flex items-center gap-4 shadow-2xl min-w-[420px]">
      {/* Mic button with animated ring */}
      <button
        onClick={isActive ? stop : start}
        onMouseDown={inputMode === "push-to-talk" && !isActive ? () => start() : undefined}
        onMouseUp={inputMode === "push-to-talk" && isActive ? () => stop() : undefined}
        className={`
          relative w-11 h-11 rounded-full flex items-center justify-center shrink-0
          transition-all duration-200
          ring-2 shadow-md
          ${isListening
            ? "bg-blue-500 ring-blue-500 shadow-blue-500/40 scale-110"
            : isSpeaking
              ? "bg-purple-500 ring-purple-500 shadow-purple-500/40"
              : isProcessing
                ? "bg-yellow-500/80 ring-yellow-500 shadow-yellow-500/30"
                : isError
                  ? "bg-red-500 ring-red-500 shadow-red-500/30"
                  : "bg-white/10 ring-transparent hover:bg-white/20"
          }
        `}
      >
        {/* Mic icon (idle/listening) */}
        {!isSpeaking && !isProcessing && (
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-white">
            <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
            <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
            <line x1="12" x2="12" y1="19" y2="22" />
          </svg>
        )}
        {/* Waveform icon (speaking) */}
        {isSpeaking && (
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="text-white">
            <path d="M2 12h2M6 8v8M10 4v16M14 6v12M18 8v8M22 12h-2" />
          </svg>
        )}
        {/* Spinner (processing) */}
        {isProcessing && !isSpeaking && (
          <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
        )}

        {/* Pulsing ring when listening */}
        {isListening && (
          <>
            <span className="absolute inset-0 rounded-full bg-blue-500 animate-ping opacity-30" />
            <span className="absolute inset-[-4px] rounded-full border-2 border-blue-400 animate-pulse opacity-50" />
          </>
        )}
      </button>

      {/* Waveform bars when listening */}
      {isListening && (
        <div className="flex items-center gap-[3px] h-6 shrink-0">
          {Array.from({ length: 7 }).map((_, i) => (
            <div
              key={i}
              className="w-[3px] rounded-full bg-blue-400"
              style={{
                animation: `voiceWave 0.8s ease-in-out ${i * 0.08}s infinite alternate`,
                height: `${6 + Math.sin(i * 0.9) * 10 + 8}px`,
              }}
            />
          ))}
          <style>{`
            @keyframes voiceWave {
              0% { transform: scaleY(0.4); opacity: 0.5; }
              100% { transform: scaleY(1); opacity: 1; }
            }
          `}</style>
        </div>
      )}

      {/* Status text */}
      <div className="flex-1 min-w-0">
        <span className={`text-sm truncate block ${
          isListening
            ? "text-blue-300"
            : isSpeaking
              ? "text-purple-300 animate-pulse"
              : isProcessing
                ? "text-yellow-300 animate-pulse"
                : isError
                  ? "text-red-400"
                  : "text-white/40"
        }`}>
          {isListening && (userTranscript || interimTranscript)
            ? `"${(userTranscript || interimTranscript).slice(-80)}"`
            : statusText
          }
        </span>

        {/* Error detail */}
        {layer2Error && !error && (
          <span className="text-xs text-yellow-400 truncate block mt-0.5">
            L2: {layer2Error}
          </span>
        )}
      </div>

      {/* Queue depth */}
      {pipelineStats.queueDepth > 0 && (
        <span className="text-xs text-yellow-400/70 shrink-0">
          {pipelineStats.queueDepth} queued
        </span>
      )}

      {/* Stop TTS button */}
      {isTTSSpeaking && (
        <button
          onClick={stopTTS}
          className="px-3 py-1.5 rounded-lg text-xs font-medium bg-yellow-500/20 text-yellow-400 hover:bg-yellow-500/30 transition-colors shrink-0"
        >
          Stop
        </button>
      )}

      {/* Mode toggle */}
      <button
        onClick={() => setInputMode(inputMode === "continuous" ? "push-to-talk" : "continuous")}
        className={`px-2 py-1 rounded-lg text-[10px] font-mono shrink-0 transition-colors ${
          inputMode === "push-to-talk"
            ? "text-blue-400/70 hover:text-blue-300 hover:bg-blue-500/10"
            : "text-white/30 hover:text-white/60 hover:bg-white/5"
        }`}
        title={inputMode === "continuous" ? "Switch to push-to-talk" : "Switch to continuous"}
      >
        {inputMode === "continuous" ? "CONT" : "PTT"}
      </button>

      {/* Server status dots */}
      <div className="flex gap-1.5 shrink-0" title={`STT: ${isSTTServerAvailable ? "OK" : "offline"} | TTS: ${isTTSServerAvailable ? "OK" : "offline"}`}>
        <div className={`w-1.5 h-1.5 rounded-full ${isSTTServerAvailable ? "bg-green-400" : "bg-red-400"}`} />
        <div className={`w-1.5 h-1.5 rounded-full ${isTTSServerAvailable ? "bg-green-400" : "bg-red-400"}`} />
      </div>
    </div>
  );
}
