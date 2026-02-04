"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import Canvas from "@/components/canvas/Canvas";
import Blob from "@/components/layer3/Blob";
import VoiceInterfaceV2 from "@/components/layer1/VoiceInterfaceV2";
import TranscriptPanel from "@/components/layer1/TranscriptPanel";
import VoiceControlBar from "@/components/layer1/VoiceControlBar";
import TextInputOverlay from "@/components/layer1/TextInputOverlay";
import StatusBar from "@/components/status-bar/StatusBar";
import { DebugPanel } from "@/components/debug";
import { commandCenterBus } from "@/lib/events";
import type { Transcript } from "@/types";

/**
 * Command Center — Main Page
 *
 * Two views toggled with Ctrl+B:
 * - "voice": Layer 1 voice interface (full canvas)
 * - "dashboard" (default): Layer 3 Blob widget grid + voice overlay
 *
 * Transcript Viewer: Icon button in top-right, shows all conversation history
 * Debug Panel: Press Ctrl+D to toggle pipeline visualization
 */
export default function CommandCenterPage() {
  const [view, setView] = useState<"voice" | "dashboard">("dashboard");
  const [showTranscripts, setShowTranscripts] = useState(false);
  const [showTextInput, setShowTextInput] = useState(false);
  const [transcripts, setTranscripts] = useState<Transcript[]>([]);
  const transcriptEndRef = useRef<HTMLDivElement>(null);

  // Ctrl+B toggles view, Ctrl+Shift+K toggles text input overlay
  if (typeof window !== "undefined") {
    // eslint-disable-next-line react-hooks/rules-of-hooks
    useState(() => {
      const handler = (e: KeyboardEvent) => {
        if (e.ctrlKey && e.key === "b") {
          e.preventDefault();
          setView((v) => (v === "voice" ? "dashboard" : "voice"));
        }
        if (e.ctrlKey && e.shiftKey && e.key === "K") {
          e.preventDefault();
          setShowTextInput((v) => !v);
        }
      };
      window.addEventListener("keydown", handler);
      return () => window.removeEventListener("keydown", handler);
    });
  }

  const handleTextSubmit = useCallback((text: string) => {
    commandCenterBus.emit({ type: "TEXT_INPUT_SUBMIT", text });
  }, []);

  // Subscribe to transcript events for the transcript viewer
  useEffect(() => {
    const unsub = commandCenterBus.on("TRANSCRIPT_UPDATE", (event) => {
      if (event.type === "TRANSCRIPT_UPDATE") {
        setTranscripts((prev) => {
          const existing = prev.findIndex((t) => t.id === event.transcript.id);
          if (existing >= 0) {
            const updated = [...prev];
            updated[existing] = event.transcript;
            return updated;
          }
          return [...prev, event.transcript];
        });
      }
    });
    return unsub;
  }, []);

  // Auto-scroll transcript viewer
  useEffect(() => {
    if (showTranscripts) {
      transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [transcripts, showTranscripts]);

  return (
    <div data-testid="command-center">
      <Canvas statusBar={<StatusBar />}>
        {view === "voice" ? (
          <>
            {/* Layer 1: Voice I/O — full canvas */}
            <VoiceInterfaceV2 />
            <TranscriptPanel />
          </>
        ) : (
          <>
            {/* Layer 3: Blob — widget dashboard */}
            <Blob />

            {/* Layer 1: Compact voice control bar — floating at bottom */}
            <div className="absolute bottom-14 left-1/2 -translate-x-1/2 z-10">
              <VoiceControlBar />
            </div>
          </>
        )}
      </Canvas>

      {/* Top-right control buttons */}
      <div className="fixed top-3 right-3 z-50 flex flex-col gap-2 items-end">
        {/* View toggle */}
        <button
          onClick={() => setView((v) => (v === "voice" ? "dashboard" : "voice"))}
          className="px-3 py-1.5 rounded-md text-xs font-mono bg-neutral-800 text-neutral-300 border border-neutral-700 hover:bg-neutral-700 transition-colors"
        >
          {view === "voice" ? "Dashboard (Ctrl+B)" : "Voice UI (Ctrl+B)"}
        </button>

        {/* Text input toggle */}
        <button
          onClick={() => setShowTextInput((v) => !v)}
          className={`w-9 h-9 rounded-md flex items-center justify-center border transition-colors ${
            showTextInput
              ? "bg-blue-600 border-blue-500 text-white"
              : "bg-neutral-800 border-neutral-700 text-neutral-400 hover:bg-neutral-700 hover:text-neutral-200"
          }`}
          title="Text input (Ctrl+Shift+K)"
          data-testid="text-input-toggle"
        >
          {/* Keyboard icon */}
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <rect x="2" y="4" width="20" height="16" rx="2" ry="2" />
            <path d="M6 8h.001M10 8h.001M14 8h.001M18 8h.001M8 12h.001M12 12h.001M16 12h.001M7 16h10" />
          </svg>
        </button>

        {/* Transcript viewer toggle */}
        <button
          onClick={() => setShowTranscripts((v) => !v)}
          className={`w-9 h-9 rounded-md flex items-center justify-center border transition-colors ${
            showTranscripts
              ? "bg-blue-600 border-blue-500 text-white"
              : "bg-neutral-800 border-neutral-700 text-neutral-400 hover:bg-neutral-700 hover:text-neutral-200"
          }`}
          title="Transcript history"
        >
          {/* Chat/transcript icon (SVG) */}
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
        </button>
      </div>

      {/* Transcript viewer panel */}
      {showTranscripts && (
        <div className="fixed top-16 right-3 z-50 w-96 max-h-[70vh] flex flex-col rounded-lg border border-neutral-700 bg-neutral-900/95 backdrop-blur-md shadow-2xl">
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-2.5 border-b border-neutral-700">
            <span className="text-sm font-medium text-neutral-200">
              Transcripts
            </span>
            <div className="flex items-center gap-2">
              <span className="text-xs text-neutral-500">
                {transcripts.length} message{transcripts.length !== 1 ? "s" : ""}
              </span>
              {transcripts.length > 0 && (
                <button
                  onClick={() => setTranscripts([])}
                  className="text-xs text-neutral-500 hover:text-neutral-300 transition-colors"
                >
                  Clear
                </button>
              )}
              <button
                onClick={() => setShowTranscripts(false)}
                className="text-neutral-500 hover:text-neutral-300 transition-colors ml-1"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6L6 18M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-3 space-y-2.5 min-h-[100px]">
            {transcripts.length === 0 ? (
              <div className="text-center text-neutral-500 text-sm py-8">
                No transcripts yet. Start speaking to see conversation history.
              </div>
            ) : (
              transcripts.map((t) => (
                <div key={t.id} className="group">
                  <div className="flex items-start gap-2">
                    {/* Role indicator */}
                    <span
                      className={`text-[10px] font-mono mt-0.5 shrink-0 px-1.5 py-0.5 rounded ${
                        t.role === "user"
                          ? "bg-blue-500/20 text-blue-400"
                          : "bg-purple-500/20 text-purple-400"
                      }`}
                    >
                      {t.role === "user" ? "YOU" : "AI"}
                    </span>

                    {/* Text */}
                    <div className="flex-1 min-w-0">
                      <p
                        className={`text-sm leading-relaxed ${
                          t.isFinal
                            ? "text-neutral-200"
                            : "text-neutral-400 italic"
                        }`}
                      >
                        {t.text}
                      </p>
                      <span className="text-[10px] text-neutral-600 opacity-0 group-hover:opacity-100 transition-opacity">
                        {new Date(t.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                  </div>
                </div>
              ))
            )}
            <div ref={transcriptEndRef} />
          </div>
        </div>
      )}

      {/* Text Input Overlay — Toggle with Ctrl+Shift+K */}
      <TextInputOverlay
        open={showTextInput}
        onClose={() => setShowTextInput(false)}
        onSubmit={handleTextSubmit}
      />

      {/* Debug Panel — Toggle with Ctrl+D */}
      <DebugPanel />
    </div>
  );
}
