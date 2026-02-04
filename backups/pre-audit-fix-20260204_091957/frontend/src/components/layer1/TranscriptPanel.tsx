"use client";

import { useEffect, useState } from "react";
import { Transcript } from "@/types";
import { commandCenterBus } from "@/lib/events";

/**
 * TranscriptPanel — Displays the running transcript from Layer 1.
 * Subscribes to TRANSCRIPT_UPDATE events from the event bus.
 * This panel shows what Layer 2 receives as input.
 */
export default function TranscriptPanel() {
  const [transcripts, setTranscripts] = useState<Transcript[]>([]);

  useEffect(() => {
    const unsub = commandCenterBus.on("TRANSCRIPT_UPDATE", (event) => {
      if (event.type === "TRANSCRIPT_UPDATE") {
        setTranscripts((prev) => {
          const existing = prev.findIndex(
            (t) => t.id === event.transcript.id
          );
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

  if (transcripts.length === 0) return null;

  return (
    <div className="absolute top-4 right-4 w-80 max-h-64 overflow-y-auto rounded-lg border border-[var(--cc-border)] bg-[var(--cc-surface)] shadow-lg">
      <div className="px-3 py-2 border-b border-[var(--cc-border)] text-xs text-[var(--cc-text-muted)] font-mono">
        Transcript → Layer 2
      </div>
      <div className="p-3 space-y-2">
        {transcripts.slice(-10).map((t) => (
          <div key={t.id} className="text-xs">
            <span className="text-[var(--cc-text-muted)]">
              [{t.role === "user" ? "USR" : "AST"}]
            </span>{" "}
            <span
              className={`${
                t.isFinal
                  ? "text-[var(--cc-text)]"
                  : "text-[var(--cc-text-muted)] italic"
              }`}
            >
              {t.text}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
