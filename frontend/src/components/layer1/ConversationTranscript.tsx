"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import { submitFeedback, getLastQueryId } from "@/lib/layer2/client";

export interface TranscriptMessage {
  id: string;
  speaker: "user" | "ai";
  text: string;
  timestamp: number;
  duration?: number; // Duration of speech in seconds
  queryId?: string; // For RL feedback tracking
}

interface ConversationTranscriptProps {
  messages: TranscriptMessage[];
  isUserSpeaking: boolean;
  isAISpeaking: boolean;
  interimText?: string; // Real-time transcription preview
}

/**
 * ResponseFeedback — Thumbs up/down buttons for RL feedback.
 */
function ResponseFeedback({
  messageId,
  queryId,
}: {
  messageId: string;
  queryId?: string;
}) {
  const [feedbackState, setFeedbackState] = useState<"up" | "down" | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const handleFeedback = useCallback(async (rating: "up" | "down") => {
    // Don't allow changing feedback after submission
    if (feedbackState !== null) return;

    setSubmitting(true);
    try {
      // Use queryId from message if available, otherwise get from storage
      const effectiveQueryId = queryId || getLastQueryId();
      if (effectiveQueryId) {
        const success = await submitFeedback(rating);
        if (success) {
          setFeedbackState(rating);
        }
      } else {
        console.warn("[Feedback] No query_id available");
      }
    } catch (err) {
      console.error("[Feedback] Submit failed:", err);
    } finally {
      setSubmitting(false);
    }
  }, [feedbackState, queryId]);

  return (
    <div className="flex items-center gap-1 mt-2 pt-1 border-t border-white/10">
      <span className="text-[10px] text-white/40 mr-1">Rate:</span>
      <button
        onClick={() => handleFeedback("up")}
        disabled={submitting || feedbackState !== null}
        className={`p-1 rounded transition-colors ${
          feedbackState === "up"
            ? "text-green-400 bg-green-500/20"
            : feedbackState === "down"
            ? "text-white/20 cursor-not-allowed"
            : "text-white/40 hover:text-green-400 hover:bg-green-500/10"
        }`}
        title="Good response"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill={feedbackState === "up" ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M7 10v12" />
          <path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z" />
        </svg>
      </button>
      <button
        onClick={() => handleFeedback("down")}
        disabled={submitting || feedbackState !== null}
        className={`p-1 rounded transition-colors ${
          feedbackState === "down"
            ? "text-red-400 bg-red-500/20"
            : feedbackState === "up"
            ? "text-white/20 cursor-not-allowed"
            : "text-white/40 hover:text-red-400 hover:bg-red-500/10"
        }`}
        title="Bad response"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill={feedbackState === "down" ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M17 14V2" />
          <path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88Z" />
        </svg>
      </button>
      {feedbackState && (
        <span className="text-[10px] text-white/30 ml-1">
          {feedbackState === "up" ? "Thanks!" : "Noted"}
        </span>
      )}
    </div>
  );
}

export default function ConversationTranscript({
  messages,
  isUserSpeaking,
  isAISpeaking,
  interimText,
}: ConversationTranscriptProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages or interim text
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, interimText]);

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  };

  const formatDuration = (duration?: number) => {
    if (!duration) return "";
    return `${duration.toFixed(1)}s`;
  };

  return (
    <div className="flex flex-col h-full bg-black/20 backdrop-blur-sm rounded-lg border border-white/10">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/10">
        <h2 className="text-sm font-medium text-white/80">Conversation</h2>
        <div className="flex gap-2">
          {isUserSpeaking && (
            <div className="flex items-center gap-1.5 text-xs text-blue-400">
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
              You're speaking
            </div>
          )}
          {isAISpeaking && (
            <div className="flex items-center gap-1.5 text-xs text-purple-400">
              <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse" />
              AI responding
            </div>
          )}
        </div>
      </div>

      {/* Messages */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 py-3 space-y-3 scrollbar-thin scrollbar-thumb-white/20 scrollbar-track-transparent"
      >
        {messages.length === 0 && !interimText && (
          <div className="flex items-center justify-center h-full text-white/40 text-sm">
            Start speaking to begin the conversation
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.speaker === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[75%] ${
                msg.speaker === "user"
                  ? "bg-blue-600/30 border-blue-500/40"
                  : "bg-purple-600/30 border-purple-500/40"
              } border rounded-lg px-3 py-2`}
            >
              {/* Speaker label and timestamp */}
              <div className="flex items-center justify-between gap-3 mb-1">
                <span
                  className={`text-xs font-medium ${
                    msg.speaker === "user" ? "text-blue-300" : "text-purple-300"
                  }`}
                >
                  {msg.speaker === "user" ? "You" : "AI Assistant"}
                </span>
                <span className="text-xs text-white/40">
                  {formatTime(msg.timestamp)}
                  {msg.duration && (
                    <span className="ml-1.5 text-white/30">
                      ({formatDuration(msg.duration)})
                    </span>
                  )}
                </span>
              </div>

              {/* Message text */}
              <div className="text-sm text-white/90 leading-relaxed">
                {msg.text}
              </div>

              {/* Feedback buttons for AI messages */}
              {msg.speaker === "ai" && (
                <ResponseFeedback messageId={msg.id} queryId={msg.queryId} />
              )}
            </div>
          </div>
        ))}

        {/* Interim (live) transcription preview */}
        {interimText && (
          <div className="flex justify-end">
            <div className="max-w-[75%] bg-blue-600/20 border-blue-500/30 border border-dashed rounded-lg px-3 py-2">
              <div className="flex items-center justify-between gap-3 mb-1">
                <span className="text-xs font-medium text-blue-300/70">
                  You (typing...)
                </span>
                <span className="text-xs text-white/30">
                  {formatTime(Date.now())}
                </span>
              </div>
              <div className="text-sm text-white/60 leading-relaxed italic">
                {interimText}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer with message count */}
      <div className="px-4 py-2 border-t border-white/10 text-xs text-white/40">
        {messages.length} {messages.length === 1 ? "message" : "messages"}
        {interimText && " • Listening..."}
      </div>
    </div>
  );
}
