"use client";

import React, { useEffect, useRef } from "react";

export interface TranscriptMessage {
  id: string;
  speaker: "user" | "ai";
  text: string;
  timestamp: number;
  duration?: number; // Duration of speech in seconds
}

interface ConversationTranscriptProps {
  messages: TranscriptMessage[];
  isUserSpeaking: boolean;
  isAISpeaking: boolean;
  interimText?: string; // Real-time transcription preview
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
