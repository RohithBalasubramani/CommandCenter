"use client";

import { useEffect, useRef, useState } from "react";

interface TextInputOverlayProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (text: string) => void;
}

/**
 * TextInputOverlay — Full-screen text input with backdrop blur.
 *
 * Toggled with Ctrl+Shift+K. Lets the user type a query directly
 * to Layer 2 (bypasses STT). Everything behind is blurred.
 */
export default function TextInputOverlay({
  open,
  onClose,
  onSubmit,
}: TextInputOverlayProps) {
  const [value, setValue] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  // Focus input when overlay opens
  useEffect(() => {
    if (open) {
      setValue("");
      // Small delay so the DOM has rendered
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  // Escape to close
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  const handleSubmit = () => {
    const trimmed = value.trim();
    if (!trimmed) return;
    onSubmit(trimmed);
    setValue("");
    onClose();
  };

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-md"
        onClick={onClose}
      />

      {/* Input container */}
      <div className="relative z-10 w-full max-w-2xl px-6">
        <div className="bg-neutral-900/90 border border-neutral-700 rounded-xl shadow-2xl overflow-hidden">
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-2.5 border-b border-neutral-800">
            <span className="text-xs font-mono text-neutral-400">
              Type your query — Enter to send, Esc to cancel
            </span>
            <span className="text-[10px] font-mono text-neutral-600">
              Ctrl+Shift+K
            </span>
          </div>

          {/* Input row */}
          <div className="flex items-center gap-3 px-4 py-3">
            {/* Prompt indicator */}
            <span className="text-blue-400 font-mono text-lg shrink-0">❯</span>

            <input
              ref={inputRef}
              type="text"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit();
                }
              }}
              placeholder="Ask anything..."
              className="flex-1 bg-transparent text-lg text-neutral-100 placeholder-neutral-600 outline-none font-sans"
              autoComplete="off"
              spellCheck={false}
            />

            {/* Send button */}
            <button
              onClick={handleSubmit}
              disabled={!value.trim()}
              className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all shrink-0 ${
                value.trim()
                  ? "bg-blue-500 text-white hover:bg-blue-600"
                  : "bg-neutral-800 text-neutral-600 cursor-not-allowed"
              }`}
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
