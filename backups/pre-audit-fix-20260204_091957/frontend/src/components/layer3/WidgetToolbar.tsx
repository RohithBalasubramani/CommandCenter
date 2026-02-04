"use client";

import React, { useCallback } from "react";
import type { WidgetSize } from "@/types";

interface WidgetToolbarProps {
  scenario: string;
  widgetKey: string;
  isPinned: boolean;
  isFocused: boolean;
  size: WidgetSize;
  onPin: () => void;
  onDismiss: () => void;
  onResize: (size: WidgetSize) => void;
  onFocus: () => void;
  onUnfocus: () => void;
  onSnapshot: () => void;
}

/** Next size in the cycle: compact → normal → expanded → compact */
const NEXT_SIZE: Record<string, WidgetSize> = {
  compact: "normal",
  normal: "expanded",
  expanded: "compact",
};

/**
 * WidgetToolbar — hover-reveal toolbar for widget interactions.
 *
 * Renders as a floating bar at top-right of the widget.
 * Appears on parent group-hover with a fade-in.
 * 5 actions: Pin, Resize, Focus, Snapshot, Dismiss.
 */
export default function WidgetToolbar({
  scenario,
  widgetKey,
  isPinned,
  isFocused,
  size,
  onPin,
  onDismiss,
  onResize,
  onFocus,
  onUnfocus,
  onSnapshot,
}: WidgetToolbarProps) {
  const handleResize = useCallback(() => {
    if (size === "hero") return; // Hero can't resize via toolbar
    onResize(NEXT_SIZE[size] || "normal");
  }, [size, onResize]);

  const handleFocusToggle = useCallback(() => {
    if (isFocused) onUnfocus();
    else onFocus();
  }, [isFocused, onFocus, onUnfocus]);

  return (
    <div
      className="absolute top-2 right-2 z-10 flex items-center gap-0.5 rounded-lg bg-neutral-800/80 backdrop-blur-sm border border-neutral-700/50 px-1 py-0.5 opacity-0 group-hover:opacity-100 transition-opacity duration-150 pointer-events-none group-hover:pointer-events-auto"
      onClick={(e) => e.stopPropagation()}
    >
      {/* Pin */}
      <ToolbarButton
        title={isPinned ? "Unpin widget" : "Pin widget"}
        onClick={onPin}
        active={isPinned}
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill={isPinned ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 17v5" />
          <path d="M9 10.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24V16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V7a1 1 0 0 1 1-1 1 1 0 0 0 1-1V4a1 1 0 0 0-1-1H8a1 1 0 0 0-1 1v1a1 1 0 0 0 1 1 1 1 0 0 1 1 1z" />
        </svg>
      </ToolbarButton>

      {/* Resize (cycle) — hidden for hero */}
      {size !== "hero" && (
        <ToolbarButton
          title={`Resize: ${size} → ${NEXT_SIZE[size] || "normal"}`}
          onClick={handleResize}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M15 3h6v6" />
            <path d="M9 21H3v-6" />
            <path d="M21 3l-7 7" />
            <path d="M3 21l7-7" />
          </svg>
        </ToolbarButton>
      )}

      {/* Focus / Unfocus */}
      <ToolbarButton
        title={isFocused ? "Back to dashboard" : "Focus — ask AI about this"}
        onClick={handleFocusToggle}
        active={isFocused}
      >
        {isFocused ? (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M9 21H3v-6" />
            <path d="M15 3h6v6" />
            <path d="M3 21l7-7" />
            <path d="M21 3l-7 7" />
          </svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <circle cx="12" cy="12" r="6" />
            <circle cx="12" cy="12" r="2" />
          </svg>
        )}
      </ToolbarButton>

      {/* Snapshot */}
      <ToolbarButton title="Save snapshot" onClick={onSnapshot}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z" />
          <circle cx="12" cy="13" r="3" />
        </svg>
      </ToolbarButton>

      {/* Dismiss */}
      <ToolbarButton title="Dismiss widget" onClick={onDismiss}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M18 6L6 18" />
          <path d="M6 6l12 12" />
        </svg>
      </ToolbarButton>
    </div>
  );
}

// --- Small icon button ---

function ToolbarButton({
  title,
  onClick,
  active,
  children,
}: {
  title: string;
  onClick: () => void;
  active?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      title={title}
      onClick={onClick}
      className={`p-1 rounded transition-colors ${
        active
          ? "text-blue-400 hover:text-blue-300"
          : "text-neutral-400 hover:text-white"
      }`}
    >
      {children}
    </button>
  );
}
