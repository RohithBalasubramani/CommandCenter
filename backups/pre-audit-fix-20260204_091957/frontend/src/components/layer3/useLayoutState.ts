"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { commandCenterBus } from "@/lib/events";
import type { LayoutJSON, WidgetSize, LayoutSnapshot } from "@/types";
import { DEFAULT_LAYOUT } from "./defaultLayout";

/** Generate a stable key for a widget instruction. */
export function widgetKey(
  scenario: string,
  fixture: string,
  index: number
): string {
  return `${scenario}-${fixture}-${index}`;
}

const SNAPSHOT_STORAGE_KEY = "cc-snapshots";
const MAX_SNAPSHOTS = 10;

/**
 * useLayoutState — manages the current LayoutJSON + widget interactions.
 *
 * - Initializes with DEFAULT_LAYOUT (project engineer dashboard)
 * - Subscribes to LAYOUT_UPDATE events from event bus
 * - Provides pin, dismiss, resize, focus, and snapshot actions
 * - Pinned widgets persist across voice-driven layout changes
 */
export function useLayoutState() {
  const [layout, setLayoutState] = useState<LayoutJSON>(DEFAULT_LAYOUT);
  const [pinnedKeys, setPinnedKeys] = useState<Set<string>>(new Set());
  const [focusedKey, setFocusedKey] = useState<string | null>(null);
  const preFocusLayoutRef = useRef<LayoutJSON | null>(null);

  // Subscribe to LAYOUT_UPDATE events — preserve pinned widgets
  useEffect(() => {
    const unsub = commandCenterBus.on("LAYOUT_UPDATE", (event) => {
      if (event.type === "LAYOUT_UPDATE") {
        console.info("[useLayoutState] LAYOUT_UPDATE received, widgets:", event.layout.widgets?.length, "heading:", event.layout.heading);
        setLayoutState((prev) => {
          // Guard: ignore empty widget arrays — keep current dashboard
          if (!event.layout.widgets || event.layout.widgets.length === 0) {
            console.warn("[useLayoutState] Ignoring empty widget array");
            return prev;
          }

          // If nothing is pinned, just replace
          if (pinnedKeys.size === 0) return event.layout;

          // Collect pinned widgets from current layout
          const pinned = prev.widgets.filter((w, i) =>
            pinnedKeys.has(widgetKey(w.scenario, w.fixture, i))
          );

          // Build new layout with pinned widgets prepended
          const newScenarios = new Set(
            event.layout.widgets.map((w) => w.scenario)
          );
          // Only keep pinned widgets that aren't already in the new layout
          const keptPinned = pinned.filter(
            (w) => !newScenarios.has(w.scenario)
          );

          return {
            ...event.layout,
            widgets: [...keptPinned, ...event.layout.widgets],
          };
        });
      }
    });
    return unsub;
  }, [pinnedKeys]);

  // Programmatic layout update (also emits to bus so other listeners can react)
  const setLayout = useCallback((newLayout: LayoutJSON) => {
    setLayoutState(newLayout);
    commandCenterBus.emit({ type: "LAYOUT_UPDATE", layout: newLayout });
  }, []);

  // --- Widget Actions ---

  /** Toggle pin on a widget. Pinned widgets survive voice layout changes. */
  const pinWidget = useCallback((key: string) => {
    setPinnedKeys((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  /** Dismiss a widget (remove from layout with exit animation). */
  const dismissWidget = useCallback((key: string) => {
    setLayoutState((prev) => ({
      ...prev,
      widgets: prev.widgets.filter(
        (w, i) => widgetKey(w.scenario, w.fixture, i) !== key
      ),
    }));
    // Also unpin if it was pinned
    setPinnedKeys((prev) => {
      const next = new Set(prev);
      next.delete(key);
      return next;
    });
  }, []);

  /** Resize a widget to a new size. */
  const resizeWidget = useCallback((key: string, newSize: WidgetSize) => {
    setLayoutState((prev) => ({
      ...prev,
      widgets: prev.widgets.map((w, i) =>
        widgetKey(w.scenario, w.fixture, i) === key
          ? { ...w, size: newSize }
          : w
      ),
    }));
  }, []);

  /** Focus a widget: expand to hero, emit drill-down event. */
  const focusWidget = useCallback(
    (key: string, scenario: string, label: string) => {
      // Save pre-focus layout for restore
      setLayoutState((prev) => {
        preFocusLayoutRef.current = prev;
        return {
          ...prev,
          widgets: prev.widgets.map((w, i) =>
            widgetKey(w.scenario, w.fixture, i) === key
              ? { ...w, size: "hero" as WidgetSize, relevance: 1.0 }
              : { ...w, size: "compact" as WidgetSize, relevance: 0.3 }
          ),
        };
      });
      setFocusedKey(key);
      // Emit focus event — voice pipeline picks this up for drill-down
      commandCenterBus.emit({ type: "WIDGET_FOCUS", scenario, label });
    },
    []
  );

  /** Unfocus: restore the pre-focus layout. */
  const unfocus = useCallback(() => {
    if (preFocusLayoutRef.current) {
      setLayoutState(preFocusLayoutRef.current);
      preFocusLayoutRef.current = null;
    }
    setFocusedKey(null);
  }, []);

  // Escape key to unfocus
  useEffect(() => {
    if (!focusedKey) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") unfocus();
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [focusedKey, unfocus]);

  /** Save current layout as a snapshot to localStorage. */
  const saveSnapshot = useCallback(() => {
    const snapshot: LayoutSnapshot = {
      id: `snap-${Date.now()}`,
      timestamp: Date.now(),
      heading: layout.heading || "Untitled",
      layout: { ...layout },
    };

    try {
      const existing = JSON.parse(
        localStorage.getItem(SNAPSHOT_STORAGE_KEY) || "[]"
      ) as LayoutSnapshot[];
      const updated = [snapshot, ...existing].slice(0, MAX_SNAPSHOTS);
      localStorage.setItem(SNAPSHOT_STORAGE_KEY, JSON.stringify(updated));
    } catch {
      // localStorage full or unavailable — ignore
    }

    commandCenterBus.emit({ type: "WIDGET_SNAPSHOT" });
  }, [layout]);

  return {
    layout,
    setLayout,
    // Widget actions
    pinnedKeys,
    focusedKey,
    pinWidget,
    dismissWidget,
    resizeWidget,
    focusWidget,
    unfocus,
    saveSnapshot,
  };
}
