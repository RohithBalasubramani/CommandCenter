"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { commandCenterBus } from "@/lib/events";
import type { LayoutJSON, WidgetSize, LayoutSnapshot, WidgetInstruction } from "@/types";
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
 * Relevance decay configuration (per README blueprint):
 *   - 5-minute decay to 0.2 baseline
 *   - Pinned widgets immune to decay
 *   - Decay tick every 30 seconds
 */
const DECAY_INTERVAL_MS = 30_000;        // tick every 30s
const DECAY_TOTAL_MS = 5 * 60_000;       // full decay over 5 minutes
const DECAY_BASELINE = 0.2;              // floor relevance value
const DECAY_PER_TICK = (1.0 - DECAY_BASELINE) / (DECAY_TOTAL_MS / DECAY_INTERVAL_MS); // ~0.08 per tick

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

  // --- Layout History (for back button) ---
  const layoutHistoryRef = useRef<LayoutJSON[]>([]);
  const [hasHistory, setHasHistory] = useState(false);

  // --- Interactive Mode ---
  const [interactiveCtx, setInteractiveCtx] = useState<{
    key: string;
    scenario: string;
    label: string;
    equipment: string;
    metric: string;
  } | null>(null);
  const preInteractiveLayoutRef = useRef<LayoutJSON | null>(null);

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

          // Push current layout to history for back navigation (max 10)
          if (prev.widgets.length > 0) {
            layoutHistoryRef.current = [...layoutHistoryRef.current.slice(-9), prev];
            setHasHistory(true);
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

  /** Enter interactive mode: full-screen widget with context-locked conversation. */
  const enterInteractive = useCallback(
    (key: string, instruction: WidgetInstruction) => {
      const dataOverride = instruction.data_override || {};
      const equipment = (dataOverride._equipment_id as string) || "";
      const metric = (dataOverride._metric as string) || "";
      const label =
        (dataOverride.label as string) ||
        ((dataOverride.demoData as Record<string, unknown>)?.label as string) ||
        instruction.scenario.replace(/-/g, " ");

      // Save current layout for restore on exit
      setLayoutState((prev) => {
        preInteractiveLayoutRef.current = prev;
        return prev;
      });

      setInteractiveCtx({
        key,
        scenario: instruction.scenario,
        label,
        equipment,
        metric,
      });
      setFocusedKey(key);

      // Emit interactive event — voice pipeline picks this up
      commandCenterBus.emit({
        type: "WIDGET_INTERACTIVE_ENTER",
        scenario: instruction.scenario,
        label,
        equipment,
        metric,
      });
    },
    []
  );

  /** Legacy focus (non-interactive): expand to hero, dim others. */
  const focusWidget = useCallback(
    (key: string, scenario: string, label: string) => {
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
      commandCenterBus.emit({ type: "WIDGET_FOCUS", scenario, label });
    },
    []
  );

  /** Go back to previous layout (layout history). */
  const goBack = useCallback(() => {
    const history = layoutHistoryRef.current;
    if (history.length === 0) return;
    const prev = history.pop()!;
    setLayoutState(prev);
    setHasHistory(history.length > 0);
  }, []);

  /** Exit interactive mode: restore pre-interactive layout. */
  const exitInteractive = useCallback(() => {
    if (preInteractiveLayoutRef.current) {
      setLayoutState(preInteractiveLayoutRef.current);
      preInteractiveLayoutRef.current = null;
    }
    setInteractiveCtx(null);
    setFocusedKey(null);
    commandCenterBus.emit({ type: "WIDGET_INTERACTIVE_EXIT" });
  }, []);

  /** Unfocus: restore the pre-focus layout (non-interactive). */
  const unfocus = useCallback(() => {
    // If in interactive mode, exit interactive instead
    if (interactiveCtx) {
      exitInteractive();
      return;
    }
    if (preFocusLayoutRef.current) {
      setLayoutState(preFocusLayoutRef.current);
      preFocusLayoutRef.current = null;
    }
    setFocusedKey(null);
  }, [interactiveCtx, exitInteractive]);

  // Escape key to unfocus / exit interactive
  useEffect(() => {
    if (!focusedKey && !interactiveCtx) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (interactiveCtx) exitInteractive();
        else unfocus();
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [focusedKey, interactiveCtx, unfocus, exitInteractive]);

  // --- Relevance Decay ---
  // Tick every DECAY_INTERVAL_MS: reduce each non-pinned widget's relevance
  // by DECAY_PER_TICK, clamped to DECAY_BASELINE floor.
  useEffect(() => {
    const timer = setInterval(() => {
      setLayoutState((prev) => {
        let changed = false;
        const widgets = prev.widgets.map((w, i) => {
          const key = widgetKey(w.scenario, w.fixture, i);
          // Pinned widgets are immune to decay
          if (pinnedKeys.has(key)) return w;
          // Already at baseline
          if (w.relevance <= DECAY_BASELINE) return w;
          changed = true;
          const newRelevance = Math.max(DECAY_BASELINE, w.relevance - DECAY_PER_TICK);
          return { ...w, relevance: parseFloat(newRelevance.toFixed(3)) };
        });
        return changed ? { ...prev, widgets } : prev;
      });
    }, DECAY_INTERVAL_MS);

    return () => clearInterval(timer);
  }, [pinnedKeys]);

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
    // Layout history
    hasHistory,
    goBack,
    // Interactive mode
    interactiveCtx,
    enterInteractive,
    exitInteractive,
  };
}
