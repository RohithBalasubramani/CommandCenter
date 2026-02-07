"use client";

import React, { useMemo, useRef, useState, useEffect, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { getWidgetComponent } from "@/components/layer4/widgetRegistry";
import WidgetWithLifecycle from "@/components/layer4/WidgetWithLifecycle";
import { FIXTURES } from "@/components/layer4/fixtureData";
import { useLayoutState } from "./useLayoutState";
import BlobGrid from "./BlobGrid";
import WidgetSlot from "./WidgetSlot";
import { commandCenterBus } from "@/lib/events";
import { config } from "@/lib/config";
import type { WidgetInstruction, WidgetSize, WidgetHeightHint, TransitionType } from "@/types";

/** Transition duration from config (BLOB_TRANSITION_DURATION flag). */
const TRANSITION_DURATION_S = config.blob.transitionDuration / 1000;

/** Map each blueprint TransitionType to framer-motion initial/exit variants. */
const TRANSITION_VARIANTS: Record<
  TransitionType,
  { initial: Record<string, number>; exit: Record<string, number> }
> = {
  "slide-in-from-top": {
    initial: { opacity: 0, y: -60, scale: 1 },
    exit:    { opacity: 0, y: -60, scale: 1 },
  },
  "slide-in-from-left": {
    initial: { opacity: 0, x: -60, scale: 1 },
    exit:    { opacity: 0, x: -60, scale: 1 },
  },
  expand: {
    initial: { opacity: 0, scale: 0.6, x: 0, y: 0 },
    exit:    { opacity: 0, scale: 0.6, x: 0, y: 0 },
  },
  shrink: {
    initial: { opacity: 0, scale: 1.3, x: 0, y: 0 },
    exit:    { opacity: 0, scale: 1.3, x: 0, y: 0 },
  },
  "fade-out": {
    initial: { opacity: 0, scale: 1, x: 0, y: 0 },
    exit:    { opacity: 0, scale: 1, x: 0, y: 0 },
  },
};

/** Default transition when layout.transitions doesn't specify one. */
const DEFAULT_TRANSITION: TransitionType = "slide-in-from-left";

/**
 * Merge fixture data with data_override from Layer 2.
 * data_override fields take precedence; fixture provides defaults.
 */
function resolveWidgetData(
  instruction: WidgetInstruction
): Record<string, unknown> {
  const scenarioMeta = FIXTURES[instruction.scenario];
  const fixtureKey = instruction.fixture || scenarioMeta?.defaultFixture;
  const fixtureData = scenarioMeta?.variants?.[fixtureKey] ?? {};

  if (!instruction.data_override) {
    return fixtureData as Record<string, unknown>;
  }

  // Shallow merge: top-level fixture fields + override fields
  // For nested demoData, do a deeper merge
  const merged = { ...fixtureData, ...instruction.data_override };

  if (
    fixtureData.demoData &&
    typeof fixtureData.demoData === "object" &&
    instruction.data_override.demoData &&
    typeof instruction.data_override.demoData === "object"
  ) {
    merged.demoData = {
      ...(fixtureData.demoData as Record<string, unknown>),
      ...(instruction.data_override.demoData as Record<string, unknown>),
    };
  }

  return merged;
}

/** Stable key for a widget instruction (scenario + fixture + index). */
function widgetKey(instruction: WidgetInstruction, index: number): string {
  return `${instruction.scenario}-${instruction.fixture}-${index}`;
}

/**
 * Blob — Layer 3 Layout Executor
 *
 * Consumes LayoutJSON (from event bus or default), resolves each widget
 * instruction to a React component + merged data, and renders them
 * inside a viewport-locked CSS Grid with slide+stagger transitions.
 */
export default function Blob() {
  const {
    layout,
    pinnedKeys,
    focusedKey,
    pinWidget,
    dismissWidget,
    resizeWidget,
    focusWidget,
    unfocus,
    saveSnapshot,
  } = useLayoutState();

  // Track layout version to detect changes for animation direction
  const layoutVersionRef = useRef(0);
  const [layoutVersion, setLayoutVersion] = useState(0);
  const prevWidgetKeysRef = useRef<Set<string>>(new Set());

  // Trust backend ordering — hero-first, relevance-ordered, row-packed.
  // Only filter out hidden widgets, no re-sorting.
  const sortedWidgets = useMemo(() => {
    return layout.widgets.filter((w) => w.size !== "hidden");
  }, [layout.widgets]);

  /** Extract a human-readable label from widget data for focus drill-down. */
  const getFocusLabel = useCallback(
    (instruction: WidgetInstruction): string => {
      const data = resolveWidgetData(instruction);
      // Try common data paths for a meaningful label
      const demo = data.demoData as Record<string, unknown> | undefined;
      if (demo?.label && typeof demo.label === "string") return demo.label;
      const inner = data.data as Record<string, unknown> | undefined;
      if (inner?.title && typeof inner.title === "string") return inner.title;
      if (data.title && typeof data.title === "string") return data.title as string;
      // Fallback to scenario name
      return instruction.scenario.replace(/-/g, " ");
    },
    []
  );

  /** Emit drill-down event when user clicks a chart element or widget body. */
  const drillDown = useCallback(
    (scenario: string, label: string, context: string) => {
      commandCenterBus.emit({ type: "WIDGET_DRILL_DOWN", scenario, label, context });
    },
    []
  );

  // Detect layout changes (new set of widgets)
  useEffect(() => {
    const currentKeys = new Set(sortedWidgets.map((w, i) => widgetKey(w, i)));
    const prevKeys = prevWidgetKeysRef.current;

    // Check if widget set actually changed
    const changed =
      currentKeys.size !== prevKeys.size ||
      Array.from(currentKeys).some((k) => !prevKeys.has(k));

    if (changed) {
      layoutVersionRef.current++;
      setLayoutVersion(layoutVersionRef.current);
      prevWidgetKeysRef.current = currentKeys;
    }
  }, [sortedWidgets]);

  return (
    <BlobGrid heading={layout.heading}>
      <AnimatePresence mode="popLayout">
        {sortedWidgets.map((instruction, index) => {
          const key = widgetKey(instruction, index);
          const WidgetComponent = getWidgetComponent(instruction.scenario);
          const isDemo = instruction.data_override?._data_source === "demo";

          // Resolve per-widget transition type from layout.transitions map
          const txType: TransitionType =
            layout.transitions?.[instruction.scenario] ||
            layout.transitions?.[key] ||
            DEFAULT_TRANSITION;
          const variants = TRANSITION_VARIANTS[txType];

          return (
            <motion.div
              key={key}
              layout
              initial={variants.initial}
              animate={{
                opacity: Math.max(0.4, instruction.relevance ?? 1),
                x: 0,
                y: 0,
                scale: 1,
              }}
              exit={variants.exit}
              transition={{
                duration: TRANSITION_DURATION_S,
                delay: 0, // All widgets appear at once after full processing
              }}
              className={sizeClasses(instruction.size, instruction.heightHint)}
              data-scenario={instruction.scenario}
              data-size={instruction.size}
              data-relevance={instruction.relevance}
            >
              {!WidgetComponent ? (
                <WidgetSlot
                  scenario={instruction.scenario}
                  size={instruction.size}
                  noGrid
                  title={getFocusLabel(instruction)}
                  description={instruction.description}
                  widgetKey={key}
                  isPinned={pinnedKeys.has(key)}
                  isFocused={focusedKey === key}
                  isDemo={isDemo}
                  onPin={() => pinWidget(key)}
                  onDismiss={() => dismissWidget(key)}
                  onResize={(s: WidgetSize) => resizeWidget(key, s)}
                  onFocus={() => focusWidget(key, instruction.scenario, getFocusLabel(instruction))}
                  onUnfocus={unfocus}
                  onSnapshot={saveSnapshot}
                  onDrillDown={(ctx) => drillDown(instruction.scenario, getFocusLabel(instruction), ctx)}
                >
                  <div className="h-full flex items-center justify-center p-4">
                    <div className="text-center">
                      <p className="text-[10px] uppercase font-bold tracking-widest text-neutral-400 mb-1">
                        Unknown Widget
                      </p>
                      <p className="text-xs text-neutral-500 font-mono">
                        {instruction.scenario}
                      </p>
                    </div>
                  </div>
                </WidgetSlot>
              ) : (
                <WidgetSlot
                  scenario={instruction.scenario}
                  size={instruction.size}
                  noGrid
                  title={getFocusLabel(instruction)}
                  description={instruction.description}
                  widgetKey={key}
                  isPinned={pinnedKeys.has(key)}
                  isFocused={focusedKey === key}
                  isDemo={isDemo}
                  onPin={() => pinWidget(key)}
                  onDismiss={() => dismissWidget(key)}
                  onResize={(s: WidgetSize) => resizeWidget(key, s)}
                  onFocus={() => focusWidget(key, instruction.scenario, getFocusLabel(instruction))}
                  onUnfocus={unfocus}
                  onSnapshot={saveSnapshot}
                  onDrillDown={(ctx) => drillDown(instruction.scenario, getFocusLabel(instruction), ctx)}
                >
                  <WidgetWithLifecycle
                    scenario={instruction.scenario}
                    data={resolveWidgetData(instruction)}
                    Component={WidgetComponent}
                  />
                </WidgetSlot>
              )}
            </motion.div>
          );
        })}
      </AnimatePresence>
    </BlobGrid>
  );
}

/**
 * Row-span + optional max-height per heightHint.
 * short/medium get max-height to prevent CSS Grid stretching them
 * to match taller siblings in the same implicit row.
 */
const HEIGHT_CLASSES: Record<WidgetHeightHint, string> = {
  short:   "row-span-1",
  medium:  "row-span-2",
  tall:    "row-span-3",
  "x-tall": "row-span-4",
};

/**
 * Size + heightHint → responsive grid column-span + row-span classes.
 *
 * Responsive breakpoints (per README blueprint):
 *   mobile (<768):     1 column  → all widgets full width
 *   tablet (768-1279): 6 columns → compact=3, normal=3, expanded=6, hero=6
 *   laptop/desktop (1280+): 12 columns → compact=3, normal=4, expanded=6, hero=12
 */
function sizeClasses(size: string, heightHint?: WidgetHeightHint): string {
  const h = heightHint ?? "medium";
  const heightCls = HEIGHT_CLASSES[h];

  switch (size) {
    case "hero":
      return "col-span-1 md:col-span-6 xl:col-span-12 row-span-4";
    case "expanded":
      return `col-span-1 md:col-span-6 xl:col-span-6 ${heightCls}`;
    case "normal":
      return `col-span-1 md:col-span-3 xl:col-span-4 ${heightCls}`;
    case "compact":
      return `col-span-1 md:col-span-3 xl:col-span-3 ${heightCls}`;
    case "hidden":
      return "hidden";
    default:
      return `col-span-1 md:col-span-3 xl:col-span-4 ${heightCls}`;
  }
}
