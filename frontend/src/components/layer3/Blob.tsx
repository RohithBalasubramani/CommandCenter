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
import type { WidgetInstruction, WidgetSize, WidgetHeightHint, TransitionType, ConfidenceEnvelope } from "@/types";

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
 *
 * Handles three data-shape mismatches between backend and widgets:
 * 1. Both fixture & override have demoData → deep merge
 * 2. Fixture has demoData object but override is flat → inject into demoData
 * 3. Fixture uses "data" key but override uses "demoData" → map across
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

  const override = instruction.data_override;

  // Shallow merge: top-level fixture fields + override fields
  const merged = { ...fixtureData, ...override };

  const fixtureDemoData = fixtureData.demoData;
  const overrideDemoData = override.demoData;
  const fixtureDataField = fixtureData.data;

  // ── Case 1: Both have demoData ──
  if (fixtureDemoData && typeof fixtureDemoData === "object" && overrideDemoData != null) {
    if (Array.isArray(overrideDemoData)) {
      // Override demoData is array (e.g. alerts list) — use directly
      merged.demoData = overrideDemoData;
    } else if (typeof overrideDemoData === "object") {
      // Both objects — deep merge (fixture defaults + override values)
      merged.demoData = {
        ...(fixtureDemoData as Record<string, unknown>),
        ...(overrideDemoData as Record<string, unknown>),
      };
    }
  }
  // ── Case 2: Fixture has demoData object, override is flat (no demoData key) ──
  // Widgets like kpi/trend/distribution read spec.demoData.* but backend
  // sends {label, value, unit, timeSeries, ...} flat in data_override.
  else if (
    fixtureDemoData &&
    typeof fixtureDemoData === "object" &&
    !Array.isArray(fixtureDemoData) &&
    overrideDemoData == null
  ) {
    const dataFields: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(override)) {
      if (!key.startsWith("_")) {
        dataFields[key] = value;
      }
    }
    if (Object.keys(dataFields).length > 0) {
      merged.demoData = {
        ...(fixtureDemoData as Record<string, unknown>),
        ...dataFields,
      };
    }
  }

  // ── Case 3: Fixture uses "data" key, override uses "demoData" ──
  // Widgets like alerts read data.data but backend sends demoData.
  if (
    fixtureDataField &&
    typeof fixtureDataField === "object" &&
    overrideDemoData != null &&
    !override.data
  ) {
    if (Array.isArray(overrideDemoData) && overrideDemoData.length > 0) {
      merged.data = overrideDemoData[0];
    } else if (typeof overrideDemoData === "object" && !Array.isArray(overrideDemoData)) {
      merged.data = {
        ...(fixtureDataField as Record<string, unknown>),
        ...(overrideDemoData as Record<string, unknown>),
      };
    }
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
    hasHistory,
    goBack,
    interactiveCtx,
    enterInteractive,
    exitInteractive,
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

  /** Render the widget grid content (shared between normal and interactive mode). */
  const renderWidgets = () => (
    <AnimatePresence mode="popLayout">
      {sortedWidgets.map((instruction, index) => {
        const key = widgetKey(instruction, index);
        const WidgetComponent = getWidgetComponent(instruction.scenario);
        const isDemo = instruction.data_override?._data_source === "demo";

        const txType: TransitionType =
          layout.transitions?.[instruction.scenario] ||
          layout.transitions?.[key] ||
          DEFAULT_TRANSITION;
        const variants = TRANSITION_VARIANTS[txType];

        const slotProps = {
          scenario: instruction.scenario,
          size: instruction.size,
          noGrid: true as const,
          title: getFocusLabel(instruction),
          description: instruction.description,
          widgetKey: key,
          isPinned: pinnedKeys.has(key),
          isFocused: focusedKey === key,
          isDemo,
          isStale: !!instruction._is_stale,
          conflictFlag: instruction._conflict_flag,
          widgetConfidence: instruction._widget_confidence,
          onPin: () => pinWidget(key),
          onDismiss: () => dismissWidget(key),
          onResize: (s: WidgetSize) => resizeWidget(key, s),
          onFocus: () => enterInteractive(key, instruction),
          onUnfocus: unfocus,
          onSnapshot: saveSnapshot,
          onDrillDown: (ctx: string) => drillDown(instruction.scenario, getFocusLabel(instruction), ctx),
        };

        return (
          <motion.div
            key={key}
            layout
            initial={variants.initial}
            animate={{
              opacity: Math.max(0.4, instruction.relevance ?? 1),
              x: 0, y: 0, scale: 1,
            }}
            exit={variants.exit}
            transition={{ duration: TRANSITION_DURATION_S, delay: 0 }}
            className={sizeClasses(instruction.size, instruction.heightHint)}
            data-scenario={instruction.scenario}
            data-size={instruction.size}
            data-relevance={instruction.relevance}
          >
            {!WidgetComponent ? (
              <WidgetSlot {...slotProps}>
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
              <WidgetSlot {...slotProps}>
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
  );

  // Extract confidence envelope from layout (injected by backend Phase 2)
  const confidence = (layout as Record<string, unknown>)._confidence as ConfidenceEnvelope | undefined;

  // ── Interactive Mode ──
  if (interactiveCtx) {
    return (
      <div className="h-full w-full flex flex-col overflow-hidden" data-interactive-mode="true">
        {/* Context bar — prominent with back button */}
        <div className="shrink-0 flex items-center gap-3 px-4 py-2.5 border-b border-indigo-500/40 bg-indigo-950/50 backdrop-blur-sm">
          <button
            onClick={exitInteractive}
            className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-sm font-medium text-white bg-indigo-600/80 hover:bg-indigo-500 border border-indigo-400/40 transition-colors shadow-sm"
            data-testid="interactive-back-btn"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M19 12H5"/><path d="m12 19-7-7 7-7"/></svg>
            Back
          </button>
          <div className="flex items-center gap-2 min-w-0">
            <span className="text-sm font-semibold text-neutral-100 truncate">
              {interactiveCtx.label}
            </span>
            {interactiveCtx.equipment && (
              <span className="shrink-0 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                {interactiveCtx.equipment}
              </span>
            )}
            {interactiveCtx.metric && (
              <span className="shrink-0 px-2 py-0.5 rounded-full text-[10px] font-mono text-neutral-400 bg-neutral-800/60 border border-neutral-700/30">
                {interactiveCtx.metric}
              </span>
            )}
          </div>
          <div className="ml-auto shrink-0 flex items-center gap-3">
            <span className="text-[10px] uppercase tracking-widest text-indigo-400 font-bold">Interactive</span>
            <span className="text-[10px] text-neutral-500">Esc to exit</span>
          </div>
        </div>
        {/* Widget grid — reuses BlobGrid */}
        <div className="flex-1 min-h-0">
          <BlobGrid heading={layout.heading} confidence={confidence}>
            {renderWidgets()}
          </BlobGrid>
        </div>
      </div>
    );
  }

  // ── Normal Mode ──
  return (
    <div className="h-full w-full flex flex-col overflow-hidden">
      {/* Back button — shown when layout history is available */}
      {hasHistory && (
        <div className="shrink-0 flex items-center gap-3 px-4 py-2 border-b border-neutral-700/30 bg-neutral-900/60 backdrop-blur-sm">
          <button
            onClick={goBack}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium text-neutral-200 bg-neutral-800/80 hover:bg-neutral-700 border border-neutral-600/50 transition-colors"
            data-testid="layout-back-btn"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M19 12H5"/><path d="m12 19-7-7 7-7"/></svg>
            Back
          </button>
          <span className="text-xs text-neutral-500">Previous dashboard</span>
        </div>
      )}
      <div className="flex-1 min-h-0">
        <BlobGrid heading={layout.heading} confidence={confidence}>
          {renderWidgets()}
        </BlobGrid>
      </div>
    </div>
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
      return "col-span-1 md:col-span-6 lg:col-span-12 row-span-4";
    case "expanded":
      return `col-span-1 md:col-span-6 lg:col-span-6 ${heightCls}`;
    case "normal":
      return `col-span-1 md:col-span-3 lg:col-span-4 ${heightCls}`;
    case "compact":
      return `col-span-1 md:col-span-3 lg:col-span-3 ${heightCls}`;
    case "hidden":
      return "hidden";
    default:
      return `col-span-1 md:col-span-3 lg:col-span-4 ${heightCls}`;
  }
}
