"use client";

import React, { useMemo, useRef, useState, useEffect, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { getWidgetComponent } from "@/components/layer4/widgetRegistry";
import { FIXTURES } from "@/components/layer4/fixtureData";
import { useLayoutState } from "./useLayoutState";
import BlobGrid from "./BlobGrid";
import WidgetSlot from "./WidgetSlot";
import { commandCenterBus } from "@/lib/events";
import type { WidgetInstruction, WidgetSize, WidgetHeightHint } from "@/types";

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

          return (
            <motion.div
              key={key}
              layout
              initial={{ opacity: 0, x: 60, scale: 0.95 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              exit={{ opacity: 0, x: -60, scale: 0.95 }}
              transition={{
                type: "spring",
                stiffness: 300,
                damping: 30,
                delay: index * 0.05, // Stagger: 50ms per widget
              }}
              className={sizeClasses(instruction.size, instruction.heightHint)}
              data-scenario={instruction.scenario}
              data-size={instruction.size}
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
                  onPin={() => pinWidget(key)}
                  onDismiss={() => dismissWidget(key)}
                  onResize={(s: WidgetSize) => resizeWidget(key, s)}
                  onFocus={() => focusWidget(key, instruction.scenario, getFocusLabel(instruction))}
                  onUnfocus={unfocus}
                  onSnapshot={saveSnapshot}
                  onDrillDown={(ctx) => drillDown(instruction.scenario, getFocusLabel(instruction), ctx)}
                >
                  <WidgetComponent data={resolveWidgetData(instruction)} />
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
 * Size + heightHint → grid column-span + row-span + max-height classes.
 * Column span comes from WidgetSize; row span from heightHint.
 * Hero always gets row-span-4 regardless of hint.
 */
function sizeClasses(size: string, heightHint?: WidgetHeightHint): string {
  const h = heightHint ?? "medium";
  const heightCls = HEIGHT_CLASSES[h];

  switch (size) {
    case "hero":
      return "col-span-12 row-span-4";
    case "expanded":
      return `col-span-6 ${heightCls}`;
    case "normal":
      return `col-span-4 ${heightCls}`;
    case "compact":
      return `col-span-3 ${heightCls}`;
    case "hidden":
      return "hidden";
    default:
      return `col-span-4 ${heightCls}`;
  }
}
