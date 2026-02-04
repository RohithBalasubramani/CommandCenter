"use client";

import React, { Suspense, useState, useMemo, useCallback, useEffect, useRef, Component, type ErrorInfo, type ReactNode } from "react";
import { FIXTURES } from "@/components/layer4/fixtureData";
import { getWidgetComponent, getAvailableScenarios } from "@/components/layer4/widgetRegistry";
import WidgetSlot from "@/components/layer3/WidgetSlot";
import type { WidgetSize } from "@/types";
import { WidgetFeedbackForm, FeedbackChecklist, PageFeedbackForm } from "./FeedbackForm";
import SampleDashboard, { SAMPLE_DASHBOARDS } from "./SampleDashboard";
import {
  exportAllFeedback,
  getAllWidgetFeedback,
  getSizeAdjustment,
  saveSizeAdjustment,
  clearSizeAdjustment,
  raiseIssueSizeAdjustment,
  type SizeAdjustment,
  type VariantSizeAdjustments,
} from "./feedbackStore";
import SimulationView from "./SimulationView";

// ── Scenario grouping ──

interface ScenarioInfo {
  key: string;
  name: string;
  variants: string[];
  isMulti: boolean;
}

function buildScenarioList(): ScenarioInfo[] {
  const allKeys = getAvailableScenarios();
  return allKeys
    .map((key) => {
      const meta = FIXTURES[key];
      const variants = meta ? Object.keys(meta.variants) : ["default-render"];
      return {
        key,
        name: meta?.name ?? key,
        variants,
        isMulti: variants.length > 1,
      };
    })
    .sort((a, b) => {
      // Multi-variant first, then alphabetical
      if (a.isMulti !== b.isMulti) return a.isMulti ? -1 : 1;
      return a.key.localeCompare(b.key);
    });
}

// ── Per-scenario responsive size presets (vw for width, vh for height) ──

type SizePreset = { w: string; h: string };
type ScenarioSizes = { [size in WidgetSize]?: SizePreset };

const DEFAULT_KPI_SIZES: ScenarioSizes = {
  compact:  { w: "20vw", h: "14vh" },
  normal:   { w: "25vw", h: "16vh" },
  expanded: { w: "35vw", h: "18vh" },
};

const DEFAULT_CHART_SIZES: ScenarioSizes = {
  normal:   { w: "35vw", h: "40vh" },
  expanded: { w: "48vw", h: "50vh" },
  hero:     { w: "65vw", h: "60vh" },
};

const SCENARIO_SIZE_OVERRIDES: Record<string, ScenarioSizes> = {
  comparison: {
    normal:   { w: "35vw", h: "18vh" },
    expanded: { w: "48vw", h: "24vh" },
    hero:     { w: "65vw", h: "28vh" },
  },
  composition: {
    normal:   { w: "30vw", h: "40vh" },
    expanded: { w: "38vw", h: "50vh" },
    hero:     { w: "45vw", h: "55vh" },
  },
  distribution: {
    normal:   { w: "35vw", h: "38vh" },
    expanded: { w: "48vw", h: "45vh" },
    hero:     { w: "65vw", h: "50vh" },
  },
  eventlogstream: {
    normal:   { w: "26vw", h: "50vh" },
    expanded: { w: "36vw", h: "60vh" },
    hero:     { w: "46vw", h: "70vh" },
  },
  "flow-sankey": {
    normal:   { w: "40vw", h: "50vh" },
    expanded: { w: "55vw", h: "55vh" },
    hero:     { w: "70vw", h: "60vh" },
  },
  "matrix-heatmap": {
    normal:   { w: "35vw", h: "40vh" },
    expanded: { w: "48vw", h: "48vh" },
    hero:     { w: "65vw", h: "55vh" },
  },
  timeline: {
    normal:   { w: "40vw", h: "24vh" },
    expanded: { w: "55vw", h: "28vh" },
    hero:     { w: "70vw", h: "32vh" },
  },
  "trends-cumulative": {
    normal:   { w: "40vw", h: "45vh" },
    expanded: { w: "55vw", h: "50vh" },
    hero:     { w: "70vw", h: "55vh" },
  },
};

// ── Per-variant size overrides (take priority over scenario-level) ──
// Key format: "scenario/variant" — exact match only

const VARIANT_SIZE_OVERRIDES: Record<string, ScenarioSizes> = {
  // ── category-bar ──
  "category-bar/shift-comparison": {
    compact:  { w: "35vw", h: "40vh" },
    normal:   { w: "48vw", h: "50vh" },
    expanded: { w: "60vw", h: "58vh" },
    hero:     { w: "75vw", h: "65vh" },
  },
  // ── comparison ──
  "comparison/delta_bar_visual-deviation-bar": {
    normal:   { w: "35vw", h: "18vh" },
    expanded: { w: "48vw", h: "18vh" },
    hero:     { w: "65vw", h: "18vh" },
  },
  "comparison/side_by_side_visual-plain-values": {
    normal:   { w: "35vw", h: "16vh" },
    expanded: { w: "48vw", h: "16vh" },
    hero:     { w: "65vw", h: "16vh" },
  },
  "comparison/grouped_bar_visual-phase-comparison": {
    normal:   { w: "35vw", h: "44vh" },
    expanded: { w: "48vw", h: "44vh" },
    hero:     { w: "65vw", h: "31vh" },
  },
  "comparison/waterfall_visual-loss-analysis": {
    normal:   { w: "35vw", h: "44vh" },
    expanded: { w: "48vw", h: "44vh" },
    hero:     { w: "65vw", h: "44vh" },
  },
  "comparison/composition_split_visual-load-type": {
    normal:   { w: "35vw", h: "27vh" },
    expanded: { w: "48vw", h: "34vh" },
    hero:     { w: "65vw", h: "40vh" },
  },
  // ── distribution ──
  "distribution/dist_energy_source_share-donut": {
    normal:   { w: "24vw", h: "34vh" },
    expanded: { w: "32vw", h: "40vh" },
    hero:     { w: "42vw", h: "46vh" },
  },
  "distribution/dist_energy_source_share-100-stacked-bar": {
    normal:   { w: "35vw", h: "6vh" },
    expanded: { w: "48vw", h: "6vh" },
    hero:     { w: "65vw", h: "6vh" },
  },
  // ── eventlogstream ──
  "eventlogstream/correlation-stack": {
    normal:   { w: "26vw", h: "64vh" },
    expanded: { w: "36vw", h: "64vh" },
    hero:     { w: "46vw", h: "70vh" },
  },
  // ── flow-sankey ──
  "flow-sankey/flow_sankey_standard-classic-left-to-right-sankey": {
    normal:   { w: "40vw", h: "50vh" },
    expanded: { w: "55vw", h: "64vh" },
    hero:     { w: "70vw", h: "77vh" },
  },
  "flow-sankey/flow_sankey_energy_balance-sankey-with-explicit-loss-branches-dropping-out": {
    normal:   { w: "31vw", h: "44vh" },
    expanded: { w: "42vw", h: "52vh" },
    hero:     { w: "55vw", h: "60vh" },
  },
  "flow-sankey/flow_sankey_multi_source-many-to-one-flow-diagram": {
    normal:   { w: "32vw", h: "56vh" },
    expanded: { w: "42vw", h: "58vh" },
    hero:     { w: "55vw", h: "58vh" },
  },
  "flow-sankey/flow_sankey_layered-multi-stage-hierarchical-flow": {
    normal:   { w: "32vw", h: "52vh" },
    expanded: { w: "42vw", h: "52vh" },
    hero:     { w: "55vw", h: "30vh" },
  },
  "flow-sankey/flow_sankey_time_sliced-sankey-with-time-scrubberplayer": {
    normal:   { w: "32vw", h: "43vh" },
    expanded: { w: "42vw", h: "42vh" },
    hero:     { w: "55vw", h: "46vh" },
  },
  // ── kpi ──
  "kpi/kpi_alert-warning-state": {
    compact:  { w: "20vw", h: "16vh" },
    normal:   { w: "25vw", h: "18vh" },
    expanded: { w: "35vw", h: "20vh" },
  },
  // ── matrix-heatmap ──
  "matrix-heatmap/value-heatmap": {
    normal:   { w: "16vw", h: "36vh" },
    expanded: { w: "16vw", h: "36vh" },
    hero:     { w: "16vw", h: "36vh" },
  },
  "matrix-heatmap/correlation-matrix": {
    normal:   { w: "35vw", h: "66vh" },
    expanded: { w: "48vw", h: "87vh" },
    hero:     { w: "60vw", h: "112vh" },
  },
  "matrix-heatmap/calendar-heatmap": {
    normal:   { w: "35vw", h: "42vh" },
    expanded: { w: "48vw", h: "42vh" },
    hero:     { w: "60vw", h: "52vh" },
  },
  // ── timeline ──
  "timeline/linear-incident-timeline": {
    normal:   { w: "40vw", h: "24vh" },
    expanded: { w: "55vw", h: "28vh" },
    hero:     { w: "70vw", h: "32vh" },
  },
  "timeline/machine-state-timeline": {
    normal:   { w: "40vw", h: "24vh" },
    expanded: { w: "55vw", h: "28vh" },
    hero:     { w: "70vw", h: "32vh" },
  },
  "timeline/multi-lane-shift-schedule": {
    compact:  { w: "30vw", h: "24vh" },
    normal:   { w: "40vw", h: "28vh" },
    expanded: { w: "55vw", h: "32vh" },
    hero:     { w: "70vw", h: "36vh" },
  },
};

function getVariantSizes(scenario: string, variant: string): ScenarioSizes {
  // Check variant-level overrides first (exact match only — no prefix/startsWith)
  const variantKey = `${scenario}/${variant}`;
  if (VARIANT_SIZE_OVERRIDES[variantKey]) return VARIANT_SIZE_OVERRIDES[variantKey];
  // Fall back to scenario-level
  if (SCENARIO_SIZE_OVERRIDES[scenario]) return SCENARIO_SIZE_OVERRIDES[scenario];
  if (scenario === "kpi") return DEFAULT_KPI_SIZES;
  return DEFAULT_CHART_SIZES;
}

// ── Error boundary — catches render crashes per-widget ──

interface EBProps { children: ReactNode; fallbackLabel?: string }
interface EBState { hasError: boolean; error?: Error }

class WidgetErrorBoundary extends Component<EBProps, EBState> {
  constructor(props: EBProps) { super(props); this.state = { hasError: false }; }
  static getDerivedStateFromError(error: Error): EBState { return { hasError: true, error }; }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error(`[WidgetError] ${this.props.fallbackLabel}:`, error, info); }
  render() {
    if (this.state.hasError) {
      return (
        <div className="h-full flex flex-col items-center justify-center text-center p-4 bg-red-950/30 border border-red-900/50 rounded-xl">
          <span className="text-xs font-bold text-red-400 mb-1">Widget Error</span>
          <span className="text-[10px] text-red-500/70 max-w-[200px] break-words">{this.props.fallbackLabel}</span>
          <span className="text-[9px] text-red-600/50 mt-1 max-w-[200px] truncate">{this.state.error?.message}</span>
        </div>
      );
    }
    return this.props.children;
  }
}

// ── Lazy load — only renders when visible in viewport ──

function LazyLoad({ children, height = "40vh", rootMargin = "200px" }: { children: ReactNode; height?: string; rootMargin?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { setVisible(true); observer.disconnect(); } },
      { rootMargin }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [rootMargin]);

  if (!visible) {
    return <div ref={ref} style={{ minHeight: height }} className="flex items-center justify-center text-neutral-700 text-xs">Loading…</div>;
  }
  return <div ref={ref}>{children}</div>;
}

// ── Widget preview at a specific size ──

function WidgetPreview({
  scenario,
  variant,
  size,
  sizePreset,
}: {
  scenario: string;
  variant: string;
  size: WidgetSize;
  sizePreset: SizePreset;
}) {
  const Component = getWidgetComponent(scenario);
  const meta = FIXTURES[scenario];
  const data = (meta?.variants?.[variant] ?? {}) as Record<string, unknown>;

  return (
    <div
      style={{ width: sizePreset.w, height: sizePreset.h, minWidth: "200px", minHeight: "80px" }}
      className="shrink-0 rounded-xl overflow-hidden"
    >
      <WidgetSlot scenario={scenario} size={size} noGrid>
        {Component ? (
          <Suspense
            fallback={
              <div className="animate-pulse h-full bg-neutral-800 rounded" />
            }
          >
            <Component data={data} />
          </Suspense>
        ) : (
          <div className="h-full flex items-center justify-center text-xs text-neutral-500">
            {scenario} (no component)
          </div>
        )}
      </WidgetSlot>
    </div>
  );
}

// ── Size adjuster inline component ──

function parseUnit(val: string): { num: number; unit: string } {
  const match = val.match(/^([\d.]+)(.*)$/);
  return match ? { num: parseFloat(match[1]), unit: match[2] || "vw" } : { num: 0, unit: "vw" };
}

function SizeAdjuster({
  scenario,
  variant,
  sizeEntries,
  defaults,
  onAdjust,
}: {
  scenario: string;
  variant: string;
  sizeEntries: { key: WidgetSize; label: string }[];
  defaults: ScenarioSizes;
  onAdjust: (adj: VariantSizeAdjustments | null) => void;
}) {
  const [open, setOpen] = useState(false);
  const [values, setValues] = useState<VariantSizeAdjustments>({});
  const [saved, setSaved] = useState(false);

  // Load persisted adjustments on mount
  useEffect(() => {
    const existing = getSizeAdjustment(scenario, variant);
    if (existing) {
      setValues(existing);
      onAdjust(existing);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scenario, variant]);

  const handleChange = useCallback(
    (sizeKey: string, field: "w" | "h", rawVal: string) => {
      setValues((prev) => {
        const def = defaults[sizeKey as WidgetSize];
        const current = prev[sizeKey] ?? { w: def?.w ?? "35vw", h: def?.h ?? "40vh" };
        const updated = { ...current, [field]: rawVal };
        const next = { ...prev, [sizeKey]: updated };
        // Persist to localStorage
        saveSizeAdjustment(scenario, variant, sizeKey, updated);
        // Notify parent for live preview
        onAdjust(next);
        return next;
      });
      setSaved(false);
    },
    [scenario, variant, defaults, onAdjust]
  );

  const handleReset = useCallback(() => {
    setValues({});
    clearSizeAdjustment(scenario, variant);
    onAdjust(null);
    setSaved(false);
  }, [scenario, variant, onAdjust]);

  const handleRaiseIssue = useCallback(() => {
    // Build defaults map for comparison
    const defMap: VariantSizeAdjustments = {};
    for (const { key } of sizeEntries) {
      const d = defaults[key];
      if (d) defMap[key] = d;
    }
    raiseIssueSizeAdjustment(scenario, variant, values, defMap);
    setSaved(true);
  }, [scenario, variant, values, sizeEntries, defaults]);

  const hasChanges = Object.keys(values).length > 0;

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="text-xs text-neutral-500 hover:text-neutral-200 border border-neutral-700/50 rounded px-2 py-0.5 transition-colors"
      >
        Adjust Sizes
      </button>
    );
  }

  return (
    <div className="mt-2 p-3 rounded-lg bg-neutral-800/40 border border-neutral-700/40 space-y-2">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[10px] uppercase font-bold tracking-wider text-neutral-400">
          Size Adjustments
        </span>
        <button
          onClick={() => setOpen(false)}
          className="text-[10px] text-neutral-500 hover:text-neutral-300"
        >
          Close
        </button>
      </div>

      {/* Grid of size inputs */}
      <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${sizeEntries.length}, 1fr)` }}>
        {sizeEntries.map(({ key, label }) => {
          const def = defaults[key];
          if (!def) return null;
          const adj = values[key];
          const wVal = adj?.w ?? def.w;
          const hVal = adj?.h ?? def.h;
          const wParsed = parseUnit(wVal);
          const hParsed = parseUnit(hVal);
          const isModified = adj && (adj.w !== def.w || adj.h !== def.h);

          return (
            <div key={key} className="space-y-1">
              <div className={`text-[10px] text-center font-medium ${isModified ? "text-amber-400" : "text-neutral-400"}`}>
                {label} {isModified && "*"}
              </div>
              <div className="flex items-center gap-1">
                <span className="text-[9px] text-neutral-500 w-3">W</span>
                <input
                  type="number"
                  value={wParsed.num}
                  onChange={(e) => handleChange(key, "w", `${e.target.value}${wParsed.unit}`)}
                  className="w-12 text-[10px] bg-neutral-900 border border-neutral-700 rounded px-1 py-0.5 text-neutral-200 text-center focus:outline-none focus:border-neutral-500"
                  step={1}
                  min={5}
                />
                <span className="text-[9px] text-neutral-600">{wParsed.unit}</span>
              </div>
              <div className="flex items-center gap-1">
                <span className="text-[9px] text-neutral-500 w-3">H</span>
                <input
                  type="number"
                  value={hParsed.num}
                  onChange={(e) => handleChange(key, "h", `${e.target.value}${hParsed.unit}`)}
                  className="w-12 text-[10px] bg-neutral-900 border border-neutral-700 rounded px-1 py-0.5 text-neutral-200 text-center focus:outline-none focus:border-neutral-500"
                  step={1}
                  min={5}
                />
                <span className="text-[9px] text-neutral-600">{hParsed.unit}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2 pt-1">
        <button
          onClick={handleRaiseIssue}
          disabled={!hasChanges}
          className={`text-[10px] px-2 py-0.5 rounded transition-colors ${
            hasChanges
              ? "bg-amber-600 hover:bg-amber-500 text-white"
              : "bg-neutral-700 text-neutral-500 cursor-not-allowed"
          }`}
        >
          {saved ? "Saved as Issue" : "Raise as Issue"}
        </button>
        {hasChanges && (
          <button
            onClick={handleReset}
            className="text-[10px] px-2 py-0.5 rounded bg-neutral-700 hover:bg-neutral-600 text-neutral-400 transition-colors"
          >
            Reset
          </button>
        )}
      </div>
    </div>
  );
}

// ── Variant card ──

function VariantCard({ scenario, variant }: { scenario: string; variant: string }) {
  const defaults = getVariantSizes(scenario, variant);
  const isKpi = scenario === "kpi";
  const [adjustments, setAdjustments] = useState<VariantSizeAdjustments | null>(null);

  // Determine which size keys to show based on what presets are available
  const sizeEntries: { key: WidgetSize; label: string }[] = isKpi
    ? [
        { key: "compact", label: "compact" },
        { key: "normal", label: "normal" },
        { key: "expanded", label: "expanded" },
      ]
    : [
        ...(defaults.compact ? [{ key: "compact" as WidgetSize, label: "compact" }] : []),
        { key: "normal", label: "normal" },
        { key: "expanded", label: "expanded" },
        { key: "hero", label: "hero" },
      ];

  // Merge defaults with live adjustments
  const getPreset = useCallback(
    (sizeKey: WidgetSize): SizePreset | undefined => {
      const def = defaults[sizeKey];
      if (!def) return undefined;
      if (adjustments?.[sizeKey]) return adjustments[sizeKey] as SizePreset;
      return def;
    },
    [defaults, adjustments]
  );

  return (
    <div className="mb-6 pb-6 border-b border-neutral-800">
      {/* Variant name + size adjuster toggle */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xs font-mono text-neutral-300 bg-neutral-800 px-2 py-0.5 rounded">
          {variant}
        </span>
        <SizeAdjuster
          scenario={scenario}
          variant={variant}
          sizeEntries={sizeEntries}
          defaults={defaults}
          onAdjust={setAdjustments}
        />
      </div>

      {/* Size previews — uses adjusted sizes if available */}
      <div className="flex items-start gap-3 overflow-x-auto pb-2">
        {sizeEntries.map(({ key, label }) => {
          const preset = getPreset(key);
          if (!preset) return null;
          const isAdjusted = adjustments?.[key] && (adjustments[key].w !== defaults[key]?.w || adjustments[key].h !== defaults[key]?.h);
          return (
            <div key={key} className="shrink-0">
              <div className={`text-[10px] mb-1 text-center ${isAdjusted ? "text-amber-400" : "text-neutral-500"}`}>
                {label} ({preset.w} × {preset.h})
              </div>
              <WidgetErrorBoundary fallbackLabel={`${scenario}/${variant} (${label})`}>
                <WidgetPreview scenario={scenario} variant={variant} size={key} sizePreset={preset} />
              </WidgetErrorBoundary>
            </div>
          );
        })}
      </div>

      {/* Feedback form */}
      <WidgetFeedbackForm scenario={scenario} variant={variant} />
    </div>
  );
}

// ── Paginated "All Widgets" view ──

const PAGE_SIZE = 20;

function AllWidgetsView({ scenarios, totalVariants }: { scenarios: ScenarioInfo[]; totalVariants: number }) {
  // Flatten all scenario/variant pairs into one list
  const allVariants = useMemo(() => {
    const list: { scenario: string; variant: string }[] = [];
    for (const s of scenarios) {
      for (const v of s.variants) {
        list.push({ scenario: s.key, variant: v });
      }
    }
    return list;
  }, [scenarios]);

  const [page, setPage] = useState(0);
  const totalPages = Math.ceil(allVariants.length / PAGE_SIZE);
  const pageItems = allVariants.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-lg font-semibold">All Widgets</h2>
          <p className="text-xs text-neutral-500">
            {scenarios.length} scenarios · {totalVariants} variants · Page {page + 1} of {totalPages}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={page === 0}
            className="text-xs px-3 py-1 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            ← Prev
          </button>
          <span className="text-xs text-neutral-400 tabular-nums">{page + 1} / {totalPages}</span>
          <button
            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
            disabled={page >= totalPages - 1}
            className="text-xs px-3 py-1 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            Next →
          </button>
        </div>
      </div>

      {pageItems.map(({ scenario, variant }) => (
        <LazyLoad key={`${scenario}/${variant}`} height="30vh">
          <div className="mb-2">
            <span className="text-[10px] text-neutral-500 uppercase tracking-widest">{scenario}</span>
          </div>
          <VariantCard scenario={scenario} variant={variant} />
        </LazyLoad>
      ))}

      {/* Bottom pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 py-6 border-t border-neutral-800 mt-4">
          <button
            onClick={() => { setPage((p) => Math.max(0, p - 1)); window.scrollTo(0, 0); }}
            disabled={page === 0}
            className="text-xs px-3 py-1.5 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            ← Previous Page
          </button>
          <div className="flex gap-1">
            {Array.from({ length: totalPages }, (_, i) => (
              <button
                key={i}
                onClick={() => { setPage(i); window.scrollTo(0, 0); }}
                className={`text-xs w-7 h-7 rounded transition-colors ${
                  i === page ? "bg-blue-600 text-white" : "bg-neutral-800 text-neutral-400 hover:bg-neutral-700"
                }`}
              >
                {i + 1}
              </button>
            ))}
          </div>
          <button
            onClick={() => { setPage((p) => Math.min(totalPages - 1, p + 1)); window.scrollTo(0, 0); }}
            disabled={page >= totalPages - 1}
            className="text-xs px-3 py-1.5 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            Next Page →
          </button>
        </div>
      )}
    </div>
  );
}

// ── Main page ──

export default function WidgetGalleryPage() {
  const scenarios = useMemo(buildScenarioList, []);
  const [activeScenario, setActiveScenario] = useState<string | null>(null);
  const [showDashboards, setShowDashboards] = useState(false);
  const [showSimulation, setShowSimulation] = useState(false);
  const [showSimulationV2, setShowSimulationV2] = useState(false);
  const [feedbackCount, setFeedbackCount] = useState(0);

  // Hydrate feedback count from localStorage after mount to avoid SSR mismatch
  useEffect(() => {
    setFeedbackCount(getAllWidgetFeedback().length);
  }, []);

  const multiVariant = scenarios.filter((s) => s.isMulti);
  const singleVariant = scenarios.filter((s) => !s.isMulti);
  const totalVariants = scenarios.reduce((sum, s) => sum + s.variants.length, 0);

  const handleExport = useCallback(() => {
    const json = exportAllFeedback();
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `widget-feedback-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, []);

  // Refresh feedback count periodically
  const refreshCount = useCallback(() => {
    setFeedbackCount(getAllWidgetFeedback().length);
  }, []);

  return (
    <div className="flex h-screen bg-neutral-950 text-neutral-100">
      {/* ── Sidebar ── */}
      <nav className="w-64 shrink-0 border-r border-neutral-800 overflow-y-auto p-4 space-y-4">
        <h1 className="text-sm font-bold tracking-tight">Widget Gallery</h1>
        <p className="text-[10px] text-neutral-500">
          {scenarios.length} scenarios · {totalVariants} variants · {feedbackCount} reviewed
        </p>

        {/* Export button */}
        <button
          onClick={handleExport}
          className="w-full text-xs px-3 py-1.5 rounded bg-blue-600 hover:bg-blue-500 text-white transition-colors"
        >
          Export All Feedback (JSON)
        </button>

        {/* Sample dashboards link */}
        <button
          onClick={() => { setShowDashboards(true); setShowSimulation(false); setShowSimulationV2(false); setActiveScenario(null); }}
          className={`w-full text-left text-xs px-2 py-1.5 rounded transition-colors ${
            showDashboards ? "bg-neutral-700 text-white" : "text-neutral-400 hover:text-white hover:bg-neutral-800"
          }`}
        >
          Sample Dashboards ({SAMPLE_DASHBOARDS.length})
        </button>

        {/* Simulation dashboards V2 link */}
        <button
          onClick={() => { setShowSimulationV2(true); setShowSimulation(false); setShowDashboards(false); setActiveScenario(null); }}
          className={`w-full text-left text-xs px-2 py-1.5 rounded transition-colors ${
            showSimulationV2 ? "bg-emerald-800 text-emerald-200" : "text-neutral-400 hover:text-white hover:bg-neutral-800"
          }`}
        >
          Simulation Dashboards V2
        </button>

        {/* Simulation dashboards V1 link */}
        <button
          onClick={() => { setShowSimulation(true); setShowSimulationV2(false); setShowDashboards(false); setActiveScenario(null); }}
          className={`w-full text-left text-xs px-2 py-1.5 rounded transition-colors ${
            showSimulation ? "bg-neutral-700 text-white" : "text-neutral-400 hover:text-white hover:bg-neutral-800"
          }`}
        >
          Simulation Dashboards V1
        </button>

        {/* Multi-variant section */}
        <div>
          <h2 className="text-[10px] uppercase font-bold text-neutral-500 tracking-widest mb-1">
            Multi-Variant ({multiVariant.length})
          </h2>
          {multiVariant.map((s) => (
            <button
              key={s.key}
              onClick={() => { setActiveScenario(s.key); setShowDashboards(false); setShowSimulation(false); setShowSimulationV2(false); refreshCount(); }}
              className={`block w-full text-left text-xs px-2 py-1 rounded transition-colors ${
                activeScenario === s.key ? "bg-neutral-700 text-white" : "text-neutral-400 hover:text-white hover:bg-neutral-800"
              }`}
            >
              {s.key}
              <span className="text-neutral-600 ml-1">({s.variants.length})</span>
            </button>
          ))}
        </div>

        {/* Single-variant section */}
        <div>
          <h2 className="text-[10px] uppercase font-bold text-neutral-500 tracking-widest mb-1">
            Single-Variant ({singleVariant.length})
          </h2>
          {singleVariant.map((s) => (
            <button
              key={s.key}
              onClick={() => { setActiveScenario(s.key); setShowDashboards(false); setShowSimulation(false); setShowSimulationV2(false); refreshCount(); }}
              className={`block w-full text-left text-xs px-2 py-1 rounded transition-colors ${
                activeScenario === s.key ? "bg-neutral-700 text-white" : "text-neutral-400 hover:text-white hover:bg-neutral-800"
              }`}
            >
              {s.key}
            </button>
          ))}
        </div>
      </nav>

      {/* ── Main content ── */}
      <main className="flex-1 overflow-y-auto p-6">
        {/* All widgets — paginated flat list */}
        {!activeScenario && !showDashboards && !showSimulation && !showSimulationV2 && (
          <AllWidgetsView scenarios={scenarios} totalVariants={totalVariants} />
        )}

        {/* Sample dashboards */}
        {showDashboards && !showSimulation && !showSimulationV2 && (
          <div>
            <div className="flex items-center gap-3 mb-1">
              <h2 className="text-lg font-semibold">Sample Dashboards</h2>
              <PageFeedbackForm pageId="dashboards" />
            </div>
            <p className="text-xs text-neutral-500 mb-6">
              Pre-built layouts for common query types. Rate each layout and note what you'd change.
            </p>
            <FeedbackChecklist pageId="dashboards" variant="dashboards" />
            {SAMPLE_DASHBOARDS.map((db) => (
              <SampleDashboard key={db.id} {...db} />
            ))}
          </div>
        )}

        {/* Simulation dashboards V2 */}
        {showSimulationV2 && <SimulationView logFile="/simulation/simulation_log.json" title="Simulation Dashboards V2" />}

        {/* Simulation dashboards V1 */}
        {showSimulation && <SimulationView logFile="/simulation/simulation_log_v1.json" title="Simulation Dashboards V1" />}

        {/* Widget variants */}
        {activeScenario && !showDashboards && !showSimulation && !showSimulationV2 && (() => {
          const info = scenarios.find((s) => s.key === activeScenario);
          if (!info) return null;
          return (
            <div>
              <div className="mb-6">
                <div className="flex items-center gap-3 mb-1">
                  <h2 className="text-lg font-semibold">{info.key}</h2>
                  <PageFeedbackForm pageId={`scenario:${info.key}`} />
                </div>
                <p className="text-xs text-neutral-500">
                  {info.name} · {info.variants.length} variant{info.variants.length > 1 ? "s" : ""}
                </p>
              </div>
              <FeedbackChecklist pageId={`scenario:${info.key}`} variant="scenario" />
              {info.variants.map((variant) => (
                <LazyLoad key={variant} height="30vh">
                  <VariantCard scenario={info.key} variant={variant} />
                </LazyLoad>
              ))}
            </div>
          );
        })()}
      </main>
    </div>
  );
}
