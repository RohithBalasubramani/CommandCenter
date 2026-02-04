"use client";

import React, { Suspense, useState, useEffect, useCallback } from "react";
import { createPortal } from "react-dom";
import { FIXTURES } from "@/components/layer4/fixtureData";
import { getWidgetComponent } from "@/components/layer4/widgetRegistry";
import WidgetSlot from "@/components/layer3/WidgetSlot";
import BlobGrid from "@/components/layer3/BlobGrid";
import type { WidgetSize, WidgetHeightHint } from "@/types";
import { DashboardFeedbackForm } from "./FeedbackForm";

interface DashboardWidget {
  scenario: string;
  fixture?: string;
  size: WidgetSize;
  heightHint: WidgetHeightHint;
  data_override?: Record<string, unknown>;
  description?: string;
}

interface SampleDashboardProps {
  id: string;
  title: string;
  description: string;
  widgets: DashboardWidget[];
}

const HEIGHT_CLASSES: Record<WidgetHeightHint, string> = {
  short: "row-span-1",
  medium: "row-span-2",
  tall: "row-span-3",
  "x-tall": "row-span-4",
};

function sizeClasses(size: WidgetSize, heightHint: WidgetHeightHint): string {
  const h = HEIGHT_CLASSES[heightHint];
  switch (size) {
    case "hero": return "col-span-12 row-span-4";
    case "expanded": return `col-span-6 ${h}`;
    case "normal": return `col-span-4 ${h}`;
    case "compact": return `col-span-3 ${h}`;
    default: return `col-span-4 ${h}`;
  }
}

function resolveData(scenario: string, fixture?: string, dataOverride?: Record<string, unknown>): Record<string, unknown> {
  const meta = FIXTURES[scenario];
  if (!meta) return dataOverride ?? {};
  const key = fixture || meta.defaultFixture;
  const fixtureData = (meta.variants?.[key] ?? {}) as Record<string, unknown>;

  if (!dataOverride) return fixtureData;

  // Merge: data_override takes precedence, fixture provides defaults
  const merged = { ...fixtureData, ...dataOverride };
  if (
    fixtureData.demoData &&
    typeof fixtureData.demoData === "object" &&
    dataOverride.demoData &&
    typeof dataOverride.demoData === "object"
  ) {
    merged.demoData = {
      ...(fixtureData.demoData as Record<string, unknown>),
      ...(dataOverride.demoData as Record<string, unknown>),
    };
  }
  return merged;
}

function DashboardContent({ title, widgets }: { title: string; widgets: DashboardWidget[] }) {
  return (
    <BlobGrid heading={title}>
      {widgets.map((w, i) => {
        const Component = getWidgetComponent(w.scenario);
        const data = resolveData(w.scenario, w.fixture, w.data_override);
        return (
          <div key={`${w.scenario}-${i}`} className={sizeClasses(w.size, w.heightHint)}>
            <WidgetSlot scenario={w.scenario} size={w.size} description={w.description} noGrid>
              {Component ? (
                <Suspense fallback={<div className="animate-pulse h-full bg-neutral-800 rounded" />}>
                  <Component data={data} />
                </Suspense>
              ) : (
                <div className="h-full flex items-center justify-center text-xs text-neutral-500">
                  {w.scenario}
                </div>
              )}
            </WidgetSlot>
          </div>
        );
      })}
    </BlobGrid>
  );
}

export default function SampleDashboard({ id, title, description, widgets }: SampleDashboardProps) {
  const [fullscreen, setFullscreen] = useState(false);

  const close = useCallback(() => setFullscreen(false), []);

  useEffect(() => {
    if (!fullscreen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [fullscreen, close]);

  return (
    <div className="mb-10">
      <div className="flex items-center gap-2 mb-1">
        <h3 className="text-sm font-semibold text-neutral-100">{title}</h3>
        <button
          onClick={() => setFullscreen(true)}
          className="text-neutral-500 hover:text-neutral-200 transition-colors"
          title="View fullscreen"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 3 21 3 21 9" />
            <polyline points="9 21 3 21 3 15" />
            <line x1="21" y1="3" x2="14" y2="10" />
            <line x1="3" y1="21" x2="10" y2="14" />
          </svg>
        </button>
      </div>
      <p className="text-xs text-neutral-500 mb-3">{description}</p>

      <div className="border border-neutral-700/50 rounded-xl overflow-hidden w-full" style={{ height: "520px" }}>
        <DashboardContent title={title} widgets={widgets} />
      </div>

      <DashboardFeedbackForm dashboardId={id} />

      {/* Fullscreen overlay — portaled to body to escape stacking contexts */}
      {fullscreen && createPortal(
        <div className="fixed inset-0 z-50 bg-neutral-950 flex flex-col">
          <div className="flex items-center justify-between px-6 py-3 border-b border-neutral-800 shrink-0">
            <div>
              <h2 className="text-sm font-semibold text-neutral-100">{title}</h2>
              <p className="text-xs text-neutral-500">{description}</p>
            </div>
            <button
              onClick={close}
              className="text-neutral-400 hover:text-white transition-colors p-1"
              title="Exit fullscreen (Esc)"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
          <div className="flex-1 overflow-auto p-4">
            <DashboardContent title={title} widgets={widgets} />
          </div>
        </div>,
        document.body
      )}
    </div>
  );
}

// ── Pre-built sample dashboards ──

export const SAMPLE_DASHBOARDS: SampleDashboardProps[] = [
  {
    id: "monitor-equipment",
    title: "Monitor All Equipment",
    description: "Broad equipment overview — device panels, KPIs on top, supporting charts below",
    widgets: [
      { scenario: "kpi", fixture: "kpi_live-standard", size: "compact", heightHint: "short" },
      { scenario: "kpi", fixture: "kpi_alert-warning-state", size: "compact", heightHint: "short" },
      { scenario: "kpi", fixture: "kpi_status-offline", size: "compact", heightHint: "short" },
      { scenario: "kpi", fixture: "kpi_accumulated-daily-total", size: "compact", heightHint: "short" },
      { scenario: "edgedevicepanel", size: "expanded", heightHint: "x-tall" },
      { scenario: "composition", fixture: "donut_pie", size: "normal", heightHint: "tall" },
      { scenario: "category-bar", fixture: "oee-by-machine", size: "expanded", heightHint: "tall" },
    ],
  },
  {
    id: "energy-consumption",
    title: "Show Energy Consumption",
    description: "Energy-focused — hero trend chart, supporting multi-line, distribution, and flow",
    widgets: [
      { scenario: "trend", fixture: "trend_live-area", size: "hero", heightHint: "tall" },
      { scenario: "kpi", fixture: "kpi_accumulated-daily-total", size: "compact", heightHint: "short" },
      { scenario: "kpi", fixture: "kpi_live-standard", size: "compact", heightHint: "short" },
      { scenario: "trend-multi-line", fixture: "power-sources-stacked", size: "expanded", heightHint: "tall" },
      { scenario: "distribution", fixture: "dist_energy_source_share-donut", size: "normal", heightHint: "tall" },
      { scenario: "flow-sankey", fixture: "flow_sankey_standard-classic-left-to-right-sankey", size: "hero", heightHint: "x-tall" },
    ],
  },
  {
    id: "maintenance-status",
    title: "Maintenance Status",
    description: "Maintenance-focused — timeline, event logs, alerts, KPI summary",
    widgets: [
      { scenario: "timeline", fixture: "machine-state-timeline", size: "hero", heightHint: "tall" },
      { scenario: "kpi", fixture: "kpi_alert-critical-state", size: "compact", heightHint: "short" },
      { scenario: "kpi", fixture: "kpi_lifecycle-progress-bar", size: "compact", heightHint: "short" },
      { scenario: "alerts", fixture: "banner-energy-peak-threshold-exceeded", size: "expanded", heightHint: "medium" },
      { scenario: "eventlogstream", fixture: "tabular-log-view", size: "expanded", heightHint: "tall" },
      { scenario: "category-bar", fixture: "downtime-duration", size: "expanded", heightHint: "tall" },
    ],
  },
  {
    id: "compare-devices",
    title: "Compare Transformer 1 vs Transformer 2",
    description: "Comparison-focused — hero comparison chart, supporting trends and KPIs",
    widgets: [
      { scenario: "comparison", fixture: "side_by_side_visual-plain-values", size: "hero", heightHint: "tall" },
      { scenario: "kpi", fixture: "kpi_live-standard", size: "compact", heightHint: "short" },
      { scenario: "kpi", fixture: "kpi_live-high-contrast", size: "compact", heightHint: "short" },
      { scenario: "trend-multi-line", fixture: "main-lt-phases-current", size: "expanded", heightHint: "tall" },
      { scenario: "matrix-heatmap", fixture: "status-matrix", size: "expanded", heightHint: "x-tall" },
    ],
  },
  {
    id: "power-quality",
    title: "Power Quality Overview",
    description: "PQ-focused — trends, distribution, heatmap, KPIs",
    widgets: [
      { scenario: "trend", fixture: "trend_phased-rgb-phase-line", size: "hero", heightHint: "tall" },
      { scenario: "kpi", fixture: "kpi_live-standard", size: "compact", heightHint: "short" },
      { scenario: "kpi", fixture: "kpi_alert-warning-state", size: "compact", heightHint: "short" },
      { scenario: "kpi", fixture: "kpi_lifecycle-dark-mode-gauge", size: "compact", heightHint: "short" },
      { scenario: "distribution", fixture: "dist_consumption_by_category-pie", size: "expanded", heightHint: "tall" },
      { scenario: "matrix-heatmap", fixture: "value-heatmap", size: "expanded", heightHint: "x-tall" },
    ],
  },
];
