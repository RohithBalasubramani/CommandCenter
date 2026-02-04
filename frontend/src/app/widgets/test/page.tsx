"use client";

import React, { Suspense, useState, useMemo } from "react";
import { FIXTURES } from "@/components/layer4/fixtureData";
import { getWidgetComponent, getAvailableScenarios } from "@/components/layer4/widgetRegistry";
import WidgetSlot from "@/components/layer3/WidgetSlot";

// ── Test data generators per scenario ──
// Each generator returns an array of test cases: { label, data }
// The "data" shape must match what the widget component expects (the fixture spec shape).

function generateTimeSeries(count: number, base: number, variance: number) {
  const now = Date.now();
  return Array.from({ length: count }, (_, i) => ({
    time: new Date(now - (count - i) * 15 * 60000).toISOString().slice(11, 16),
    value: Math.round((base + (Math.random() - 0.5) * variance) * 10) / 10,
  }));
}

function generateTimeSeriesISO(count: number, base: number, variance: number) {
  const now = Date.now();
  return Array.from({ length: count }, (_, i) => {
    const ts = new Date(now - (count - i) * 15 * 60000).toISOString();
    const raw = Math.round((base + (Math.random() - 0.5) * variance) * 100) / 100;
    return { x: ts, raw, cumulative: raw * (i + 1) * 0.25 };
  });
}

// ── KPI test cases ──
const KPI_TESTS = [
  {
    label: "Live Standard — Grid Voltage",
    fixture: "kpi_live-standard",
    data: {
      coreWidget: "KPI", variant: "KPI_LIVE", representation: "Standard", encoding: "Text",
      layout: { padding: "p-6", radius: "rounded-xl", zones: "flex-col" },
      visual: { background: "bg-white", border: "border-gray-100", theme: "light" },
      states: { default: "text-neutral-900" },
      demoData: { label: "Grid Voltage", value: "241.7", unit: "V" },
    },
  },
  {
    label: "Live High Contrast — Active Power",
    fixture: "kpi_live-high-contrast",
    data: {
      coreWidget: "KPI", variant: "KPI_LIVE", representation: "High Contrast", encoding: "Text",
      layout: { padding: "p-6", radius: "rounded-xl", zones: "flex-col" },
      visual: { background: "bg-neutral-900", border: "border-neutral-800", theme: "dark" },
      states: { default: "text-white" },
      demoData: { label: "Active Power", value: "512", unit: "kW" },
    },
  },
  {
    label: "Warning — Hydraulic Pressure",
    fixture: "kpi_alert-warning-state",
    data: {
      coreWidget: "KPI", variant: "KPI_ALERT", representation: "Warning State", encoding: "Icon+Text",
      layout: { padding: "p-6", radius: "rounded-xl", zones: "flex-row" },
      visual: { background: "bg-amber-50", border: "border-amber-100", theme: "light" },
      states: { warning: "text-amber-700" },
      demoData: { label: "Hydraulic Pressure", value: "840", unit: "PSI", state: "warning" },
    },
  },
  {
    label: "Critical — Core Temperature",
    fixture: "kpi_alert-critical-state",
    data: {
      coreWidget: "KPI", variant: "KPI_ALERT", representation: "Critical State", encoding: "Icon+Text",
      layout: { padding: "p-6", radius: "rounded-xl", zones: "flex-row" },
      visual: { background: "bg-red-50", border: "border-red-100", theme: "light" },
      states: { critical: "text-red-700" },
      demoData: { label: "Core Temp", value: "97.2", unit: "\u00b0C", state: "critical" },
    },
  },
  {
    label: "Status Badge — Pump Running",
    fixture: "kpi_status-badge",
    data: {
      coreWidget: "KPI", variant: "KPI_STATUS", representation: "Badge", encoding: "Icon+Text",
      layout: { padding: "p-6", radius: "rounded-xl", zones: "flex-col" },
      visual: { background: "bg-white", border: "border-gray-100", theme: "light" },
      states: { default: "text-neutral-900" },
      demoData: { label: "Pump P-101", status: "Running", statusColor: "bg-green-500" },
    },
  },
  {
    label: "Lifecycle Gauge — Motor Load",
    fixture: "kpi_lifecycle-dark-mode-gauge",
    data: {
      coreWidget: "KPI", variant: "KPI_LIFECYCLE", representation: "Gauge", encoding: "Text",
      layout: { padding: "p-6", radius: "rounded-xl", zones: "flex-col" },
      visual: { background: "bg-neutral-900", border: "border-neutral-800", theme: "dark" },
      states: { default: "text-white" },
      demoData: { label: "Motor Load", value: "72", unit: "%", max: 100 },
    },
  },
  {
    label: "Accumulated Daily — Energy",
    fixture: "kpi_accumulated-daily-total",
    data: {
      coreWidget: "KPI", variant: "KPI_ACCUMULATED", representation: "Daily Total", encoding: "Text+Subtext",
      layout: { padding: "p-6", radius: "rounded-xl", zones: "stack" },
      visual: { background: "bg-white", border: "border-gray-100", theme: "light" },
      states: { default: "text-neutral-900" },
      demoData: { label: "Daily Energy", value: "1,842", unit: "kWh", period: "Today (Jan 31)" },
    },
  },
  {
    label: "Status Offline — Chiller",
    fixture: "kpi_status-offline",
    data: {
      coreWidget: "KPI", variant: "KPI_STATUS", representation: "Offline", encoding: "Icon+Text",
      layout: { padding: "p-6", radius: "rounded-xl", zones: "flex-col" },
      visual: { background: "bg-neutral-100", border: "border-neutral-200", theme: "light" },
      states: { default: "text-neutral-400" },
      demoData: { label: "Chiller CH-02", status: "Offline", statusColor: "bg-gray-400" },
    },
  },
  {
    label: "Progress Bar — Filter Life",
    fixture: "kpi_lifecycle-progress-bar",
    data: {
      coreWidget: "KPI", variant: "KPI_LIFECYCLE", representation: "Progress Bar", encoding: "Text",
      layout: { padding: "p-6", radius: "rounded-xl", zones: "flex-col" },
      visual: { background: "bg-white", border: "border-gray-100", theme: "light" },
      states: { default: "text-neutral-900" },
      demoData: { label: "Filter Life Remaining", value: "68", unit: "%", max: 100 },
    },
  },
];

// ── Alerts test cases ──
const ALERTS_TESTS = [
  {
    label: "Banner — Energy Peak",
    fixture: "banner-energy-peak-threshold-exceeded",
    data: null, // use fixture
  },
  {
    label: "Toast — Power Factor Critical",
    fixture: "toast-power-factor-critical-low",
    data: null,
  },
  {
    label: "Card — DG Started",
    fixture: "card-dg-02-started-successfully",
    data: null,
  },
  {
    label: "Badge — AHU Temperature",
    fixture: "badge-ahu-01-high-temperature",
    data: null,
  },
  {
    label: "Modal — UPS Battery Critical",
    fixture: "modal-ups-battery-critical",
    data: null,
  },
];

// ── Trend test cases ──
const TREND_TESTS = [
  {
    label: "Live Line — Temperature",
    fixture: "trend_live-line",
    data: {
      coreWidget: "Trend", variant: "TREND_LIVE", representation: "Line", encoding: "Chart",
      layout: { padding: "p-4", radius: "rounded-xl", zones: "flex-col" },
      visual: { background: "bg-white", border: "border-gray-100", theme: "light", colors: ["#2563eb"] },
      states: {},
      demoData: { label: "Bearing Temperature", timeRange: "Last 4h", timeSeries: generateTimeSeries(16, 65, 8) },
    },
  },
  {
    label: "Area — Energy Consumption",
    fixture: "trend_live-area",
    data: {
      coreWidget: "Trend", variant: "TREND_LIVE", representation: "Area", encoding: "Chart",
      layout: { padding: "p-4", radius: "rounded-xl", zones: "flex-col" },
      visual: { background: "bg-white", border: "border-gray-100", theme: "light", colors: ["#16a34a"] },
      states: {},
      demoData: { label: "Energy Consumption", timeRange: "Last 4h", timeSeries: generateTimeSeries(16, 420, 60) },
    },
  },
  {
    label: "Threshold Line — Vibration Alert",
    fixture: "trend_alert_context-line-threshold",
    data: {
      coreWidget: "Trend", variant: "TREND_ALERT_CONTEXT", representation: "Line", encoding: "Chart",
      layout: { padding: "p-4", radius: "rounded-xl", zones: "flex-col" },
      visual: { background: "bg-white", border: "border-gray-100", theme: "light", colors: ["#ef4444"] },
      states: {},
      demoData: { label: "Vibration Level", timeRange: "Last 6h", timeSeries: generateTimeSeries(24, 78, 20) },
    },
  },
  {
    label: "Step Line — Pump State",
    fixture: "trend_standard-step-line",
    data: {
      coreWidget: "Trend", variant: "TREND_STANDARD", representation: "Step Line", encoding: "Chart",
      layout: { padding: "p-4", radius: "rounded-xl", zones: "flex-col" },
      visual: { background: "bg-white", border: "border-gray-100", theme: "light", colors: ["#525252"] },
      states: {},
      demoData: { label: "Pump State (On/Off)", timeRange: "Last 8h", timeSeries: Array.from({ length: 32 }, (_, i) => ({ time: `${String(Math.floor(i / 4) + 6).padStart(2, "0")}:${String((i % 4) * 15).padStart(2, "0")}`, value: Math.random() > 0.3 ? 1 : 0 })) },
    },
  },
  {
    label: "RGB Phase — 3-Phase Current",
    fixture: "trend_phased-rgb-phase-line",
    data: {
      coreWidget: "Trend", variant: "TREND_LIVE", representation: "RGB Phase Line", encoding: "Chart",
      layout: { padding: "p-4", radius: "rounded-xl", zones: "flex-col" },
      visual: { background: "bg-white", border: "border-gray-100", theme: "light", colors: ["#ef4444", "#eab308", "#3b82f6"] },
      states: {},
      demoData: { label: "3-Phase Current", timeRange: "Last 2h", timeSeries: Array.from({ length: 8 }, (_, i) => ({ time: `${String(14 + Math.floor(i / 4)).padStart(2, "0")}:${String((i % 4) * 15).padStart(2, "0")}`, r: 42 + Math.random() * 5, y: 40 + Math.random() * 5, b: 41 + Math.random() * 5 })) },
    },
  },
  {
    label: "Heatmap — Weekly Usage",
    fixture: "trend_pattern-heatmap",
    data: {
      coreWidget: "Trend", variant: "TREND_LIVE", representation: "Heatmap", encoding: "Chart",
      layout: { padding: "p-4", radius: "rounded-xl", zones: "flex-col" },
      visual: { background: "bg-white", border: "border-gray-100", theme: "light", colors: ["#2563eb"] },
      states: {},
      demoData: { label: "Weekly Usage Pattern", timeRange: "This Week", buckets: Array.from({ length: 28 }, () => ({ intensity: Math.random() })) },
    },
  },
];

// ── Category Bar test cases ──
const CATEGORY_BAR_TESTS = [
  { label: "OEE by Machine", fixture: "oee-by-machine", data: null },
  { label: "Downtime Duration", fixture: "downtime-duration", data: null },
  { label: "Production States", fixture: "production-states", data: null },
  { label: "Shift Comparison", fixture: "shift-comparison", data: null },
  { label: "Efficiency Deviation", fixture: "efficiency-deviation", data: null },
];

// ── Timeline test cases ──
const TIMELINE_TESTS = [
  { label: "Linear Incident Timeline", fixture: "linear-incident-timeline", data: null },
  { label: "Machine State Timeline", fixture: "machine-state-timeline", data: null },
  { label: "Multi-Lane Shift Schedule", fixture: "multi-lane-shift-schedule", data: null },
  { label: "Forensic Annotated View", fixture: "forensic-annotated-view", data: null },
  { label: "Log Density / Burst", fixture: "log-density-burst-analysis", data: null },
];

// ── EventLogStream test cases ──
const EVENTLOGSTREAM_TESTS = [
  { label: "Chronological Timeline", fixture: "chronological-timeline", data: null },
  { label: "Compact Card Feed", fixture: "compact-card-feed", data: null },
  { label: "Tabular Log View", fixture: "tabular-log-view", data: null },
  { label: "Correlation Stack", fixture: "correlation-stack", data: null },
  { label: "Grouped by Asset", fixture: "grouped-by-asset", data: null },
];

// ── Trend Multi-Line test cases ──
const TREND_MULTI_LINE_TESTS = [
  { label: "Power Sources Stacked", fixture: "power-sources-stacked", data: null },
  { label: "Main LT Phases Current", fixture: "main-lt-phases-current", data: null },
  { label: "UPS Health Dual Axis", fixture: "ups-health-dual-axis", data: null },
  { label: "Power Quality", fixture: "power-quality", data: null },
  { label: "HVAC Performance", fixture: "hvac-performance", data: null },
  { label: "Energy Demand", fixture: "energy-demand", data: null },
];

// ── Trends Cumulative test cases ──
const TRENDS_CUMULATIVE_TESTS = [
  { label: "Energy Consumption", fixture: "energy-consumption", data: null },
  { label: "Instantaneous Power", fixture: "instantaneous-power", data: null },
  { label: "Source Mix", fixture: "source-mix", data: null },
  { label: "Performance vs Baseline", fixture: "performance-vs-baseline", data: null },
  { label: "Cost vs Budget", fixture: "cost-vs-budget", data: null },
  { label: "Batch Production", fixture: "batch-production", data: null },
];

// ── Distribution test cases ──
const DISTRIBUTION_TESTS = [
  { label: "Energy Source Share (Donut)", fixture: "dist_energy_source_share-donut", data: null },
  { label: "100% Stacked Bar", fixture: "dist_energy_source_share-100-stacked-bar", data: null },
  { label: "Load by Asset (Horizontal)", fixture: "dist_load_by_asset-horizontal-bar", data: null },
  { label: "Consumption by Category (Pie)", fixture: "dist_consumption_by_category-pie", data: null },
  { label: "Consumption by Shift", fixture: "dist_consumption_by_shift-grouped-bar", data: null },
  { label: "Downtime Pareto", fixture: "dist_downtime_top_contributors-pareto-bar", data: null },
];

// ── Comparison test cases ──
const COMPARISON_TESTS = [
  { label: "Side by Side Plain", fixture: "side_by_side_visual-plain-values", data: null },
  { label: "Waterfall Loss Analysis", fixture: "waterfall_visual-loss-analysis", data: null },
  { label: "Grouped Bar Phase", fixture: "grouped_bar_visual-phase-comparison", data: null },
  { label: "Delta Deviation Bar", fixture: "delta_bar_visual-deviation-bar", data: null },
  { label: "Small Multiples Grid", fixture: "small_multiples_visual-temp-grid", data: null },
  { label: "Composition Split", fixture: "composition_split_visual-load-type", data: null },
];

// ── Composition test cases ──
const COMPOSITION_TESTS = [
  { label: "Stacked Bar", fixture: "stacked_bar", data: null },
  { label: "Stacked Area", fixture: "stacked_area", data: null },
  { label: "Donut Pie", fixture: "donut_pie", data: null },
  { label: "Waterfall", fixture: "waterfall", data: null },
  { label: "Treemap", fixture: "treemap", data: null },
];

// ── Flow Sankey test cases ──
const FLOW_SANKEY_TESTS = [
  { label: "Classic Sankey", fixture: "flow_sankey_standard-classic-left-to-right-sankey", data: null },
  { label: "Energy Balance (Loss Branches)", fixture: "flow_sankey_energy_balance-sankey-with-explicit-loss-branches-dropping-out", data: null },
  { label: "Multi Source (Many-to-One)", fixture: "flow_sankey_multi_source-many-to-one-flow-diagram", data: null },
  { label: "Layered Multi-Stage", fixture: "flow_sankey_layered-multi-stage-hierarchical-flow", data: null },
  { label: "Time Sliced Sankey", fixture: "flow_sankey_time_sliced-sankey-with-time-scrubberplayer", data: null },
];

// ── Matrix Heatmap test cases ──
const MATRIX_HEATMAP_TESTS = [
  { label: "Value Heatmap", fixture: "value-heatmap", data: null },
  { label: "Correlation Matrix", fixture: "correlation-matrix", data: null },
  { label: "Calendar Heatmap", fixture: "calendar-heatmap", data: null },
  { label: "Status Matrix", fixture: "status-matrix", data: null },
  { label: "Density Matrix", fixture: "density-matrix", data: null },
];

// ── Single-variant scenarios ──
const SINGLE_VARIANT_TESTS = [
  { scenario: "edgedevicepanel", label: "Edge Device Panel", fixture: "default-render" },
  { scenario: "chatstream", label: "Chat Stream", fixture: "default-render" },
  { scenario: "peoplehexgrid", label: "People Hex Grid", fixture: "default-render" },
  { scenario: "peoplenetwork", label: "People Network", fixture: "default-render" },
  { scenario: "peopleview", label: "People View", fixture: "default-render" },
  { scenario: "supplychainglobe", label: "Supply Chain Globe", fixture: "default-render" },
];

// ── Scenario registry for test page ──
interface TestCase {
  label: string;
  fixture: string;
  data: Record<string, unknown> | null; // null = use fixture data as-is
}

interface ScenarioTestSuite {
  scenario: string;
  name: string;
  tests: TestCase[];
  dataSource: "generated" | "fixture"; // whether tests use generated or fixture data
}

const ALL_SUITES: ScenarioTestSuite[] = [
  { scenario: "kpi", name: "KPI", tests: KPI_TESTS, dataSource: "generated" },
  { scenario: "alerts", name: "Alerts", tests: ALERTS_TESTS, dataSource: "fixture" },
  { scenario: "trend", name: "Trend", tests: TREND_TESTS, dataSource: "generated" },
  { scenario: "trend-multi-line", name: "Trend Multi-Line", tests: TREND_MULTI_LINE_TESTS, dataSource: "fixture" },
  { scenario: "trends-cumulative", name: "Trends Cumulative", tests: TRENDS_CUMULATIVE_TESTS, dataSource: "fixture" },
  { scenario: "distribution", name: "Distribution", tests: DISTRIBUTION_TESTS, dataSource: "fixture" },
  { scenario: "comparison", name: "Comparison", tests: COMPARISON_TESTS, dataSource: "fixture" },
  { scenario: "composition", name: "Composition", tests: COMPOSITION_TESTS, dataSource: "fixture" },
  { scenario: "flow-sankey", name: "Flow (Sankey)", tests: FLOW_SANKEY_TESTS, dataSource: "fixture" },
  { scenario: "matrix-heatmap", name: "Matrix / Heatmap", tests: MATRIX_HEATMAP_TESTS, dataSource: "fixture" },
  { scenario: "timeline", name: "Timeline", tests: TIMELINE_TESTS, dataSource: "fixture" },
  { scenario: "eventlogstream", name: "Event Log Stream", tests: EVENTLOGSTREAM_TESTS, dataSource: "fixture" },
  { scenario: "category-bar", name: "Category Bar", tests: CATEGORY_BAR_TESTS, dataSource: "fixture" },
];

// ── Widget Test Card ──

function TestCard({
  scenario,
  fixture,
  testData,
  label,
}: {
  scenario: string;
  fixture: string;
  testData: Record<string, unknown> | null;
  label: string;
}) {
  const Component = getWidgetComponent(scenario);
  const meta = FIXTURES[scenario];

  // Resolve data: use test data if provided, otherwise use fixture data
  const data = useMemo(() => {
    if (testData) return testData;
    return (meta?.variants?.[fixture] ?? {}) as Record<string, unknown>;
  }, [testData, meta, fixture]);

  const isFixtureData = !testData;
  const hasData = data && Object.keys(data).length > 0;

  return (
    <div className="border border-neutral-800 rounded-xl overflow-hidden bg-neutral-900/50">
      {/* Header */}
      <div className="px-3 py-2 border-b border-neutral-800 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-neutral-200">{label}</span>
          <span className="text-[9px] font-mono text-neutral-500 bg-neutral-800 px-1.5 py-0.5 rounded">
            {fixture}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <span
            className={`text-[9px] font-mono px-1.5 py-0.5 rounded ${
              isFixtureData
                ? "bg-amber-900/30 text-amber-400 border border-amber-800/50"
                : "bg-emerald-900/30 text-emerald-400 border border-emerald-800/50"
            }`}
          >
            {isFixtureData ? "FIXTURE DATA" : "TEST DATA"}
          </span>
          {!hasData && (
            <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-red-900/30 text-red-400 border border-red-800/50">
              NO DATA
            </span>
          )}
        </div>
      </div>

      {/* Widget render */}
      <div className="h-[280px]">
        <WidgetSlot scenario={scenario} size="normal" noGrid>
          {Component ? (
            <Suspense
              fallback={<div className="animate-pulse h-full bg-neutral-800 rounded" />}
            >
              <Component data={data} />
            </Suspense>
          ) : (
            <div className="h-full flex items-center justify-center text-xs text-neutral-500">
              {scenario} — no component registered
            </div>
          )}
        </WidgetSlot>
      </div>
    </div>
  );
}

// ── Single Variant Test Card (for edgedevicepanel, chatstream, etc.) ──

function SingleVariantCard({
  scenario,
  label,
  fixture,
}: {
  scenario: string;
  label: string;
  fixture: string;
}) {
  const Component = getWidgetComponent(scenario);
  const meta = FIXTURES[scenario];
  const data = (meta?.variants?.[fixture] ?? {}) as Record<string, unknown>;

  return (
    <div className="border border-neutral-800 rounded-xl overflow-hidden bg-neutral-900/50">
      <div className="px-3 py-2 border-b border-neutral-800 flex items-center justify-between">
        <span className="text-xs font-medium text-neutral-200">{label}</span>
        <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-blue-900/30 text-blue-400 border border-blue-800/50">
          SINGLE VARIANT
        </span>
      </div>
      <div className="h-[300px]">
        <WidgetSlot scenario={scenario} size="expanded" noGrid>
          {Component ? (
            <Suspense
              fallback={<div className="animate-pulse h-full bg-neutral-800 rounded" />}
            >
              <Component data={data} />
            </Suspense>
          ) : (
            <div className="h-full flex items-center justify-center text-xs text-neutral-500">
              {scenario} — no component registered
            </div>
          )}
        </WidgetSlot>
      </div>
    </div>
  );
}

// ── Main Page ──

export default function WidgetTestPage() {
  const [activeSuite, setActiveSuite] = useState<string | null>(null);
  const [showSingleVariants, setShowSingleVariants] = useState(false);

  // Count stats
  const totalTests = ALL_SUITES.reduce((s, suite) => s + suite.tests.length, 0) + SINGLE_VARIANT_TESTS.length;
  const generatedCount = ALL_SUITES.filter((s) => s.dataSource === "generated").reduce((s, suite) => s + suite.tests.length, 0);
  const fixtureCount = totalTests - generatedCount;

  const activeSuiteData = ALL_SUITES.find((s) => s.scenario === activeSuite);

  return (
    <div className="flex h-screen bg-neutral-950 text-neutral-100">
      {/* Sidebar */}
      <nav className="w-64 shrink-0 border-r border-neutral-800 overflow-y-auto p-4 space-y-4">
        <div>
          <h1 className="text-sm font-bold tracking-tight">Widget Test Suite</h1>
          <p className="text-[10px] text-neutral-500 mt-1">
            {ALL_SUITES.length + 1} groups &middot; {totalTests} tests
          </p>
          <p className="text-[10px] text-neutral-500">
            <span className="text-emerald-500">{generatedCount} generated</span> &middot;{" "}
            <span className="text-amber-500">{fixtureCount} fixture</span>
          </p>
        </div>

        <div className="border-t border-neutral-800 pt-3">
          <a
            href="/widgets"
            className="block text-xs text-blue-400 hover:text-blue-300 mb-3"
          >
            &larr; Back to Widget Gallery
          </a>
        </div>

        {/* Multi-variant suites */}
        <div>
          <h2 className="text-[10px] uppercase font-bold text-neutral-500 tracking-widest mb-1">
            Scenarios ({ALL_SUITES.length})
          </h2>
          {ALL_SUITES.map((suite) => (
            <button
              key={suite.scenario}
              onClick={() => { setActiveSuite(suite.scenario); setShowSingleVariants(false); }}
              className={`block w-full text-left text-xs px-2 py-1 rounded transition-colors ${
                activeSuite === suite.scenario
                  ? "bg-neutral-700 text-white"
                  : "text-neutral-400 hover:text-white hover:bg-neutral-800"
              }`}
            >
              {suite.name}
              <span className="text-neutral-600 ml-1">
                ({suite.tests.length})
                {suite.dataSource === "generated" && (
                  <span className="text-emerald-600 ml-1">G</span>
                )}
              </span>
            </button>
          ))}
        </div>

        {/* Single-variant */}
        <div>
          <h2 className="text-[10px] uppercase font-bold text-neutral-500 tracking-widest mb-1">
            Single Variant ({SINGLE_VARIANT_TESTS.length})
          </h2>
          <button
            onClick={() => { setShowSingleVariants(true); setActiveSuite(null); }}
            className={`block w-full text-left text-xs px-2 py-1 rounded transition-colors ${
              showSingleVariants
                ? "bg-neutral-700 text-white"
                : "text-neutral-400 hover:text-white hover:bg-neutral-800"
            }`}
          >
            All Single Variants
          </button>
        </div>
      </nav>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto p-6">
        {/* Landing */}
        {!activeSuite && !showSingleVariants && (
          <div className="h-full flex items-center justify-center">
            <div className="text-center max-w-lg">
              <h2 className="text-lg font-semibold mb-2">Widget Test Suite</h2>
              <p className="text-sm text-neutral-400 mb-4">
                Comprehensive test cases for all {ALL_SUITES.length} multi-variant scenarios
                and {SINGLE_VARIANT_TESTS.length} single-variant scenarios.
              </p>
              <div className="grid grid-cols-2 gap-3 text-left max-w-sm mx-auto">
                <div className="bg-emerald-900/20 border border-emerald-800/50 rounded-lg px-3 py-2">
                  <div className="text-emerald-400 text-lg font-bold">{generatedCount}</div>
                  <div className="text-[10px] text-emerald-500 uppercase tracking-wider">
                    Generated Test Data
                  </div>
                  <div className="text-[10px] text-neutral-500 mt-1">
                    Fresh data per page load
                  </div>
                </div>
                <div className="bg-amber-900/20 border border-amber-800/50 rounded-lg px-3 py-2">
                  <div className="text-amber-400 text-lg font-bold">{fixtureCount}</div>
                  <div className="text-[10px] text-amber-500 uppercase tracking-wider">
                    Fixture Data (Static)
                  </div>
                  <div className="text-[10px] text-neutral-500 mt-1">
                    Pre-defined demo data
                  </div>
                </div>
              </div>
              <p className="text-xs text-neutral-500 mt-4">
                Select a scenario from the sidebar to view its test cases.
              </p>
            </div>
          </div>
        )}

        {/* Active suite */}
        {activeSuiteData && (
          <div>
            <div className="mb-6">
              <h2 className="text-lg font-semibold">{activeSuiteData.name}</h2>
              <p className="text-xs text-neutral-500">
                {activeSuiteData.scenario} &middot; {activeSuiteData.tests.length} test cases &middot;{" "}
                <span className={activeSuiteData.dataSource === "generated" ? "text-emerald-500" : "text-amber-500"}>
                  {activeSuiteData.dataSource === "generated" ? "Generated test data" : "Static fixture data"}
                </span>
              </p>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-2 2xl:grid-cols-3 gap-4">
              {activeSuiteData.tests.map((test) => (
                <TestCard
                  key={test.fixture}
                  scenario={activeSuiteData.scenario}
                  fixture={test.fixture}
                  testData={test.data as Record<string, unknown> | null}
                  label={test.label}
                />
              ))}
            </div>
          </div>
        )}

        {/* Single variants */}
        {showSingleVariants && (
          <div>
            <div className="mb-6">
              <h2 className="text-lg font-semibold">Single Variant Scenarios</h2>
              <p className="text-xs text-neutral-500">
                {SINGLE_VARIANT_TESTS.length} scenarios with one fixture variant each
              </p>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              {SINGLE_VARIANT_TESTS.map((sv) => (
                <SingleVariantCard
                  key={sv.scenario}
                  scenario={sv.scenario}
                  label={sv.label}
                  fixture={sv.fixture}
                />
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
