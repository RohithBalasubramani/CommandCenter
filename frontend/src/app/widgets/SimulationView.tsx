"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import SampleDashboard from "./SampleDashboard";
import { FeedbackChecklist } from "./FeedbackForm";
import type { WidgetSize, WidgetHeightHint } from "@/types";

// ── Types matching the simulation runner output ──

interface SimulationRunMeta {
  run_id: string;
  tag: string;
  started_at: string;
  completed_at: string;
  backend_url: string;
  total_questions: number;
  successful: number;
  failed: number;
}

interface SimulationWidget {
  scenario: string;
  fixture?: string;
  relevance?: number;
  size: WidgetSize;
  heightHint?: WidgetHeightHint;
  position?: string;
  data_override?: Record<string, unknown>;
  description?: string;
}

interface SimulationResult {
  question_id: string;
  question: string;
  category: string;
  expected_characteristics: string[];
  expected_scenarios: string[];
  difficulty: string;
  response: {
    voice_response: string;
    layout_json: {
      heading: string;
      widgets: SimulationWidget[];
    } | null;
    intent: {
      type: string;
      domains: string[];
      confidence: number;
    } | null;
    processing_time_ms: number;
  } | null;
  analysis: {
    total_time_ms: number;
    widget_count?: number;
    scenarios_used?: string[];
    fixtures_used?: string[];
    heading?: string;
    has_layout?: boolean;
    has_voice_response?: boolean;
    domain_match?: { expected: string[]; detected: string[]; match: boolean };
    scenario_coverage?: {
      expected: string[];
      present: string[];
      missing: string[];
      unexpected: string[];
    };
    error: string | null;
  } | null;
}

interface SimulationLog {
  run_meta: SimulationRunMeta;
  results: SimulationResult[];
}

// ── Size fallback: if heightHint is missing, infer from size ──

function inferHeightHint(size: WidgetSize): WidgetHeightHint {
  switch (size) {
    case "hero": return "tall";
    case "expanded": return "tall";
    case "normal": return "medium";
    case "compact": return "short";
    default: return "medium";
  }
}

// ── Category badge colors ──

const CATEGORY_COLORS: Record<string, string> = {
  comparison: "bg-purple-900 text-purple-300",
  trend: "bg-blue-900 text-blue-300",
  distribution: "bg-teal-900 text-teal-300",
  maintenance: "bg-orange-900 text-orange-300",
  shift: "bg-yellow-900 text-yellow-300",
  work_orders: "bg-red-900 text-red-300",
  energy: "bg-green-900 text-green-300",
  health_status: "bg-cyan-900 text-cyan-300",
  flow_sankey: "bg-indigo-900 text-indigo-300",
  cumulative: "bg-emerald-900 text-emerald-300",
  multi_source: "bg-violet-900 text-violet-300",
  power_quality: "bg-amber-900 text-amber-300",
  hvac: "bg-sky-900 text-sky-300",
  ups_dg: "bg-rose-900 text-rose-300",
  top_consumers: "bg-pink-900 text-pink-300",
  cross_domain: "bg-neutral-700 text-neutral-300",
};

// ── Single simulation result card ──

function SimulationCard({ result }: { result: SimulationResult }) {
  const layout = result.response?.layout_json;
  const widgets = layout?.widgets ?? [];
  const analysis = result.analysis;
  const hasError = analysis?.error != null;

  // Map simulation widgets to SampleDashboard format (include data_override + description)
  const dashboardWidgets = widgets.map((w) => ({
    scenario: w.scenario,
    fixture: w.fixture,
    size: w.size || "normal" as WidgetSize,
    heightHint: w.heightHint || inferHeightHint(w.size || "normal"),
    data_override: w.data_override as Record<string, unknown> | undefined,
    description: w.description as string | undefined,
  }));

  const categoryColor = CATEGORY_COLORS[result.category] || "bg-neutral-800 text-neutral-400";

  return (
    <div className="mb-8">
      {/* Metadata bar */}
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="text-[10px] font-mono text-neutral-500">{result.question_id}</span>
        <span className={`text-[10px] px-1.5 py-0.5 rounded ${categoryColor}`}>
          {result.category}
        </span>
        <span className={`text-[10px] px-1.5 py-0.5 rounded ${
          result.difficulty === "complex" ? "bg-red-900/50 text-red-400" :
          result.difficulty === "medium" ? "bg-yellow-900/50 text-yellow-400" :
          "bg-neutral-800 text-neutral-500"
        }`}>
          {result.difficulty}
        </span>
        {analysis && !hasError && (
          <>
            <span className="text-[10px] text-neutral-600">
              {analysis.total_time_ms}ms
            </span>
            <span className="text-[10px] text-neutral-600">
              {analysis.widget_count ?? 0} widgets
            </span>
            {analysis.scenario_coverage?.missing && analysis.scenario_coverage.missing.length > 0 && (
              <span className="text-[10px] text-amber-500">
                missing: {analysis.scenario_coverage.missing.join(", ")}
              </span>
            )}
          </>
        )}
        {hasError && (
          <span className="text-[10px] text-red-400">
            Error: {analysis?.error}
          </span>
        )}
      </div>

      {/* Voice response preview */}
      {result.response?.voice_response && (
        <p className="text-[11px] text-neutral-500 italic mb-2 line-clamp-2">
          &ldquo;{result.response.voice_response}&rdquo;
        </p>
      )}

      {/* Dashboard render */}
      {dashboardWidgets.length > 0 ? (
        <SampleDashboard
          id={`sim:${result.question_id}`}
          title={layout?.heading || result.question}
          description={`Q: "${result.question}"`}
          widgets={dashboardWidgets}
        />
      ) : (
        <div className="border border-neutral-800 rounded-xl p-6 text-center">
          <p className="text-sm text-neutral-500">
            {hasError ? "Failed to generate dashboard" : "No widgets generated for this question"}
          </p>
          <p className="text-xs text-neutral-600 mt-1">{result.question}</p>
        </div>
      )}
    </div>
  );
}

// ── Main simulation view ──

export default function SimulationView({ logFile = "/simulation/simulation_log.json", title = "Simulation Dashboards" }: { logFile?: string; title?: string }) {
  const [log, setLog] = useState<SimulationLog | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [categoryFilter, setCategoryFilter] = useState<string>("all");
  const PAGE_SIZE = 5;
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);

  useEffect(() => {
    setLoading(true);
    setError(null);
    setLog(null);
    setCategoryFilter("all");
    setVisibleCount(PAGE_SIZE);
    fetch(logFile)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: SimulationLog) => {
        setLog(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, [logFile]);

  const categories = useMemo(() => {
    if (!log) return [];
    const cats = new Set(log.results.map((r) => r.category));
    return Array.from(cats).sort();
  }, [log]);

  const filteredResults = useMemo(() => {
    if (!log) return [];
    if (categoryFilter === "all") return log.results;
    return log.results.filter((r) => r.category === categoryFilter);
  }, [log, categoryFilter]);

  // Reset pagination when filter changes
  useEffect(() => {
    setVisibleCount(PAGE_SIZE);
  }, [categoryFilter]);

  const visibleResults = filteredResults.slice(0, visibleCount);
  const hasMore = visibleCount < filteredResults.length;
  const loadMore = useCallback(() => setVisibleCount((v) => v + PAGE_SIZE), []);

  const successCount = log?.results.filter((r) => !r.analysis?.error).length ?? 0;
  const failCount = log?.results.filter((r) => r.analysis?.error).length ?? 0;

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <div className="animate-pulse text-sm text-neutral-400">Loading simulation results...</div>
        </div>
      </div>
    );
  }

  if (error || !log) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center max-w-md">
          <h2 className="text-lg font-semibold mb-2">No Simulation Data</h2>
          <p className="text-sm text-neutral-400 mb-4">
            Run the simulation first to generate dashboard results:
          </p>
          <code className="text-xs bg-neutral-800 px-3 py-2 rounded block text-neutral-300">
            python scripts/simulation/run_simulation.py --tag baseline
          </code>
          {error && (
            <p className="text-xs text-red-400 mt-3">Fetch error: {error}</p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div>
      {/* Header */}
      <div className="mb-4">
        <h2 className="text-lg font-semibold">{title}</h2>
        <p className="text-xs text-neutral-500 mt-1">
          Run: {log.run_meta.run_id}
          {log.run_meta.tag && <span className="ml-2 text-blue-400">[{log.run_meta.tag}]</span>}
          {" "}&middot; {successCount} OK, {failCount} failed
          {" "}&middot; {new Date(log.run_meta.started_at).toLocaleString()}
        </p>
      </div>

      {/* Category filter */}
      <div className="flex items-center gap-2 mb-4 flex-wrap">
        <button
          onClick={() => setCategoryFilter("all")}
          className={`text-[10px] px-2 py-1 rounded transition-colors ${
            categoryFilter === "all" ? "bg-neutral-600 text-white" : "bg-neutral-800 text-neutral-400 hover:bg-neutral-700"
          }`}
        >
          All ({log.results.length})
        </button>
        {categories.map((cat) => {
          const count = log.results.filter((r) => r.category === cat).length;
          const color = CATEGORY_COLORS[cat] || "bg-neutral-800 text-neutral-400";
          return (
            <button
              key={cat}
              onClick={() => setCategoryFilter(cat)}
              className={`text-[10px] px-2 py-1 rounded transition-colors ${
                categoryFilter === cat ? color : "bg-neutral-800 text-neutral-500 hover:bg-neutral-700"
              }`}
            >
              {cat} ({count})
            </button>
          );
        })}
      </div>

      {/* Feedback checklist for simulation dashboards */}
      <FeedbackChecklist pageId="simulation" variant="dashboards" />

      {/* Results (paginated) */}
      {visibleResults.map((result) => (
        <SimulationCard key={result.question_id} result={result} />
      ))}

      {/* Load more */}
      {hasMore && (
        <div className="text-center py-6">
          <button
            onClick={loadMore}
            className="text-xs px-4 py-2 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 transition-colors"
          >
            Load More ({filteredResults.length - visibleCount} remaining)
          </button>
        </div>
      )}

      {/* End marker */}
      {!hasMore && filteredResults.length > 0 && (
        <p className="text-center text-[10px] text-neutral-600 py-4">
          Showing all {filteredResults.length} results
        </p>
      )}
    </div>
  );
}
