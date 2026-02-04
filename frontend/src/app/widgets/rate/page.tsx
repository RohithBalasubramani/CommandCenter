"use client";

import React, { Suspense, useState, useMemo, useEffect, useCallback, useRef, Component as ReactComponent, type ErrorInfo, type ReactNode } from "react";
import { FIXTURES } from "@/components/layer4/fixtureData";
import { getWidgetComponent } from "@/components/layer4/widgetRegistry";
import WidgetSlot from "@/components/layer3/WidgetSlot";
import { config } from "@/lib/config";

// ── Types ──

interface ExhaustiveEntry {
  entry_id: string;
  question_id: string;
  question: string;
  category: string;
  scenario: string;
  fixture: string;
  size: string;
  natural: boolean;
  forced: boolean;
  pipeline_selected_fixture: string | null;
  widget_index: number;
  data_override: Record<string, unknown>;
  layout_context: {
    heading: string;
    total_widgets: number;
    position: number;
  };
  pipeline_meta: {
    intent: Record<string, unknown>;
    voice_response: string;
    processing_time_ms: number;
  };
  rating: "up" | "down" | null;
  tags: string[];
  notes: string;
}

interface RatingsStore {
  [entryId: string]: {
    rating: "up" | "down";
    tags: string[];
    notes: string;
    rated_at: string;
  };
}

// ── Constants ──

const RATING_TAGS = [
  "relevant",
  "good-data",
  "perfect",
  "wrong-chart-type",
  "bad-fixture",
  "data-mismatch",
  "ugly",
];

const STORAGE_KEY = "cc_widget_ratings";
const DEVICE_ID_KEY = "cc_device_id";
const API_BASE = config.api.baseUrl;

/** Get or create a stable device identifier for this browser. */
function getDeviceId(): string {
  if (typeof window === "undefined") return "ssr";
  let id = localStorage.getItem(DEVICE_ID_KEY);
  if (!id) {
    id = `device-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    localStorage.setItem(DEVICE_ID_KEY, id);
  }
  return id;
}

// ── Helpers ──

function resolveWidgetData(scenario: string, fixture: string, dataOverride: Record<string, unknown>) {
  const scenarioMeta = (FIXTURES as Record<string, { defaultFixture?: string; variants?: Record<string, Record<string, unknown>> }>)[scenario];
  const fixtureKey = fixture || scenarioMeta?.defaultFixture || "";
  const fixtureData = scenarioMeta?.variants?.[fixtureKey] ?? {};

  if (!dataOverride || Object.keys(dataOverride).length === 0) {
    return fixtureData as Record<string, unknown>;
  }

  const merged = { ...fixtureData, ...dataOverride };
  const fd = fixtureData as Record<string, unknown>;
  if (
    fd.demoData &&
    typeof fd.demoData === "object" &&
    dataOverride.demoData &&
    typeof dataOverride.demoData === "object"
  ) {
    merged.demoData = {
      ...(fd.demoData as Record<string, unknown>),
      ...(dataOverride.demoData as Record<string, unknown>),
    };
  }
  return merged;
}

function loadRatings(): RatingsStore {
  if (typeof window === "undefined") return {};
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function saveRatings(ratings: RatingsStore) {
  if (typeof window === "undefined") return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(ratings));
}

/** POST a single rating to the backend (fire-and-forget). */
function syncRatingToBackend(entryId: string, payload: RatingsStore[string]) {
  const deviceId = getDeviceId();
  fetch(`${API_BASE}/api/ratings/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      entry_id: entryId,
      rating: payload.rating,
      tags: payload.tags,
      notes: payload.notes,
      rated_at: payload.rated_at,
      device_id: deviceId,
    }),
  }).catch((err) => console.warn("[Ratings] Failed to sync to backend:", err));
}

/** Fetch all ratings from backend and merge into local store. */
async function fetchRemoteRatings(): Promise<RatingsStore> {
  try {
    const res = await fetch(`${API_BASE}/api/ratings/`);
    if (!res.ok) return {};
    const data: Array<{
      entry_id: string;
      rating: "up" | "down";
      tags: string[];
      notes: string;
      rated_at: string;
      device_id: string;
    }> = await res.json();
    const store: RatingsStore = {};
    for (const r of data) {
      // Keep the latest rating per entry_id (backend already ordered by -rated_at)
      if (!store[r.entry_id]) {
        store[r.entry_id] = {
          rating: r.rating,
          tags: r.tags,
          notes: r.notes,
          rated_at: r.rated_at,
        };
      }
    }
    return store;
  } catch (err) {
    console.warn("[Ratings] Failed to fetch from backend:", err);
    return {};
  }
}

/** Bulk-push local ratings to backend (initial sync). */
function bulkSyncToBackend(ratings: RatingsStore) {
  if (Object.keys(ratings).length === 0) return;
  const deviceId = getDeviceId();
  fetch(`${API_BASE}/api/ratings/bulk/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ratings, device_id: deviceId }),
  }).catch((err) => console.warn("[Ratings] Bulk sync failed:", err));
}

// ── Main Component ──

export default function RatingPage() {
  const [entries, setEntries] = useState<ExhaustiveEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [ratings, setRatings] = useState<RatingsStore>({});
  const [notesInput, setNotesInput] = useState("");
  const [selectedTags, setSelectedTags] = useState<string[]>([]);

  // Filters
  const [filterCategory, setFilterCategory] = useState<string>("all");
  const [filterScenario, setFilterScenario] = useState<string>("all");
  const [filterNatural, setFilterNatural] = useState<string>("all"); // "all" | "natural" | "forced"
  const [filterRating, setFilterRating] = useState<string>("all"); // "all" | "unrated" | "up" | "down"

  const hasSyncedRef = useRef(false);

  // Load data + sync ratings with backend
  useEffect(() => {
    fetch("/simulation/exhaustive_data.json")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => {
        setEntries(data.entries || []);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setLoading(false);
      });

    // Load local ratings first (instant)
    const local = loadRatings();
    setRatings(local);

    // Then fetch remote ratings and merge
    if (!hasSyncedRef.current) {
      hasSyncedRef.current = true;
      fetchRemoteRatings().then((remote) => {
        // Merge: remote ratings fill in gaps, local takes priority for conflicts
        const merged = { ...remote, ...local };
        setRatings(merged);
        saveRatings(merged);
        // Push any local-only ratings to backend
        const localOnlyKeys = Object.keys(local).filter((k) => !remote[k]);
        if (localOnlyKeys.length > 0) {
          const localOnly: RatingsStore = {};
          for (const k of localOnlyKeys) localOnly[k] = local[k];
          bulkSyncToBackend(localOnly);
        }
      });
    }
  }, []);

  // Filtered entries
  const filtered = useMemo(() => {
    return entries.filter((e) => {
      if (filterCategory !== "all" && e.category !== filterCategory) return false;
      if (filterScenario !== "all" && e.scenario !== filterScenario) return false;
      if (filterNatural === "natural" && !e.natural) return false;
      if (filterNatural === "forced" && !e.forced) return false;
      if (filterRating === "unrated" && ratings[e.entry_id]) return false;
      if (filterRating === "up" && ratings[e.entry_id]?.rating !== "up") return false;
      if (filterRating === "down" && ratings[e.entry_id]?.rating !== "down") return false;
      return true;
    });
  }, [entries, filterCategory, filterScenario, filterNatural, filterRating, ratings]);

  const currentEntry = filtered[currentIndex] || null;

  // Unique values for filters
  const categories = useMemo(() => Array.from(new Set(entries.map((e) => e.category))).sort(), [entries]);
  const scenarios = useMemo(() => Array.from(new Set(entries.map((e) => e.scenario))).sort(), [entries]);

  // Stats
  const stats = useMemo(() => {
    const total = entries.length;
    const rated = Object.keys(ratings).length;
    const liked = Object.values(ratings).filter((r) => r.rating === "up").length;
    const disliked = Object.values(ratings).filter((r) => r.rating === "down").length;
    return { total, rated, liked, disliked, remaining: total - rated };
  }, [entries, ratings]);

  // Reset notes/tags when entry changes
  useEffect(() => {
    if (currentEntry) {
      const existing = ratings[currentEntry.entry_id];
      setNotesInput(existing?.notes || "");
      setSelectedTags(existing?.tags || []);
    }
  }, [currentIndex, currentEntry?.entry_id]); // eslint-disable-line react-hooks/exhaustive-deps

  // Navigation
  const goNext = useCallback(() => {
    setCurrentIndex((i) => Math.min(i + 1, filtered.length - 1));
  }, [filtered.length]);

  const goPrev = useCallback(() => {
    setCurrentIndex((i) => Math.max(i - 1, 0));
  }, []);

  const jumpToUnrated = useCallback(() => {
    const idx = filtered.findIndex((e) => !ratings[e.entry_id]);
    if (idx >= 0) setCurrentIndex(idx);
  }, [filtered, ratings]);

  // Rate
  const rate = useCallback(
    (value: "up" | "down") => {
      if (!currentEntry) return;
      const payload = {
        rating: value,
        tags: selectedTags,
        notes: notesInput,
        rated_at: new Date().toISOString(),
      };
      const updated = {
        ...ratings,
        [currentEntry.entry_id]: payload,
      };
      setRatings(updated);
      saveRatings(updated);
      // Sync to backend (fire-and-forget)
      syncRatingToBackend(currentEntry.entry_id, payload);
      // Auto-advance
      setTimeout(goNext, 150);
    },
    [currentEntry, ratings, selectedTags, notesInput, goNext]
  );

  // Toggle tag
  const toggleTag = useCallback((tag: string) => {
    setSelectedTags((prev) =>
      prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]
    );
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      switch (e.key) {
        case "ArrowRight":
          goNext();
          break;
        case "ArrowLeft":
          goPrev();
          break;
        case "ArrowUp":
          e.preventDefault();
          rate("up");
          break;
        case "ArrowDown":
          e.preventDefault();
          rate("down");
          break;
        case "s":
          goNext();
          break;
        case "n":
          document.getElementById("notes-input")?.focus();
          break;
        case "u":
          jumpToUnrated();
          break;
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [goNext, goPrev, rate, jumpToUnrated]);

  // Export ratings
  const exportRatings = useCallback(() => {
    const blob = new Blob(
      [
        JSON.stringify(
          {
            exported_at: new Date().toISOString(),
            stats,
            ratings,
          },
          null,
          2
        ),
      ],
      { type: "application/json" }
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `widget_ratings_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [ratings, stats]);

  // ── Render ──

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0a0a0f] flex items-center justify-center text-neutral-400">
        Loading exhaustive data...
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-[#0a0a0f] flex items-center justify-center">
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-6 text-red-400 max-w-md">
          <h2 className="font-bold text-lg mb-2">Failed to load data</h2>
          <p className="text-sm">{error}</p>
          <p className="text-xs mt-3 text-red-400/60">
            Run <code className="bg-red-500/10 px-1 rounded">python run_exhaustive.py</code> first
            to generate the data.
          </p>
        </div>
      </div>
    );
  }

  if (entries.length === 0) {
    return (
      <div className="min-h-screen bg-[#0a0a0f] flex items-center justify-center text-neutral-400">
        No entries found. Run the exhaustive simulation first.
      </div>
    );
  }

  const entryRating = currentEntry ? ratings[currentEntry.entry_id] : null;

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-neutral-200 flex flex-col">
      {/* Header */}
      <div className="border-b border-neutral-800 px-6 py-3 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-4">
          <h1 className="text-sm font-bold tracking-wider uppercase text-neutral-400">
            Widget Rating
          </h1>
          <div className="flex gap-2 text-xs">
            <span className="px-2 py-0.5 rounded bg-green-500/10 text-green-400">
              {stats.liked} liked
            </span>
            <span className="px-2 py-0.5 rounded bg-red-500/10 text-red-400">
              {stats.disliked} disliked
            </span>
            <span className="px-2 py-0.5 rounded bg-neutral-800 text-neutral-400">
              {stats.remaining} remaining
            </span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={exportRatings}
            className="text-xs px-3 py-1.5 rounded bg-indigo-600 hover:bg-indigo-500 text-white transition-colors"
          >
            Export Ratings
          </button>
          <a
            href="/widgets/test"
            className="text-xs px-3 py-1.5 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 transition-colors"
          >
            Test Page
          </a>
        </div>
      </div>

      {/* Filters */}
      <div className="border-b border-neutral-800 px-6 py-2 flex items-center gap-4 shrink-0">
        <select
          value={filterCategory}
          onChange={(e) => {
            setFilterCategory(e.target.value);
            setCurrentIndex(0);
          }}
          className="text-xs bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-neutral-300"
        >
          <option value="all">All Categories</option>
          {categories.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
        <select
          value={filterScenario}
          onChange={(e) => {
            setFilterScenario(e.target.value);
            setCurrentIndex(0);
          }}
          className="text-xs bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-neutral-300"
        >
          <option value="all">All Scenarios</option>
          {scenarios.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <select
          value={filterNatural}
          onChange={(e) => {
            setFilterNatural(e.target.value);
            setCurrentIndex(0);
          }}
          className="text-xs bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-neutral-300"
        >
          <option value="all">Natural + Forced</option>
          <option value="natural">Natural Only</option>
          <option value="forced">Forced Only</option>
        </select>
        <select
          value={filterRating}
          onChange={(e) => {
            setFilterRating(e.target.value);
            setCurrentIndex(0);
          }}
          className="text-xs bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-neutral-300"
        >
          <option value="all">All Ratings</option>
          <option value="unrated">Unrated</option>
          <option value="up">Liked</option>
          <option value="down">Disliked</option>
        </select>
        <span className="text-xs text-neutral-500 ml-auto">
          {filtered.length} entries
        </span>
      </div>

      {/* Main Content */}
      {currentEntry ? (
        <div className="flex-1 flex min-h-0 overflow-hidden">
          {/* Left: Question + Widget */}
          <div className="flex-1 flex flex-col p-6 overflow-auto">
            {/* Question */}
            <div className="mb-4">
              <div className="text-xs text-neutral-500 mb-1">
                {currentEntry.question_id} / {currentEntry.category}
              </div>
              <div className="text-sm font-medium text-neutral-200">
                &ldquo;{currentEntry.question}&rdquo;
              </div>
              {currentEntry.layout_context.heading && (
                <div className="text-xs text-neutral-500 mt-1">
                  Dashboard: {currentEntry.layout_context.heading}
                </div>
              )}
            </div>

            {/* Widget Rendering */}
            <div className="flex-1 min-h-[300px] max-h-[500px] bg-neutral-900/50 rounded-xl border border-neutral-800 overflow-hidden">
              <Suspense
                fallback={
                  <div className="w-full h-full flex items-center justify-center text-neutral-500 text-sm">
                    Loading widget...
                  </div>
                }
              >
                <WidgetRenderer entry={currentEntry} />
              </Suspense>
            </div>

            {/* Meta badges */}
            <div className="flex gap-2 mt-3 flex-wrap">
              <span
                className={`text-[10px] px-2 py-0.5 rounded-full font-bold uppercase tracking-wide ${
                  currentEntry.natural
                    ? "bg-green-500/10 text-green-400"
                    : "bg-amber-500/10 text-amber-400"
                }`}
              >
                {currentEntry.natural ? "Natural" : "Forced"}
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 font-mono">
                {currentEntry.scenario}
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-neutral-800 text-neutral-400 font-mono">
                {currentEntry.fixture}
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-neutral-800 text-neutral-400">
                Size: {currentEntry.size}
              </span>
              {currentEntry.pipeline_selected_fixture && (
                <span
                  className={`text-[10px] px-2 py-0.5 rounded-full ${
                    currentEntry.fixture === currentEntry.pipeline_selected_fixture
                      ? "bg-green-500/10 text-green-400"
                      : "bg-neutral-800 text-neutral-500"
                  }`}
                >
                  {currentEntry.fixture === currentEntry.pipeline_selected_fixture
                    ? "Pipeline-selected"
                    : `Pipeline chose: ${currentEntry.pipeline_selected_fixture}`}
                </span>
              )}
            </div>
          </div>

          {/* Right: Rating Panel */}
          <div className="w-80 border-l border-neutral-800 p-6 flex flex-col shrink-0">
            {/* Rating buttons */}
            <div className="flex gap-3 mb-6">
              <button
                onClick={() => rate("up")}
                className={`flex-1 py-3 rounded-lg text-lg font-bold transition-all ${
                  entryRating?.rating === "up"
                    ? "bg-green-500 text-white shadow-lg shadow-green-500/30"
                    : "bg-neutral-800 text-neutral-400 hover:bg-green-500/20 hover:text-green-400"
                }`}
              >
                &#x1F44D;
              </button>
              <button
                onClick={() => rate("down")}
                className={`flex-1 py-3 rounded-lg text-lg font-bold transition-all ${
                  entryRating?.rating === "down"
                    ? "bg-red-500 text-white shadow-lg shadow-red-500/30"
                    : "bg-neutral-800 text-neutral-400 hover:bg-red-500/20 hover:text-red-400"
                }`}
              >
                &#x1F44E;
              </button>
              <button
                onClick={goNext}
                className="px-4 py-3 rounded-lg bg-neutral-800 text-neutral-400 hover:bg-neutral-700 transition-colors text-sm"
              >
                Skip
              </button>
            </div>

            {/* Tags */}
            <div className="mb-4">
              <div className="text-xs text-neutral-500 mb-2 uppercase tracking-wider font-bold">
                Tags
              </div>
              <div className="flex flex-wrap gap-1.5">
                {RATING_TAGS.map((tag) => (
                  <button
                    key={tag}
                    onClick={() => toggleTag(tag)}
                    className={`text-[10px] px-2 py-1 rounded-full transition-colors ${
                      selectedTags.includes(tag)
                        ? "bg-indigo-500/20 text-indigo-400 ring-1 ring-indigo-500/50"
                        : "bg-neutral-800 text-neutral-500 hover:text-neutral-300"
                    }`}
                  >
                    {tag}
                  </button>
                ))}
              </div>
            </div>

            {/* Notes */}
            <div className="mb-6">
              <div className="text-xs text-neutral-500 mb-2 uppercase tracking-wider font-bold">
                Notes
              </div>
              <textarea
                id="notes-input"
                value={notesInput}
                onChange={(e) => setNotesInput(e.target.value)}
                placeholder="Optional notes..."
                rows={3}
                className="w-full bg-neutral-900 border border-neutral-700 rounded-lg px-3 py-2 text-xs text-neutral-300 placeholder-neutral-600 resize-none focus:outline-none focus:ring-1 focus:ring-indigo-500/50"
              />
            </div>

            {/* Context Info */}
            <div className="border-t border-neutral-800 pt-4 space-y-2 text-xs text-neutral-500">
              <div className="flex justify-between">
                <span>Scenario</span>
                <span className="text-neutral-300 font-mono">{currentEntry.scenario}</span>
              </div>
              <div className="flex justify-between">
                <span>Fixture</span>
                <span className="text-neutral-300 font-mono text-[10px]">
                  {currentEntry.fixture.length > 25
                    ? currentEntry.fixture.slice(0, 25) + "..."
                    : currentEntry.fixture}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Source</span>
                <span className={currentEntry.natural ? "text-green-400" : "text-amber-400"}>
                  {currentEntry.natural ? "Pipeline" : "Synthetic"}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Widget #</span>
                <span className="text-neutral-300">
                  {currentEntry.widget_index >= 0
                    ? `${currentEntry.widget_index + 1} of ${currentEntry.layout_context.total_widgets}`
                    : "N/A (forced)"}
                </span>
              </div>
              {currentEntry.pipeline_meta.processing_time_ms > 0 && (
                <div className="flex justify-between">
                  <span>Pipeline</span>
                  <span className="text-neutral-300">
                    {(currentEntry.pipeline_meta.processing_time_ms / 1000).toFixed(1)}s
                  </span>
                </div>
              )}
            </div>

            {/* Keyboard shortcuts */}
            <div className="mt-auto pt-4 border-t border-neutral-800">
              <div className="text-[10px] text-neutral-600 space-y-0.5">
                <div>
                  <kbd className="bg-neutral-800 px-1 rounded">&#8593;</kbd> Like &nbsp;
                  <kbd className="bg-neutral-800 px-1 rounded">&#8595;</kbd> Dislike
                </div>
                <div>
                  <kbd className="bg-neutral-800 px-1 rounded">&#8592;</kbd>{" "}
                  <kbd className="bg-neutral-800 px-1 rounded">&#8594;</kbd> Navigate &nbsp;
                  <kbd className="bg-neutral-800 px-1 rounded">s</kbd> Skip
                </div>
                <div>
                  <kbd className="bg-neutral-800 px-1 rounded">n</kbd> Focus notes &nbsp;
                  <kbd className="bg-neutral-800 px-1 rounded">u</kbd> Jump unrated
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center text-neutral-500">
          No entries match the current filters.
        </div>
      )}

      {/* Footer Navigation */}
      <div className="border-t border-neutral-800 px-6 py-2 flex items-center justify-between shrink-0">
        <button
          onClick={goPrev}
          disabled={currentIndex === 0}
          className="text-xs px-3 py-1.5 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 disabled:opacity-30 transition-colors"
        >
          Prev
        </button>
        <div className="text-xs text-neutral-500">
          Entry {filtered.length > 0 ? currentIndex + 1 : 0} / {filtered.length}
          {stats.rated > 0 && (
            <span className="ml-3 text-neutral-600">
              ({Math.round((stats.rated / stats.total) * 100)}% rated)
            </span>
          )}
        </div>
        <div className="flex gap-2">
          <button
            onClick={jumpToUnrated}
            className="text-xs px-3 py-1.5 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 transition-colors"
          >
            Jump to unrated
          </button>
          <button
            onClick={goNext}
            disabled={currentIndex >= filtered.length - 1}
            className="text-xs px-3 py-1.5 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 disabled:opacity-30 transition-colors"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Error Boundary ──

interface EBProps { children: ReactNode; label?: string }
interface EBState { hasError: boolean; error?: Error }

class WidgetErrorBoundary extends ReactComponent<EBProps, EBState> {
  constructor(props: EBProps) { super(props); this.state = { hasError: false }; }
  static getDerivedStateFromError(error: Error): EBState { return { hasError: true, error }; }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error(`[WidgetError] ${this.props.label}:`, error, info); }
  render() {
    if (this.state.hasError) {
      return (
        <div className="w-full h-full flex flex-col items-center justify-center text-center p-4 bg-red-950/30 border border-red-900/50 rounded-xl">
          <span className="text-sm font-bold text-red-400 mb-1">Widget Error</span>
          <span className="text-xs text-red-500/70">{this.props.label}</span>
          <span className="text-[10px] text-red-600/50 mt-1 max-w-[300px] truncate">{this.state.error?.message}</span>
        </div>
      );
    }
    return this.props.children;
  }
}

// ── Widget Renderer ──

function WidgetRenderer({ entry }: { entry: ExhaustiveEntry }) {
  const WidgetComponent = getWidgetComponent(entry.scenario);
  const data = useMemo(
    () => resolveWidgetData(entry.scenario, entry.fixture, entry.data_override),
    [entry.scenario, entry.fixture, entry.data_override]
  );

  if (!WidgetComponent) {
    return (
      <div className="w-full h-full flex items-center justify-center text-neutral-500 text-sm">
        Unknown scenario: {entry.scenario}
      </div>
    );
  }

  return (
    <div className="w-full h-full">
      <WidgetErrorBoundary label={`${entry.scenario}/${entry.fixture} (${entry.size})`}>
        <WidgetSlot scenario={entry.scenario} title={entry.scenario} size={entry.size as "compact" | "normal" | "expanded" | "hero"} noGrid>
          <WidgetComponent data={data} />
        </WidgetSlot>
      </WidgetErrorBoundary>
    </div>
  );
}
