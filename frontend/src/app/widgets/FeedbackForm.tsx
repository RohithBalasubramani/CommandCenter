"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  type WidgetFeedback,
  type DashboardFeedback,
  type PageFeedback,
  FEEDBACK_TAGS,
  saveWidgetFeedback,
  saveDashboardFeedback,
  getWidgetFeedbackForScenario,
  closeWidgetFeedback,
  reopenWidgetFeedback,
  getAllDashboardFeedback,
  closeDashboardFeedback,
  reopenDashboardFeedback,
  getPageFeedbackForPage,
  savePageFeedback,
  closePageFeedback,
  reopenPageFeedback,
} from "./feedbackStore";

// ── Star rating ──

function StarRating({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  return (
    <div className="flex gap-1">
      {[1, 2, 3, 4, 5].map((star) => (
        <button
          key={star}
          onClick={() => onChange(star)}
          className={`text-lg transition-colors ${
            star <= value ? "text-yellow-400" : "text-neutral-600"
          } hover:text-yellow-300`}
        >
          ★
        </button>
      ))}
    </div>
  );
}

// ── Widget feedback form ──
// Always starts blank. Each save creates a new issue.

export function WidgetFeedbackForm({
  scenario,
  variant,
  onSaved,
}: {
  scenario: string;
  variant: string;
  onSaved?: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [rating, setRating] = useState(0);
  const [tags, setTags] = useState<string[]>([]);
  const [notes, setNotes] = useState("");

  const toggleTag = (id: string) => {
    setTags((prev) => (prev.includes(id) ? prev.filter((t) => t !== id) : [...prev, id]));
  };

  const handleSave = () => {
    if (!notes.trim() && rating === 0 && tags.length === 0) return;
    saveWidgetFeedback({
      scenario,
      variant,
      rating,
      tags,
      notes,
      timestamp: Date.now(),
    });
    // Reset form for next issue
    setRating(0);
    setTags([]);
    setNotes("");
    setOpen(false);
    onSaved?.();
  };

  if (!open) {
    return (
      <div className="flex items-center gap-3 mt-2">
        <button
          onClick={() => setOpen(true)}
          className="text-xs text-neutral-400 hover:text-white border border-neutral-700 rounded px-2 py-1 transition-colors"
        >
          Raise Issue
        </button>
      </div>
    );
  }

  return (
    <div className="mt-3 p-3 rounded-lg bg-neutral-800/60 border border-neutral-700/50 space-y-3">
      {/* Rating */}
      <div className="flex items-center gap-3">
        <span className="text-xs text-neutral-400 w-14">Rating</span>
        <StarRating value={rating} onChange={setRating} />
      </div>

      {/* Tags */}
      <div className="flex flex-wrap gap-1.5">
        {FEEDBACK_TAGS.map((tag) => (
          <button
            key={tag.id}
            onClick={() => toggleTag(tag.id)}
            className={`text-[10px] px-2 py-0.5 rounded-full border transition-colors ${
              tags.includes(tag.id)
                ? "bg-blue-500/20 border-blue-500/50 text-blue-300"
                : "bg-neutral-800 border-neutral-700 text-neutral-400 hover:text-white"
            }`}
          >
            {tag.label}
          </button>
        ))}
      </div>

      {/* Notes */}
      <textarea
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        placeholder="What would you change? Be specific..."
        className="w-full h-16 text-xs bg-neutral-900 border border-neutral-700 rounded px-2 py-1.5 text-neutral-200 placeholder-neutral-600 resize-none focus:outline-none focus:border-neutral-500"
      />

      {/* Actions */}
      <div className="flex items-center gap-2">
        <button
          onClick={handleSave}
          className="text-xs px-3 py-1 rounded bg-blue-600 hover:bg-blue-500 text-white transition-colors"
        >
          Save
        </button>
        <button
          onClick={() => { setOpen(false); setRating(0); setTags([]); setNotes(""); }}
          className="text-xs px-3 py-1 rounded bg-neutral-700 hover:bg-neutral-600 text-neutral-300 transition-colors"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}

// ── Dashboard feedback form ──
// Always starts blank. Each save creates a new issue.

export function DashboardFeedbackForm({
  dashboardId,
  onSaved,
}: {
  dashboardId: string;
  existingFeedback?: DashboardFeedback | null;
  onSaved?: () => void;
}) {
  const [rating, setRating] = useState(0);
  const [notes, setNotes] = useState("");
  const [open, setOpen] = useState(false);

  const handleSave = () => {
    if (!notes.trim() && rating === 0) return;
    saveDashboardFeedback({
      dashboardId,
      rating,
      notes,
      timestamp: Date.now(),
    });
    setRating(0);
    setNotes("");
    setOpen(false);
    onSaved?.();
  };

  if (!open) {
    return (
      <div className="mt-3">
        <button
          onClick={() => setOpen(true)}
          className="text-xs text-neutral-400 hover:text-white border border-neutral-700 rounded px-2 py-1 transition-colors"
        >
          Raise Issue
        </button>
      </div>
    );
  }

  return (
    <div className="mt-3 p-3 rounded-lg bg-neutral-800/60 border border-neutral-700/50 space-y-3">
      <div className="flex items-center gap-3">
        <span className="text-xs text-neutral-400 w-14">Rating</span>
        <StarRating value={rating} onChange={setRating} />
      </div>
      <textarea
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        placeholder="What would you change about this layout? Arrangement, sizes, hierarchy..."
        className="w-full h-16 text-xs bg-neutral-900 border border-neutral-700 rounded px-2 py-1.5 text-neutral-200 placeholder-neutral-600 resize-none focus:outline-none focus:border-neutral-500"
      />
      <div className="flex items-center gap-2">
        <button
          onClick={handleSave}
          className="text-xs px-3 py-1 rounded bg-blue-600 hover:bg-blue-500 text-white transition-colors"
        >
          Save
        </button>
        <button
          onClick={() => { setOpen(false); setRating(0); setNotes(""); }}
          className="text-xs px-3 py-1 rounded bg-neutral-700 hover:bg-neutral-600 text-neutral-300 transition-colors"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}

// ── Page-level feedback form ──
// Always starts blank. Each save creates a new issue.

export function PageFeedbackForm({ pageId, onSaved }: { pageId: string; onSaved?: () => void }) {
  const [open, setOpen] = useState(false);
  const [notes, setNotes] = useState("");

  const handleSave = () => {
    if (!notes.trim()) return;
    savePageFeedback({ pageId, notes, timestamp: Date.now() });
    setNotes("");
    setOpen(false);
    onSaved?.();
  };

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="text-xs text-neutral-400 hover:text-white border border-neutral-700 rounded px-2 py-1 transition-colors"
      >
        Raise Page Issue
      </button>
    );
  }

  return (
    <div className="p-3 rounded-lg bg-neutral-800/60 border border-neutral-700/50 space-y-3">
      <textarea
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        placeholder="Overall feedback for this page..."
        className="w-full h-16 text-xs bg-neutral-900 border border-neutral-700 rounded px-2 py-1.5 text-neutral-200 placeholder-neutral-600 resize-none focus:outline-none focus:border-neutral-500"
      />
      <div className="flex items-center gap-2">
        <button
          onClick={handleSave}
          className="text-xs px-3 py-1 rounded bg-blue-600 hover:bg-blue-500 text-white transition-colors"
        >
          Save
        </button>
        <button
          onClick={() => { setOpen(false); setNotes(""); }}
          className="text-xs px-3 py-1 rounded bg-neutral-700 hover:bg-neutral-600 text-neutral-300 transition-colors"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}

// ── Feedback checklist (issue tracker) ──

interface FeedbackItem {
  id: string;
  label: string;
  notes: string;
  closed: boolean;
  timestamp: number;
  kind: "widget" | "dashboard" | "page";
}

function formatRelativeTime(ts: number): string {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

export function FeedbackChecklist({
  pageId,
  variant,
}: {
  pageId: string;
  variant: "scenario" | "dashboards";
}) {
  const [items, setItems] = useState<FeedbackItem[]>([]);
  const [ver, setVer] = useState(0);
  const reload = useCallback(() => setVer((v) => v + 1), []);

  useEffect(() => {
    const all: FeedbackItem[] = [];

    // Page-level feedback items
    const pageFbs = getPageFeedbackForPage(pageId);
    for (const f of pageFbs) {
      all.push({
        id: f.id,
        label: "Page feedback",
        notes: f.notes,
        closed: !!f.closed,
        timestamp: f.timestamp,
        kind: "page",
      });
    }

    if (variant === "scenario") {
      const scenario = pageId.replace("scenario:", "");
      const fbs = getWidgetFeedbackForScenario(scenario);
      for (const f of fbs) {
        all.push({
          id: f.id,
          label: f.variant,
          notes: f.notes,
          closed: !!f.closed,
          timestamp: f.timestamp,
          kind: "widget",
        });
      }
    } else {
      const fbs = getAllDashboardFeedback();
      for (const f of fbs) {
        all.push({
          id: f.id,
          label: f.dashboardId,
          notes: f.notes,
          closed: !!f.closed,
          timestamp: f.timestamp,
          kind: "dashboard",
        });
      }
    }

    // Sort newest first
    all.sort((a, b) => b.timestamp - a.timestamp);
    setItems(all);
  }, [pageId, variant, ver]);

  const handleToggle = (item: FeedbackItem) => {
    if (item.kind === "page") {
      item.closed ? reopenPageFeedback(item.id) : closePageFeedback(item.id);
    } else if (item.kind === "widget") {
      item.closed ? reopenWidgetFeedback(item.id) : closeWidgetFeedback(item.id);
    } else {
      item.closed ? reopenDashboardFeedback(item.id) : closeDashboardFeedback(item.id);
    }
    reload();
  };

  const pending = items.filter((i) => !i.closed);
  const closed = items.filter((i) => i.closed);

  if (!items.length) return null;

  return (
    <div className="mb-6 p-4 rounded-lg bg-neutral-900/60 border border-neutral-700/50">
      <h3 className="text-xs font-bold uppercase tracking-widest text-neutral-400 mb-3">
        Raised Issues ({pending.length} pending{closed.length > 0 ? ` · ${closed.length} closed` : ""})
      </h3>

      {/* Pending items */}
      {pending.map((item) => (
        <label
          key={item.id}
          className="flex items-start gap-2 py-2 px-2 rounded mb-1 bg-neutral-800/60 hover:bg-neutral-800 transition-colors cursor-pointer"
        >
          <input
            type="checkbox"
            checked={false}
            onChange={() => handleToggle(item)}
            className="mt-0.5 accent-green-500"
          />
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              {item.kind === "page" && (
                <span className="text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-300">
                  page
                </span>
              )}
              <span className="text-xs font-mono text-neutral-200 truncate">{item.label}</span>
              <span className="text-[10px] text-neutral-500">{formatRelativeTime(item.timestamp)}</span>
            </div>
            <p className="text-xs text-neutral-400 mt-0.5 leading-relaxed line-clamp-2">{item.notes}</p>
          </div>
        </label>
      ))}

      {/* Closed items */}
      {closed.length > 0 && (
        <details className="mt-3">
          <summary className="text-[10px] font-bold uppercase tracking-widest text-neutral-500 cursor-pointer hover:text-neutral-300 transition-colors">
            Closed ({closed.length})
          </summary>
          <div className="mt-2">
            {closed.map((item) => (
              <label
                key={item.id}
                className="flex items-start gap-2 py-2 px-2 rounded mb-1 bg-neutral-800/30 transition-colors cursor-pointer"
              >
                <input
                  type="checkbox"
                  checked={true}
                  onChange={() => handleToggle(item)}
                  className="mt-0.5 accent-green-500"
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    {item.kind === "page" && (
                      <span className="text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded bg-green-500/20 text-green-400">
                        page
                      </span>
                    )}
                    <span className="text-xs font-mono text-neutral-500 line-through truncate">{item.label}</span>
                    <span className="text-[10px] text-neutral-600">{formatRelativeTime(item.timestamp)}</span>
                  </div>
                  <p className="text-xs text-neutral-600 mt-0.5 leading-relaxed line-through line-clamp-2">{item.notes}</p>
                </div>
              </label>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
