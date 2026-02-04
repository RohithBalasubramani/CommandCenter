/**
 * feedbackStore — localStorage persistence for widget feedback.
 *
 * Three collections:
 *  1. Widget feedback: per scenario+variant (rating, tags, notes)
 *  2. Dashboard feedback: per sample dashboard (rating, notes)
 *  3. Page feedback: per page/section (notes)
 *
 * Each feedback item has a unique `id` and a `closed` flag for issue tracking.
 * Saving new feedback always appends — it never replaces existing items.
 * Closed items stay in their list and can be reopened.
 */

export interface WidgetFeedback {
  id: string;
  scenario: string;
  variant: string;
  rating: number;       // 1–5
  tags: string[];       // e.g. ["looks-good", "needs-redesign", "wrong-data"]
  notes: string;
  timestamp: number;
  closed?: boolean;
}

export interface DashboardFeedback {
  id: string;
  dashboardId: string;
  rating: number;
  notes: string;
  timestamp: number;
  closed?: boolean;
}

export interface PageFeedback {
  id: string;
  pageId: string;        // e.g. "scenario:kpi", "dashboards"
  notes: string;
  timestamp: number;
  closed?: boolean;
}

const WIDGET_KEY = "cc-widget-feedback";
const DASHBOARD_KEY = "cc-dashboard-feedback";
const PAGE_KEY = "cc-page-feedback";

function uid(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

// ── Widget feedback ──

export function getAllWidgetFeedback(): WidgetFeedback[] {
  if (typeof window === "undefined") return [];
  try {
    const raw: unknown[] = JSON.parse(localStorage.getItem(WIDGET_KEY) || "[]");
    // Migrate old entries that lack an id
    return raw.map((f: any) => ({ ...f, id: f.id || `${f.timestamp}-${f.variant}` }));
  } catch {
    return [];
  }
}

export function getFeedbackFor(scenario: string, variant: string): WidgetFeedback | null {
  return getAllWidgetFeedback().find(
    (f) => f.scenario === scenario && f.variant === variant
  ) ?? null;
}

export function getWidgetFeedbackForScenario(scenario: string): WidgetFeedback[] {
  return getAllWidgetFeedback().filter((f) => f.scenario === scenario);
}

export function saveWidgetFeedback(fb: Omit<WidgetFeedback, "id">): void {
  const all = getAllWidgetFeedback();
  all.push({ ...fb, id: uid() });
  localStorage.setItem(WIDGET_KEY, JSON.stringify(all));
  autoExportToDisk();
}

export function closeWidgetFeedback(id: string): void {
  const all = getAllWidgetFeedback().map((f) =>
    f.id === id ? { ...f, closed: true } : f
  );
  localStorage.setItem(WIDGET_KEY, JSON.stringify(all));
  autoExportToDisk();
}

export function reopenWidgetFeedback(id: string): void {
  const all = getAllWidgetFeedback().map((f) =>
    f.id === id ? { ...f, closed: false } : f
  );
  localStorage.setItem(WIDGET_KEY, JSON.stringify(all));
  autoExportToDisk();
}

// ── Dashboard feedback ──

export function getAllDashboardFeedback(): DashboardFeedback[] {
  if (typeof window === "undefined") return [];
  try {
    const raw: unknown[] = JSON.parse(localStorage.getItem(DASHBOARD_KEY) || "[]");
    return raw.map((f: any) => ({ ...f, id: f.id || `${f.timestamp}-${f.dashboardId}` }));
  } catch {
    return [];
  }
}

export function saveDashboardFeedback(fb: Omit<DashboardFeedback, "id">): void {
  const all = getAllDashboardFeedback();
  all.push({ ...fb, id: uid() });
  localStorage.setItem(DASHBOARD_KEY, JSON.stringify(all));
  autoExportToDisk();
}

export function closeDashboardFeedback(id: string): void {
  const all = getAllDashboardFeedback().map((f) =>
    f.id === id ? { ...f, closed: true } : f
  );
  localStorage.setItem(DASHBOARD_KEY, JSON.stringify(all));
  autoExportToDisk();
}

export function reopenDashboardFeedback(id: string): void {
  const all = getAllDashboardFeedback().map((f) =>
    f.id === id ? { ...f, closed: false } : f
  );
  localStorage.setItem(DASHBOARD_KEY, JSON.stringify(all));
  autoExportToDisk();
}

// ── Page feedback ──

export function getAllPageFeedback(): PageFeedback[] {
  if (typeof window === "undefined") return [];
  try {
    const raw: unknown[] = JSON.parse(localStorage.getItem(PAGE_KEY) || "[]");
    return raw.map((f: any) => ({ ...f, id: f.id || `${f.timestamp}-${f.pageId}` }));
  } catch {
    return [];
  }
}

export function getPageFeedbackForPage(pageId: string): PageFeedback[] {
  return getAllPageFeedback().filter((f) => f.pageId === pageId);
}

export function savePageFeedback(fb: Omit<PageFeedback, "id">): void {
  const all = getAllPageFeedback();
  all.push({ ...fb, id: uid() });
  localStorage.setItem(PAGE_KEY, JSON.stringify(all));
  autoExportToDisk();
}

export function closePageFeedback(id: string): void {
  const all = getAllPageFeedback().map((f) =>
    f.id === id ? { ...f, closed: true } : f
  );
  localStorage.setItem(PAGE_KEY, JSON.stringify(all));
  autoExportToDisk();
}

export function reopenPageFeedback(id: string): void {
  const all = getAllPageFeedback().map((f) =>
    f.id === id ? { ...f, closed: false } : f
  );
  localStorage.setItem(PAGE_KEY, JSON.stringify(all));
  autoExportToDisk();
}

// ── Size adjustments (live tweaks stored separately) ──

export interface SizeAdjustment {
  w: string; // e.g. "35vw"
  h: string; // e.g. "40vh"
}

export interface VariantSizeAdjustments {
  [sizeKey: string]: SizeAdjustment; // sizeKey = "compact" | "normal" | "expanded" | "hero"
}

const SIZE_ADJ_KEY = "cc-size-adjustments";

/** Get all size adjustments. Key format: "scenario/variant" */
export function getAllSizeAdjustments(): Record<string, VariantSizeAdjustments> {
  if (typeof window === "undefined") return {};
  try {
    return JSON.parse(localStorage.getItem(SIZE_ADJ_KEY) || "{}");
  } catch {
    return {};
  }
}

/** Get size adjustments for a specific scenario/variant. */
export function getSizeAdjustment(scenario: string, variant: string): VariantSizeAdjustments | null {
  const all = getAllSizeAdjustments();
  return all[`${scenario}/${variant}`] ?? null;
}

/** Save size adjustment for a specific scenario/variant/size. */
export function saveSizeAdjustment(
  scenario: string,
  variant: string,
  sizeKey: string,
  adjustment: SizeAdjustment
): void {
  const all = getAllSizeAdjustments();
  const key = `${scenario}/${variant}`;
  if (!all[key]) all[key] = {};
  all[key][sizeKey] = adjustment;
  localStorage.setItem(SIZE_ADJ_KEY, JSON.stringify(all));
}

/** Clear size adjustment for a specific scenario/variant. */
export function clearSizeAdjustment(scenario: string, variant: string): void {
  const all = getAllSizeAdjustments();
  delete all[`${scenario}/${variant}`];
  localStorage.setItem(SIZE_ADJ_KEY, JSON.stringify(all));
}

/** Raise a feedback issue from size adjustments — saves as widget feedback with structured notes. */
export function raiseIssueSizeAdjustment(
  scenario: string,
  variant: string,
  adjustments: VariantSizeAdjustments,
  defaults: VariantSizeAdjustments
): void {
  const lines: string[] = ["[SIZE ADJUSTMENT]"];
  for (const sizeKey of Object.keys(adjustments)) {
    const adj = adjustments[sizeKey];
    const def = defaults[sizeKey];
    if (def) {
      lines.push(`${sizeKey}: ${def.w}×${def.h} → ${adj.w}×${adj.h}`);
    } else {
      lines.push(`${sizeKey}: (new) ${adj.w}×${adj.h}`);
    }
  }
  saveWidgetFeedback({
    scenario,
    variant,
    rating: 0,
    tags: ["size-adjustment"],
    notes: lines.join("\n"),
    timestamp: Date.now(),
  });
}

// ── Clear all feedback + adjustments ──

/** Clear all widget feedback, dashboard feedback, page feedback, and size adjustments from localStorage. */
export function clearAllFeedback(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(WIDGET_KEY);
  localStorage.removeItem(DASHBOARD_KEY);
  localStorage.removeItem(PAGE_KEY);
  localStorage.removeItem(SIZE_ADJ_KEY);
  autoExportToDisk();
}

// ── Export ──

export function exportAllFeedback(): string {
  return JSON.stringify({
    widgets: getAllWidgetFeedback(),
    dashboards: getAllDashboardFeedback(),
    pages: getAllPageFeedback(),
    exportedAt: new Date().toISOString(),
  }, null, 2);
}

/** Auto-save: write the full feedback JSON to disk via the /api/widget-feedback endpoint. */
function autoExportToDisk(): void {
  const json = exportAllFeedback();
  fetch("/api/widget-feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: json,
  }).catch(() => {
    // Silent fail — dev-only convenience
  });
}

// ── Feedback tags ──

export const FEEDBACK_TAGS = [
  { id: "looks-good", label: "Looks Good" },
  { id: "needs-redesign", label: "Needs Redesign" },
  { id: "wrong-data", label: "Wrong Data Mapping" },
  { id: "too-tall", label: "Too Tall" },
  { id: "too-short", label: "Too Short" },
  { id: "title-missing", label: "Title Missing" },
  { id: "colors-wrong", label: "Colors Wrong" },
  { id: "layout-broken", label: "Layout Broken" },
  { id: "font-too-small", label: "Font Too Small" },
  { id: "keep-as-is", label: "Keep As-Is" },
  { id: "size-adjustment", label: "Size Adjustment" },
] as const;
