/**
 * Phase 1-3 Confidence Indicators & Widget Badges — E2E Tests
 *
 * Tests frontend rendering of Phase 1-3 metadata:
 *   - Confidence indicator bar + percentage + caveats (BlobGrid)
 *   - Stale badge (WidgetSlot)
 *   - Conflict badge (WidgetSlot)
 *   - Low-confidence badge (WidgetSlot)
 *
 * All tests use layout injection via event bus — no backend needed.
 *
 * Run: npx playwright test e2e/tests/phase123-confidence-badges.spec.ts --config playwright-audit.config.ts
 */
import { test, expect, Page } from "@playwright/test";
import * as fs from "fs";
import * as path from "path";

const BASE = process.env.FRONTEND_URL || "http://localhost:3100";
const EVIDENCE_DIR = path.join(process.cwd(), "e2e-audit-evidence", "phase123-badges");

// Ensure evidence dir exists
if (!fs.existsSync(EVIDENCE_DIR)) fs.mkdirSync(EVIDENCE_DIR, { recursive: true });

async function screenshot(page: Page, name: string): Promise<string> {
  const p = path.join(EVIDENCE_DIR, `${name}.png`);
  await page.screenshot({ path: p, fullPage: false }).catch(() => {});
  return p;
}

/** Wait for Next.js CSR hydration — page must have >10 DOM nodes */
async function waitForHydration(page: Page, timeout = 60000) {
  await page
    .waitForFunction(() => document.getElementsByTagName("*").length > 10, {
      timeout,
    })
    .catch(() => {});
}

/** Get the real event bus from globalThis (same singleton used by React). */
async function ensureEventBus(page: Page) {
  // The real bus is stored at globalThis.__commandCenterBus__ (double underscore)
  // by the EventBus module. We just verify it exists after hydration.
  const hasBus = await page.evaluate(() => {
    return !!(globalThis as any).__commandCenterBus__;
  });
  if (!hasBus) {
    // Fallback: wait a bit more for React to mount and create the bus
    await page.waitForTimeout(2000);
  }
}

/** Inject a LAYOUT_UPDATE event with custom confidence and widget metadata. */
async function injectLayout(
  page: Page,
  options: {
    confidence?: Record<string, unknown>;
    widgets?: Array<Record<string, unknown>>;
    heading?: string;
  } = {}
) {
  await ensureEventBus(page);

  const defaultWidgets = [
    {
      scenario: "kpi",
      fixture: "kpi_live-standard",
      size: "compact",
      position: null,
      relevance: 0.9,
      data_override: {
        demoData: { label: "Pump 4 Power", value: 42.5, unit: "kW" },
        _data_source: "live",
      },
    },
    {
      scenario: "trend",
      fixture: "trend-standard",
      size: "expanded",
      position: null,
      relevance: 0.85,
      data_override: {
        demoData: { label: "Power Trend" },
        _data_source: "live",
      },
    },
  ];

  const layout: Record<string, unknown> = {
    heading: options.heading || "Test Dashboard",
    widgets: options.widgets || defaultWidgets,
    transitions: {},
  };

  if (options.confidence) {
    layout._confidence = options.confidence;
  }

  await page.evaluate((layoutData) => {
    // Access the real singleton event bus used by the React app
    const bus = (globalThis as any).__commandCenterBus__;
    if (bus) {
      bus.emit({ type: "LAYOUT_UPDATE", layout: layoutData });
    }
  }, layout);

  // Wait for React to process the layout update
  await page.waitForTimeout(600);
}

/** Wait for widgets to appear. */
async function waitForWidgets(page: Page, minCount = 1, timeout = 15000) {
  await page
    .waitForFunction(
      (min) => document.querySelectorAll("[data-scenario]").length >= min,
      minCount,
      { timeout }
    )
    .catch(() => {});
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

test.describe("Phase 1-3: Confidence Indicators & Widget Badges", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`${BASE}/`);
    await page.waitForLoadState("networkidle").catch(() => {});
    await waitForHydration(page);
  });

  // ═════════════════════════════════════════════════════════════════════════════
  // Confidence Indicator (BlobGrid)
  // ═════════════════════════════════════════════════════════════════════════════

  test("C1: Confidence indicator hidden for full_dashboard action", async ({ page }) => {
    await injectLayout(page, {
      confidence: {
        intent: 1.0, retrieval: 0.95, freshness: 0.9,
        widget_fit: 1.0, data_fill: 0.95,
        overall: 0.95, action: "full_dashboard", caveats: [],
      },
    });
    await waitForWidgets(page);
    await screenshot(page, "C1-full-dashboard");

    const indicator = page.locator('[data-testid="confidence-indicator"]');
    await expect(indicator).toHaveCount(0);
  });

  test("C2: Confidence indicator visible for partial_with_caveats", async ({ page }) => {
    await injectLayout(page, {
      heading: "Partial Results",
      confidence: {
        intent: 0.8, retrieval: 0.6, freshness: 0.5,
        widget_fit: 0.7, data_fill: 0.6,
        overall: 0.62, action: "partial_with_caveats",
        caveats: ["Some data gaps detected"],
      },
    });
    await waitForWidgets(page);
    await screenshot(page, "C2-partial-caveats");

    const indicator = page.locator('[data-testid="confidence-indicator"]');
    await expect(indicator).toBeVisible();

    // Should show percentage text
    const pctText = indicator.locator("span").last();
    await expect(pctText).toContainText("62%");
  });

  test("C3: Confidence bar color — emerald for high confidence", async ({ page }) => {
    await injectLayout(page, {
      confidence: {
        intent: 0.9, retrieval: 0.8, freshness: 0.85,
        widget_fit: 0.9, data_fill: 0.8,
        overall: 0.78, action: "partial_with_caveats", caveats: [],
      },
    });
    await waitForWidgets(page);
    await screenshot(page, "C3-emerald-bar");

    const bar = page.locator('[data-testid="confidence-indicator"] .rounded-full').first();
    // Inner bar (the colored one) should have emerald class
    const innerBar = page.locator('[data-testid="confidence-indicator"] .bg-emerald-500');
    const count = await innerBar.count();
    // >= 0.75 should show emerald
    expect(count).toBeGreaterThan(0);
  });

  test("C4: Confidence bar color — yellow for medium confidence", async ({ page }) => {
    await injectLayout(page, {
      confidence: {
        intent: 0.7, retrieval: 0.5, freshness: 0.6,
        widget_fit: 0.6, data_fill: 0.5,
        overall: 0.58, action: "partial_with_caveats", caveats: ["Data gaps"],
      },
    });
    await waitForWidgets(page);
    await screenshot(page, "C4-yellow-bar");

    const yellowBar = page.locator('[data-testid="confidence-indicator"] .bg-yellow-500');
    const count = await yellowBar.count();
    expect(count).toBeGreaterThan(0);
  });

  test("C5: Confidence bar color — orange for low confidence", async ({ page }) => {
    await injectLayout(page, {
      confidence: {
        intent: 0.4, retrieval: 0.3, freshness: 0.4,
        widget_fit: 0.4, data_fill: 0.3,
        overall: 0.38, action: "reduced_layout", caveats: ["Limited data"],
      },
    });
    await waitForWidgets(page);
    await screenshot(page, "C5-orange-bar");

    const orangeBar = page.locator('[data-testid="confidence-indicator"] .bg-orange-500');
    const count = await orangeBar.count();
    expect(count).toBeGreaterThan(0);
  });

  test("C6: Caveats text displayed when present", async ({ page }) => {
    await injectLayout(page, {
      heading: "Dashboard with Caveats",
      confidence: {
        intent: 0.7, retrieval: 0.5, freshness: 0.5,
        widget_fit: 0.6, data_fill: 0.5,
        overall: 0.55, action: "partial_with_caveats",
        caveats: ["3 data gaps detected", "2 stale widgets"],
      },
    });
    await waitForWidgets(page);
    await screenshot(page, "C6-caveats");

    const caveats = page.locator('[data-testid="confidence-caveats"]');
    await expect(caveats).toBeVisible();
    await expect(caveats).toContainText("data gaps");
  });

  test("C7: Caveats hidden when empty array", async ({ page }) => {
    await injectLayout(page, {
      confidence: {
        intent: 0.7, retrieval: 0.6, freshness: 0.6,
        widget_fit: 0.7, data_fill: 0.6,
        overall: 0.62, action: "partial_with_caveats", caveats: [],
      },
    });
    await waitForWidgets(page);
    await screenshot(page, "C7-no-caveats");

    const caveats = page.locator('[data-testid="confidence-caveats"]');
    await expect(caveats).toHaveCount(0);
  });

  // ═════════════════════════════════════════════════════════════════════════════
  // Stale Badge (WidgetSlot)
  // ═════════════════════════════════════════════════════════════════════════════

  test("S1: Stale badge visible when _is_stale is true", async ({ page }) => {
    await injectLayout(page, {
      widgets: [
        {
          scenario: "kpi",
          fixture: "kpi_live-standard",
          size: "compact",
          position: null,
          relevance: 0.9,
          _is_stale: true,
          _staleness_seconds: 7200,
          data_override: {
            demoData: { label: "Stale Pump Data", value: 40.0, unit: "kW" },
            _data_source: "live",
          },
        },
      ],
    });
    await waitForWidgets(page);
    await screenshot(page, "S1-stale-badge");

    const staleBadge = page.locator('[data-testid="stale-badge"]');
    await expect(staleBadge).toBeVisible();
    await expect(staleBadge).toContainText("Stale");
  });

  test("S2: Stale badge hidden when _is_stale is false", async ({ page }) => {
    await injectLayout(page, {
      widgets: [
        {
          scenario: "kpi",
          fixture: "kpi_live-standard",
          size: "compact",
          position: null,
          relevance: 0.9,
          _is_stale: false,
          data_override: {
            demoData: { label: "Fresh Data", value: 42.0, unit: "kW" },
            _data_source: "live",
          },
        },
      ],
    });
    await waitForWidgets(page);
    await screenshot(page, "S2-no-stale");

    const staleBadge = page.locator('[data-testid="stale-badge"]');
    await expect(staleBadge).toHaveCount(0);
  });

  test("S3: Stale badge hidden when _is_stale is undefined", async ({ page }) => {
    await injectLayout(page); // default widgets — no _is_stale
    await waitForWidgets(page);

    const staleBadge = page.locator('[data-testid="stale-badge"]');
    await expect(staleBadge).toHaveCount(0);
  });

  // ═════════════════════════════════════════════════════════════════════════════
  // Conflict Badge (WidgetSlot)
  // ═════════════════════════════════════════════════════════════════════════════

  test("F1: Conflict badge visible when _conflict_flag is set", async ({ page }) => {
    await injectLayout(page, {
      widgets: [
        {
          scenario: "kpi",
          fixture: "kpi_live-standard",
          size: "compact",
          position: null,
          relevance: 0.9,
          _conflict_flag: "KPI shows normal but alert is critical",
          data_override: {
            demoData: { label: "Conflict Widget", value: 42.0, unit: "kW" },
            _data_source: "live",
          },
        },
      ],
    });
    await waitForWidgets(page);
    await screenshot(page, "F1-conflict-badge");

    const conflictBadge = page.locator('[data-testid="conflict-badge"]');
    await expect(conflictBadge).toBeVisible();
    await expect(conflictBadge).toContainText("Conflict");

    // Title should contain the conflict message
    const title = await conflictBadge.getAttribute("title");
    expect(title).toContain("KPI shows normal");
  });

  test("F2: Conflict badge hidden when _conflict_flag is absent", async ({ page }) => {
    await injectLayout(page); // default widgets — no _conflict_flag
    await waitForWidgets(page);

    const conflictBadge = page.locator('[data-testid="conflict-badge"]');
    await expect(conflictBadge).toHaveCount(0);
  });

  // ═════════════════════════════════════════════════════════════════════════════
  // Low-Confidence Badge (WidgetSlot)
  // ═════════════════════════════════════════════════════════════════════════════

  test("L1: Low-confidence badge visible when _widget_confidence < 0.5", async ({ page }) => {
    await injectLayout(page, {
      widgets: [
        {
          scenario: "kpi",
          fixture: "kpi_live-standard",
          size: "compact",
          position: null,
          relevance: 0.9,
          _widget_confidence: 0.3,
          data_override: {
            demoData: { label: "Low Conf Widget", value: 10.0, unit: "kW" },
            _data_source: "live",
          },
        },
      ],
    });
    await waitForWidgets(page);
    // Re-inject to ensure React processes the layout (handles race with default layout)
    await injectLayout(page, {
      widgets: [
        {
          scenario: "kpi",
          fixture: "kpi_live-standard",
          size: "compact",
          position: null,
          relevance: 0.9,
          _widget_confidence: 0.3,
          data_override: {
            demoData: { label: "Low Conf Widget", value: 10.0, unit: "kW" },
            _data_source: "live",
          },
        },
      ],
    });
    await waitForWidgets(page);
    await screenshot(page, "L1-low-confidence");

    const lowConfBadge = page.locator('[data-testid="low-confidence-badge"]');
    await expect(lowConfBadge).toBeVisible();
    await expect(lowConfBadge).toContainText("30% conf");
  });

  test("L2: Low-confidence badge hidden when _widget_confidence >= 0.5", async ({ page }) => {
    await injectLayout(page, {
      widgets: [
        {
          scenario: "kpi",
          fixture: "kpi_live-standard",
          size: "compact",
          position: null,
          relevance: 0.9,
          _widget_confidence: 0.85,
          data_override: {
            demoData: { label: "High Conf Widget", value: 42.0, unit: "kW" },
            _data_source: "live",
          },
        },
      ],
    });
    await waitForWidgets(page);
    await screenshot(page, "L2-high-confidence");

    const lowConfBadge = page.locator('[data-testid="low-confidence-badge"]');
    await expect(lowConfBadge).toHaveCount(0);
  });

  test("L3: Low-confidence badge hidden when _widget_confidence is undefined", async ({ page }) => {
    await injectLayout(page); // default widgets — no _widget_confidence
    await waitForWidgets(page);

    const lowConfBadge = page.locator('[data-testid="low-confidence-badge"]');
    await expect(lowConfBadge).toHaveCount(0);
  });

  // ═════════════════════════════════════════════════════════════════════════════
  // Multiple Badges Stacking
  // ═════════════════════════════════════════════════════════════════════════════

  test("M1: Multiple badges stack vertically in same widget", async ({ page }) => {
    await injectLayout(page, {
      widgets: [
        {
          scenario: "kpi",
          fixture: "kpi_live-standard",
          size: "normal",
          position: null,
          relevance: 0.9,
          _is_stale: true,
          _conflict_flag: "Data mismatch",
          _widget_confidence: 0.2,
          data_override: {
            demoData: { label: "Multi-Badge Widget", value: 99.0, unit: "kW" },
            _data_source: "demo",
          },
        },
      ],
    });
    await waitForWidgets(page);
    await screenshot(page, "M1-multi-badges");

    // All four badges should be visible
    const demoBadge = page.locator('[data-testid="demo-badge"]');
    const staleBadge = page.locator('[data-testid="stale-badge"]');
    const conflictBadge = page.locator('[data-testid="conflict-badge"]');
    const lowConfBadge = page.locator('[data-testid="low-confidence-badge"]');

    await expect(demoBadge).toBeVisible();
    await expect(staleBadge).toBeVisible();
    await expect(conflictBadge).toBeVisible();
    await expect(lowConfBadge).toBeVisible();
    await expect(lowConfBadge).toContainText("20% conf");
  });
});
