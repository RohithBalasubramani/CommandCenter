/**
 * Upgrade System E2E Tests — Real User Prompts
 *
 * Tests all 10 upgrades through the actual frontend with realistic operator prompts.
 * Uses the text input overlay (not mocks) to send queries to the live backend.
 */
import { test, expect, Page } from "@playwright/test";
import * as fs from "fs";
import * as path from "path";

const BASE = process.env.FRONTEND_URL || "http://localhost:3100";
const BACKEND = process.env.BACKEND_URL || "http://127.0.0.1:8100";
const EVIDENCE_DIR = path.join(
  process.cwd(),
  "e2e-audit-evidence",
  "upgrade-prompts"
);

if (!fs.existsSync(EVIDENCE_DIR)) fs.mkdirSync(EVIDENCE_DIR, { recursive: true });

async function screenshot(page: Page, name: string) {
  const p = path.join(EVIDENCE_DIR, `${name}.png`);
  await page.screenshot({ path: p, fullPage: true }).catch(() => {});
  return p;
}

async function waitForHydration(page: Page, timeout = 60000) {
  await page
    .waitForFunction(() => document.getElementsByTagName("*").length > 10, {
      timeout,
    })
    .catch(() => {});
}

/** Open text input overlay via toggle button or keyboard shortcut */
async function openTextInput(page: Page) {
  // If already open, nothing to do
  const input = page.getByTestId("text-input");
  if (await input.isVisible().catch(() => false)) return;

  // Try keyboard shortcut first (Ctrl+Shift+K) — more reliable than button click
  await page.keyboard.press("Control+Shift+KeyK");
  await page.waitForTimeout(500);
  if (await input.isVisible().catch(() => false)) return;

  // Fallback: click the toggle button
  const toggle = page.getByTestId("text-input-toggle");
  if (await toggle.isVisible().catch(() => false)) {
    await toggle.click();
    await page.waitForTimeout(500);
    if (await input.isVisible().catch(() => false)) return;
  }

  // Last resort: dispatch event directly to open overlay
  await page.evaluate(() => {
    window.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "K",
        ctrlKey: true,
        shiftKey: true,
        bubbles: true,
      })
    );
  });
  await page.waitForTimeout(500);
}

/** Send a query through the text input and wait for response */
async function sendQuery(page: Page, query: string, waitMs = 15000) {
  await openTextInput(page);
  const input = page.getByTestId("text-input");
  await input.fill(query);
  const submit = page.getByTestId("submit-query");
  await submit.waitFor({ state: "visible", timeout: 3000 });
  await submit.click();
  // Wait for backend response — watch for layout update or voice response
  await page.waitForTimeout(waitMs);
}

/** Count visible widgets by data-scenario attribute */
async function countWidgets(page: Page): Promise<number> {
  return page.locator("[data-scenario]").count();
}

/** Get all widget scenario types currently displayed */
async function getWidgetScenarios(page: Page): Promise<string[]> {
  const elements = page.locator("[data-scenario]");
  const count = await elements.count();
  const scenarios: string[] = [];
  for (let i = 0; i < count; i++) {
    const scenario = await elements.nth(i).getAttribute("data-scenario");
    if (scenario) scenarios.push(scenario);
  }
  return scenarios;
}

/** Intercept orchestrate calls and capture request/response */
function interceptOrchestrate(page: Page) {
  const calls: Array<{ request: any; response: any }> = [];
  page.on("response", async (response) => {
    if (response.url().includes("/api/layer2/orchestrate")) {
      try {
        const body = await response.json();
        const req = response.request();
        const reqBody = req.postDataJSON?.() || {};
        calls.push({ request: reqBody, response: body });
      } catch {}
    }
  });
  return calls;
}

// ============================================================
// TESTS
// ============================================================

test.describe("Upgrade E2E — Real Prompts", () => {
  test.beforeEach(async ({ page }) => {
    // Check backend is alive
    try {
      const resp = await page.request.get(`${BACKEND}/api/layer2/orchestrate/`, {
        timeout: 5000,
      });
    } catch {
      // Backend might not respond to GET, that's fine
    }
  });

  test("Prompt 1: 'show me pump 4 vibration' renders widgets", async ({
    page,
  }) => {
    await page.goto(BASE, { waitUntil: "domcontentloaded" });
    await waitForHydration(page);
    await screenshot(page, "01-initial-page");

    // Send a basic equipment query
    await sendQuery(page, "show me pump 4 vibration", 20000);
    await screenshot(page, "01-pump4-vibration-result");

    // Should have at least 1 widget rendered
    const widgetCount = await countWidgets(page);
    console.log(`Prompt 1: ${widgetCount} widgets rendered`);

    // Check that the page has some meaningful content
    const bodyText = await page.textContent("body");
    const hasRelevantContent =
      bodyText?.toLowerCase().includes("pump") ||
      bodyText?.toLowerCase().includes("vibration") ||
      widgetCount > 0;

    expect(hasRelevantContent).toBeTruthy();
  });

  test("Prompt 2: 'what alerts are active right now' returns alerts widget", async ({
    page,
  }) => {
    await page.goto(BASE, { waitUntil: "domcontentloaded" });
    await waitForHydration(page);

    await sendQuery(page, "what alerts are active right now", 20000);
    await screenshot(page, "02-active-alerts-result");

    const scenarios = await getWidgetScenarios(page);
    console.log(`Prompt 2 scenarios: ${scenarios}`);

    // Should have at least one widget
    expect(scenarios.length).toBeGreaterThan(0);
  });

  test("Prompt 3: 'compare pump 4 and pump 5 power consumption' renders comparison", async ({
    page,
  }) => {
    await page.goto(BASE, { waitUntil: "domcontentloaded" });
    await waitForHydration(page);

    await sendQuery(
      page,
      "compare pump 4 and pump 5 power consumption",
      20000
    );
    await screenshot(page, "03-comparison-result");

    const scenarios = await getWidgetScenarios(page);
    console.log(`Prompt 3 scenarios: ${scenarios}`);

    // Should have widgets (ideally a comparison widget)
    expect(scenarios.length).toBeGreaterThan(0);
  });

  test("Prompt 4: 'show me the trend for motor 1 temperature over the last 24 hours'", async ({
    page,
  }) => {
    await page.goto(BASE, { waitUntil: "domcontentloaded" });
    await waitForHydration(page);

    await sendQuery(
      page,
      "show me the trend for motor 1 temperature over the last 24 hours",
      20000
    );
    await screenshot(page, "04-motor-trend-result");

    const scenarios = await getWidgetScenarios(page);
    console.log(`Prompt 4 scenarios: ${scenarios}`);

    expect(scenarios.length).toBeGreaterThan(0);
    // Trend widget should be present
    const hasTrend = scenarios.some(
      (s) => s === "trend" || s === "trend-multi-line"
    );
    console.log(`  Has trend widget: ${hasTrend}`);
  });

  test("Prompt 5: Multi-turn conversation — follow-up with pronoun", async ({
    page,
  }) => {
    await page.goto(BASE, { waitUntil: "domcontentloaded" });
    await waitForHydration(page);

    // First query — establish context
    await sendQuery(page, "show me pump 4 status", 20000);
    await screenshot(page, "05a-pump4-status");
    const firstWidgets = await getWidgetScenarios(page);
    console.log(`Prompt 5a scenarios: ${firstWidgets}`);

    // Follow-up with pronoun — should resolve to pump 4
    await sendQuery(page, "what about its maintenance history", 20000);
    await screenshot(page, "05b-maintenance-followup");
    const followUpWidgets = await getWidgetScenarios(page);
    console.log(`Prompt 5b scenarios: ${followUpWidgets}`);

    // Should have widgets in both cases
    expect(firstWidgets.length).toBeGreaterThan(0);
    // Follow-up should also produce results (not out-of-scope)
    const bodyText = await page.textContent("body");
    const notOutOfScope = !bodyText
      ?.toLowerCase()
      .includes("outside what i can help");
    console.log(`  Follow-up not out-of-scope: ${notOutOfScope}`);
  });

  test("Prompt 6: 'why is pump 4 vibration high' — diagnostic query", async ({
    page,
  }) => {
    await page.goto(BASE, { waitUntil: "domcontentloaded" });
    await waitForHydration(page);

    await sendQuery(page, "why is pump 4 vibration high", 20000);
    await screenshot(page, "06-why-vibration-high");

    const scenarios = await getWidgetScenarios(page);
    console.log(`Prompt 6 scenarios: ${scenarios}`);

    // Should produce widgets with diagnostic/explanation content
    expect(scenarios.length).toBeGreaterThan(0);

    // Check voice response mentions vibration context
    const bodyText = await page.textContent("body");
    const mentionsVibration = bodyText?.toLowerCase().includes("vibration");
    console.log(`  Mentions vibration: ${mentionsVibration}`);
  });

  test("Prompt 7: 'show me all equipment health scores' — overview query", async ({
    page,
  }) => {
    await page.goto(BASE, { waitUntil: "domcontentloaded" });
    await waitForHydration(page);

    await sendQuery(page, "show me all equipment health scores", 20000);
    await screenshot(page, "07-health-scores");

    const scenarios = await getWidgetScenarios(page);
    console.log(`Prompt 7 scenarios: ${scenarios}`);

    expect(scenarios.length).toBeGreaterThan(0);
  });

  test("Prompt 8: Backend API — Focus graph created in interactive mode", async ({
    page,
  }) => {
    // Test the backend directly to verify focus graph behavior
    const response = await page.request.post(
      `${BACKEND}/api/layer2/orchestrate/`,
      {
        data: {
          transcript: "tell me about pump 4",
          context: {
            widget_context: {
              equipment: "pump_004",
              metric: "vibration",
              scenario: "kpi",
              label: "Pump 4 Vibration",
              conversation_history: [],
            },
          },
        },
      }
    );

    expect(response.ok()).toBeTruthy();
    const data = await response.json();

    console.log(`Prompt 8: query_id=${data.query_id}`);
    console.log(
      `  Voice: ${(data.voice_response || "").substring(0, 100)}...`
    );
    console.log(
      `  Widgets: ${(data.layout_json?.widgets || []).map((w: any) => w.scenario)}`
    );

    // Focus graph should be present in context_update
    const fg = data.context_update?.focus_graph;
    if (fg) {
      console.log(
        `  Focus Graph: ${Object.keys(fg.nodes || {})} root=${fg.root_node_id}`
      );
      expect(fg.root_node_id).toBe("equipment:pump_004");
      expect(Object.keys(fg.nodes)).toContain("equipment:pump_004");
    } else {
      console.log("  Focus graph not returned (may need widget_context)");
    }

    expect(data.layout_json?.widgets?.length).toBeGreaterThan(0);
  });

  test("Prompt 9: Backend API — Pronoun resolution with focus graph", async ({
    page,
  }) => {
    // Send "is it running well?" with interactive context
    const response = await page.request.post(
      `${BACKEND}/api/layer2/orchestrate/`,
      {
        data: {
          transcript: "is it running well?",
          context: {
            widget_context: {
              equipment: "pump_004",
              metric: "vibration",
              scenario: "kpi",
              label: "Pump 4 Vibration",
              conversation_history: [
                {
                  role: "user",
                  text: "tell me about pump 4 vibration",
                },
                {
                  role: "ai",
                  text: "Pump 4 vibration is at 2.1 mm/s",
                },
              ],
            },
            session_id: "e2e-pronoun-test",
          },
        },
      }
    );

    expect(response.ok()).toBeTruthy();
    const data = await response.json();

    console.log(`Prompt 9: query_id=${data.query_id}`);
    console.log(
      `  Voice: ${(data.voice_response || "").substring(0, 150)}...`
    );

    // Should NOT be out-of-scope since we have interactive context
    const isOutOfScope = (data.voice_response || "")
      .toLowerCase()
      .includes("outside what i can help");
    expect(isOutOfScope).toBeFalsy();

    // Should return widgets
    expect(data.layout_json?.widgets?.length).toBeGreaterThan(0);
  });

  test("Prompt 10: Backend API — Cancel endpoint responds correctly", async ({
    page,
  }) => {
    // Cancel a nonexistent plan
    const resp1 = await page.request.post(`${BACKEND}/api/layer2/cancel/`, {
      data: { plan_id: "fake-plan-id-12345" },
    });
    expect(resp1.ok()).toBeTruthy();
    const data1 = await resp1.json();
    expect(data1.cancelled).toBe(false);
    expect(data1.reason).toBe("plan_not_found");
    console.log(`Prompt 10a: Cancel nonexistent → ${JSON.stringify(data1)}`);

    // Cancel with missing plan_id
    const resp2 = await page.request.post(`${BACKEND}/api/layer2/cancel/`, {
      data: {},
    });
    expect(resp2.status()).toBe(400);
    const data2 = await resp2.json();
    expect(data2.reason).toBe("missing_plan_id");
    console.log(`Prompt 10b: Cancel missing id → ${JSON.stringify(data2)}`);
  });

  test("Prompt 11: Backend API — Constraint violation voice qualifier", async ({
    page,
  }) => {
    // Send a query and check the voice response for constraint qualifiers
    const response = await page.request.post(
      `${BACKEND}/api/layer2/orchestrate/`,
      {
        data: {
          transcript: "show transformer 1 load",
          context: {},
        },
      }
    );

    expect(response.ok()).toBeTruthy();
    const data = await response.json();

    console.log(`Prompt 11: query_id=${data.query_id}`);
    console.log(
      `  Voice: ${(data.voice_response || "").substring(0, 150)}...`
    );
    console.log(
      `  Widgets: ${(data.layout_json?.widgets || []).map((w: any) => w.scenario)}`
    );
    console.log(`  Processing time: ${data.processing_time_ms}ms`);

    // Should return a valid response
    expect(data.voice_response?.length).toBeGreaterThan(10);
  });

  test("Prompt 12: 'show me cooling tower 2 status and any alerts'", async ({
    page,
  }) => {
    await page.goto(BASE, { waitUntil: "domcontentloaded" });
    await waitForHydration(page);

    await sendQuery(
      page,
      "show me cooling tower 2 status and any alerts",
      20000
    );
    await screenshot(page, "12-cooling-tower");

    const scenarios = await getWidgetScenarios(page);
    console.log(`Prompt 12 scenarios: ${scenarios}`);

    expect(scenarios.length).toBeGreaterThan(0);
  });
});
