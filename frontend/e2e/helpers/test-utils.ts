/**
 * Command Center E2E Test Utilities
 *
 * Provides helpers for multi-modal testing, widget validation,
 * performance measurement, and forensic capture.
 */
import { Page, expect, Locator } from '@playwright/test';

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export interface WidgetInfo {
  scenario: string;
  fixture: string;
  size: string;
  element: Locator;
}

export interface PerformanceMetrics {
  layoutRenderTime: number;
  widgetCount: number;
  totalDOMNodes: number;
  memoryUsage?: number;
  fps?: number;
}

export interface TestResult {
  testName: string;
  passed: boolean;
  duration: number;
  widgetsRendered: number;
  errors: string[];
  screenshots: string[];
  metrics: PerformanceMetrics;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PAGE INTERACTIONS
// ═══════════════════════════════════════════════════════════════════════════════

export class CommandCenterPage {
  constructor(public page: Page) {}

  // ── Navigation ──

  async goto() {
    await this.page.goto('/');
    await this.page.waitForLoadState('networkidle');
  }

  async waitForReady() {
    // Wait for main UI to be ready
    await this.page.waitForSelector('[data-testid="command-center"]', { timeout: 30000 }).catch(() => {
      // Fallback: wait for any main container
      return this.page.waitForSelector('main', { timeout: 30000 });
    });
  }

  // ── Text Input ──

  async openTextInput() {
    // Check if text input is already visible
    const textInput = this.page.locator('[data-testid="text-input"]');
    if (await textInput.isVisible().catch(() => false)) {
      return; // Already open
    }

    // Click the text input toggle button
    const toggleButton = this.page.locator('[data-testid="text-input-toggle"]');
    if (await toggleButton.isVisible().catch(() => false)) {
      await toggleButton.click();
      // Wait for the overlay to appear
      await this.page.waitForSelector('[data-testid="text-input"]', { timeout: 5000 });
      return;
    }

    // Fallback: try keyboard shortcut
    await this.page.keyboard.press('Control+Shift+K');
    await this.page.waitForSelector('[data-testid="text-input"]', { timeout: 5000 });
  }

  async typeQuery(query: string) {
    // First, ensure text input overlay is open
    await this.openTextInput();

    // Look for text input
    const inputSelectors = [
      '[data-testid="text-input"]',
      '[data-testid="query-input"]',
      'input[type="text"]',
      'textarea',
    ];

    for (const selector of inputSelectors) {
      const input = this.page.locator(selector).first();
      if (await input.isVisible().catch(() => false)) {
        await input.fill(query);
        return;
      }
    }

    // If no text input, try voice interface text mode
    const textOverlay = this.page.locator('[data-testid="text-overlay"]');
    if (await textOverlay.isVisible().catch(() => false)) {
      await textOverlay.fill(query);
      return;
    }

    throw new Error('No text input found on page');
  }

  async submitQuery() {
    // Try various submit methods
    const submitSelectors = [
      '[data-testid="submit-query"]',
      'button[type="submit"]',
      '[data-testid="send-button"]',
    ];

    for (const selector of submitSelectors) {
      const button = this.page.locator(selector).first();
      if (await button.isVisible().catch(() => false)) {
        await button.click();
        return;
      }
    }

    // Fallback: press Enter
    await this.page.keyboard.press('Enter');
  }

  async sendQuery(query: string) {
    await this.typeQuery(query);
    await this.submitQuery();
  }

  // ── Voice Input (Simulated) ──

  async simulateVoiceInput(transcriptText: string) {
    // Inject transcript via event bus or API
    await this.page.evaluate((text) => {
      // Try to dispatch transcript event
      const event = new CustomEvent('TRANSCRIPT_UPDATE', {
        detail: { transcript: text, isFinal: true }
      });
      window.dispatchEvent(event);

      // Also try event bus if available
      if ((window as any).commandCenterBus) {
        (window as any).commandCenterBus.emit('TRANSCRIPT_UPDATE', {
          transcript: text,
          isFinal: true
        });
      }
    }, transcriptText);
  }

  // ── Layout and Widgets ──

  async waitForLayout(timeout = 30000) {
    // Wait for layout update
    await this.page.waitForSelector('[data-testid="blob-grid"]', { timeout }).catch(() => {
      // Fallback: wait for any grid container
      return this.page.waitForSelector('.grid', { timeout });
    });
  }

  async getWidgets(): Promise<WidgetInfo[]> {
    const widgets: WidgetInfo[] = [];

    // Find all widget slots
    const widgetSlots = await this.page.locator('[data-testid^="widget-"]').all();

    for (const slot of widgetSlots) {
      const scenario = await slot.getAttribute('data-scenario') || 'unknown';
      const fixture = await slot.getAttribute('data-fixture') || 'unknown';
      const size = await slot.getAttribute('data-size') || 'normal';

      widgets.push({ scenario, fixture, size, element: slot });
    }

    // Fallback: look for common widget patterns
    if (widgets.length === 0) {
      const containers = await this.page.locator('.widget-container, [class*="widget"]').all();
      for (const container of containers) {
        widgets.push({
          scenario: 'unknown',
          fixture: 'unknown',
          size: 'normal',
          element: container
        });
      }
    }

    return widgets;
  }

  async getWidgetCount(): Promise<number> {
    const widgets = await this.getWidgets();
    return widgets.length;
  }

  async clickWidget(index: number) {
    const widgets = await this.getWidgets();
    if (index < widgets.length) {
      await widgets[index].element.click();
    }
  }

  // ── Voice Response ──

  async getVoiceResponse(): Promise<string | null> {
    // Check transcript panel or response display
    const responseSelectors = [
      '[data-testid="voice-response"]',
      '[data-testid="ai-response"]',
      '.response-text',
    ];

    for (const selector of responseSelectors) {
      const element = this.page.locator(selector).last();
      if (await element.isVisible().catch(() => false)) {
        return await element.textContent();
      }
    }

    return null;
  }

  // ── Performance Metrics ──

  async getPerformanceMetrics(): Promise<PerformanceMetrics> {
    const metrics = await this.page.evaluate(() => {
      const widgets = document.querySelectorAll('[data-testid^="widget-"], .widget-container');

      // Get memory if available
      let memoryUsage: number | undefined;
      if ((performance as any).memory) {
        memoryUsage = (performance as any).memory.usedJSHeapSize / 1024 / 1024;
      }

      return {
        layoutRenderTime: performance.now(),
        widgetCount: widgets.length,
        totalDOMNodes: document.getElementsByTagName('*').length,
        memoryUsage,
      };
    });

    return metrics;
  }

  // ── Console and Network Errors ──

  async captureErrors(): Promise<string[]> {
    const errors: string[] = [];

    // Collect console errors
    this.page.on('console', (msg) => {
      if (msg.type() === 'error') {
        errors.push(`Console: ${msg.text()}`);
      }
    });

    // Collect page errors
    this.page.on('pageerror', (err) => {
      errors.push(`Page Error: ${err.message}`);
    });

    return errors;
  }

  // ── Layout JSON Validation ──

  async validateLayoutJSON(): Promise<{ valid: boolean; errors: string[] }> {
    const result = await this.page.evaluate(() => {
      const errors: string[] = [];

      // Check if layout state exists
      if (!(window as any).__layoutState) {
        return { valid: true, errors: [] }; // No layout state to validate
      }

      const layout = (window as any).__layoutState;

      // Validate widget structure
      if (layout.widgets) {
        for (const widget of layout.widgets) {
          if (!widget.scenario) {
            errors.push(`Widget missing scenario: ${JSON.stringify(widget)}`);
          }
          if (!widget.fixture) {
            errors.push(`Widget missing fixture: ${JSON.stringify(widget)}`);
          }
          if (!['hero', 'expanded', 'normal', 'compact', 'hidden'].includes(widget.size)) {
            errors.push(`Widget has invalid size: ${widget.size}`);
          }
        }
      }

      return { valid: errors.length === 0, errors };
    });

    return result;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST SCENARIOS
// ═══════════════════════════════════════════════════════════════════════════════

export const REALISTIC_SCENARIOS = [
  // Industrial queries
  { query: "What is the current status of all pumps?", expectedDomains: ['industrial'], minWidgets: 1 },
  { query: "Show me the temperature trend for the past 24 hours", expectedDomains: ['industrial'], minWidgets: 1 },
  { query: "Compare pump 1 and pump 2 performance", expectedDomains: ['industrial'], minWidgets: 1 },
  { query: "What is the energy consumption breakdown by equipment?", expectedDomains: ['industrial'], minWidgets: 1 },
  { query: "Show transformer load distribution across all substations", expectedDomains: ['industrial'], minWidgets: 1 },
  { query: "What is the chiller efficiency today?", expectedDomains: ['industrial'], minWidgets: 1 },
  { query: "Display the HVAC system status", expectedDomains: ['industrial'], minWidgets: 1 },
  { query: "Show me the power quality metrics", expectedDomains: ['industrial'], minWidgets: 1 },
  { query: "What equipment needs maintenance based on health scores?", expectedDomains: ['industrial'], minWidgets: 1 },
  { query: "Show the top 5 energy consumers", expectedDomains: ['industrial'], minWidgets: 1 },

  // Alert queries
  { query: "Are there any critical alerts right now?", expectedDomains: ['alerts'], minWidgets: 1 },
  { query: "Show me all active alarms", expectedDomains: ['alerts'], minWidgets: 1 },
  { query: "What warnings have we had in the past hour?", expectedDomains: ['alerts'], minWidgets: 1 },
  { query: "Display threshold breach alerts", expectedDomains: ['alerts'], minWidgets: 1 },
  { query: "Show me the alert history for transformer TR-001", expectedDomains: ['alerts', 'industrial'], minWidgets: 1 },

  // People queries
  { query: "Who is on shift right now?", expectedDomains: ['people'], minWidgets: 1 },
  { query: "Show me the technician availability", expectedDomains: ['people'], minWidgets: 1 },
  { query: "Display the operator schedule for this week", expectedDomains: ['people'], minWidgets: 1 },
  { query: "Who is responsible for the chiller plant?", expectedDomains: ['people'], minWidgets: 1 },

  // Task queries
  { query: "What work orders are pending?", expectedDomains: ['tasks'], minWidgets: 1 },
  { query: "Show overdue maintenance tasks", expectedDomains: ['tasks'], minWidgets: 1 },
  { query: "Display all open tickets", expectedDomains: ['tasks'], minWidgets: 1 },
  { query: "What projects are in progress?", expectedDomains: ['tasks'], minWidgets: 1 },

  // Supply queries
  { query: "What is the current inventory status?", expectedDomains: ['supply'], minWidgets: 1 },
  { query: "Show spare parts stock levels", expectedDomains: ['supply'], minWidgets: 1 },
  { query: "Display supplier delivery schedule", expectedDomains: ['supply'], minWidgets: 1 },
  { query: "What items are below reorder point?", expectedDomains: ['supply'], minWidgets: 1 },

  // Complex multi-domain queries
  { query: "Show me equipment with alerts that need maintenance", expectedDomains: ['industrial', 'alerts'], minWidgets: 2 },
  { query: "Who can fix the pump that's showing a warning?", expectedDomains: ['people', 'alerts'], minWidgets: 1 },
  { query: "Compare this week's energy vs last week with any alerts highlighted", expectedDomains: ['industrial', 'alerts'], minWidgets: 2 },
];

export const CONTEXT_STRESS_SCENARIOS = [
  // Multi-turn conversations
  [
    "Show me all pumps",
    "Focus on pump 1",
    "What's its temperature trend?",
    "Compare it to pump 2",
    "Any alerts on either?",
    "Who can fix the issues?",
    "Create a work order",
    "Show the pending work orders",
    "Go back to the pumps overview",
    "Add the chiller status too",
  ],
  [
    "What's the current energy consumption?",
    "Break it down by building",
    "Show me Building A",
    "What equipment is in Building A?",
    "Any alerts there?",
    "Show the historical trend",
    "Compare to yesterday",
    "Add the HVAC data",
    "Remove the comparison",
    "Export this view",
  ],
  [
    "Are there any critical alerts?",
    "Show me the details",
    "What equipment is affected?",
    "Who is on shift to handle this?",
    "Assign it to them",
    "Show me similar past incidents",
    "What was the root cause?",
    "Recommend preventive actions",
    "Create a maintenance schedule",
    "Go back to current alerts",
  ],
];

export const ADVERSARIAL_SCENARIOS = [
  // Intent changes
  { query: "Show me the pumps... no wait, show the chillers instead", expected: "chiller" },
  { query: "Turn on pump 1... actually, never mind", expected: "cancel" },
  { query: "What is the weather... I mean, what is the pump status?", expected: "pump" },

  // Contradictions
  { query: "Show more details but make it simpler", expected: "clarify" },
  { query: "Hide everything but keep it visible", expected: "clarify" },

  // Impossible requests
  { query: "Show me tomorrow's alerts", expected: "future" },
  { query: "Delete all the equipment data", expected: "reject" },
  { query: "Make the pumps run faster", expected: "scope" },

  // Ambiguous
  { query: "Show me that thing from earlier", expected: "clarify" },
  { query: "The other one", expected: "clarify" },
  { query: "Fix it", expected: "clarify" },

  // Gibberish
  { query: "asdf jkl; qwerty", expected: "error" },
  { query: "12345 67890", expected: "error" },

  // Very long
  { query: "Show me the pump status and the chiller status and the transformer status and the AHU status and all the alerts and all the work orders and everyone on shift and the inventory levels and the energy consumption and the temperature trends and the pressure readings and the flow rates for all equipment in all buildings for the past month with daily breakdowns and highlight any anomalies", expected: "complex" },

  // Empty/whitespace
  { query: "", expected: "empty" },
  { query: "   ", expected: "empty" },

  // Special characters
  { query: "SELECT * FROM equipment; DROP TABLE alerts;--", expected: "reject" },
  { query: "<script>alert('xss')</script>", expected: "reject" },

  // Interruptions
  { query: "Show me the--", expected: "incomplete" },
  { query: "What is", expected: "incomplete" },
];

export const ALL_WIDGETS = [
  'kpi',
  'trend',
  'trend-multi-line',
  'trends-cumulative',
  'distribution',
  'comparison',
  'composition',
  'flow-sankey',
  'matrix-heatmap',
  'category-bar',
  'timeline',
  'eventlogstream',
  'chatstream',
  'alerts',
  'edgedevicepanel',
  // AUDIT FIX: Removed agentsview and vaultview - not implemented in widget catalog
  'peoplehexgrid',
  'peoplenetwork',
  'peopleview',
  'supplychainglobe',
];

// Queries designed to trigger specific widgets
export const WIDGET_TRIGGER_QUERIES: Record<string, string> = {
  'kpi': 'Show me the key performance indicators',
  'trend': 'Show the temperature trend over time',
  'trend-multi-line': 'Compare temperature trends across multiple sensors',
  'trends-cumulative': 'Show cumulative energy consumption over time',
  'distribution': 'Show the distribution of equipment by type',
  'comparison': 'Compare pump 1 and pump 2 performance',
  'composition': 'Show the energy breakdown by source',
  'flow-sankey': 'Show the energy flow diagram',
  'matrix-heatmap': 'Show the equipment health matrix',
  'category-bar': 'Show alerts by category',
  'timeline': 'Show the event timeline',
  'eventlogstream': 'Show the live event log',
  'chatstream': 'Open the chat interface',
  'alerts': 'Show all active alerts',
  'edgedevicepanel': 'Show the equipment panel',
  // AUDIT FIX: Removed agentsview and vaultview - not implemented in widget catalog
  'peoplehexgrid': 'Show the team on shift',
  'peoplenetwork': 'Show the team network',
  'peopleview': 'Show the personnel directory',
  'supplychainglobe': 'Show the supply chain map',
};
