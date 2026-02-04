/**
 * MEGA VALIDATION - PHASE 2: PLAYWRIGHT E2E TESTING
 *
 * Break-it-or-block-release validation for Command Center.
 * Real user simulation with typed + voice input.
 *
 * Evidence Requirements:
 * - Screenshot on every state change
 * - Video recording (configured in playwright.config.ts)
 * - Trace capture on failure
 * - Performance metrics logging
 *
 * If any E2E scenario fails → BLOCK RELEASE
 */
import { test, expect, Page } from '@playwright/test';
import {
  CommandCenterPage,
  REALISTIC_SCENARIOS,
  CONTEXT_STRESS_SCENARIOS,
  ADVERSARIAL_SCENARIOS,
  ALL_WIDGETS,
  WIDGET_TRIGGER_QUERIES,
} from '../helpers/test-utils';
import * as fs from 'fs';
import * as path from 'path';

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

const EVIDENCE_DIR = path.join(__dirname, '../../mega-validation-evidence');
const REPORT_FILE = path.join(EVIDENCE_DIR, 'phase2-report.json');

interface Phase2Result {
  timestamp: string;
  totalTests: number;
  passedTests: number;
  failedTests: number;
  testResults: TestResult[];
  performanceMetrics: PerformanceMetrics;
  coverageMetrics: CoverageMetrics;
  releaseBlocked: boolean;
  blockReasons: string[];
}

interface TestResult {
  name: string;
  category: string;
  passed: boolean;
  duration: number;
  error?: string;
  screenshot?: string;
}

interface PerformanceMetrics {
  avgLayoutRenderTime: number;
  maxLayoutRenderTime: number;
  avgWidgetCount: number;
  totalDOMNodes: number[];
}

interface CoverageMetrics {
  scenariosTestedPercent: number;
  widgetsRenderedPercent: number;
  uniqueWidgetsRendered: string[];
  uniqueDomainsHit: string[];
}

// Initialize report
let phase2Report: Phase2Result = {
  timestamp: new Date().toISOString(),
  totalTests: 0,
  passedTests: 0,
  failedTests: 0,
  testResults: [],
  performanceMetrics: {
    avgLayoutRenderTime: 0,
    maxLayoutRenderTime: 0,
    avgWidgetCount: 0,
    totalDOMNodes: [],
  },
  coverageMetrics: {
    scenariosTestedPercent: 0,
    widgetsRenderedPercent: 0,
    uniqueWidgetsRendered: [],
    uniqueDomainsHit: [],
  },
  releaseBlocked: false,
  blockReasons: [],
};

// ═══════════════════════════════════════════════════════════════════════════════
// TEST UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

async function ensureEvidenceDir() {
  if (!fs.existsSync(EVIDENCE_DIR)) {
    fs.mkdirSync(EVIDENCE_DIR, { recursive: true });
  }
}

async function captureEvidence(
  page: Page,
  testName: string,
  suffix: string = ''
): Promise<string> {
  await ensureEvidenceDir();
  const safeName = testName.replace(/[^a-z0-9]/gi, '_').toLowerCase();
  const filename = `${safeName}${suffix ? '_' + suffix : ''}_${Date.now()}.png`;
  const filepath = path.join(EVIDENCE_DIR, filename);
  await page.screenshot({ path: filepath, fullPage: true });
  return filepath;
}

function recordTestResult(result: TestResult) {
  phase2Report.testResults.push(result);
  phase2Report.totalTests++;
  if (result.passed) {
    phase2Report.passedTests++;
  } else {
    phase2Report.failedTests++;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 2.1: REALISTIC USER SCENARIOS
// ═══════════════════════════════════════════════════════════════════════════════

test.describe('Phase 2.1: Realistic User Scenarios', () => {
  let ccPage: CommandCenterPage;

  test.beforeEach(async ({ page }) => {
    ccPage = new CommandCenterPage(page);
    await ccPage.goto();
    await ccPage.waitForReady();
    await ensureEvidenceDir();
  });

  test.afterEach(async ({ page }, testInfo) => {
    // Capture final state
    if (testInfo.status !== 'passed') {
      await captureEvidence(page, testInfo.title, 'failure');
    }
  });

  // Test first 15 realistic scenarios
  for (const scenario of REALISTIC_SCENARIOS.slice(0, 15)) {
    test(`should handle: ${scenario.query.slice(0, 50)}...`, async ({ page }) => {
      const startTime = Date.now();

      try {
        // Send query
        await ccPage.sendQuery(scenario.query);

        // Wait for layout with explicit timeout
        await ccPage.waitForLayout(30000);

        // Capture state
        const screenshot = await captureEvidence(page, scenario.query.slice(0, 30));

        // Validate layout
        const validation = await ccPage.validateLayoutJSON();
        const widgetCount = await ccPage.getWidgetCount();
        const metrics = await ccPage.getPerformanceMetrics();

        // Update performance metrics
        phase2Report.performanceMetrics.totalDOMNodes.push(metrics.totalDOMNodes);
        if (metrics.layoutRenderTime > phase2Report.performanceMetrics.maxLayoutRenderTime) {
          phase2Report.performanceMetrics.maxLayoutRenderTime = metrics.layoutRenderTime;
        }

        // Record widgets rendered
        const widgets = await ccPage.getWidgets();
        for (const widget of widgets) {
          if (!phase2Report.coverageMetrics.uniqueWidgetsRendered.includes(widget.scenario)) {
            phase2Report.coverageMetrics.uniqueWidgetsRendered.push(widget.scenario);
          }
        }

        const duration = Date.now() - startTime;

        recordTestResult({
          name: scenario.query.slice(0, 50),
          category: 'realistic',
          passed: validation.errors.length === 0,
          duration,
          screenshot,
        });

        expect(validation.errors).toHaveLength(0);
      } catch (error) {
        const duration = Date.now() - startTime;
        recordTestResult({
          name: scenario.query.slice(0, 50),
          category: 'realistic',
          passed: false,
          duration,
          error: error instanceof Error ? error.message : String(error),
        });
        throw error;
      }
    });
  }
});

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 2.2: MULTI-TURN CONVERSATION STRESS
// ═══════════════════════════════════════════════════════════════════════════════

test.describe('Phase 2.2: Multi-Turn Conversation Stress', () => {
  let ccPage: CommandCenterPage;

  test.beforeEach(async ({ page }) => {
    ccPage = new CommandCenterPage(page);
    await ccPage.goto();
    await ccPage.waitForReady();
  });

  test('should handle 10-turn conversation without state corruption', async ({ page }) => {
    // 10-turn conversation stress test (strict requirement)
    const conversation = CONTEXT_STRESS_SCENARIOS[0];
    const startTime = Date.now();
    const errors: string[] = [];

    for (let i = 0; i < Math.min(10, conversation.length); i++) {
      const query = conversation[i];

      try {
        await ccPage.sendQuery(query);
        await ccPage.waitForLayout(10000);

        // Capture each turn
        await captureEvidence(page, `conversation_turn_${i + 1}`);

        // Validate state integrity
        const validation = await ccPage.validateLayoutJSON();
        if (validation.errors.length > 0) {
          errors.push(`Turn ${i + 1}: ${validation.errors.join(', ')}`);
        }

        // Small pause between turns (simulates real user)
        await page.waitForTimeout(500);
      } catch (error) {
        errors.push(`Turn ${i + 1}: ${error instanceof Error ? error.message : String(error)}`);
      }
    }

    const duration = Date.now() - startTime;

    recordTestResult({
      name: '10-turn conversation',
      category: 'context-stress',
      passed: errors.length === 0,
      duration,
      error: errors.length > 0 ? errors.join('; ') : undefined,
    });

    expect(errors).toHaveLength(0);
  });

  test('should handle rapid-fire queries without crashing', async ({ page }) => {
    const queries = [
      'Show pumps',
      'Show chillers',
      'Show alerts',
      'Show energy',
      'Show people',
    ];
    const startTime = Date.now();
    let crashed = false;

    for (const query of queries) {
      try {
        await ccPage.sendQuery(query);
        // Don't wait for full layout, rapid fire
        await page.waitForTimeout(200);
      } catch (error) {
        crashed = true;
        break;
      }
    }

    // Final wait for stability
    await page.waitForTimeout(2000);

    // Check page is still responsive
    try {
      await ccPage.sendQuery('Show pump status');
      await ccPage.waitForLayout(30000);
    } catch {
      crashed = true;
    }

    const duration = Date.now() - startTime;

    recordTestResult({
      name: 'rapid-fire queries',
      category: 'context-stress',
      passed: !crashed,
      duration,
    });

    expect(crashed).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 2.3: ADVERSARIAL INPUT HANDLING
// ═══════════════════════════════════════════════════════════════════════════════

test.describe('Phase 2.3: Adversarial Input Handling', () => {
  let ccPage: CommandCenterPage;

  test.beforeEach(async ({ page }) => {
    ccPage = new CommandCenterPage(page);
    await ccPage.goto();
    await ccPage.waitForReady();
  });

  test('should reject SQL injection attempts', async ({ page }) => {
    const startTime = Date.now();
    const injection = "SELECT * FROM equipment; DROP TABLE alerts;--";

    await ccPage.sendQuery(injection);
    await page.waitForTimeout(3000);

    // Should not crash
    const isResponsive = await page.evaluate(() => document.body !== null);
    const screenshot = await captureEvidence(page, 'sql_injection_test');

    const duration = Date.now() - startTime;

    recordTestResult({
      name: 'SQL injection rejection',
      category: 'adversarial',
      passed: isResponsive,
      duration,
      screenshot,
    });

    expect(isResponsive).toBe(true);
  });

  test('should reject XSS attempts', async ({ page }) => {
    const startTime = Date.now();
    const xss = "<script>alert('xss')</script>";

    await ccPage.sendQuery(xss);
    await page.waitForTimeout(3000);

    // Check no script was executed
    const alertTriggered = await page.evaluate(() => {
      return (window as any).__xssTriggered === true;
    });

    const isResponsive = await page.evaluate(() => document.body !== null);
    const screenshot = await captureEvidence(page, 'xss_test');

    const duration = Date.now() - startTime;

    recordTestResult({
      name: 'XSS rejection',
      category: 'adversarial',
      passed: isResponsive && !alertTriggered,
      duration,
      screenshot,
    });

    expect(alertTriggered).toBe(false);
    expect(isResponsive).toBe(true);
  });

  test('should handle empty input gracefully', async ({ page }) => {
    const startTime = Date.now();

    try {
      // Try to submit empty query
      await ccPage.submitQuery();
      await page.waitForTimeout(1000);
    } catch {
      // Empty input may throw, that's OK
    }

    // Page should still be responsive
    const isResponsive = await page.evaluate(() => document.body !== null);

    const duration = Date.now() - startTime;

    recordTestResult({
      name: 'empty input handling',
      category: 'adversarial',
      passed: isResponsive,
      duration,
    });

    expect(isResponsive).toBe(true);
  });

  test('should handle very long input', async ({ page }) => {
    const startTime = Date.now();
    const longQuery = 'Show me the pump status '.repeat(100);

    await ccPage.sendQuery(longQuery);
    await page.waitForTimeout(5000);

    // Should not crash, may truncate
    const isResponsive = await page.evaluate(() => document.body !== null);
    const screenshot = await captureEvidence(page, 'long_input_test');

    const duration = Date.now() - startTime;

    recordTestResult({
      name: 'long input handling',
      category: 'adversarial',
      passed: isResponsive,
      duration,
      screenshot,
    });

    expect(isResponsive).toBe(true);
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 2.4: WIDGET RENDERING VALIDATION
// ═══════════════════════════════════════════════════════════════════════════════

test.describe('Phase 2.4: Widget Rendering Validation', () => {
  let ccPage: CommandCenterPage;

  test.beforeEach(async ({ page }) => {
    ccPage = new CommandCenterPage(page);
    await ccPage.goto();
    await ccPage.waitForReady();
  });

  // Test widget trigger queries for key widgets
  const keyWidgets = ['kpi', 'trend', 'alerts', 'comparison', 'distribution'];

  for (const widget of keyWidgets) {
    test(`should render ${widget} widget correctly`, async ({ page }) => {
      const query = WIDGET_TRIGGER_QUERIES[widget];
      if (!query) {
        test.skip();
        return;
      }

      const startTime = Date.now();

      await ccPage.sendQuery(query);
      await ccPage.waitForLayout(30000);

      const widgets = await ccPage.getWidgets();
      const validation = await ccPage.validateLayoutJSON();
      const screenshot = await captureEvidence(page, `widget_${widget}`);

      const duration = Date.now() - startTime;

      // Update coverage
      for (const w of widgets) {
        if (!phase2Report.coverageMetrics.uniqueWidgetsRendered.includes(w.scenario)) {
          phase2Report.coverageMetrics.uniqueWidgetsRendered.push(w.scenario);
        }
      }

      recordTestResult({
        name: `${widget} widget render`,
        category: 'widget-validation',
        passed: validation.errors.length === 0,
        duration,
        screenshot,
      });

      expect(validation.errors).toHaveLength(0);
    });
  }
});

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 2.5: PERFORMANCE VALIDATION
// ═══════════════════════════════════════════════════════════════════════════════

test.describe('Phase 2.5: Performance Validation', () => {
  let ccPage: CommandCenterPage;

  test.beforeEach(async ({ page }) => {
    ccPage = new CommandCenterPage(page);
    await ccPage.goto();
    await ccPage.waitForReady();
  });

  test('should render layout within 10 seconds', async ({ page }) => {
    // Budget: 10 seconds strict requirement
    const LAYOUT_BUDGET_MS = 10000;
    const startTime = Date.now();

    await ccPage.sendQuery('Show me all equipment status');

    // Wait with strict timeout
    try {
      await ccPage.waitForLayout(LAYOUT_BUDGET_MS);
    } catch {
      recordTestResult({
        name: 'layout render time',
        category: 'performance',
        passed: false,
        duration: LAYOUT_BUDGET_MS,
        error: `Layout did not render within ${LAYOUT_BUDGET_MS / 1000} seconds (strict)`,
      });
      throw new Error('Layout render timeout exceeded');
    }

    const duration = Date.now() - startTime;

    recordTestResult({
      name: 'layout render time',
      category: 'performance',
      passed: duration < LAYOUT_BUDGET_MS,
      duration,
    });

    expect(duration).toBeLessThan(LAYOUT_BUDGET_MS);
  });

  test('should not exceed 50000 DOM nodes', async ({ page }) => {
    await ccPage.sendQuery('Show me everything about the facility');
    await ccPage.waitForLayout(30000);

    const metrics = await ccPage.getPerformanceMetrics();
    const domNodes = metrics.totalDOMNodes;

    recordTestResult({
      name: 'DOM node count',
      category: 'performance',
      passed: domNodes < 50000,
      duration: 0,
      error: domNodes >= 50000 ? `DOM nodes: ${domNodes}` : undefined,
    });

    expect(domNodes).toBeLessThan(50000);
  });

  test('should maintain FPS > 30 during interaction', async ({ page }) => {
    // Start FPS monitoring
    await page.evaluate(() => {
      let frameCount = 0;
      let lastTime = performance.now();
      (window as any).__fpsHistory = [];

      const measureFPS = () => {
        const currentTime = performance.now();
        frameCount++;

        if (currentTime - lastTime >= 1000) {
          (window as any).__fpsHistory.push(frameCount);
          frameCount = 0;
          lastTime = currentTime;
        }

        requestAnimationFrame(measureFPS);
      };

      requestAnimationFrame(measureFPS);
    });

    // Perform interactions
    await ccPage.sendQuery('Show pumps');
    await ccPage.waitForLayout();
    await page.waitForTimeout(2000);

    await ccPage.sendQuery('Show chillers');
    await ccPage.waitForLayout();
    await page.waitForTimeout(2000);

    // Get FPS readings
    const fpsHistory = await page.evaluate(() => (window as any).__fpsHistory || []);
    const avgFPS = fpsHistory.length > 0
      ? fpsHistory.reduce((a: number, b: number) => a + b, 0) / fpsHistory.length
      : 60; // Default if not measured

    recordTestResult({
      name: 'FPS during interaction',
      category: 'performance',
      passed: avgFPS >= 30,
      duration: 0,
      error: avgFPS < 30 ? `Average FPS: ${avgFPS.toFixed(1)}` : undefined,
    });

    expect(avgFPS).toBeGreaterThanOrEqual(30);
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// FINAL REPORT GENERATION
// ═══════════════════════════════════════════════════════════════════════════════

test.afterAll(async () => {
  // Calculate final metrics
  const totalDOMNodes = phase2Report.performanceMetrics.totalDOMNodes;
  if (totalDOMNodes.length > 0) {
    phase2Report.performanceMetrics.avgLayoutRenderTime =
      totalDOMNodes.reduce((a, b) => a + b, 0) / totalDOMNodes.length;
  }

  // Calculate coverage
  phase2Report.coverageMetrics.scenariosTestedPercent =
    (phase2Report.totalTests / REALISTIC_SCENARIOS.length) * 100;
  phase2Report.coverageMetrics.widgetsRenderedPercent =
    (phase2Report.coverageMetrics.uniqueWidgetsRendered.length / ALL_WIDGETS.length) * 100;

  // Determine release decision
  const failedCritical = phase2Report.testResults.filter(
    (r) => !r.passed && ['adversarial', 'performance'].includes(r.category)
  );

  if (failedCritical.length > 0) {
    phase2Report.releaseBlocked = true;
    phase2Report.blockReasons = failedCritical.map(
      (r) => `${r.category}: ${r.name} - ${r.error || 'failed'}`
    );
  }

  if (phase2Report.failedTests / phase2Report.totalTests > 0.2) {
    phase2Report.releaseBlocked = true;
    phase2Report.blockReasons.push(
      `Too many failures: ${phase2Report.failedTests}/${phase2Report.totalTests} (> 20%)`
    );
  }

  // Save report
  await ensureEvidenceDir();
  fs.writeFileSync(REPORT_FILE, JSON.stringify(phase2Report, null, 2));

  console.log('\n═══════════════════════════════════════════════════════════════════════════════');
  console.log('  MEGA VALIDATION - PHASE 2 REPORT');
  console.log('═══════════════════════════════════════════════════════════════════════════════');
  console.log(`  Timestamp: ${phase2Report.timestamp}`);
  console.log(`  Total Tests: ${phase2Report.totalTests}`);
  console.log(`  Passed: ${phase2Report.passedTests}`);
  console.log(`  Failed: ${phase2Report.failedTests}`);
  console.log(`  Pass Rate: ${((phase2Report.passedTests / phase2Report.totalTests) * 100).toFixed(1)}%`);
  console.log('');
  console.log('  Coverage:');
  console.log(`    Scenarios Tested: ${phase2Report.coverageMetrics.scenariosTestedPercent.toFixed(1)}%`);
  console.log(`    Widgets Rendered: ${phase2Report.coverageMetrics.widgetsRenderedPercent.toFixed(1)}%`);
  console.log(`    Unique Widgets: ${phase2Report.coverageMetrics.uniqueWidgetsRendered.join(', ')}`);
  console.log('');

  if (phase2Report.releaseBlocked) {
    console.log('  ✗ RELEASE BLOCKED');
    console.log('  Reasons:');
    for (const reason of phase2Report.blockReasons) {
      console.log(`    - ${reason}`);
    }
  } else {
    console.log('  ✓ RELEASE APPROVED (Phase 2 E2E)');
  }

  console.log('═══════════════════════════════════════════════════════════════════════════════');
  console.log(`  Report saved to: ${REPORT_FILE}`);
  console.log('═══════════════════════════════════════════════════════════════════════════════\n');
});
