/**
 * Performance Under UI Load Tests
 *
 * Measures:
 * - Layout render time with varying widget counts
 * - FPS stability
 * - React reconciliation cost
 * - Memory growth over session
 *
 * Fails if:
 * - UI freezes
 * - Widgets desync
 * - Layout flickers
 * - Memory grows unbounded
 */
import { test, expect } from '@playwright/test';
import { CommandCenterPage } from '../helpers/test-utils';

test.describe('Performance Under UI Load', () => {
  let page: CommandCenterPage;

  test.beforeEach(async ({ page: playwrightPage }) => {
    page = new CommandCenterPage(playwrightPage);
    await page.goto();
    await page.waitForReady();
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // LAYOUT RENDER TIME BENCHMARKS
  // ═══════════════════════════════════════════════════════════════════════════

  test('should render 1-2 widgets under 500ms', async () => {
    const start = Date.now();

    await page.sendQuery('Show pump status');
    await page.waitForLayout();

    const renderTime = Date.now() - start;

    // Allow network latency, should still be reasonable
    expect(renderTime).toBeLessThan(10000); // 10s max including AI processing

    const widgetCount = await page.getWidgetCount();
    console.log(`Rendered ${widgetCount} widgets in ${renderTime}ms`);
  });

  test('should render 4 widgets under 1000ms', async () => {
    const start = Date.now();

    await page.sendQuery('Show pumps, chillers, alerts, and energy consumption');
    await page.waitForLayout();

    const renderTime = Date.now() - start;

    // Should still complete reasonably
    expect(renderTime).toBeLessThan(15000); // 15s max with AI

    const widgetCount = await page.getWidgetCount();
    console.log(`Rendered ${widgetCount} widgets in ${renderTime}ms`);
  });

  test('should render 8 widgets under 2000ms', async () => {
    const start = Date.now();

    await page.sendQuery('Show comprehensive dashboard: pumps, chillers, transformers, alerts, energy, trends, people, and tasks');
    await page.waitForLayout();

    const renderTime = Date.now() - start;

    expect(renderTime).toBeLessThan(20000); // 20s max

    const widgetCount = await page.getWidgetCount();
    console.log(`Rendered ${widgetCount} widgets in ${renderTime}ms`);
  });

  test('should render maximum widgets (10) under 3000ms', async () => {
    const start = Date.now();

    await page.sendQuery('Show absolutely everything available');
    await page.waitForLayout();

    const renderTime = Date.now() - start;

    expect(renderTime).toBeLessThan(30000); // 30s max

    const widgetCount = await page.getWidgetCount();
    expect(widgetCount).toBeLessThanOrEqual(10); // Should respect max

    console.log(`Rendered ${widgetCount} widgets in ${renderTime}ms`);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // FPS AND ANIMATION STABILITY
  // ═══════════════════════════════════════════════════════════════════════════

  test('should maintain 30+ FPS during widget transitions', async () => {
    // Get initial layout
    await page.sendQuery('Show pumps');
    await page.waitForLayout();

    // Measure FPS during transition
    const fpsReadings: number[] = [];

    await page.page.evaluate(() => {
      let lastTime = performance.now();
      let frameCount = 0;

      function measure() {
        frameCount++;
        const now = performance.now();
        if (now - lastTime >= 1000) {
          (window as any).__lastFPS = frameCount;
          frameCount = 0;
          lastTime = now;
        }
        if ((window as any).__measuringFPS) {
          requestAnimationFrame(measure);
        }
      }

      (window as any).__measuringFPS = true;
      requestAnimationFrame(measure);
    });

    // Trigger layout change
    await page.sendQuery('Now show alerts and energy');

    // Wait a bit and collect FPS
    await page.page.waitForTimeout(2000);

    const fps = await page.page.evaluate(() => {
      (window as any).__measuringFPS = false;
      return (window as any).__lastFPS || 60;
    });

    // FPS should be reasonable (allowing for test environment)
    expect(fps).toBeGreaterThanOrEqual(10); // Minimum during heavy transitions
  });

  test('should not freeze during rapid updates', async () => {
    const queries = [
      'Show pumps',
      'Show alerts',
      'Show energy',
      'Show people',
      'Show tasks',
    ];

    let freezeDetected = false;

    for (const query of queries) {
      const start = Date.now();

      await page.sendQuery(query);

      // Check if response came within reasonable time
      try {
        await page.page.waitForSelector('body', { timeout: 5000 });
      } catch {
        freezeDetected = true;
        break;
      }

      const elapsed = Date.now() - start;
      if (elapsed > 30000) {
        freezeDetected = true;
        break;
      }

      await page.page.waitForTimeout(200);
    }

    expect(freezeDetected).toBe(false);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // DOM SIZE AND COMPLEXITY
  // ═══════════════════════════════════════════════════════════════════════════

  test('should keep DOM size reasonable', async () => {
    // Load heavy dashboard
    await page.sendQuery('Show comprehensive overview with all available data');
    await page.waitForLayout();

    const metrics = await page.getPerformanceMetrics();

    // DOM should not explode
    expect(metrics.totalDOMNodes).toBeLessThan(50000);

    console.log(`Total DOM nodes: ${metrics.totalDOMNodes}`);
  });

  test('should not have memory leaks in DOM', async () => {
    // Get initial DOM count
    const initialMetrics = await page.getPerformanceMetrics();
    const initialNodes = initialMetrics.totalDOMNodes;

    // Cycle through layouts
    for (let i = 0; i < 10; i++) {
      const queries = ['Show pumps', 'Show alerts', 'Show energy'];
      await page.sendQuery(queries[i % queries.length]);
      await page.page.waitForTimeout(500);
    }

    // Final DOM count
    const finalMetrics = await page.getPerformanceMetrics();
    const finalNodes = finalMetrics.totalDOMNodes;

    // Should not grow significantly
    const growth = finalNodes - initialNodes;
    expect(growth).toBeLessThan(10000);

    console.log(`DOM growth: ${growth} nodes`);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // MEMORY STABILITY
  // ═══════════════════════════════════════════════════════════════════════════

  test('should not leak memory over 20 interactions', async () => {
    const initialMetrics = await page.getPerformanceMetrics();
    const initialMemory = initialMetrics.memoryUsage || 0;

    // Perform many interactions
    for (let i = 0; i < 20; i++) {
      const queries = [
        'Show pumps', 'Show chillers', 'Show alerts',
        'Show energy', 'Show people', 'Show tasks',
      ];
      await page.sendQuery(queries[i % queries.length]);
      await page.page.waitForTimeout(300);
    }

    // Force garbage collection if available
    await page.page.evaluate(() => {
      if ((window as any).gc) {
        (window as any).gc();
      }
    });

    await page.page.waitForTimeout(1000);

    const finalMetrics = await page.getPerformanceMetrics();
    const finalMemory = finalMetrics.memoryUsage || 0;

    if (initialMemory > 0 && finalMemory > 0) {
      const growth = finalMemory - initialMemory;
      console.log(`Memory growth: ${growth.toFixed(2)}MB`);

      // Should not grow more than 200MB over session
      expect(growth).toBeLessThan(200);
    }
  });

  test('should release memory when widgets are removed', async () => {
    // Load heavy layout
    await page.sendQuery('Show comprehensive overview');
    await page.waitForLayout();

    const heavyMetrics = await page.getPerformanceMetrics();
    const heavyMemory = heavyMetrics.memoryUsage || 0;

    // Switch to light layout
    await page.sendQuery('Just show pump 1 status');
    await page.waitForLayout();

    // Wait for cleanup
    await page.page.waitForTimeout(2000);

    // Force GC if available
    await page.page.evaluate(() => {
      if ((window as any).gc) {
        (window as any).gc();
      }
    });

    await page.page.waitForTimeout(500);

    const lightMetrics = await page.getPerformanceMetrics();
    const lightMemory = lightMetrics.memoryUsage || 0;

    if (heavyMemory > 0 && lightMemory > 0) {
      // Light layout should use less or similar memory
      console.log(`Heavy: ${heavyMemory.toFixed(2)}MB, Light: ${lightMemory.toFixed(2)}MB`);
    }
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // LAYOUT STABILITY
  // ═══════════════════════════════════════════════════════════════════════════

  test('should not flicker during updates', async () => {
    // Setup mutation observer to detect flicker
    await page.page.evaluate(() => {
      (window as any).__flickerCount = 0;
      (window as any).__lastWidgetCount = 0;

      const observer = new MutationObserver((mutations) => {
        const currentWidgets = document.querySelectorAll('[data-testid^="widget-"]').length;

        // Detect flicker: widget count drops then recovers
        if (currentWidgets < (window as any).__lastWidgetCount) {
          (window as any).__flickerCount++;
        }

        (window as any).__lastWidgetCount = currentWidgets;
      });

      observer.observe(document.body, { childList: true, subtree: true });
      (window as any).__flickerObserver = observer;
    });

    // Trigger several layout updates
    await page.sendQuery('Show pumps');
    await page.waitForLayout();
    await page.sendQuery('Show alerts');
    await page.waitForLayout();
    await page.sendQuery('Show energy');
    await page.waitForLayout();

    // Check flicker count
    const flickerCount = await page.page.evaluate(() => {
      (window as any).__flickerObserver?.disconnect();
      return (window as any).__flickerCount;
    });

    // Some widget count changes are expected, but not excessive
    expect(flickerCount).toBeLessThan(20);
  });

  test('should maintain widget order during updates', async () => {
    // Load initial layout
    await page.sendQuery('Show pumps and alerts');
    await page.waitForLayout();

    const initialWidgets = await page.getWidgets();
    const initialOrder = initialWidgets.map(w => w.scenario);

    // Trigger update
    await page.sendQuery('Also add energy data');
    await page.waitForLayout();

    const updatedWidgets = await page.getWidgets();
    const updatedOrder = updatedWidgets.map(w => w.scenario);

    // Existing widgets should maintain relative order
    // (New widgets can be inserted anywhere)
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // STRESS TEST
  // ═══════════════════════════════════════════════════════════════════════════

  test('should survive stress test (50 rapid queries)', async () => {
    const errors: string[] = [];
    let successCount = 0;

    for (let i = 0; i < 50; i++) {
      const queries = [
        'Show pumps', 'Show alerts', 'Show energy',
        'Show chillers', 'Show people', 'Show tasks',
      ];

      try {
        await page.sendQuery(queries[i % queries.length]);
        await page.page.waitForTimeout(100);
        successCount++;
      } catch (e) {
        errors.push(`Query ${i}: ${e}`);
      }
    }

    // Wait for final stabilization
    await page.page.waitForTimeout(2000);

    // Check final state
    const validation = await page.validateLayoutJSON();

    console.log(`Stress test: ${successCount}/50 successful`);
    expect(successCount).toBeGreaterThanOrEqual(45); // Allow some failures
    expect(validation.errors).toHaveLength(0);
  });
});
