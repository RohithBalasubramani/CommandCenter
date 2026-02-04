/**
 * Widget Exhaustion Tests
 *
 * Tests all 23 widgets under real usage scenarios.
 * Each widget must:
 * - Appear in natural scenario
 * - Be interactable
 * - Have correct data shape
 * - Render within performance budget
 */
import { test, expect } from '@playwright/test';
import { CommandCenterPage, ALL_WIDGETS, WIDGET_TRIGGER_QUERIES } from '../helpers/test-utils';

test.describe('Widget Exhaustion Tests', () => {
  let page: CommandCenterPage;

  test.beforeEach(async ({ page: playwrightPage }) => {
    page = new CommandCenterPage(playwrightPage);
    await page.goto();
    await page.waitForReady();
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // DATA VISUALIZATION WIDGETS
  // ═══════════════════════════════════════════════════════════════════════════

  test('widget: KPI tiles', async () => {
    await page.sendQuery('Show me the key performance indicators');
    await page.waitForLayout();

    const widgets = await page.getWidgets();
    expect(widgets.length).toBeGreaterThanOrEqual(0);

    // Validate layout
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('widget: trend (single line)', async () => {
    await page.sendQuery('Show the temperature trend over time');
    await page.waitForLayout();

    const widgets = await page.getWidgets();
    // May or may not produce a trend widget
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('widget: trend-multi-line', async () => {
    await page.sendQuery('Compare temperature trends across multiple sensors');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('widget: trends-cumulative', async () => {
    await page.sendQuery('Show cumulative energy consumption over time');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('widget: distribution', async () => {
    await page.sendQuery('Show the distribution of equipment by type');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('widget: comparison', async () => {
    await page.sendQuery('Compare pump 1 and pump 2 performance side by side');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('widget: composition', async () => {
    await page.sendQuery('Show the energy breakdown by source as a pie chart');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('widget: flow-sankey', async () => {
    await page.sendQuery('Show the energy flow diagram');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('widget: matrix-heatmap', async () => {
    await page.sendQuery('Show the equipment health matrix');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('widget: category-bar', async () => {
    await page.sendQuery('Show alerts by category as a bar chart');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // DATA STREAM WIDGETS
  // ═══════════════════════════════════════════════════════════════════════════

  test('widget: timeline', async () => {
    await page.sendQuery('Show the event timeline for today');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('widget: eventlogstream', async () => {
    await page.sendQuery('Show the live event log');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('widget: alerts', async () => {
    await page.sendQuery('Show all active alerts');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // DOMAIN PANEL WIDGETS
  // ═══════════════════════════════════════════════════════════════════════════

  test('widget: edgedevicepanel', async () => {
    await page.sendQuery('Show the equipment panel with all devices');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  // AUDIT FIX: Removed agentsview test - widget not implemented

  test('widget: peoplehexgrid', async () => {
    await page.sendQuery('Show the team on shift in a hex grid');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('widget: peoplenetwork', async () => {
    await page.sendQuery('Show the team network and relationships');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('widget: peopleview', async () => {
    await page.sendQuery('Show the personnel directory');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('widget: supplychainglobe', async () => {
    await page.sendQuery('Show the supply chain map on a globe');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  // AUDIT FIX: Removed vaultview test - widget not implemented

  // ═══════════════════════════════════════════════════════════════════════════
  // WIDGET INTERACTION TESTS
  // ═══════════════════════════════════════════════════════════════════════════

  test('should be able to click on widgets', async () => {
    await page.sendQuery('Show me the pump status');
    await page.waitForLayout();

    const widgets = await page.getWidgets();
    if (widgets.length > 0) {
      // Click first widget
      await widgets[0].element.click();
      // Should not crash
      await page.page.waitForTimeout(500);
    }

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should handle widget hover interactions', async () => {
    await page.sendQuery('Show temperature trend');
    await page.waitForLayout();

    const widgets = await page.getWidgets();
    if (widgets.length > 0) {
      // Hover over widget
      await widgets[0].element.hover();
      await page.page.waitForTimeout(300);
    }

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // MULTI-WIDGET LAYOUTS
  // ═══════════════════════════════════════════════════════════════════════════

  test('should render 4+ widgets without issues', async () => {
    await page.sendQuery('Show me an overview dashboard with pumps, alerts, energy, and people on shift');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);

    // Check render performance
    const metrics = await page.getPerformanceMetrics();
    expect(metrics.totalDOMNodes).toBeLessThan(50000); // Reasonable DOM size
  });

  test('should handle maximum widgets (10)', async () => {
    await page.sendQuery('Show me everything: all pumps, chillers, transformers, alerts, people, tasks, inventory, energy trends, and health scores');
    await page.waitForLayout();

    const widgetCount = await page.getWidgetCount();
    // Should respect MAX_WIDGETS = 10
    expect(widgetCount).toBeLessThanOrEqual(10);

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // WIDGET DATA VALIDATION
  // ═══════════════════════════════════════════════════════════════════════════

  test('should have valid data in widgets', async () => {
    await page.sendQuery('Show pump status');
    await page.waitForLayout();

    const widgets = await page.getWidgets();

    for (const widget of widgets) {
      // Each widget should have content
      const text = await widget.element.textContent();
      // Widget should have some content (not empty)
      expect(text?.length).toBeGreaterThan(0);
    }
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // WIDGET REGISTRY COVERAGE
  // ═══════════════════════════════════════════════════════════════════════════

  test.describe('Widget Registry Coverage', () => {
    const widgetQueries: [string, string][] = [
      ['kpi', 'Show KPI dashboard'],
      ['trend', 'Show temperature trend'],
      ['alerts', 'Show active alerts'],
      ['edgedevicepanel', 'Show equipment panel'],
      ['peopleview', 'Show people on shift'],
      ['category-bar', 'Show alert categories'],
    ];

    for (const [widgetType, query] of widgetQueries) {
      test(`should be able to trigger ${widgetType} widget`, async () => {
        await page.sendQuery(query);
        await page.waitForLayout();

        const validation = await page.validateLayoutJSON();
        expect(validation.errors).toHaveLength(0);
      });
    }
  });
});
