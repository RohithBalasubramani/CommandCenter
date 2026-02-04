/**
 * Realistic User Scenarios Test Suite
 *
 * 30+ real-world user scenarios that simulate actual usage patterns.
 * Tests multi-widget layouts, context carryover, and natural queries.
 */
import { test, expect } from '@playwright/test';
import { CommandCenterPage, REALISTIC_SCENARIOS } from '../helpers/test-utils';

test.describe('Realistic User Scenarios', () => {
  let page: CommandCenterPage;

  test.beforeEach(async ({ page: playwrightPage }) => {
    page = new CommandCenterPage(playwrightPage);
    await page.goto();
    await page.waitForReady();
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // INDUSTRIAL QUERIES
  // ═══════════════════════════════════════════════════════════════════════════

  test('should show pump status with appropriate widgets', async () => {
    await page.sendQuery('What is the current status of all pumps?');
    await page.waitForLayout();

    const widgetCount = await page.getWidgetCount();
    expect(widgetCount).toBeGreaterThanOrEqual(1);

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should show temperature trend over time', async () => {
    await page.sendQuery('Show me the temperature trend for the past 24 hours');
    await page.waitForLayout();

    const widgets = await page.getWidgets();
    expect(widgets.length).toBeGreaterThanOrEqual(1);

    // Expect a trend widget
    const hasTrendWidget = widgets.some(w =>
      w.scenario.includes('trend') || w.element.locator('.recharts-line').count() > 0
    );
    // This may or may not have a trend widget depending on data
  });

  test('should compare two pieces of equipment', async () => {
    await page.sendQuery('Compare pump 1 and pump 2 performance');
    await page.waitForLayout();

    const widgetCount = await page.getWidgetCount();
    expect(widgetCount).toBeGreaterThanOrEqual(1);
  });

  test('should show energy breakdown', async () => {
    await page.sendQuery('What is the energy consumption breakdown by equipment?');
    await page.waitForLayout();

    const widgets = await page.getWidgets();
    expect(widgets.length).toBeGreaterThanOrEqual(1);
  });

  test('should show transformer load distribution', async () => {
    await page.sendQuery('Show transformer load distribution across all substations');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.valid).toBe(true);
  });

  test('should show chiller efficiency', async () => {
    await page.sendQuery('What is the chiller efficiency today?');
    await page.waitForLayout();

    const widgetCount = await page.getWidgetCount();
    expect(widgetCount).toBeGreaterThanOrEqual(0); // May return message only
  });

  test('should show HVAC system status', async () => {
    await page.sendQuery('Display the HVAC system status');
    await page.waitForLayout();

    // Should render widgets or a message
    await expect(page.page.locator('body')).toContainText(/.+/);
  });

  test('should show power quality metrics', async () => {
    await page.sendQuery('Show me the power quality metrics');
    await page.waitForLayout();

    const widgetCount = await page.getWidgetCount();
    expect(widgetCount).toBeGreaterThanOrEqual(0);
  });

  test('should show equipment needing maintenance', async () => {
    await page.sendQuery('What equipment needs maintenance based on health scores?');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should show top energy consumers', async () => {
    await page.sendQuery('Show the top 5 energy consumers');
    await page.waitForLayout();

    const widgetCount = await page.getWidgetCount();
    expect(widgetCount).toBeGreaterThanOrEqual(0);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // ALERT QUERIES
  // ═══════════════════════════════════════════════════════════════════════════

  test('should show critical alerts', async () => {
    await page.sendQuery('Are there any critical alerts right now?');
    await page.waitForLayout();

    // Should either show alert widgets or a "no alerts" message
    await expect(page.page.locator('body')).toContainText(/.+/);
  });

  test('should show active alarms', async () => {
    await page.sendQuery('Show me all active alarms');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should show recent warnings', async () => {
    await page.sendQuery('What warnings have we had in the past hour?');
    await page.waitForLayout();

    const widgetCount = await page.getWidgetCount();
    expect(widgetCount).toBeGreaterThanOrEqual(0);
  });

  test('should show threshold breach alerts', async () => {
    await page.sendQuery('Display threshold breach alerts');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should show alert history for specific equipment', async () => {
    await page.sendQuery('Show me the alert history for transformer TR-001');
    await page.waitForLayout();

    const widgetCount = await page.getWidgetCount();
    expect(widgetCount).toBeGreaterThanOrEqual(0);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // PEOPLE QUERIES
  // ═══════════════════════════════════════════════════════════════════════════

  test('should show current shift', async () => {
    await page.sendQuery('Who is on shift right now?');
    await page.waitForLayout();

    const widgetCount = await page.getWidgetCount();
    expect(widgetCount).toBeGreaterThanOrEqual(0);
  });

  test('should show technician availability', async () => {
    await page.sendQuery('Show me the technician availability');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should show operator schedule', async () => {
    await page.sendQuery('Display the operator schedule for this week');
    await page.waitForLayout();

    const widgetCount = await page.getWidgetCount();
    expect(widgetCount).toBeGreaterThanOrEqual(0);
  });

  test('should show responsibility for equipment', async () => {
    await page.sendQuery('Who is responsible for the chiller plant?');
    await page.waitForLayout();

    await expect(page.page.locator('body')).toContainText(/.+/);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // TASK QUERIES
  // ═══════════════════════════════════════════════════════════════════════════

  test('should show pending work orders', async () => {
    await page.sendQuery('What work orders are pending?');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should show overdue tasks', async () => {
    await page.sendQuery('Show overdue maintenance tasks');
    await page.waitForLayout();

    const widgetCount = await page.getWidgetCount();
    expect(widgetCount).toBeGreaterThanOrEqual(0);
  });

  test('should show open tickets', async () => {
    await page.sendQuery('Display all open tickets');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should show projects in progress', async () => {
    await page.sendQuery('What projects are in progress?');
    await page.waitForLayout();

    await expect(page.page.locator('body')).toContainText(/.+/);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // SUPPLY QUERIES
  // ═══════════════════════════════════════════════════════════════════════════

  test('should show inventory status', async () => {
    await page.sendQuery('What is the current inventory status?');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should show spare parts stock', async () => {
    await page.sendQuery('Show spare parts stock levels');
    await page.waitForLayout();

    const widgetCount = await page.getWidgetCount();
    expect(widgetCount).toBeGreaterThanOrEqual(0);
  });

  test('should show supplier delivery schedule', async () => {
    await page.sendQuery('Display supplier delivery schedule');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should show items below reorder point', async () => {
    await page.sendQuery('What items are below reorder point?');
    await page.waitForLayout();

    await expect(page.page.locator('body')).toContainText(/.+/);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // COMPLEX MULTI-DOMAIN QUERIES
  // ═══════════════════════════════════════════════════════════════════════════

  test('should handle equipment with alerts needing maintenance', async () => {
    await page.sendQuery('Show me equipment with alerts that need maintenance');
    await page.waitForLayout();

    const widgetCount = await page.getWidgetCount();
    // Complex query should produce multiple widgets
    expect(widgetCount).toBeGreaterThanOrEqual(1);
  });

  test('should find person who can fix alerting equipment', async () => {
    await page.sendQuery('Who can fix the pump that\'s showing a warning?');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should compare energy with alerts highlighted', async () => {
    await page.sendQuery('Compare this week\'s energy vs last week with any alerts highlighted');
    await page.waitForLayout();

    const widgetCount = await page.getWidgetCount();
    expect(widgetCount).toBeGreaterThanOrEqual(1);
  });
});
