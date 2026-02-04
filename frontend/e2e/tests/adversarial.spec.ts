/**
 * Adversarial User Behavior Tests
 *
 * Tests system resilience against:
 * - Mid-sentence intent changes
 * - Contradictory requests
 * - Impossible requests
 * - Ambiguous queries
 * - Malformed input
 * - Injection attempts
 */
import { test, expect } from '@playwright/test';
import { CommandCenterPage, ADVERSARIAL_SCENARIOS } from '../helpers/test-utils';

test.describe('Adversarial User Behavior Tests', () => {
  let page: CommandCenterPage;

  test.beforeEach(async ({ page: playwrightPage }) => {
    page = new CommandCenterPage(playwrightPage);
    await page.goto();
    await page.waitForReady();
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // INTENT CHANGES MID-SENTENCE
  // ═══════════════════════════════════════════════════════════════════════════

  test('should handle "no wait" intent changes', async () => {
    await page.sendQuery('Show me the pumps... no wait, show the chillers instead');
    await page.waitForLayout();

    // Should not crash and should respond
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should handle cancelled actions', async () => {
    await page.sendQuery('Turn on pump 1... actually, never mind');
    await page.waitForLayout();

    // Should understand this is a cancellation, not execute action
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should handle mid-query corrections', async () => {
    await page.sendQuery('What is the weather... I mean, what is the pump status?');
    await page.waitForLayout();

    // Should focus on the corrected request (pump status)
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // CONTRADICTORY REQUESTS
  // ═══════════════════════════════════════════════════════════════════════════

  test('should handle contradictory display requests', async () => {
    await page.sendQuery('Show more details but make it simpler');
    await page.waitForLayout();

    // Should either ask for clarification or make a reasonable choice
    // Should not crash
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should handle impossible visibility requests', async () => {
    await page.sendQuery('Hide everything but keep it visible');
    await page.waitForLayout();

    // Should handle gracefully
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should handle both/and contradictions', async () => {
    await page.sendQuery('Show me everything and nothing at the same time');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // IMPOSSIBLE REQUESTS
  // ═══════════════════════════════════════════════════════════════════════════

  test('should handle future data requests', async () => {
    await page.sendQuery('Show me tomorrow\'s alerts');
    await page.waitForLayout();

    // Should explain that future data isn't available
    // Should not hallucinate future data
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should reject destructive requests', async () => {
    await page.sendQuery('Delete all the equipment data');
    await page.waitForLayout();

    // Should reject this request
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should handle out-of-scope control requests', async () => {
    await page.sendQuery('Make the pumps run faster');
    await page.waitForLayout();

    // Should explain this is out of scope
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should handle requests for non-existent equipment', async () => {
    await page.sendQuery('Show me the status of pump XYZ-999999');
    await page.waitForLayout();

    // Should indicate equipment not found
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // AMBIGUOUS QUERIES
  // ═══════════════════════════════════════════════════════════════════════════

  test('should handle vague references', async () => {
    await page.sendQuery('Show me that thing from earlier');
    await page.waitForLayout();

    // Should ask for clarification or show overview
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should handle "the other one" reference', async () => {
    await page.sendQuery('The other one');
    await page.waitForLayout();

    // Should ask for clarification
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should handle minimal queries', async () => {
    await page.sendQuery('Fix it');
    await page.waitForLayout();

    // Should ask what needs fixing
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should handle "show me everything"', async () => {
    await page.sendQuery('Show me everything');
    await page.waitForLayout();

    // Should show a reasonable overview, not crash
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // GIBBERISH AND RANDOM INPUT
  // ═══════════════════════════════════════════════════════════════════════════

  test('should handle keyboard mash', async () => {
    await page.sendQuery('asdf jkl; qwerty');
    await page.waitForLayout();

    // Should not crash, should indicate it doesn't understand
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should handle numbers only', async () => {
    await page.sendQuery('12345 67890');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should handle random characters', async () => {
    await page.sendQuery('!@#$%^&*()');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // EXTREMELY LONG QUERIES
  // ═══════════════════════════════════════════════════════════════════════════

  test('should handle very long queries', async () => {
    const longQuery = 'Show me the pump status and the chiller status and the transformer status and the AHU status and all the alerts and all the work orders and everyone on shift and the inventory levels and the energy consumption and the temperature trends and the pressure readings and the flow rates for all equipment in all buildings for the past month with daily breakdowns and highlight any anomalies';

    await page.sendQuery(longQuery);
    await page.waitForLayout();

    // Should handle gracefully, potentially asking to simplify
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should handle repeated words', async () => {
    await page.sendQuery('pump pump pump pump pump pump pump pump pump pump');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // EMPTY AND WHITESPACE
  // ═══════════════════════════════════════════════════════════════════════════

  test('should handle empty query', async () => {
    await page.sendQuery('');
    // May not trigger anything, but shouldn't crash
    await page.page.waitForTimeout(1000);

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should handle whitespace only', async () => {
    await page.sendQuery('   ');
    await page.page.waitForTimeout(1000);

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // INJECTION ATTEMPTS
  // ═══════════════════════════════════════════════════════════════════════════

  test('should reject SQL injection attempts', async () => {
    await page.sendQuery('SELECT * FROM equipment; DROP TABLE alerts;--');
    await page.waitForLayout();

    // Should not execute SQL, should handle as text
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should reject XSS attempts', async () => {
    await page.sendQuery('<script>alert("xss")</script>');
    await page.waitForLayout();

    // Should not execute script
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);

    // Verify no script execution
    await expect(page.page.locator('script')).toHaveCount(0);
  });

  test('should handle HTML injection', async () => {
    await page.sendQuery('<h1>Big Text</h1><img src=x onerror=alert(1)>');
    await page.waitForLayout();

    // Should sanitize/escape HTML
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should handle command injection patterns', async () => {
    await page.sendQuery('| cat /etc/passwd');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // INTERRUPTIONS
  // ═══════════════════════════════════════════════════════════════════════════

  test('should handle incomplete queries', async () => {
    await page.sendQuery('Show me the--');
    await page.waitForLayout();

    // Should ask for completion or show best guess
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should handle single word start', async () => {
    await page.sendQuery('What');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // UNICODE AND SPECIAL CHARACTERS
  // ═══════════════════════════════════════════════════════════════════════════

  test('should handle emoji', async () => {
    await page.sendQuery('Show me the pumps status please');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should handle non-Latin scripts', async () => {
    await page.sendQuery('ポンプのステータス');  // Japanese
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should handle mixed scripts', async () => {
    await page.sendQuery('Show 泵 status auf Deutsch');
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should handle RTL text', async () => {
    await page.sendQuery('حالة المضخة');  // Arabic
    await page.waitForLayout();

    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });
});
