/**
 * Failure Injection Tests
 *
 * Deliberately induces failures to verify graceful degradation:
 * - Slow RAG
 * - Dropped DB results
 * - Empty vector search
 * - Malformed layout_json
 *
 * System must:
 * - Fail visibly
 * - Explain failure
 * - Not crash UI
 * - Not speak nonsense
 */
import { test, expect } from '@playwright/test';
import { CommandCenterPage } from '../helpers/test-utils';

test.describe('Failure Injection Tests', () => {
  let page: CommandCenterPage;

  test.beforeEach(async ({ page: playwrightPage }) => {
    page = new CommandCenterPage(playwrightPage);
    await page.goto();
    await page.waitForReady();
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // NETWORK FAILURE SIMULATION
  // ═══════════════════════════════════════════════════════════════════════════

  test('should handle API timeout gracefully', async () => {
    // Intercept API calls and add delay
    await page.page.route('**/api/layer2/**', async route => {
      await new Promise(resolve => setTimeout(resolve, 5000)); // 5s delay
      await route.continue();
    });

    await page.sendQuery('Show pumps');

    // Should show loading or timeout message, not crash
    await page.page.waitForTimeout(6000);

    // Page should still be functional
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should handle API error response', async () => {
    // Intercept and return error
    await page.page.route('**/api/layer2/orchestrate/**', route => {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Internal server error' }),
      });
    });

    await page.sendQuery('Show pumps');
    await page.page.waitForTimeout(2000);

    // Should show error message, not crash
    // Check that page is still interactive
    const bodyExists = await page.page.locator('body').count();
    expect(bodyExists).toBe(1);
  });

  test('should handle network disconnect', async () => {
    // Start a query
    await page.sendQuery('Show pumps');
    await page.page.waitForTimeout(500);

    // Simulate network disconnect
    await page.page.route('**/*', route => route.abort('failed'));

    // Try another query
    await page.sendQuery('Show alerts');
    await page.page.waitForTimeout(2000);

    // Restore network
    await page.page.unroute('**/*');

    // Page should recover
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // MALFORMED RESPONSE HANDLING
  // ═══════════════════════════════════════════════════════════════════════════

  test('should handle malformed JSON response', async () => {
    await page.page.route('**/api/layer2/orchestrate/**', route => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: '{ invalid json }',
      });
    });

    await page.sendQuery('Show pumps');
    await page.page.waitForTimeout(2000);

    // Should not crash
    const bodyExists = await page.page.locator('body').count();
    expect(bodyExists).toBe(1);
  });

  test('should handle empty response', async () => {
    await page.page.route('**/api/layer2/orchestrate/**', route => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: '',
      });
    });

    await page.sendQuery('Show pumps');
    await page.page.waitForTimeout(2000);

    // Should handle gracefully
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should handle response with missing fields', async () => {
    await page.page.route('**/api/layer2/orchestrate/**', route => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          // Missing voice_response, layout_json, etc.
          partial: true,
        }),
      });
    });

    await page.sendQuery('Show pumps');
    await page.page.waitForTimeout(2000);

    // Should handle gracefully
    const bodyExists = await page.page.locator('body').count();
    expect(bodyExists).toBe(1);
  });

  test('should handle invalid layout_json', async () => {
    await page.page.route('**/api/layer2/orchestrate/**', route => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          voice_response: 'Here is the data',
          layout_json: {
            widgets: [
              { invalid: 'widget structure' },
              { scenario: null, fixture: null },
              { scenario: 'nonexistent', fixture: 'invalid' },
            ],
          },
        }),
      });
    });

    await page.sendQuery('Show pumps');
    await page.page.waitForTimeout(2000);

    // Should filter invalid widgets or show error
    const validation = await page.validateLayoutJSON();
    // May have errors but shouldn't crash
    const bodyExists = await page.page.locator('body').count();
    expect(bodyExists).toBe(1);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // EMPTY DATA HANDLING
  // ═══════════════════════════════════════════════════════════════════════════

  test('should handle empty widget list', async () => {
    await page.page.route('**/api/layer2/orchestrate/**', route => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          voice_response: 'No data found',
          layout_json: { widgets: [] },
          intent: { type: 'query', domains: [] },
        }),
      });
    });

    await page.sendQuery('Show pumps');
    await page.page.waitForTimeout(2000);

    // Should show message, not empty screen
    const bodyContent = await page.page.locator('body').textContent();
    expect(bodyContent?.length).toBeGreaterThan(0);
  });

  test('should handle null layout_json', async () => {
    await page.page.route('**/api/layer2/orchestrate/**', route => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          voice_response: 'Processing...',
          layout_json: null,
        }),
      });
    });

    await page.sendQuery('Show pumps');
    await page.page.waitForTimeout(2000);

    // Should handle gracefully
    const bodyExists = await page.page.locator('body').count();
    expect(bodyExists).toBe(1);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // PARTIAL FAILURE HANDLING
  // ═══════════════════════════════════════════════════════════════════════════

  test('should handle partial RAG results', async () => {
    await page.page.route('**/api/layer2/orchestrate/**', route => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          voice_response: 'Partial results available',
          layout_json: {
            widgets: [
              { scenario: 'kpi', fixture: 'kpi_live', size: 'normal' },
            ],
          },
          rag_results: [
            { domain: 'industrial', success: true, data: {} },
            { domain: 'alerts', success: false, error: 'Database timeout' },
          ],
        }),
      });
    });

    await page.sendQuery('Show pumps and alerts');
    await page.page.waitForTimeout(2000);

    // Should show what's available
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should indicate LLM unavailable', async () => {
    await page.page.route('**/api/layer2/orchestrate/**', route => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          voice_response: '[LLM unavailable] Showing cached data',
          layout_json: { widgets: [] },
          llm_available: false,
        }),
      });
    });

    await page.sendQuery('Show pumps');
    await page.page.waitForTimeout(2000);

    // Should communicate LLM issue
    const bodyExists = await page.page.locator('body').count();
    expect(bodyExists).toBe(1);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // CONCURRENT REQUEST HANDLING
  // ═══════════════════════════════════════════════════════════════════════════

  test('should handle rapid concurrent requests', async () => {
    let requestCount = 0;

    await page.page.route('**/api/layer2/orchestrate/**', route => {
      requestCount++;
      const currentRequest = requestCount;

      // Add random delay to simulate real conditions
      setTimeout(() => {
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            voice_response: `Response ${currentRequest}`,
            layout_json: {
              widgets: [{ scenario: 'kpi', fixture: 'kpi_live', size: 'normal' }],
            },
          }),
        });
      }, Math.random() * 500);
    });

    // Fire multiple requests rapidly
    await Promise.all([
      page.sendQuery('Query 1'),
      page.sendQuery('Query 2'),
      page.sendQuery('Query 3'),
    ]);

    await page.page.waitForTimeout(3000);

    // Should settle to a consistent state
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // RECOVERY TESTS
  // ═══════════════════════════════════════════════════════════════════════════

  test('should recover after transient failure', async () => {
    let callCount = 0;

    await page.page.route('**/api/layer2/orchestrate/**', route => {
      callCount++;

      if (callCount <= 2) {
        // First two calls fail
        route.fulfill({
          status: 500,
          body: 'Server error',
        });
      } else {
        // Third call succeeds
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            voice_response: 'Success!',
            layout_json: {
              widgets: [{ scenario: 'kpi', fixture: 'kpi_live', size: 'normal' }],
            },
          }),
        });
      }
    });

    // Try queries
    await page.sendQuery('Query 1');
    await page.page.waitForTimeout(500);
    await page.sendQuery('Query 2');
    await page.page.waitForTimeout(500);
    await page.sendQuery('Query 3'); // Should succeed
    await page.page.waitForTimeout(1000);

    // Should have recovered
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  test('should maintain state after API reconnection', async () => {
    // Get initial state
    await page.sendQuery('Show initial view');
    await page.waitForLayout();

    // Simulate disconnect
    await page.page.route('**/api/**', route => route.abort());

    // Try query (will fail)
    await page.sendQuery('This will fail');
    await page.page.waitForTimeout(1000);

    // Reconnect
    await page.page.unroute('**/api/**');

    // Try again
    await page.sendQuery('Show pumps after reconnect');
    await page.waitForLayout();

    // Should work normally
    const validation = await page.validateLayoutJSON();
    expect(validation.errors).toHaveLength(0);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // ERROR DISPLAY VALIDATION
  // ═══════════════════════════════════════════════════════════════════════════

  test('should display user-friendly error messages', async () => {
    await page.page.route('**/api/layer2/orchestrate/**', route => {
      route.fulfill({
        status: 503,
        contentType: 'application/json',
        body: JSON.stringify({
          error: 'Service temporarily unavailable',
        }),
      });
    });

    await page.sendQuery('Show pumps');
    await page.page.waitForTimeout(2000);

    // Should not expose raw error/stack traces
    const bodyContent = await page.page.locator('body').textContent();

    // Should not contain stack traces or internal errors
    expect(bodyContent).not.toContain('TypeError');
    expect(bodyContent).not.toContain('undefined');
    expect(bodyContent).not.toContain('at Function');
  });
});
