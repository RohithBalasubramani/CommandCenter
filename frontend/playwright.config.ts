import { defineConfig, devices } from '@playwright/test';

/**
 * Command Center E2E Playwright Configuration
 *
 * Mega Test Suite for Pre-Release Validation
 */
export default defineConfig({
  testDir: './e2e',

  // Run tests in sequence for stability
  fullyParallel: false,

  // Fail fast on CI
  forbidOnly: !!process.env.CI,

  // Retry failed tests
  retries: process.env.CI ? 2 : 1,

  // Single worker for deterministic state
  workers: 1,

  // Rich reporting
  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['json', { outputFile: 'e2e-results.json' }],
    ['list'],
  ],

  // Global settings
  use: {
    // Base URL for the frontend
    baseURL: process.env.FRONTEND_URL || 'http://localhost:3100',

    // Capture traces on failure
    trace: 'on-first-retry',

    // Screenshot on failure
    screenshot: 'only-on-failure',

    // Video recording for all tests
    video: 'on',

    // Viewport
    viewport: { width: 1920, height: 1080 },

    // Timeout per action
    actionTimeout: 30000,

    // Navigation timeout
    navigationTimeout: 60000,
  },

  // Global timeout per test
  timeout: 120000,

  // Expect timeout
  expect: {
    timeout: 15000,
  },

  // Test projects
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'],
        },
      },
    },
    {
      name: 'firefox',
      use: {
        ...devices['Desktop Firefox'],
        launchOptions: {
          firefoxUserPrefs: {
            'media.navigator.streams.fake': true,
            'media.navigator.permission.disabled': true,
          },
        },
      },
    },
  ],

  // Output folder for artifacts
  outputDir: 'e2e-artifacts',

  // Web server configuration (starts frontend if not running)
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:3100',
    reuseExistingServer: !process.env.CI,
    timeout: 120000,
  },
});
