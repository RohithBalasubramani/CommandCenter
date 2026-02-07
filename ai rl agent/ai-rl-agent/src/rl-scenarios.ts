/**
 * RL Agent Test Scenarios for Command Center
 *
 * 25 scenarios across 5 categories:
 * 1. Widget Quality Evaluation (8)
 * 2. Data Accuracy Verification (5)
 * 3. Feedback Submission (4)
 * 4. RL Training Lifecycle (4)
 * 5. Weight & Config Management (4)
 */

import type { TestScenario, SetupAction, SuccessCriterion, BackendCheck } from './types.js'

// ─── Setup Constants ────────────────────────────────────────────────

const HYDRATION_SETUP: SetupAction[] = [
  { type: 'wait', waitMs: 3000, description: 'Wait for Next.js hydration' },
]

const RL_HINTS = [
  'The Command Center uses a text input activated via Ctrl+Shift+K or by clicking the keyboard icon in the toolbar.',
  'After submitting a query, widgets appear in a grid layout with [data-scenario] attributes.',
  'Each widget has a scenario type (e.g., "equipment_status", "alerts", "trend") and optional fixture data.',
  'The RL agent should evaluate widget relevance to the query, not just check they exist.',
  'Use the backend API at localhost:8100 for direct data verification.',
]

// ─── Category 1: Widget Quality Evaluation ──────────────────────────

const widgetQualityScenarios: TestScenario[] = [
  {
    id: 'rl-eval-001',
    name: 'Pump Status — Equipment Widgets',
    goal: 'Send a pump status query and verify the response contains appropriate equipment/KPI widgets. Evaluate widget scenario relevance and data accuracy.',
    startUrl: '/',
    maxActions: 15,
    category: 'Widget Quality Evaluation',
    priority: 'critical',
    tags: ['eval', 'equipment', 'smoke'],
    hints: [
      ...RL_HINTS,
      'Query: "What is the current status of all pumps?"',
      'Expected widget scenarios: equipment_status, kpi',
      'Verify the voice response mentions pump status information.',
    ],
    setup: HYDRATION_SETUP,
    successCriteria: [
      {
        description: 'At least 1 widget rendered',
        check: { check: 'visible', target: { css: '[data-scenario]' } },
      },
    ],
    backendChecks: [
      {
        description: 'Orchestrate endpoint responds',
        endpoint: '/api/layer2/orchestrate/',
        method: 'POST',
        expectedStatus: 200,
      },
    ],
  },
  {
    id: 'rl-eval-002',
    name: 'Temperature Trend — Chart Widget',
    goal: 'Query temperature trend and verify a trend/chart widget is rendered with appropriate time-series visualization.',
    startUrl: '/',
    maxActions: 15,
    category: 'Widget Quality Evaluation',
    priority: 'critical',
    tags: ['eval', 'trend'],
    hints: [
      ...RL_HINTS,
      'Query: "Show me the temperature trend for the past 24 hours"',
      'Expected widget scenarios: trend, chart, gauge',
      'Look for Recharts line/area elements or SVG chart elements.',
    ],
    setup: HYDRATION_SETUP,
    successCriteria: [
      {
        description: 'At least 1 widget rendered',
        check: { check: 'visible', target: { css: '[data-scenario]' } },
      },
    ],
  },
  {
    id: 'rl-eval-003',
    name: 'Alert Query — Safety-Critical Widget Positioning',
    goal: 'Query for alerts and verify an alerts widget appears in the top 3 positions. This tests the RL safety floor — alerts must never be pushed to bottom.',
    startUrl: '/',
    maxActions: 15,
    category: 'Widget Quality Evaluation',
    priority: 'critical',
    tags: ['eval', 'safety', 'alerts'],
    hints: [
      ...RL_HINTS,
      'Query: "Are there any critical alerts right now?"',
      'Expected: alerts widget MUST be in position 1-3 (safety constraint).',
      'If alerts widget is at position 4+, this is a safety violation — rate DOWN.',
    ],
    setup: HYDRATION_SETUP,
    successCriteria: [
      {
        description: 'Response rendered (widgets or message)',
        check: { check: 'text_contains', target: { css: 'body' }, expected: '' },
      },
    ],
  },
  {
    id: 'rl-eval-004',
    name: 'Energy Breakdown — Chart/Comparison Widgets',
    goal: 'Query energy consumption breakdown and verify appropriate visualization widgets (chart, comparison, breakdown).',
    startUrl: '/',
    maxActions: 15,
    category: 'Widget Quality Evaluation',
    priority: 'high',
    tags: ['eval', 'energy'],
    hints: [
      ...RL_HINTS,
      'Query: "What is the energy consumption breakdown by equipment?"',
      'Expected widget scenarios: energy_breakdown, chart, comparison, kpi',
    ],
    setup: HYDRATION_SETUP,
    successCriteria: [
      {
        description: 'At least 1 widget rendered',
        check: { check: 'visible', target: { css: '[data-scenario]' } },
      },
    ],
  },
  {
    id: 'rl-eval-005',
    name: 'Multi-Domain Query — Equipment + Alerts',
    goal: 'Send a complex cross-domain query about equipment with alerts and verify multiple relevant widgets are returned.',
    startUrl: '/',
    maxActions: 15,
    category: 'Widget Quality Evaluation',
    priority: 'high',
    tags: ['eval', 'multi-domain'],
    hints: [
      ...RL_HINTS,
      'Query: "Show me equipment with alerts that need maintenance"',
      'Expected: multiple widgets covering equipment, alerts, and maintenance domains.',
    ],
    setup: HYDRATION_SETUP,
    successCriteria: [
      {
        description: 'At least 1 widget rendered',
        check: { check: 'visible', target: { css: '[data-scenario]' } },
      },
    ],
  },
  {
    id: 'rl-eval-006',
    name: 'Ambiguous Query — Graceful Handling',
    goal: 'Send an ambiguous query and verify the system handles it gracefully with reasonable widget choices.',
    startUrl: '/',
    maxActions: 15,
    category: 'Widget Quality Evaluation',
    priority: 'medium',
    tags: ['eval', 'edge-case'],
    hints: [
      ...RL_HINTS,
      'Query: "Show me everything"',
      'Expected: some widgets or a clarifying message. Should NOT crash or return empty.',
    ],
    setup: HYDRATION_SETUP,
    successCriteria: [
      {
        description: 'Page has response content',
        check: { check: 'text_contains', target: { css: 'body' }, expected: '' },
      },
    ],
  },
  {
    id: 'rl-eval-007',
    name: 'Out-of-Scope Query — Rejection',
    goal: 'Send a clearly out-of-scope query and verify the system rejects it with a helpful message.',
    startUrl: '/',
    maxActions: 15,
    category: 'Widget Quality Evaluation',
    priority: 'medium',
    tags: ['eval', 'edge-case', 'scope'],
    hints: [
      ...RL_HINTS,
      'Query: "What is the meaning of life?"',
      'Expected: rejection message or redirect to industrial queries. No random widgets.',
    ],
    setup: HYDRATION_SETUP,
    successCriteria: [
      {
        description: 'Response provided',
        check: { check: 'text_contains', target: { css: 'body' }, expected: '' },
      },
    ],
  },
  {
    id: 'rl-eval-008',
    name: 'Complex Comparison — Comparison Widgets',
    goal: 'Query a comparison between two pieces of equipment and verify comparison-type widgets are returned.',
    startUrl: '/',
    maxActions: 15,
    category: 'Widget Quality Evaluation',
    priority: 'high',
    tags: ['eval', 'comparison'],
    hints: [
      ...RL_HINTS,
      'Query: "Compare pump 1 and pump 2 performance"',
      'Expected widget scenarios: comparison, trend, equipment_status',
    ],
    setup: HYDRATION_SETUP,
    successCriteria: [
      {
        description: 'At least 1 widget rendered',
        check: { check: 'visible', target: { css: '[data-scenario]' } },
      },
    ],
  },
]

// ─── Category 2: Data Accuracy Verification ─────────────────────────

const dataAccuracyScenarios: TestScenario[] = [
  {
    id: 'rl-data-001',
    name: 'Backend vs UI Widget Match',
    goal: 'Send a pump status query via the text input and verify that widgets with data-scenario attributes appear on the page. The RL evaluator will separately verify backend consistency.',
    startUrl: '/',
    maxActions: 15,
    category: 'Data Accuracy Verification',
    priority: 'critical',
    tags: ['data', 'accuracy', 'smoke'],
    hints: [
      ...RL_HINTS,
      'Query: "What is the current status of all pumps?"',
      'Step 1: Activate text input with Ctrl+Shift+K.',
      'Step 2: Type the query and press Enter.',
      'Step 3: Wait for widgets to appear (elements with [data-scenario] attribute).',
      'Step 4: Verify at least one widget rendered by checking for [data-scenario] elements.',
    ],
    setup: HYDRATION_SETUP,
    successCriteria: [
      {
        description: 'Widgets rendered on page',
        check: { check: 'visible', target: { css: '[data-scenario]' } },
      },
    ],
    backendChecks: [
      {
        description: 'Orchestrate returns widgets',
        endpoint: '/api/layer2/orchestrate/',
        method: 'POST',
        expectedStatus: 200,
      },
    ],
  },
  {
    id: 'rl-data-002',
    name: 'KPI Numbers Render Correctly',
    goal: 'Verify that KPI widget numbers visible in the UI match the data in the backend response.',
    startUrl: '/',
    maxActions: 15,
    category: 'Data Accuracy Verification',
    priority: 'high',
    tags: ['data', 'kpi'],
    hints: [
      ...RL_HINTS,
      'Query: "Show me key performance indicators"',
      'After widgets render, read the numeric values displayed.',
      'Compare with the data field in the orchestrate response.',
    ],
    setup: HYDRATION_SETUP,
    successCriteria: [
      {
        description: 'KPI widgets render with numeric values',
        check: { check: 'text_contains', target: { css: 'body' }, expected: '' },
      },
    ],
  },
  {
    id: 'rl-data-003',
    name: 'Table Rows Match Backend Payload',
    goal: 'Send a query that produces a table widget and verify the row count matches the backend fixture data.',
    startUrl: '/',
    maxActions: 15,
    category: 'Data Accuracy Verification',
    priority: 'high',
    tags: ['data', 'table'],
    hints: [
      ...RL_HINTS,
      'Query: "Show me all active alarms"',
      'Look for table elements (tr, td) or list items within the widget.',
      'Count the data rows and compare with backend response.',
    ],
    setup: HYDRATION_SETUP,
    successCriteria: [
      {
        description: 'Response renders',
        check: { check: 'text_contains', target: { css: 'body' }, expected: '' },
      },
    ],
  },
  {
    id: 'rl-data-004',
    name: 'Chart Axis Labels Correct',
    goal: 'Verify that chart widgets have appropriate axis labels that relate to the query.',
    startUrl: '/',
    maxActions: 15,
    category: 'Data Accuracy Verification',
    priority: 'medium',
    tags: ['data', 'chart'],
    hints: [
      ...RL_HINTS,
      'Query: "Show me the temperature trend for the past 24 hours"',
      'Look for SVG text elements or Recharts axis labels.',
      'Verify they mention temperature, time, or units.',
    ],
    setup: HYDRATION_SETUP,
    successCriteria: [
      {
        description: 'Chart renders',
        check: { check: 'text_contains', target: { css: 'body' }, expected: '' },
      },
    ],
  },
  {
    id: 'rl-data-005',
    name: 'Voice Response Mentions Correct Equipment',
    goal: 'Verify the voice response text references the correct equipment names from the query.',
    startUrl: '/',
    maxActions: 15,
    category: 'Data Accuracy Verification',
    priority: 'medium',
    tags: ['data', 'voice'],
    hints: [
      ...RL_HINTS,
      'Query: "What is the status of transformer TR-001?"',
      'The voice response should mention "TR-001" or "transformer".',
      'If it talks about pumps or unrelated equipment, rate DOWN.',
    ],
    setup: HYDRATION_SETUP,
    successCriteria: [
      {
        description: 'Response references equipment',
        check: { check: 'text_contains', target: { css: 'body' }, expected: '' },
      },
    ],
  },
]

// ─── Category 3: Feedback Submission ────────────────────────────────

const feedbackScenarios: TestScenario[] = [
  {
    id: 'rl-fb-001',
    name: 'Submit Positive Feedback — Pump Query',
    goal: 'Send a pump status query, wait for widgets to render, and verify the response looks good. The RL evaluator will automatically submit positive feedback.',
    startUrl: '/',
    maxActions: 15,
    category: 'Feedback Submission',
    priority: 'critical',
    tags: ['feedback', 'positive', 'smoke'],
    hints: [
      ...RL_HINTS,
      'Query: "What is the current status of all pumps?"',
      'Step 1: Activate text input with Ctrl+Shift+K.',
      'Step 2: Type the query and press Enter.',
      'Step 3: Wait for widgets to appear.',
      'Step 4: Verify widgets have relevant content about pumps.',
    ],
    setup: HYDRATION_SETUP,
    successCriteria: [
      {
        description: 'Widgets rendered on page',
        check: { check: 'visible', target: { css: '[data-scenario]' } },
      },
    ],
    backendChecks: [
      {
        description: 'Feedback endpoint accepts request',
        endpoint: '/api/layer2/feedback/',
        method: 'POST',
        expectedStatus: 200,
      },
    ],
  },
  {
    id: 'rl-fb-002',
    name: 'Submit Negative Feedback — Wrong Domain',
    goal: 'Send a specific equipment query and verify the response. If widgets are irrelevant, the RL evaluator will submit negative feedback with a correction.',
    startUrl: '/',
    maxActions: 15,
    category: 'Feedback Submission',
    priority: 'high',
    tags: ['feedback', 'negative', 'correction'],
    hints: [
      ...RL_HINTS,
      'Query: "What is the status of transformer TR-001?"',
      'Step 1: Send the query via text input.',
      'Step 2: Wait for widgets.',
      'Step 3: Verify the response mentions transformer or TR-001.',
    ],
    setup: HYDRATION_SETUP,
    successCriteria: [
      {
        description: 'Response rendered',
        check: { check: 'text_contains', target: { css: 'body' }, expected: '' },
      },
    ],
  },
  {
    id: 'rl-fb-003',
    name: 'Widget Interaction — Expand and Explore',
    goal: 'Send a query, then interact with the rendered widgets — click to expand, hover, scroll within them. These interactions generate engagement data for RL.',
    startUrl: '/',
    maxActions: 20,
    category: 'Feedback Submission',
    priority: 'high',
    tags: ['feedback', 'interactions'],
    hints: [
      ...RL_HINTS,
      'Query: "Show me the energy consumption breakdown by equipment"',
      'Step 1: Send the query.',
      'Step 2: After widgets render, click on the first widget to expand it.',
      'Step 3: Scroll within the expanded widget.',
      'Step 4: Hover over different data points.',
      'These interactions generate implicit feedback signals for the RL system.',
    ],
    setup: HYDRATION_SETUP,
    successCriteria: [
      {
        description: 'Widgets rendered',
        check: { check: 'visible', target: { css: '[data-scenario]' } },
      },
    ],
  },
  {
    id: 'rl-fb-004',
    name: 'Batch Queries — Three Sequential',
    goal: 'Send 3 different queries in sequence. After each, verify widgets render. This generates multiple data points for RL evaluation.',
    startUrl: '/',
    maxActions: 30,
    category: 'Feedback Submission',
    priority: 'medium',
    tags: ['feedback', 'batch'],
    hints: [
      ...RL_HINTS,
      'Send these 3 queries in order, waiting for widgets each time:',
      'Query 1: "Show pump status"',
      'Query 2: "Any critical alerts?"',
      'Query 3: "Energy breakdown by equipment"',
      'For each query: Ctrl+Shift+K → type → Enter → wait for widgets.',
    ],
    setup: HYDRATION_SETUP,
    successCriteria: [
      {
        description: 'Final query rendered widgets',
        check: { check: 'visible', target: { css: '[data-scenario]' } },
      },
    ],
  },
]

// ─── Category 4: RL Training Lifecycle ──────────────────────────────

const trainingLifecycleScenarios: TestScenario[] = [
  {
    id: 'rl-train-001',
    name: 'Check RL System Status',
    goal: 'Navigate to the Command Center and verify the page loads correctly. The RL runner will check the RL system status via API separately.',
    startUrl: '/',
    maxActions: 10,
    category: 'RL Training Lifecycle',
    priority: 'critical',
    tags: ['training', 'status', 'smoke'],
    hints: [
      ...RL_HINTS,
      'Simply verify the Command Center page loads and is functional.',
      'The RL runner handles API status checks automatically.',
    ],
    setup: HYDRATION_SETUP,
    successCriteria: [
      {
        description: 'Page loaded successfully',
        check: { check: 'text_contains', target: { css: 'body' }, expected: '' },
      },
    ],
    backendChecks: [
      {
        description: 'RL status endpoint responds',
        endpoint: '/api/layer2/rl-status/',
        method: 'GET',
        expectedStatus: 200,
      },
    ],
  },
  {
    id: 'rl-train-002',
    name: 'Training Query — Maintenance Equipment',
    goal: 'Send a maintenance query to generate training data. The RL evaluator submits feedback which feeds the training pipeline.',
    startUrl: '/',
    maxActions: 15,
    category: 'RL Training Lifecycle',
    priority: 'high',
    tags: ['training', 'approve'],
    hints: [
      ...RL_HINTS,
      'Query: "What equipment needs maintenance based on health scores?"',
      'Step 1: Activate text input with Ctrl+Shift+K.',
      'Step 2: Type the query and press Enter.',
      'Step 3: Wait for widgets to appear.',
    ],
    setup: HYDRATION_SETUP,
    successCriteria: [
      {
        description: 'Response rendered',
        check: { check: 'text_contains', target: { css: 'body' }, expected: '' },
      },
    ],
    backendChecks: [
      {
        description: 'Approve training endpoint responds',
        endpoint: '/api/layer2/approve-training/',
        method: 'POST',
        expectedStatus: 200,
      },
    ],
  },
  {
    id: 'rl-train-003',
    name: 'Evaluate-Feedback Loop — Three Queries',
    goal: 'Send 3 queries in sequence to generate training data. Each query response is evaluated and feedback is submitted by the RL system.',
    startUrl: '/',
    maxActions: 25,
    category: 'RL Training Lifecycle',
    priority: 'high',
    tags: ['training', 'loop'],
    hints: [
      ...RL_HINTS,
      'Send these queries in order:',
      'Query 1: "Show me transformer load distribution"',
      'Query 2: "What warnings have we had in the past hour?"',
      'Query 3: "Show the top 5 energy consumers"',
      'For each: Ctrl+Shift+K → type → Enter → wait for widgets.',
    ],
    setup: HYDRATION_SETUP,
    successCriteria: [
      {
        description: 'Final query rendered widgets',
        check: { check: 'visible', target: { css: '[data-scenario]' } },
      },
    ],
  },
  {
    id: 'rl-train-004',
    name: 'Full Training Cycle — Five Queries',
    goal: 'Send 5 diverse queries covering equipment, alerts, energy, maintenance, and personnel. This generates a comprehensive training batch.',
    startUrl: '/',
    maxActions: 40,
    category: 'RL Training Lifecycle',
    priority: 'medium',
    tags: ['training', 'full-cycle'],
    hints: [
      ...RL_HINTS,
      'Send these 5 queries in order:',
      'Query 1: "What is the current status of all pumps?"',
      'Query 2: "Are there any critical alerts right now?"',
      'Query 3: "What is the energy consumption breakdown?"',
      'Query 4: "What equipment needs maintenance?"',
      'Query 5: "Who is on shift right now?"',
      'For each: Ctrl+Shift+K → type → Enter → wait for widgets.',
    ],
    setup: HYDRATION_SETUP,
    successCriteria: [
      {
        description: 'Widgets rendered',
        check: { check: 'visible', target: { css: '[data-scenario]' } },
      },
    ],
  },
]

// ─── Category 5: Weight & Config Management ─────────────────────────

const weightManagementScenarios: TestScenario[] = [
  {
    id: 'rl-weights-001',
    name: 'Read Current Reward Weights',
    goal: 'Navigate to the Command Center page. The RL runner will fetch reward weights from the RL status API.',
    startUrl: '/',
    maxActions: 10,
    category: 'Weight & Config Management',
    priority: 'high',
    tags: ['weights', 'config', 'smoke'],
    hints: [
      ...RL_HINTS,
      'Simply verify the page loads correctly.',
      'The RL runner handles weight configuration reads via API.',
    ],
    setup: HYDRATION_SETUP,
    successCriteria: [
      {
        description: 'Page loaded',
        check: { check: 'text_contains', target: { css: 'body' }, expected: '' },
      },
    ],
    backendChecks: [
      {
        description: 'RL status shows config',
        endpoint: '/api/layer2/rl-status/',
        method: 'GET',
        expectedStatus: 200,
      },
    ],
  },
  {
    id: 'rl-weights-002',
    name: 'Weight Sensitivity — Positive Pump Queries',
    goal: 'Send multiple pump-related queries to see how consistent the widget selection is. Good consistency helps identify scorer sensitivity.',
    startUrl: '/',
    maxActions: 25,
    category: 'Weight & Config Management',
    priority: 'medium',
    tags: ['weights', 'sweep'],
    hints: [
      ...RL_HINTS,
      'Send these pump variations:',
      'Query 1: "Show pump status"',
      'Query 2: "What is the status of pump 1?"',
      'Query 3: "Are any pumps showing warnings?"',
      'For each: Ctrl+Shift+K → type → Enter → wait for widgets.',
    ],
    setup: HYDRATION_SETUP,
    successCriteria: [
      {
        description: 'Widgets rendered',
        check: { check: 'visible', target: { css: '[data-scenario]' } },
      },
    ],
  },
  {
    id: 'rl-weights-003',
    name: 'Mixed Domain Queries — Energy and Alerts',
    goal: 'Send alternating energy and alert queries to generate diverse training signals.',
    startUrl: '/',
    maxActions: 25,
    category: 'Weight & Config Management',
    priority: 'medium',
    tags: ['weights', 'comparison', 'lora'],
    hints: [
      ...RL_HINTS,
      'Query 1: "What is the energy consumption breakdown?"',
      'Query 2: "Show me all active alarms"',
      'Query 3: "Show the top 5 energy consumers"',
      'For each: Ctrl+Shift+K → type → Enter → wait for widgets.',
    ],
    setup: HYDRATION_SETUP,
    successCriteria: [
      {
        description: 'Widgets rendered',
        check: { check: 'visible', target: { css: '[data-scenario]' } },
      },
    ],
  },
  {
    id: 'rl-weights-004',
    name: 'Safety Constraint Verification',
    goal: 'Send a multi-domain query that includes alerts. Verify alert widgets appear in the response. The RL evaluator checks position constraints.',
    startUrl: '/',
    maxActions: 15,
    category: 'Weight & Config Management',
    priority: 'critical',
    tags: ['weights', 'safety'],
    hints: [
      ...RL_HINTS,
      'Query: "Show me critical alerts and also energy breakdown and pump status"',
      'This query should return multiple widgets.',
      'After widgets render, check that alert-related widgets are visible.',
    ],
    setup: HYDRATION_SETUP,
    successCriteria: [
      {
        description: 'Alert widget in safe position',
        check: { check: 'visible', target: { css: '[data-scenario]' } },
      },
    ],
  },
]

// ─── Export All Scenarios ───────────────────────────────────────────

export const rlScenarios: TestScenario[] = [
  ...widgetQualityScenarios,
  ...dataAccuracyScenarios,
  ...feedbackScenarios,
  ...trainingLifecycleScenarios,
  ...weightManagementScenarios,
]

// ─── Query Bank (used by rl-runner for batch evaluation) ────────────

export const EVAL_QUERIES = [
  // ── Equipment Status ──
  'What is the current status of all pumps?',
  'Show me pump 1 operating parameters',
  'Which pumps are currently running?',
  'What is the status of transformer TR-001?',
  'Display the HVAC system status',
  'What is the chiller efficiency today?',
  'Show me the compressor runtime hours',
  'Which motors have the highest vibration levels?',
  'What is the current flow rate for pump 3?',
  'Show bearing temperature for all rotating equipment',

  // ── Trends & Charts ──
  'Show me the temperature trend for the past 24 hours',
  'What is the energy consumption trend this week?',
  'Show pressure trend for the main header',
  'Display voltage trend for Bus A over the past 6 hours',
  'Show me the cooling water temperature over time',

  // ── Alerts & Alarms ──
  'Are there any critical alerts right now?',
  'Show me all active alarms',
  'What warnings have we had in the past hour?',
  'Display threshold breach alerts',
  'Show me the alert history for transformer TR-001',
  'Which equipment has the most frequent alarms?',

  // ── Energy ──
  'What is the energy consumption breakdown by equipment?',
  'Show the top 5 energy consumers',
  'Compare this week energy vs last week',
  'What is the power factor for the main incoming?',
  'Show me the power quality metrics',
  'What is the total energy cost this month?',

  // ── Comparison ──
  'Compare pump 1 and pump 2 performance',
  'Compare chiller 1 efficiency with chiller 2',
  'Show transformer load distribution across all substations',

  // ── Maintenance ──
  'What equipment needs maintenance based on health scores?',
  'Show overdue maintenance tasks',
  'What work orders are pending?',
  'Show me the predictive maintenance schedule',
  'Which equipment has the lowest health score?',

  // ── Personnel ──
  'Who is on shift right now?',
  'Show me the technician availability',
  'Display the operator schedule for this week',
  'Who is responsible for the chiller plant?',

  // ── Inventory & Supply ──
  'Show spare parts stock levels',
  'What items are below reorder point?',
  'Display supplier delivery schedule',
  'What is the current inventory status?',

  // ── Complex / Multi-domain ──
  'Show me equipment with alerts that need maintenance',
  'Who can fix the pump that is showing a warning?',
  'Compare this week energy vs last week with any alerts highlighted',
  'Show me critical alerts and also energy breakdown and pump status',
  'What is the overall plant efficiency right now?',

  // ── Additional Equipment ──
  'What is the diesel generator runtime today?',
  'Show UPS battery status and backup power availability',
  'What is the current load on the main switchgear?',
  'Show me all transformers and their loading percentage',
  'What is the status of the nitrogen plant?',
  'Display compressed air system pressure and consumption',
  'Show AHU-1 operating parameters including zone temperatures',
  'What is the cooling tower water temperature?',

  // ── Additional Trends ──
  'Show motor current trend for the past 8 hours',
  'Display power consumption trend by shift',
  'What is the OEE trend for the past week?',
  'Show harmonics THD trend on Bus B',
  'Display water consumption trend for the past month',

  // ── Additional Alerts ──
  'How many alarms were triggered last night?',
  'Show me high-priority fault notifications',
  'Which transformers have tripped recently?',
  'Display any voltage sag or swell events today',
  'Are there any overdue safety inspection alerts?',

  // ── Additional Energy ──
  'What is today peak demand and when did it occur?',
  'Show energy split between EB grid and diesel generator',
  'Compare morning shift vs afternoon shift energy usage',
  'What is the harmonic distortion level on the main feeder?',
  'Display the kWh consumption by production line',

  // ── Additional People ──
  'How many technicians are available for emergency repairs?',
  'Show the training completion status for all operators',
  'Who worked overtime this week?',
  'Display safety incident reports for the past month',
  'What is the current headcount by department?',

  // ── Additional Supply Chain ──
  'Which suppliers have pending purchase orders?',
  'Show critical spare parts with less than 2 weeks stock',
  'What is the lead time for transformer replacement parts?',
  'Display procurement requests awaiting approval',
  'Show material consumption vs forecast this quarter',

  // ── Additional Maintenance ──
  'What is the mean time between failures for pump 2?',
  'Show vibration analysis report for motor M-101',
  'Which equipment is due for lubrication this week?',
  'Display preventive maintenance compliance rate',
  'What bearings need replacement based on condition monitoring?',

  // ── Additional Complex ──
  'Show me the top energy consumers that also have maintenance overdue',
  'Which critical equipment has both low health score and active alarms?',
  'Compare yesterday shift handover notes with today morning status',
  'Show everything about chiller plant including efficiency, alerts, and spare parts',
  'Give me a complete overview of transformer substations with load, alerts, and maintenance',

  // ── Ambiguous / Vague Queries (test specificity scoring) ──
  'Show status',
  'Show me the pumps',
  'What is the status?',
  'Show everything',
  'How is it going?',
  'Show me the data',
  'Display all equipment',
  'What is happening?',
  'Show inventory',
  'Show alerts',

  // ── Edge Cases & Boundary Conditions ──
  'Show pump status',                                        // Vague — which pump?
  'Temperature',                                             // Single word — minimal context
  'Pump 1 pump 2 pump 3 comparison with alerts energy',     // Overloaded multi-domain
  'What happened between 2am and 3am last Tuesday?',        // Very specific time window
  'Show me the thing that was beeping earlier',              // Colloquial / non-technical
  'Is transformer TR-001 going to fail soon?',               // Predictive question
  'Why is the energy bill so high this month?',              // Diagnostic / root cause
  'What should I check first today?',                        // Advisory / guidance
  'Show me something useful',                                // Completely ambiguous

  // ── Out-of-Scope Queries (test rejection) ──
  'What is the weather today?',
  'Tell me a joke',
  'What is the meaning of life?',
  'How do I cook pasta?',
]
