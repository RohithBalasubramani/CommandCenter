/**
 * RL Agent Configuration for Command Center
 *
 * Extends the CC QA agent config with RL-specific settings
 * for evaluation thresholds, API endpoints, and training parameters.
 */

import type { AgentConfig, RunnerConfig } from './config.js'

// ─── RL-Specific Configuration ──────────────────────────────────────

export interface RLConfig {
  apiBaseUrl: string
  feedbackApiKey: string
  evaluationThreshold: number
  latencyBudgetMs: number
  maxEvalRetries: number
  batchSize: number
  cooldownMs: number
}

// ─── Agent Config ───────────────────────────────────────────────────

export const rlAgentConfig: AgentConfig = {
  appName: 'Command Center RL Agent',
  baseUrl: 'http://localhost:3100',
  apiBaseUrl: 'http://localhost:8100',
  uiFramework: 'tailwind',

  customSelectors: {
    buttons: ['button', '[role="button"]', '.btn'],
    textInputs: ['input[type="text"]', 'textarea', '[contenteditable]'],
    selects: ['select', '[role="combobox"]', '[role="listbox"]'],
    checkboxes: ['input[type="checkbox"]', '[role="checkbox"]'],
    links: ['a[href]', '[role="link"]'],
    tabs: ['[role="tab"]'],
    menuItems: ['[role="menuitem"]'],
    toasts: ['.toast', '[role="alert"]', '.notification'],
    errors: ['.error', '.text-red-500', '[role="alert"]'],
    modals: ['dialog', '[role="dialog"]', '.modal'],
    loading: ['.loading', '.spinner', '[aria-busy="true"]'],
  },

  llm: {
    model: 'sonnet',
    useVision: true,
    maxTokens: 1024,
    timeout: 900_000,
    temperature: 0.1,
  },

  actionDelay: 1500,
  actionTimeout: 15_000,
  pageLoadTimeout: 60_000,
  stabilityTimeout: 8_000,
  screenshotEveryStep: true,
  verifyAfterAction: true,
  waitForStable: true,
  detectActionReplay: true,
  maxRepeatedActions: 4,
  evidenceDir: './evidence/rl',
}

export const rlRunnerConfig: RunnerConfig = {
  agent: rlAgentConfig,
  maxRetries: 1,
  parallelCount: 1,
  browser: 'chromium',
  headed: false,
  recordTrace: true,
  recordVideo: false,
  viewport: { width: 1440, height: 900 },
  permissions: ['microphone'],
}

// ─── RL Evaluation Settings ─────────────────────────────────────────

export const rlConfig: RLConfig = {
  apiBaseUrl: process.env.RL_API_BASE_URL || 'http://localhost:8100',
  feedbackApiKey: process.env.FEEDBACK_API_KEY || '',
  evaluationThreshold: 0.6,
  latencyBudgetMs: 8_000,
  maxEvalRetries: 2,
  batchSize: 10,
  cooldownMs: 2_000,
}

// ─── Scoring Weights ────────────────────────────────────────────────

export const EVAL_WEIGHTS = {
  widgetCountMatch: 0.10,
  scenarioRelevance: 0.30,
  dataAccuracy: 0.20,
  responseQuality: 0.15,
  latencyScore: 0.10,
  querySpecificity: 0.15,
} as const

// ─── Expected Widget Mappings ───────────────────────────────────────
// Maps query intent keywords to expected widget scenarios.
// Used by the evaluator to compute scenario relevance.

export const EXPECTED_SCENARIOS: Record<string, string[]> = {
  // Equipment
  pump: ['equipment_status', 'kpi', 'trend', 'health_status'],
  motor: ['equipment_status', 'kpi', 'trend', 'vibration'],
  transformer: ['equipment_status', 'kpi', 'distribution', 'load'],
  chiller: ['equipment_status', 'kpi', 'efficiency', 'trend'],
  compressor: ['equipment_status', 'kpi', 'trend'],
  hvac: ['equipment_status', 'kpi', 'zone_temperature'],
  ahu: ['equipment_status', 'kpi', 'zone_temperature'],
  generator: ['equipment_status', 'kpi', 'runtime'],
  ups: ['equipment_status', 'kpi', 'battery_status'],
  switchgear: ['equipment_status', 'kpi', 'distribution'],

  // Monitoring
  temperature: ['trend', 'kpi', 'gauge'],
  pressure: ['trend', 'kpi', 'gauge'],
  vibration: ['trend', 'kpi', 'spectrum'],
  flow: ['trend', 'kpi', 'gauge'],
  voltage: ['trend', 'kpi', 'distribution'],
  current: ['trend', 'kpi'],
  bearing: ['trend', 'kpi', 'health_status'],

  // Alerts
  alert: ['alerts', 'kpi', 'trend'],
  alarm: ['alerts', 'kpi'],
  warning: ['alerts', 'kpi'],
  fault: ['alerts', 'kpi'],
  threshold: ['alerts', 'trend'],

  // Energy
  energy: ['energy_breakdown', 'kpi', 'trend', 'comparison'],
  power: ['kpi', 'trend', 'distribution'],
  consumption: ['trend', 'kpi', 'comparison'],
  harmonic: ['trend', 'kpi', 'spectrum'],
  'power factor': ['kpi', 'trend'],

  // Operations
  maintenance: ['maintenance_schedule', 'equipment_status', 'alerts', 'work_orders'],
  'work order': ['work_orders', 'kpi'],
  efficiency: ['kpi', 'trend', 'comparison'],
  oee: ['kpi', 'trend', 'breakdown'],
  runtime: ['kpi', 'trend'],
  health: ['health_status', 'kpi', 'alerts'],

  // People
  shift: ['shift_schedule', 'personnel'],
  technician: ['personnel', 'shift_schedule'],
  operator: ['personnel', 'shift_schedule'],
  worker: ['personnel', 'kpi'],
  safety: ['alerts', 'kpi', 'trend'],

  // Supply Chain
  inventory: ['inventory_status', 'kpi', 'alerts'],
  stock: ['inventory_status', 'kpi'],
  spare: ['inventory_status', 'kpi'],
  supplier: ['supplier_status', 'kpi'],
  procurement: ['procurement', 'kpi'],
  delivery: ['delivery_schedule', 'kpi'],

  // Comparison/complex
  compare: ['comparison', 'trend', 'kpi'],
  overview: ['kpi', 'alerts', 'trend', 'distribution'],
  breakdown: ['breakdown', 'distribution', 'kpi'],
}

// ─── Query Ambiguity Keywords ──────────────────────────────────────
// Words that make a query more specific vs. vague

export const SPECIFICITY_INDICATORS = {
  specific: [
    /\b[A-Z]{1,4}[-_]\d{2,5}\b/,                   // Equipment IDs: TR-001, AHU-1, P-003
    /\bpump\s*\d+\b/i,                               // pump 1, pump 3
    /\bchiller\s*\d+\b/i,                             // chiller 1
    /\bmotor\s*[A-Z]?[-_]?\d+\b/i,                   // motor M-101
    /\bbus\s*[A-Z]\b/i,                               // Bus A, Bus B
    /\bline\s*\d+\b/i,                                // line 1
    /\bzone\s*\d+\b/i,                                // zone 2
    /past\s+\d+\s+(hour|day|week|month)/i,            // past 24 hours
    /last\s+\d+\s+(hour|day|week|month)/i,            // last 7 days
    /this\s+(morning|afternoon|week|month|shift)/i,   // this week
    /\b(today|yesterday|tonight)\b/i,                 // today
    /top\s+\d+/i,                                     // top 5
    /\d+\s*(kw|kva|%|degrees?|°|psi|bar|rpm)\b/i,    // numeric thresholds
  ],
  vague: [
    /^show\s+(me\s+)?(the\s+)?status$/i,             // "show status"
    /^show\s+(me\s+)?(the\s+)?\w+$/i,                // "show inventory", "show pumps"
    /^(what|how)\s+is\s+the\s+\w+\??$/i,             // "what is the status?"
    /everything/i,                                     // "show everything"
    /\ball\b/i,                                        // "show all pumps" (less specific than pump-003)
  ],
} as const
