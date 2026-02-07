/**
 * AI RL Agent — Autonomous reinforcement learning agent for Command Center.
 *
 * Built on the AI QA Agent framework, this package adds:
 * - Response evaluation (widget quality, data accuracy, latency)
 * - Structured feedback submission to the RL training system
 * - RL training lifecycle management (approve, monitor, compare)
 * - A/B model comparison and weight sensitivity analysis
 *
 * @example
 * ```typescript
 * import { RLClient, RLEvaluator } from '@neurareport/ai-rl-agent'
 *
 * const client = new RLClient({ apiBaseUrl: 'http://localhost:8100' })
 * const status = await client.getStatus()
 * console.log(status.buffer, status.trainer)
 * ```
 */

// ─── Types ───────────────────────────────────────────────────────────

export type {
  // Action types
  AgentActionType,
  AgentAction,
  ActionTarget,
  VerifyAssertion,

  // Observation types
  PageObservation,
  InteractiveElement,
  ApiCallRecord,

  // Scenario types
  TestScenario,
  QaAgentProfile,
  PersonaModifier,
  SuccessCriterion,
  BackendCheck,
  SetupAction,
  TeardownAction,

  // Result types
  ScenarioResult,
  ActionLogEntry,
  CriterionResult,
  BackendCheckResult,
  FailureCategory,

  // Progress tracking
  GoalLedger,
  PerceptionEntry,

  // Learning
  CachedActionSequence,
  LessonLearned,

  // Reports
  AuditReport,

  // Failure mode mitigations
  ActionVerification,
  PageStability,
  ReplayDetection,
} from './types'

// ─── Configuration ───────────────────────────────────────────────────

export type {
  UIFramework,
  ElementDiscoveryConfig,
  LLMConfig,
  AgentConfig,
  RunnerConfig,
} from './config'

export {
  FRAMEWORK_PRESETS,
  DEFAULT_LLM_CONFIG,
  DEFAULT_AGENT_CONFIG,
  DEFAULT_RUNNER_CONFIG,
  createConfig,
  createRunnerConfig,
  getElementSelectors,
} from './config'

// ─── Browser Agent ───────────────────────────────────────────────────

export { BrowserAgent } from './browser-agent'

// ─── Agent Brain ─────────────────────────────────────────────────────

export { AgentBrain } from './agent-brain'

// ─── Test Runner ─────────────────────────────────────────────────────

export type { AITestRunnerConfig } from './test-runner'

export {
  AITestRunner,
  generateAuditReport,
  runScenarios,
} from './test-runner'

// ─── RL Client ──────────────────────────────────────────────────────

export { RLClient } from './rl-client'

export type {
  OrchestrateRequest,
  OrchestrateResponse,
  WidgetData,
  FeedbackRequest,
  FeedbackResponse,
  RLStatus,
  RLHistory,
  ApproveTrainingResponse,
  HealthResponse,
  RLClientConfig,
} from './rl-client'

// ─── RL Evaluator ───────────────────────────────────────────────────

export { RLEvaluator } from './rl-evaluator'

export type {
  RLEvaluation,
  ABComparison,
  FeedbackPayload,
  EvaluationBatch,
  InteractionRecord,
  PageWidget,
} from './rl-evaluator'

// ─── RL Config ──────────────────────────────────────────────────────

export {
  rlAgentConfig,
  rlRunnerConfig,
  rlConfig,
  EVAL_WEIGHTS,
  EXPECTED_SCENARIOS,
  SPECIFICITY_INDICATORS,
} from './rl-config'

export type { RLConfig } from './rl-config'

// ─── RL Scenarios ───────────────────────────────────────────────────

export { rlScenarios, EVAL_QUERIES } from './rl-scenarios'
