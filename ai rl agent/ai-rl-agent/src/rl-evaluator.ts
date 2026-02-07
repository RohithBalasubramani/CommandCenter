/**
 * RL Evaluator — Core evaluation logic for Command Center responses.
 *
 * Evaluates AI responses across 6 dimensions:
 *   1. Widget count match (API vs rendered)
 *   2. Scenario relevance (returned widgets vs expected for query)
 *   3. Data accuracy (backend vs page widget match)
 *   4. Response quality (voice response relevance & length)
 *   5. Latency score (processing time budget)
 *   6. Query specificity (vague queries get lower weight)
 *
 * Generates structured feedback including:
 *   - Ambiguity detection (vague query vs. system mismatch)
 *   - Correction suggestions per domain
 *   - Deterministic interaction records
 */

import type { OrchestrateResponse, FeedbackRequest } from './rl-client.js'
import { EVAL_WEIGHTS, EXPECTED_SCENARIOS, SPECIFICITY_INDICATORS, type RLConfig } from './rl-config.js'
import type { AgentConfig } from './config.js'

// ─── Types ──────────────────────────────────────────────────────────

export interface InteractionRecord {
  widget_index: number
  action: string
  duration_ms: number
}

export interface RLEvaluation {
  queryId: string
  query: string
  overallScore: number
  rating: 'up' | 'down'
  widgetCountMatch: boolean
  scenarioRelevance: number
  dataAccuracy: number
  responseQuality: number
  latencyScore: number
  querySpecificity: number
  queryClarity: 'specific' | 'moderate' | 'vague'
  interactions: InteractionRecord[]
  correction?: string
  reasoning: string
  timestamp: string
  processingTimeMs: number
}

export interface ABComparison {
  query: string
  resultA: RLEvaluation
  resultB: RLEvaluation
  winner: 'A' | 'B' | 'tie'
  scoreDelta: number
  reasoning: string
}

export interface FeedbackPayload {
  query_id: string
  rating: 'up' | 'down'
  interactions: Array<{ widget_index: number; action: string; duration_ms: number }>
  correction?: string
}

export interface EvaluationBatch {
  evaluations: RLEvaluation[]
  summary: {
    total: number
    passed: number
    failed: number
    averageScore: number
    averageLatencyMs: number
    averageSpecificity: number
    vagueQueryCount: number
  }
  timestamp: string
}

// ─── Evaluator ──────────────────────────────────────────────────────

export class RLEvaluator {
  private config: RLConfig
  private agentConfig: AgentConfig

  constructor(config: RLConfig, agentConfig: AgentConfig) {
    this.config = config
    this.agentConfig = agentConfig
  }

  /**
   * Evaluate a single response from the orchestrator.
   */
  async evaluateResponse(
    query: string,
    result: OrchestrateResponse,
    pageWidgets: PageWidget[],
  ): Promise<RLEvaluation> {
    const widgetCountMatch = this.checkWidgetCount(result, pageWidgets)
    const scenarioRelevance = this.checkScenarioRelevance(query, result)
    const dataAccuracy = this.checkDataAccuracy(result, pageWidgets)
    const responseQuality = await this.checkResponseQuality(query, result)
    const latencyScore = this.computeLatencyScore(result.processing_time_ms)
    const { specificity, clarity } = this.computeQuerySpecificity(query)

    const normalizedScore =
      (widgetCountMatch ? EVAL_WEIGHTS.widgetCountMatch : 0) +
      scenarioRelevance * EVAL_WEIGHTS.scenarioRelevance +
      dataAccuracy * EVAL_WEIGHTS.dataAccuracy +
      responseQuality * EVAL_WEIGHTS.responseQuality +
      latencyScore * EVAL_WEIGHTS.latencyScore +
      specificity * EVAL_WEIGHTS.querySpecificity

    const rating: 'up' | 'down' = normalizedScore >= this.config.evaluationThreshold ? 'up' : 'down'
    const interactions = this.generateInteractions(result, normalizedScore)

    const correction = rating === 'down'
      ? this.suggestCorrection(query, result, clarity, scenarioRelevance)
      : undefined

    const reasoning = this.buildReasoning(
      query, widgetCountMatch, scenarioRelevance, dataAccuracy,
      responseQuality, latencyScore, specificity, clarity, normalizedScore, rating,
    )

    return {
      queryId: result.query_id,
      query,
      overallScore: normalizedScore,
      rating,
      widgetCountMatch,
      scenarioRelevance,
      dataAccuracy,
      responseQuality,
      latencyScore,
      querySpecificity: specificity,
      queryClarity: clarity,
      interactions,
      correction,
      reasoning,
      timestamp: new Date().toISOString(),
      processingTimeMs: result.processing_time_ms,
    }
  }

  /**
   * Convert evaluation to feedback payload for the RL API.
   */
  generateFeedback(evaluation: RLEvaluation): FeedbackPayload {
    return {
      query_id: evaluation.queryId,
      rating: evaluation.rating,
      interactions: evaluation.interactions,
      correction: evaluation.correction,
    }
  }

  /**
   * Compare two responses for A/B testing.
   */
  async compareAB(
    query: string,
    resultA: OrchestrateResponse,
    resultB: OrchestrateResponse,
    pageWidgetsA: PageWidget[],
    pageWidgetsB: PageWidget[],
  ): Promise<ABComparison> {
    const evalA = await this.evaluateResponse(query, resultA, pageWidgetsA)
    const evalB = await this.evaluateResponse(query, resultB, pageWidgetsB)

    const scoreDelta = evalA.overallScore - evalB.overallScore
    const winner: 'A' | 'B' | 'tie' =
      Math.abs(scoreDelta) < 0.05 ? 'tie' : scoreDelta > 0 ? 'A' : 'B'

    return {
      query,
      resultA: evalA,
      resultB: evalB,
      winner,
      scoreDelta,
      reasoning: `A scored ${evalA.overallScore.toFixed(3)} vs B scored ${evalB.overallScore.toFixed(3)}. ` +
        `Winner: ${winner}. Delta: ${Math.abs(scoreDelta).toFixed(3)}`,
    }
  }

  /**
   * Create a summary from a batch of evaluations.
   */
  summarizeBatch(evaluations: RLEvaluation[]): EvaluationBatch {
    const passed = evaluations.filter(e => e.rating === 'up').length
    const totalLatency = evaluations.reduce((sum, e) => sum + e.processingTimeMs, 0)
    const totalScore = evaluations.reduce((sum, e) => sum + e.overallScore, 0)
    const totalSpec = evaluations.reduce((sum, e) => sum + e.querySpecificity, 0)
    const vagueCount = evaluations.filter(e => e.queryClarity === 'vague').length

    return {
      evaluations,
      summary: {
        total: evaluations.length,
        passed,
        failed: evaluations.length - passed,
        averageScore: evaluations.length > 0 ? totalScore / evaluations.length : 0,
        averageLatencyMs: evaluations.length > 0 ? totalLatency / evaluations.length : 0,
        averageSpecificity: evaluations.length > 0 ? totalSpec / evaluations.length : 0,
        vagueQueryCount: vagueCount,
      },
      timestamp: new Date().toISOString(),
    }
  }

  // ─── Private Evaluation Methods ─────────────────────────────────

  private checkWidgetCount(
    result: OrchestrateResponse,
    pageWidgets: PageWidget[],
  ): boolean {
    const expected = result.layout_json?.widgets?.length || 0
    const actual = pageWidgets.length
    return Math.abs(expected - actual) <= 1
  }

  private checkScenarioRelevance(
    query: string,
    result: OrchestrateResponse,
  ): number {
    const widgets = result.layout_json?.widgets || []
    if (widgets.length === 0) return 0.5

    const queryLower = query.toLowerCase()
    const matchedScenarios: string[] = []

    for (const [keyword, expectedScenarios] of Object.entries(EXPECTED_SCENARIOS)) {
      if (queryLower.includes(keyword)) {
        matchedScenarios.push(...expectedScenarios)
      }
    }

    if (matchedScenarios.length === 0) return 0.7 // Unknown query type, give benefit of doubt

    // Deduplicate
    const uniqueExpected = [...new Set(matchedScenarios)]

    // Check how many returned widgets have relevant scenarios
    let relevantCount = 0
    for (const widget of widgets) {
      const scenario = widget.scenario?.toLowerCase() || ''
      if (uniqueExpected.some(ms => scenario.includes(ms) || ms.includes(scenario))) {
        relevantCount++
      }
    }

    return widgets.length > 0 ? relevantCount / widgets.length : 0
  }

  private checkDataAccuracy(
    result: OrchestrateResponse,
    pageWidgets: PageWidget[],
  ): number {
    const widgets = result.layout_json?.widgets || []
    if (widgets.length === 0 || pageWidgets.length === 0) return 0.5

    let matchCount = 0
    const checkCount = Math.min(widgets.length, pageWidgets.length)

    for (let i = 0; i < checkCount; i++) {
      const expected = widgets[i]
      const actual = pageWidgets[i]

      if (actual.scenario && expected.scenario) {
        const scenarioMatch =
          actual.scenario.toLowerCase() === expected.scenario.toLowerCase() ||
          actual.scenario.toLowerCase().includes(expected.scenario.toLowerCase())
        if (scenarioMatch) matchCount++
      }
    }

    return checkCount > 0 ? matchCount / checkCount : 0.5
  }

  private async checkResponseQuality(
    query: string,
    result: OrchestrateResponse,
  ): Promise<number> {
    const voice = result.voice_response || ''
    if (!voice) return 0.3

    let score = 0.5 // Base score for having a response

    // Length check
    if (voice.length > 20 && voice.length < 2000) score += 0.2
    if (voice.length >= 2000) score += 0.1

    // Check if voice response references query terms
    const queryWords = query.toLowerCase().split(/\s+/).filter(w => w.length > 3)
    const voiceLower = voice.toLowerCase()
    const mentionedWords = queryWords.filter(w => voiceLower.includes(w))
    if (queryWords.length > 0) {
      score += 0.3 * (mentionedWords.length / queryWords.length)
    }

    return Math.min(1.0, score)
  }

  private computeLatencyScore(processingTimeMs: number): number {
    const budget = this.config.latencyBudgetMs
    if (processingTimeMs <= budget) return 1.0
    if (processingTimeMs <= budget * 2) {
      return 1.0 - 0.5 * ((processingTimeMs - budget) / budget)
    }
    if (processingTimeMs <= budget * 4) {
      return 0.5 - 0.5 * ((processingTimeMs - budget * 2) / (budget * 2))
    }
    return 0.0
  }

  /**
   * Score how specific vs. vague a query is.
   * Specific queries (with equipment IDs, time ranges) get higher scores.
   * Vague queries ("show status") get lower scores.
   */
  private computeQuerySpecificity(query: string): { specificity: number; clarity: 'specific' | 'moderate' | 'vague' } {
    let specificity = 0.5 // Baseline

    // Check specificity indicators
    let specificMatches = 0
    for (const pattern of SPECIFICITY_INDICATORS.specific) {
      if (pattern.test(query)) specificMatches++
    }

    let vagueMatches = 0
    for (const pattern of SPECIFICITY_INDICATORS.vague) {
      if (pattern.test(query)) vagueMatches++
    }

    // More words = more specific (usually)
    const wordCount = query.split(/\s+/).length
    if (wordCount >= 8) specificity += 0.1
    if (wordCount >= 12) specificity += 0.1

    specificity += Math.min(0.3, specificMatches * 0.15)
    specificity -= Math.min(0.3, vagueMatches * 0.15)

    specificity = Math.max(0.1, Math.min(1.0, specificity))

    const clarity: 'specific' | 'moderate' | 'vague' =
      specificity >= 0.7 ? 'specific' : specificity >= 0.4 ? 'moderate' : 'vague'

    return { specificity, clarity }
  }

  /**
   * Generate deterministic interaction records (no randomness).
   */
  private generateInteractions(
    result: OrchestrateResponse,
    score: number,
  ): InteractionRecord[] {
    const widgets = result.layout_json?.widgets || []
    const interactions: InteractionRecord[] = []
    const engagementMultiplier = score > 0.6 ? 2.0 : 0.5

    for (let i = 0; i < widgets.length; i++) {
      // Deterministic duration based on widget index and score
      const baseDuration = 3000 + (i * 500)

      interactions.push({
        widget_index: i,
        action: 'view',
        duration_ms: Math.floor(baseDuration * engagementMultiplier),
      })

      // First widget gets an expand action if score is good
      if (i === 0 && score > 0.5) {
        interactions.push({
          widget_index: i,
          action: 'expand',
          duration_ms: Math.floor(5000 * engagementMultiplier),
        })
      }
    }

    return interactions
  }

  /**
   * Generate correction text for 'down' rated queries.
   * Differentiates between query ambiguity and system mismatch.
   */
  private suggestCorrection(
    query: string,
    result: OrchestrateResponse,
    clarity: 'specific' | 'moderate' | 'vague',
    scenarioRelevance: number,
  ): string | undefined {
    const widgets = result.layout_json?.widgets || []
    const queryLower = query.toLowerCase()
    const suggestions: string[] = []

    // No widgets at all
    if (widgets.length === 0) {
      return 'No widgets returned for this query — expected at least a KPI or status widget'
    }

    // Vague query — suggest being more specific
    if (clarity === 'vague') {
      suggestions.push(
        `Query "${query}" is vague — try specifying an equipment ID, time range, or specific metric for better results`
      )
    }

    // Low scenario relevance — widgets don't match query intent
    if (scenarioRelevance < 0.3) {
      const returnedScenarios = widgets.map(w => w.scenario).filter(Boolean).join(', ')
      suggestions.push(
        `Widget scenarios [${returnedScenarios}] don't match query intent — expected scenarios relevant to "${query.slice(0, 50)}"`
      )
    }

    // Safety-critical: alerts query without alerts widget
    if (
      (queryLower.includes('alert') || queryLower.includes('alarm') || queryLower.includes('warning') || queryLower.includes('fault')) &&
      !widgets.some(w => w.scenario?.toLowerCase().includes('alert'))
    ) {
      suggestions.push('Safety constraint: query mentions alerts/alarms but no alert widget was included')
    }

    // Comparison queries without comparison widget
    if (
      queryLower.includes('compare') &&
      !widgets.some(w => w.scenario?.toLowerCase().includes('comparison'))
    ) {
      suggestions.push('Query asks for comparison but no comparison widget was included')
    }

    // Trend queries without trend widget
    if (
      queryLower.includes('trend') &&
      !widgets.some(w => w.scenario?.toLowerCase().includes('trend'))
    ) {
      suggestions.push('Query asks for trend but no trend/chart widget was included')
    }

    // Maintenance queries without maintenance widget
    if (
      queryLower.includes('maintenance') &&
      !widgets.some(w => {
        const s = w.scenario?.toLowerCase() || ''
        return s.includes('maintenance') || s.includes('work_order')
      })
    ) {
      suggestions.push('Query about maintenance but no maintenance or work order widget included')
    }

    // Personnel queries without personnel widget
    if (
      (queryLower.includes('shift') || queryLower.includes('technician') || queryLower.includes('operator') || queryLower.includes('who')) &&
      !widgets.some(w => {
        const s = w.scenario?.toLowerCase() || ''
        return s.includes('personnel') || s.includes('shift') || s.includes('schedule')
      })
    ) {
      suggestions.push('Query about personnel/shifts but no personnel or schedule widget included')
    }

    // Inventory queries without inventory widget
    if (
      (queryLower.includes('inventory') || queryLower.includes('stock') || queryLower.includes('spare')) &&
      !widgets.some(w => {
        const s = w.scenario?.toLowerCase() || ''
        return s.includes('inventory') || s.includes('stock') || s.includes('supply')
      })
    ) {
      suggestions.push('Query about inventory/stock but no inventory widget included')
    }

    return suggestions.length > 0 ? suggestions.join('; ') : undefined
  }

  private buildReasoning(
    query: string,
    widgetCountMatch: boolean,
    scenarioRelevance: number,
    dataAccuracy: number,
    responseQuality: number,
    latencyScore: number,
    querySpecificity: number,
    queryClarity: string,
    overallScore: number,
    rating: 'up' | 'down',
  ): string {
    const parts: string[] = [
      `Query: "${query.slice(0, 80)}"`,
      `Clarity: ${queryClarity} (${(querySpecificity * 100).toFixed(0)}%)`,
      `Widget count ${widgetCountMatch ? 'match' : 'MISMATCH'}`,
      `Relevance: ${(scenarioRelevance * 100).toFixed(0)}%`,
      `Accuracy: ${(dataAccuracy * 100).toFixed(0)}%`,
      `Quality: ${(responseQuality * 100).toFixed(0)}%`,
      `Latency: ${(latencyScore * 100).toFixed(0)}%`,
      `Overall: ${(overallScore * 100).toFixed(1)}% → ${rating}`,
    ]
    return parts.join(' | ')
  }
}

// ─── Helper Types ───────────────────────────────────────────────────

export interface PageWidget {
  scenario: string
  fixture?: string
  size?: string
  textContent?: string
}
