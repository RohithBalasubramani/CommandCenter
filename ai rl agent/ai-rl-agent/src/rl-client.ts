/**
 * RL API Client — HTTP client for Command Center RL backend endpoints.
 *
 * Provides typed access to:
 * - /api/layer2/orchestrate/      — Send queries, get layout + widgets
 * - /api/layer2/feedback/         — Submit ratings, interactions, corrections
 * - /api/layer2/approve-training/ — Approve LoRA training
 * - /api/layer2/rl-status/        — RL system status (live)
 * - /api/layer2/rl-history/       — Historical RL training data
 * - /api/layer2/rag/industrial/health/ — RAG health check
 */

// ─── Types ──────────────────────────────────────────────────────────

export interface OrchestrateRequest {
  transcript: string
  session_id?: string
  context?: Record<string, any>
  user_id?: string
}

export interface WidgetData {
  scenario: string
  size?: string
  why?: string
  fixture?: string
  data?: Record<string, any>
  heightHint?: number
}

export interface OrchestrateResponse {
  voice_response: string
  filler_text: string
  layout_json: {
    heading?: string
    widgets: WidgetData[]
    select_method?: string
  }
  context_update: Record<string, any>
  intent: {
    domain?: string
    action?: string
    entities?: string[]
    confidence?: number
    type?: string
    primary_characteristic?: string
    domains?: string[]
  }
  query_id: string
  processing_time_ms: number
}

export interface FeedbackRequest {
  query_id: string
  rating: 'up' | 'down'
  interactions?: Array<{
    widget_index: number
    action: string
    duration_ms: number
  }>
  correction?: string
}

export interface FeedbackResponse {
  status: string
  updated: boolean
}

// Matches actual backend GET /api/layer2/rl-status/ response
export interface RLStatus {
  running: boolean
  buffer: {
    total_experiences: number
    with_feedback: number
    without_feedback: number
    ratings: { up: number; down: number; none: number }
    max_size: number
    redis_connected: boolean
  }
  trainer: {
    running: boolean
    training_steps: number
    total_samples_trained: number
    avg_reward_trend: number
    recent_rewards: number[]
    tier1_scorer: {
      type: string
      rank: number
      parameters: number
      device: string
      training_steps: number
      total_feedback_events: number
      avg_loss: number
      recent_losses: number[]
    }
    tier2_lora: {
      training_in_progress: boolean
      pending_pairs: number
      min_pairs_for_training: number
      total_trainings: number
      total_pairs_trained: number
      last_loss: number | null
      current_version: number
      last_training_time: string | null
    }
  }
  config: {
    train_widget_selector: boolean
    train_fixture_selector: boolean
    train_interval: number
    min_batch_size: number
  }
}

// Matches GET /api/layer2/rl-history/ response
export interface RLHistory {
  reward_timeline: Array<{ timestamp: string; reward: number; rating: string }>
  feedback_distribution: { up: number; down: number; none: number }
  latency_buckets: Array<{ range: string; count: number }>
  intent_distribution: Record<string, number>
  scenario_frequency: Record<string, number>
  processing_time_trend: Array<{ timestamp: string; ms: number }>
  training_loss_curve: Array<{
    step: number
    loss: number | null
    accuracy: number | null
    margins: number | null
  }>
  evaluation_summary: {
    count: number
    avg_overall: number
    avg_scenario_relevance: number
    avg_data_accuracy: number
    avg_response_quality: number
    avg_latency_score: number
    scores: Array<{ score: number; query: string }>
  }
  query_details: Array<{
    query_id: string
    timestamp: string
    query: string
    rating: 'up' | 'down' | null
    reward: number | null
    processing_time_ms: number
    widget_count: number
    scenarios: string[]
    intent_type: string
    domains: string[]
    primary_characteristic: string
    feedback_source: 'user_direct' | 'eval_agent' | 'both' | 'implicit_only'
    query_clarity: 'clear' | 'ambiguous_query' | 'system_mismatch'
  }>
  query_aggregates?: {
    avg_processing_ms: number
    avg_widget_count: number
    total_experiences: number
    scorer_steps: number
    dpo_pairs_generated: number
    characteristic_counts: Record<string, number>
  }
}

export interface ApproveTrainingResponse {
  status: string
  file: string
}

export interface HealthResponse {
  status: string
  [key: string]: any
}

export interface RLClientConfig {
  apiBaseUrl: string
  feedbackApiKey?: string
  timeoutMs?: number
}

// ─── Client ─────────────────────────────────────────────────────────

export class RLClient {
  private baseUrl: string
  private feedbackApiKey: string
  private timeoutMs: number

  constructor(config: RLClientConfig) {
    this.baseUrl = config.apiBaseUrl.replace(/\/$/, '')
    this.feedbackApiKey = config.feedbackApiKey || ''
    this.timeoutMs = config.timeoutMs || 60_000
  }

  private async request<T>(
    method: 'GET' | 'POST',
    path: string,
    body?: Record<string, any>,
    extraHeaders?: Record<string, string>,
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...extraHeaders,
    }

    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs)

    try {
      const resp = await fetch(url, {
        method,
        headers,
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      })

      clearTimeout(timeout)

      if (!resp.ok) {
        const errBody = await resp.text().catch(() => '')
        throw new Error(`HTTP ${resp.status} ${resp.statusText}: ${errBody}`)
      }

      return (await resp.json()) as T
    } catch (err: any) {
      clearTimeout(timeout)
      if (err.name === 'AbortError') {
        throw new Error(`Request to ${path} timed out after ${this.timeoutMs}ms`)
      }
      throw err
    }
  }

  /**
   * Send a query to the orchestrator and get the layout response.
   */
  async orchestrate(
    transcript: string,
    sessionId?: string,
    context?: Record<string, any>,
    userId?: string,
  ): Promise<OrchestrateResponse> {
    const body: OrchestrateRequest = {
      transcript,
      session_id: sessionId,
      context: context || {},
      user_id: userId || 'rl-agent',
    }
    return this.request<OrchestrateResponse>('POST', '/api/layer2/orchestrate/', body)
  }

  /**
   * Submit feedback for a query response.
   * Retries with exponential backoff on 429 (rate-limited) responses.
   */
  async submitFeedback(feedback: FeedbackRequest, maxRetries = 3): Promise<FeedbackResponse> {
    const headers: Record<string, string> = {}
    if (this.feedbackApiKey) {
      headers['X-Feedback-Key'] = this.feedbackApiKey
    }

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await this.request<FeedbackResponse>('POST', '/api/layer2/feedback/', feedback, headers)
      } catch (err: any) {
        const is429 = err.message?.includes('429')
        if (is429 && attempt < maxRetries) {
          const delay = Math.pow(2, attempt + 2) * 1000
          await new Promise(resolve => setTimeout(resolve, delay))
          continue
        }
        throw err
      }
    }

    throw new Error('submitFeedback: exhausted retries')
  }

  /**
   * Approve pending LoRA training.
   */
  async approveTraining(): Promise<ApproveTrainingResponse> {
    return this.request<ApproveTrainingResponse>('POST', '/api/layer2/approve-training/')
  }

  /**
   * Get current RL system status (live data).
   */
  async getStatus(): Promise<RLStatus> {
    return this.request<RLStatus>('GET', '/api/layer2/rl-status/')
  }

  /**
   * Get historical RL training data (experience buffer, training curves, evaluations).
   */
  async getHistory(limit = 500): Promise<RLHistory> {
    return this.request<RLHistory>('GET', `/api/layer2/rl-history/?limit=${limit}`)
  }

  /**
   * Check RAG pipeline health.
   */
  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('GET', '/api/layer2/rag/industrial/health/')
  }

  /**
   * Check if both frontend and backend are reachable.
   */
  async checkServers(frontendUrl: string): Promise<{ frontend: boolean; backend: boolean }> {
    const check = async (url: string): Promise<boolean> => {
      try {
        const controller = new AbortController()
        const timeout = setTimeout(() => controller.abort(), 5000)
        const resp = await fetch(url, { signal: controller.signal })
        clearTimeout(timeout)
        return resp.status < 500
      } catch {
        return false
      }
    }

    const [frontend, backend] = await Promise.all([
      check(frontendUrl),
      check(`${this.baseUrl}/api/layer2/rl-status/`),
    ])

    return { frontend, backend }
  }
}
