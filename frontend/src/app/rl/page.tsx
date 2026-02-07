"use client";

import { useState, useEffect, useCallback } from "react";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

// ─── Constants ──────────────────────────────────────────────────────

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8100";

const C = {
  accent: "#6366f1",
  accentDim: "#4f46e5",
  success: "#22c55e",
  warning: "#f59e0b",
  error: "#ef4444",
  cyan: "#06b6d4",
  purple: "#a855f7",
  pink: "#ec4899",
  muted: "#8888a0",
  surface: "#111118",
  border: "#2a2a3a",
  text: "#e4e4ef",
  textMuted: "#8888a0",
  bg: "#0a0a0f",
};

const tooltipStyle = {
  backgroundColor: C.surface,
  borderColor: C.border,
  borderRadius: 8,
  fontSize: 11,
  color: C.text,
};

// ─── Types ──────────────────────────────────────────────────────────

interface RLStatus {
  running: boolean;
  buffer: {
    total_experiences: number;
    with_feedback: number;
    without_feedback: number;
    ratings: { up: number; down: number; none: number };
    max_size: number;
    redis_connected: boolean;
  };
  trainer: {
    running: boolean;
    training_steps: number;
    total_samples_trained: number;
    avg_reward_trend: number;
    recent_rewards: number[];
    tier1_scorer: {
      type: string;
      rank: number;
      parameters: number;
      device: string;
      training_steps: number;
      total_feedback_events: number;
      avg_loss: number;
      recent_losses: number[];
    };
    tier2_lora: {
      training_in_progress: boolean;
      pending_pairs: number;
      min_pairs_for_training: number;
      total_trainings: number;
      total_pairs_trained: number;
      last_loss: number | null;
      current_version: number;
      last_training_time: string | null;
    };
  };
  config: Record<string, unknown>;
}

interface RLHistory {
  reward_timeline: Array<{ timestamp: string; reward: number; rating: string }>;
  feedback_distribution: { up: number; down: number; none: number };
  latency_buckets: Array<{ range: string; count: number }>;
  intent_distribution: Record<string, number>;
  scenario_frequency: Record<string, number>;
  processing_time_trend: Array<{ timestamp: string; ms: number }>;
  training_loss_curve: Array<{
    step: number;
    loss: number | null;
    accuracy: number | null;
    margins: number | null;
    grad_norm: number | null;
    lr: number | null;
  }>;
  evaluation_summary: {
    count: number;
    avg_overall: number;
    avg_scenario_relevance: number;
    avg_data_accuracy: number;
    avg_response_quality: number;
    avg_latency_score: number;
    scores: Array<{ score: number; query: string }>;
  };
  query_details: QueryDetail[];
  query_aggregates?: {
    avg_processing_ms: number;
    avg_widget_count: number;
    total_experiences: number;
    scorer_steps: number;
    dpo_pairs_generated: number;
    characteristic_counts: Record<string, number>;
  };
}

interface QueryDetail {
  query_id: string;
  timestamp: string;
  query: string;
  rating: "up" | "down" | null;
  reward: number | null;
  processing_time_ms: number;
  widget_count: number;
  scenarios: string[];
  intent_type: string;
  domains: string[];
  primary_characteristic: string;
  confidence: number | null;
  select_method: string;
  heading: string;
  eval_overall: number | null;
  eval_relevance: number | null;
  eval_accuracy: number | null;
  eval_quality: number | null;
  eval_latency: number | null;
  eval_rating: "up" | "down" | null;
  correction: string | null;
  feedback_source: "user_direct" | "eval_agent" | "both" | "implicit_only";
  query_clarity: "clear" | "ambiguous_query" | "system_mismatch";
  latency_signal: "fast" | "normal" | "slow";
  widget_signal: "rich" | "normal" | "sparse";
  diversity_signal: "rare" | "uncommon" | "common";
  buffer_position: number;
}

// ─── Helpers ────────────────────────────────────────────────────────

function fmtTime(ts: string) {
  if (!ts) return "";
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function fmtDate(ts: string) {
  if (!ts) return "";
  const d = new Date(ts);
  return `${d.getMonth() + 1}/${d.getDate()} ${d.getHours()}:${String(d.getMinutes()).padStart(2, "0")}`;
}

// ─── Sub-components ─────────────────────────────────────────────────

function StatCard({
  label,
  value,
  sub,
  spark,
  accent = C.accent,
}: {
  label: string;
  value: string | number;
  sub?: string;
  spark?: number[];
  accent?: string;
}) {
  const sparkData = spark?.map((v, i) => ({ i, v }));
  return (
    <div className="rounded-xl border border-[var(--cc-border)] bg-[var(--cc-surface)] p-4 flex flex-col">
      <span className="text-[10px] uppercase tracking-[0.12em] font-bold text-[var(--cc-text-muted)]">
        {label}
      </span>
      <span className="text-2xl font-bold text-[var(--cc-text)] mt-1 font-mono">
        {value}
      </span>
      {sub && (
        <span className="text-[11px] text-[var(--cc-text-muted)] mt-0.5">
          {sub}
        </span>
      )}
      {sparkData && sparkData.length > 1 && (
        <div className="h-8 mt-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={sparkData}>
              <Line
                type="monotone"
                dataKey="v"
                stroke={accent}
                strokeWidth={1.5}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function ChartCard({
  title,
  subtitle,
  children,
  className = "",
  tall = false,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  className?: string;
  tall?: boolean;
}) {
  return (
    <div
      className={`rounded-xl border border-[var(--cc-border)] bg-[var(--cc-surface)] p-5 ${className}`}
    >
      <div className="mb-3">
        <h3 className="text-sm font-semibold text-[var(--cc-text)]">{title}</h3>
        {subtitle && (
          <p className="text-[10px] text-[var(--cc-text-muted)] mt-0.5">
            {subtitle}
          </p>
        )}
      </div>
      <div className={tall ? "h-[300px]" : "h-[240px]"}>{children}</div>
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return (
    <div className="h-full flex items-center justify-center text-[var(--cc-text-muted)] text-xs">
      {text}
    </div>
  );
}

// ─── Main Page ──────────────────────────────────────────────────────

export default function RLDashboardPage() {
  const [status, setStatus] = useState<RLStatus | null>(null);
  const [history, setHistory] = useState<RLHistory | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [loading, setLoading] = useState(true);
  const [querySearch, setQuerySearch] = useState("");
  const [queryFilter, setQueryFilter] = useState<"all" | "up" | "down" | "none">("all");
  const [expandedQuery, setExpandedQuery] = useState<string | null>(null);
  const [queryPage, setQueryPage] = useState(0);
  const QUERIES_PER_PAGE = 25;

  // Poll live status every 5s
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/layer2/rl-status/`);
        if (res.ok) {
          setStatus(await res.json());
          setLastUpdated(new Date());
        }
      } catch {
        /* ignore */
      }
    };
    fetchStatus();
    const iv = setInterval(fetchStatus, 5000);
    return () => clearInterval(iv);
  }, []);

  // Fetch history once on mount
  const fetchHistory = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/layer2/rl-history/`);
      if (res.ok) setHistory(await res.json());
    } catch {
      /* ignore */
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  // Derived data
  const t1 = status?.trainer?.tier1_scorer;
  const t2 = status?.trainer?.tier2_lora;
  const recentRewards = status?.trainer?.recent_rewards || [];
  const recentLosses = t1?.recent_losses || [];
  const avgReward = status?.trainer?.avg_reward_trend ?? 0;
  const fb = history?.feedback_distribution || { up: 0, down: 0, none: 0 };
  const evalSummary = history?.evaluation_summary;

  // Radar data for evaluation
  const radarData = evalSummary
    ? [
        { axis: "Overall", value: evalSummary.avg_overall },
        { axis: "Relevance", value: evalSummary.avg_scenario_relevance },
        { axis: "Accuracy", value: evalSummary.avg_data_accuracy },
        { axis: "Quality", value: evalSummary.avg_response_quality },
        { axis: "Latency", value: evalSummary.avg_latency_score },
      ]
    : [];

  // Scenario frequency as bar data
  const scenarioData = Object.entries(history?.scenario_frequency || {})
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count);

  // Pie data
  const pieData = [
    { name: "Positive", value: fb.up, color: C.success },
    { name: "Negative", value: fb.down, color: C.error },
    { name: "No Rating", value: fb.none, color: C.muted },
  ].filter((d) => d.value > 0);

  // Latency bar colors
  const latencyColors = [C.success, "#84cc16", C.warning, "#f97316", C.error];

  // LoRA progress
  const loraPairs = t2?.pending_pairs ?? 0;
  const loraMin = t2?.min_pairs_for_training ?? 50;
  const loraProgress = Math.min(100, Math.round((loraPairs / loraMin) * 100));

  return (
    <div className="h-screen overflow-y-auto bg-[var(--cc-bg)] text-[var(--cc-text)]">
      {/* ── Header ─────────────────────────────────────────── */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--cc-border)] sticky top-0 bg-[var(--cc-bg)]/90 backdrop-blur z-10">
        <div>
          <h1 className="text-xl font-bold tracking-tight">
            RL Training Monitor
          </h1>
          <p className="text-[11px] text-[var(--cc-text-muted)] font-mono mt-0.5">
            Continuous Reinforcement Learning — Two-Tier Architecture
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <span
              className={`w-2 h-2 rounded-full ${status?.running ? "bg-green-500 animate-pulse" : "bg-red-500"}`}
            />
            <span className="text-xs text-[var(--cc-text-muted)]">
              {status?.running ? "Live" : "Offline"}
            </span>
          </div>
          {lastUpdated && (
            <span className="text-[10px] font-mono text-[var(--cc-text-muted)]">
              {lastUpdated.toLocaleTimeString()}
            </span>
          )}
          <button
            onClick={fetchHistory}
            className="text-xs px-3 py-1.5 rounded-lg border border-[var(--cc-border)] hover:bg-[var(--cc-surface-hover)] transition-colors text-[var(--cc-text-muted)]"
          >
            {loading ? "Loading..." : "Refresh"}
          </button>
        </div>
      </div>

      <div className="px-6 py-5 space-y-5">
        {/* ── KPI Cards ──────────────────────────────────── */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            label="Experience Buffer"
            value={status?.buffer?.total_experiences ?? "—"}
            sub={`${status?.buffer?.with_feedback ?? 0} with feedback / ${status?.buffer?.max_size ?? 10000} max`}
          />
          <StatCard
            label="Training Steps"
            value={status?.trainer?.training_steps ?? "—"}
            sub={`${status?.trainer?.total_samples_trained ?? 0} samples trained`}
          />
          <StatCard
            label="Avg Reward"
            value={avgReward > 0 ? `+${avgReward.toFixed(4)}` : avgReward.toFixed(4)}
            sub="Last 10 batch average"
            spark={recentRewards}
            accent={avgReward >= 0 ? C.success : C.error}
          />
          <StatCard
            label="LoRA Model"
            value={
              t2?.training_in_progress
                ? "Training..."
                : `v${t2?.current_version ?? 0}`
            }
            sub={`${loraPairs}/${loraMin} DPO pairs (${loraProgress}%)`}
          />
        </div>

        {/* ── LoRA Progress Bar ──────────────────────────── */}
        <div className="rounded-xl border border-[var(--cc-border)] bg-[var(--cc-surface)] px-5 py-3">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-[10px] uppercase tracking-[0.12em] font-bold text-[var(--cc-text-muted)]">
              LoRA DPO Pair Accumulation
            </span>
            <span className="text-xs font-mono text-[var(--cc-text-muted)]">
              {loraPairs} / {loraMin} pairs
              {t2?.training_in_progress && (
                <span className="ml-2 text-amber-400 animate-pulse">
                  Training in progress...
                </span>
              )}
            </span>
          </div>
          <div className="w-full h-2 rounded-full bg-[var(--cc-border)] overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-700"
              style={{
                width: `${loraProgress}%`,
                background:
                  loraProgress >= 100
                    ? C.success
                    : `linear-gradient(90deg, ${C.accent}, ${C.purple})`,
              }}
            />
          </div>
        </div>

        {/* ── Reward Timeline (full width) ───────────────── */}
        <ChartCard
          title="Reward Timeline"
          subtitle={`${history?.reward_timeline?.length ?? 0} data points — computed rewards over time`}
          tall
        >
          {history?.reward_timeline?.length ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={history.reward_timeline}>
                <defs>
                  <linearGradient
                    id="rewardGrad"
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop
                      offset="5%"
                      stopColor={C.accent}
                      stopOpacity={0.3}
                    />
                    <stop
                      offset="95%"
                      stopColor={C.accent}
                      stopOpacity={0}
                    />
                  </linearGradient>
                </defs>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke={C.border}
                  vertical={false}
                />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={fmtDate}
                  tick={{ fontSize: 10, fill: C.textMuted }}
                  stroke={C.border}
                  interval="preserveStartEnd"
                />
                <YAxis
                  domain={[-2, 2]}
                  tick={{ fontSize: 10, fill: C.textMuted }}
                  stroke={C.border}
                  width={40}
                />
                <Tooltip
                  contentStyle={tooltipStyle}
                  labelFormatter={(l) => fmtDate(l as string)}
                  formatter={(v: number) => [v.toFixed(4), "Reward"]}
                />
                <Area
                  type="monotone"
                  dataKey="reward"
                  stroke={C.accent}
                  fill="url(#rewardGrad)"
                  strokeWidth={1.5}
                  dot={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState text="No reward data yet — run the RL agent to generate training data" />
          )}
        </ChartCard>

        {/* ── Two-column grid ────────────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          {/* Training Loss Curve */}
          <ChartCard
            title="DPO Training Loss"
            subtitle={`${history?.training_loss_curve?.length ?? 0} steps — loss, accuracy, reward margins`}
          >
            {history?.training_loss_curve?.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={history.training_loss_curve}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke={C.border}
                    vertical={false}
                  />
                  <XAxis
                    dataKey="step"
                    tick={{ fontSize: 10, fill: C.textMuted }}
                    stroke={C.border}
                    label={{
                      value: "Step",
                      position: "insideBottomRight",
                      offset: -5,
                      fontSize: 10,
                      fill: C.textMuted,
                    }}
                  />
                  <YAxis
                    tick={{ fontSize: 10, fill: C.textMuted }}
                    stroke={C.border}
                    width={50}
                  />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Line
                    type="monotone"
                    dataKey="loss"
                    stroke={C.error}
                    strokeWidth={2}
                    dot={false}
                    name="Loss"
                  />
                  <Line
                    type="monotone"
                    dataKey="accuracy"
                    stroke={C.success}
                    strokeWidth={2}
                    dot={false}
                    name="Accuracy"
                  />
                  <Line
                    type="monotone"
                    dataKey="margins"
                    stroke={C.cyan}
                    strokeWidth={1.5}
                    dot={false}
                    name="Margins"
                    strokeDasharray="4 2"
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState text="No DPO training data yet — LoRA training triggers at 50+ DPO pairs" />
            )}
          </ChartCard>

          {/* Evaluation Radar */}
          <ChartCard
            title="Evaluation Dimensions"
            subtitle={
              evalSummary
                ? `${evalSummary.count} evaluations — avg ${(evalSummary.avg_overall * 100).toFixed(1)}%`
                : "No evaluation data"
            }
          >
            {radarData.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData} cx="50%" cy="50%">
                  <PolarGrid stroke={C.border} />
                  <PolarAngleAxis
                    dataKey="axis"
                    tick={{ fontSize: 11, fill: C.textMuted }}
                  />
                  <PolarRadiusAxis
                    domain={[0, 1]}
                    tick={{ fontSize: 9, fill: C.textMuted }}
                    stroke={C.border}
                  />
                  <Radar
                    dataKey="value"
                    stroke={C.purple}
                    fill={C.purple}
                    fillOpacity={0.25}
                    strokeWidth={2}
                  />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    formatter={(v: number) => [
                      `${(v * 100).toFixed(1)}%`,
                      "Score",
                    ]}
                  />
                </RadarChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState text="No evaluation data yet — run the RL agent in api-eval mode" />
            )}
          </ChartCard>

          {/* Feedback Distribution */}
          <ChartCard
            title="Feedback Distribution"
            subtitle={`${fb.up + fb.down + fb.none} total feedback signals`}
          >
            {pieData.length ? (
              <div className="h-full flex items-center">
                <div className="w-1/2 h-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={pieData}
                        cx="50%"
                        cy="50%"
                        innerRadius={55}
                        outerRadius={85}
                        dataKey="value"
                        stroke="none"
                      >
                        {pieData.map((entry, idx) => (
                          <Cell key={idx} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip contentStyle={tooltipStyle} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="w-1/2 space-y-3 pl-4">
                  {pieData.map((d) => (
                    <div key={d.name} className="flex items-center gap-2">
                      <span
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: d.color }}
                      />
                      <span className="text-xs text-[var(--cc-text)]">
                        {d.name}
                      </span>
                      <span className="text-xs font-mono text-[var(--cc-text-muted)] ml-auto">
                        {d.value}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <EmptyState text="No feedback data yet" />
            )}
          </ChartCard>

          {/* Latency Distribution */}
          <ChartCard
            title="Response Latency"
            subtitle="Processing time distribution across queries"
          >
            {history?.latency_buckets?.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={history.latency_buckets} barSize={40}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke={C.border}
                    vertical={false}
                  />
                  <XAxis
                    dataKey="range"
                    tick={{ fontSize: 11, fill: C.textMuted }}
                    stroke={C.border}
                  />
                  <YAxis
                    tick={{ fontSize: 10, fill: C.textMuted }}
                    stroke={C.border}
                    width={35}
                  />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {history.latency_buckets.map((_, idx) => (
                      <Cell key={idx} fill={latencyColors[idx] || C.muted} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState text="No latency data" />
            )}
          </ChartCard>

          {/* Scenario Frequency */}
          <ChartCard
            title="Widget Scenario Frequency"
            subtitle={`${scenarioData.length} unique scenarios selected`}
          >
            {scenarioData.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={scenarioData.slice(0, 12)}
                  layout="vertical"
                  barSize={16}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke={C.border}
                    horizontal={false}
                  />
                  <XAxis
                    type="number"
                    tick={{ fontSize: 10, fill: C.textMuted }}
                    stroke={C.border}
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    tick={{ fontSize: 10, fill: C.textMuted }}
                    stroke={C.border}
                    width={100}
                  />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Bar
                    dataKey="count"
                    fill={C.accent}
                    radius={[0, 4, 4, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState text="No scenario data" />
            )}
          </ChartCard>

          {/* Tier Details */}
          <ChartCard title="Tier Details" subtitle="Scorer (Tier 1) + LoRA (Tier 2)">
            <div className="h-full flex flex-col gap-4 overflow-y-auto">
              {/* Tier 1 */}
              <div className="flex-1 space-y-2">
                <div className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                  <span className="text-xs font-semibold text-[var(--cc-text)]">
                    Tier 1: Low-Rank Scorer
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-3 text-center">
                  <div>
                    <div className="text-lg font-mono font-bold text-[var(--cc-text)]">
                      {t1?.parameters ?? "—"}
                    </div>
                    <div className="text-[9px] uppercase tracking-wider text-[var(--cc-text-muted)]">
                      Parameters
                    </div>
                  </div>
                  <div>
                    <div className="text-lg font-mono font-bold text-[var(--cc-text)]">
                      {t1?.training_steps ?? "—"}
                    </div>
                    <div className="text-[9px] uppercase tracking-wider text-[var(--cc-text-muted)]">
                      Steps
                    </div>
                  </div>
                  <div>
                    <div className="text-lg font-mono font-bold text-[var(--cc-text)]">
                      {t1?.avg_loss != null
                        ? t1.avg_loss.toFixed(4)
                        : "—"}
                    </div>
                    <div className="text-[9px] uppercase tracking-wider text-[var(--cc-text-muted)]">
                      Avg Loss
                    </div>
                  </div>
                </div>
                {recentLosses.length > 1 && (
                  <div className="h-10">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={recentLosses.map((v, i) => ({ i, v }))}
                      >
                        <Line
                          type="monotone"
                          dataKey="v"
                          stroke={C.cyan}
                          strokeWidth={1.5}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>

              <div className="border-t border-[var(--cc-border)]" />

              {/* Tier 2 */}
              <div className="flex-1 space-y-2">
                <div className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-purple-400" />
                  <span className="text-xs font-semibold text-[var(--cc-text)]">
                    Tier 2: LoRA DPO
                  </span>
                  {t2?.training_in_progress && (
                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-400 animate-pulse">
                      Training
                    </span>
                  )}
                </div>
                <div className="grid grid-cols-3 gap-3 text-center">
                  <div>
                    <div className="text-lg font-mono font-bold text-[var(--cc-text)]">
                      v{t2?.current_version ?? 0}
                    </div>
                    <div className="text-[9px] uppercase tracking-wider text-[var(--cc-text-muted)]">
                      Version
                    </div>
                  </div>
                  <div>
                    <div className="text-lg font-mono font-bold text-[var(--cc-text)]">
                      {t2?.total_trainings ?? 0}
                    </div>
                    <div className="text-[9px] uppercase tracking-wider text-[var(--cc-text-muted)]">
                      Runs
                    </div>
                  </div>
                  <div>
                    <div className="text-lg font-mono font-bold text-[var(--cc-text)]">
                      {t2?.last_loss != null
                        ? t2.last_loss.toFixed(4)
                        : "—"}
                    </div>
                    <div className="text-[9px] uppercase tracking-wider text-[var(--cc-text-muted)]">
                      Last Loss
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2 text-[10px] text-[var(--cc-text-muted)]">
                  <span>{t2?.total_pairs_trained ?? 0} pairs trained total</span>
                  {t2?.last_training_time && (
                    <span className="ml-auto">
                      Last: {fmtTime(t2.last_training_time)}
                    </span>
                  )}
                </div>
              </div>
            </div>
          </ChartCard>
        </div>

        {/* ── Processing Time Trend (full width) ─────────── */}
        {history?.processing_time_trend &&
          history.processing_time_trend.length > 5 && (
            <ChartCard
              title="Processing Time Trend"
              subtitle={`${history.processing_time_trend.length} queries — end-to-end latency over time`}
            >
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={history.processing_time_trend}>
                  <defs>
                    <linearGradient
                      id="latGrad"
                      x1="0"
                      y1="0"
                      x2="0"
                      y2="1"
                    >
                      <stop
                        offset="5%"
                        stopColor={C.cyan}
                        stopOpacity={0.2}
                      />
                      <stop
                        offset="95%"
                        stopColor={C.cyan}
                        stopOpacity={0}
                      />
                    </linearGradient>
                  </defs>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke={C.border}
                    vertical={false}
                  />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={fmtDate}
                    tick={{ fontSize: 10, fill: C.textMuted }}
                    stroke={C.border}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    tick={{ fontSize: 10, fill: C.textMuted }}
                    stroke={C.border}
                    width={45}
                    label={{
                      value: "ms",
                      position: "insideTopLeft",
                      offset: 0,
                      fontSize: 10,
                      fill: C.textMuted,
                    }}
                  />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    labelFormatter={(l) => fmtDate(l as string)}
                    formatter={(v: number) => [`${v.toLocaleString()}ms`, "Latency"]}
                  />
                  <Area
                    type="monotone"
                    dataKey="ms"
                    stroke={C.cyan}
                    fill="url(#latGrad)"
                    strokeWidth={1.5}
                    dot={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </ChartCard>
          )}

        {/* ── Evaluation Score Distribution (full width) ──── */}
        {evalSummary?.scores && evalSummary.scores.length > 0 && (
          <ChartCard
            title="Evaluation Score Distribution"
            subtitle={`${evalSummary.count} queries evaluated — individual scores`}
          >
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={evalSummary.scores} barSize={6}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke={C.border}
                  vertical={false}
                />
                <XAxis
                  dataKey="query"
                  tick={false}
                  stroke={C.border}
                  label={{
                    value: "Queries",
                    position: "insideBottomRight",
                    offset: -5,
                    fontSize: 10,
                    fill: C.textMuted,
                  }}
                />
                <YAxis
                  domain={[0, 1]}
                  tick={{ fontSize: 10, fill: C.textMuted }}
                  stroke={C.border}
                  width={35}
                  tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                />
                <Tooltip
                  contentStyle={tooltipStyle}
                  formatter={(v: number) => [
                    `${(v * 100).toFixed(1)}%`,
                    "Score",
                  ]}
                  labelFormatter={(l) => l as string}
                />
                <Bar dataKey="score" radius={[2, 2, 0, 0]}>
                  {evalSummary.scores.map((entry, idx) => (
                    <Cell
                      key={idx}
                      fill={
                        entry.score >= 0.8
                          ? C.success
                          : entry.score >= 0.6
                            ? C.warning
                            : C.error
                      }
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        )}

        {/* ── Individual Query Feedback ──────────────── */}
        <QueryFeedbackSection
          details={history?.query_details || []}
          aggregates={history?.query_aggregates}
          search={querySearch}
          setSearch={setQuerySearch}
          filter={queryFilter}
          setFilter={setQueryFilter}
          expanded={expandedQuery}
          setExpanded={setExpandedQuery}
          page={queryPage}
          setPage={setQueryPage}
          perPage={QUERIES_PER_PAGE}
        />

        {/* Bottom spacer */}
        <div className="h-4" />
      </div>
    </div>
  );
}

// ─── Query Feedback Section ──────────────────────────────────────────

function SignalPill({
  label,
  value,
  detail,
}: {
  label: string;
  value: "fast" | "normal" | "slow" | "rich" | "sparse" | "rare" | "uncommon" | "common";
  detail: string;
}) {
  const colorMap: Record<string, string> = {
    fast: C.success,
    rich: C.success,
    rare: C.warning,
    normal: C.cyan,
    uncommon: C.purple,
    common: C.muted,
    slow: C.error,
    sparse: C.error,
  };
  const color = colorMap[value] || C.muted;
  return (
    <div className="flex flex-col items-center">
      <span
        className="text-[10px] font-semibold px-2 py-0.5 rounded-full"
        style={{
          color,
          backgroundColor: `${color}15`,
          border: `1px solid ${color}30`,
        }}
      >
        {value}
      </span>
      <span className="text-[8px] text-[var(--cc-text-muted)] mt-0.5">{label}</span>
      <span className="text-[7px] text-[var(--cc-text-muted)]">{detail}</span>
    </div>
  );
}

function ScoreBadge({ value, label }: { value: number | null; label: string }) {
  if (value == null) return null;
  const pct = value * 100;
  const color =
    pct >= 80 ? C.success : pct >= 60 ? C.warning : C.error;
  return (
    <div className="flex flex-col items-center">
      <div
        className="text-xs font-mono font-bold"
        style={{ color }}
      >
        {pct.toFixed(0)}%
      </div>
      <div className="text-[8px] text-[var(--cc-text-muted)] uppercase tracking-wider">
        {label}
      </div>
    </div>
  );
}

function RatingBadge({ rating }: { rating: "up" | "down" | null }) {
  if (rating === "up")
    return (
      <span className="inline-flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400">
        ▲ Positive
      </span>
    );
  if (rating === "down")
    return (
      <span className="inline-flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-full bg-red-500/20 text-red-400">
        ▼ Negative
      </span>
    );
  return (
    <span className="inline-flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-full bg-gray-500/20 text-gray-400">
      — None
    </span>
  );
}

function QueryFeedbackSection({
  details,
  aggregates,
  search,
  setSearch,
  filter,
  setFilter,
  expanded,
  setExpanded,
  page,
  setPage,
  perPage,
}: {
  details: QueryDetail[];
  aggregates?: RLHistory["query_aggregates"];
  search: string;
  setSearch: (s: string) => void;
  filter: "all" | "up" | "down" | "none";
  setFilter: (f: "all" | "up" | "down" | "none") => void;
  expanded: string | null;
  setExpanded: (id: string | null) => void;
  page: number;
  setPage: (p: number) => void;
  perPage: number;
}) {
  // Filter + search
  const filtered = details.filter((d) => {
    if (filter === "up" && d.rating !== "up" && d.eval_rating !== "up") return false;
    if (filter === "down" && d.rating !== "down" && d.eval_rating !== "down") return false;
    if (filter === "none" && (d.rating != null || d.eval_rating != null)) return false;
    if (search) {
      const s = search.toLowerCase();
      return (
        d.query.toLowerCase().includes(s) ||
        d.scenarios.some((sc) => sc.toLowerCase().includes(s)) ||
        d.primary_characteristic.toLowerCase().includes(s) ||
        d.domains.some((dom) => dom.toLowerCase().includes(s))
      );
    }
    return true;
  });

  // Sort: most recent first
  const sorted = [...filtered].reverse();
  const totalPages = Math.ceil(sorted.length / perPage);
  const pageItems = sorted.slice(page * perPage, (page + 1) * perPage);

  // Stats
  const withEval = details.filter((d) => d.eval_overall != null);
  const avgScore = withEval.length > 0
    ? withEval.reduce((s, d) => s + (d.eval_overall ?? 0), 0) / withEval.length
    : 0;
  const positiveCount = details.filter((d) => d.rating === "up" || d.eval_rating === "up").length;
  const negativeCount = details.filter((d) => d.rating === "down" || d.eval_rating === "down").length;

  if (details.length === 0) {
    return (
      <div className="rounded-xl border border-[var(--cc-border)] bg-[var(--cc-surface)] p-6">
        <h3 className="text-sm font-semibold text-[var(--cc-text)]">
          Individual Query Feedback
        </h3>
        <p className="text-xs text-[var(--cc-text-muted)] mt-2">
          No query data yet — run the RL agent to generate training data
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-[var(--cc-border)] bg-[var(--cc-surface)] p-5">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold text-[var(--cc-text)]">
            Individual Query Feedback
          </h3>
          <p className="text-[10px] text-[var(--cc-text-muted)] mt-0.5">
            {details.length} queries — {withEval.length} evaluated — avg{" "}
            {(avgScore * 100).toFixed(1)}% — {positiveCount} positive —{" "}
            {negativeCount} negative
          </p>
        </div>

        {/* Summary pills */}
        <div className="flex gap-2">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            <span className="text-[10px] font-semibold text-emerald-400">{positiveCount}</span>
          </div>
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-red-500/10 border border-red-500/20">
            <span className="w-1.5 h-1.5 rounded-full bg-red-400" />
            <span className="text-[10px] font-semibold text-red-400">{negativeCount}</span>
          </div>
        </div>
      </div>

      {/* Search + Filter bar */}
      <div className="flex gap-2 mb-3">
        <input
          type="text"
          placeholder="Search queries, scenarios, domains..."
          value={search}
          onChange={(e) => { setSearch(e.target.value); setPage(0); }}
          className="flex-1 bg-[var(--cc-bg)] border border-[var(--cc-border)] rounded-lg px-3 py-1.5 text-xs text-[var(--cc-text)] placeholder:text-[var(--cc-text-muted)] focus:outline-none focus:border-[var(--cc-accent)]"
        />
        {(["all", "up", "down", "none"] as const).map((f) => (
          <button
            key={f}
            onClick={() => { setFilter(f); setPage(0); }}
            className={`px-3 py-1.5 rounded-lg text-[10px] font-semibold uppercase tracking-wider border transition-colors ${
              filter === f
                ? "bg-[var(--cc-accent)] border-[var(--cc-accent)] text-white"
                : "bg-[var(--cc-bg)] border-[var(--cc-border)] text-[var(--cc-text-muted)] hover:border-[var(--cc-accent)]"
            }`}
          >
            {f === "none" ? "no rating" : f}
          </button>
        ))}
      </div>

      {/* Result count */}
      <div className="text-[10px] text-[var(--cc-text-muted)] mb-2">
        Showing {page * perPage + 1}–{Math.min((page + 1) * perPage, sorted.length)} of{" "}
        {sorted.length} results
      </div>

      {/* Query list */}
      <div className="space-y-1.5">
        {pageItems.map((d) => {
          const isExpanded = expanded === d.query_id;
          const effectiveRating = d.rating || d.eval_rating;
          const hasEval = d.eval_overall != null;
          const latMs = d.processing_time_ms;

          return (
            <div key={d.query_id} className="group">
              {/* Row */}
              <button
                onClick={() => setExpanded(isExpanded ? null : d.query_id)}
                className="w-full text-left flex items-center gap-3 px-3 py-2.5 rounded-lg bg-[var(--cc-bg)] border border-transparent hover:border-[var(--cc-border)] transition-colors"
              >
                {/* Reward indicator bar */}
                <div
                  className="w-1 h-8 rounded-full flex-shrink-0"
                  style={{
                    backgroundColor:
                      effectiveRating === "up"
                        ? C.success
                        : effectiveRating === "down"
                          ? C.error
                          : C.border,
                  }}
                />

                {/* Query text + metadata */}
                <div className="flex-1 min-w-0">
                  <div className="text-xs text-[var(--cc-text)] truncate font-medium">
                    {d.query || "(empty query)"}
                  </div>
                  <div className="flex items-center gap-2 mt-0.5">
                    <span className="text-[9px] text-[var(--cc-text-muted)]">
                      {fmtDate(d.timestamp)}
                    </span>
                    <span className="text-[9px] px-1.5 py-0 rounded bg-[var(--cc-surface)] text-[var(--cc-text-muted)]">
                      {d.intent_type}
                    </span>
                    {d.domains.map((dom) => (
                      <span
                        key={dom}
                        className="text-[9px] px-1.5 py-0 rounded bg-indigo-500/10 text-indigo-400"
                      >
                        {dom}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Scenarios */}
                <div className="flex gap-1 flex-shrink-0 max-w-[200px] overflow-hidden">
                  {d.scenarios.slice(0, 4).map((sc, i) => (
                    <span
                      key={i}
                      className="text-[8px] px-1.5 py-0.5 rounded-full bg-[var(--cc-surface)] text-[var(--cc-text-muted)] whitespace-nowrap"
                    >
                      {sc}
                    </span>
                  ))}
                  {d.scenarios.length > 4 && (
                    <span className="text-[8px] text-[var(--cc-text-muted)]">
                      +{d.scenarios.length - 4}
                    </span>
                  )}
                </div>

                {/* Latency */}
                <div className="flex-shrink-0 w-16 text-right">
                  <span
                    className="text-[10px] font-mono"
                    style={{
                      color:
                        latMs < 1000 ? C.success : latMs < 5000 ? C.cyan : latMs < 10000 ? C.warning : C.error,
                    }}
                  >
                    {latMs < 1000 ? `${latMs}ms` : `${(latMs / 1000).toFixed(1)}s`}
                  </span>
                </div>

                {/* Score */}
                <div className="flex-shrink-0 w-12 text-right">
                  {hasEval ? (
                    <span
                      className="text-xs font-mono font-bold"
                      style={{
                        color:
                          (d.eval_overall ?? 0) >= 0.8
                            ? C.success
                            : (d.eval_overall ?? 0) >= 0.6
                              ? C.warning
                              : C.error,
                      }}
                    >
                      {((d.eval_overall ?? 0) * 100).toFixed(0)}%
                    </span>
                  ) : (
                    <span className="text-[10px] text-[var(--cc-text-muted)]">—</span>
                  )}
                </div>

                {/* Rating badge */}
                <div className="flex-shrink-0">
                  <RatingBadge rating={effectiveRating} />
                </div>

                {/* Expand arrow */}
                <span
                  className="text-[var(--cc-text-muted)] text-xs transition-transform"
                  style={{ transform: isExpanded ? "rotate(180deg)" : "rotate(0deg)" }}
                >
                  ▼
                </span>
              </button>

              {/* Expanded detail */}
              {isExpanded && (
                <div className="mx-3 mt-1 mb-2 p-4 rounded-lg bg-[var(--cc-surface)] border border-[var(--cc-border)]">
                  {/* Top: heading + reward */}
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <div className="text-xs font-semibold text-[var(--cc-text)]">
                        {d.heading || d.query}
                      </div>
                      <div className="text-[10px] text-[var(--cc-text-muted)] mt-0.5">
                        {d.select_method && `Method: ${d.select_method}`}
                        {d.confidence != null && ` · Confidence: ${(d.confidence * 100).toFixed(0)}%`}
                        {d.primary_characteristic && ` · Type: ${d.primary_characteristic}`}
                      </div>
                    </div>
                    {d.reward != null && (
                      <div className="text-right">
                        <div
                          className="text-lg font-mono font-bold"
                          style={{ color: d.reward >= 0 ? C.success : C.error }}
                        >
                          {d.reward >= 0 ? "+" : ""}
                          {d.reward.toFixed(3)}
                        </div>
                        <div className="text-[8px] uppercase text-[var(--cc-text-muted)]">
                          Computed Reward
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Evaluation scores grid */}
                  {hasEval ? (
                    <div className="flex items-center gap-6 mb-3 py-2 px-3 rounded-lg bg-[var(--cc-bg)]">
                      <ScoreBadge value={d.eval_overall} label="Overall" />
                      <ScoreBadge value={d.eval_relevance} label="Relevance" />
                      <ScoreBadge value={d.eval_accuracy} label="Accuracy" />
                      <ScoreBadge value={d.eval_quality} label="Quality" />
                      <ScoreBadge value={d.eval_latency} label="Latency" />

                      {/* Mini bar visualization */}
                      <div className="flex-1 flex items-end gap-0.5 h-6 ml-4">
                        {[
                          { v: d.eval_overall, c: C.accent },
                          { v: d.eval_relevance, c: C.purple },
                          { v: d.eval_accuracy, c: C.success },
                          { v: d.eval_quality, c: C.cyan },
                          { v: d.eval_latency, c: C.warning },
                        ].map((bar, i) => (
                          <div
                            key={i}
                            className="flex-1 rounded-t"
                            style={{
                              height: `${(bar.v ?? 0) * 100}%`,
                              backgroundColor: bar.c,
                              opacity: 0.7,
                            }}
                          />
                        ))}
                      </div>
                    </div>
                  ) : d.feedback_source === "user_direct" ? (
                    <div className="flex items-center gap-3 mb-3 py-2 px-3 rounded-lg bg-[var(--cc-bg)] border border-dashed border-[var(--cc-border)]">
                      <span className="text-[10px] text-[var(--cc-text-muted)]">
                        No AI evaluation scores — this query was rated by a user directly
                        ({d.rating === "up" ? "thumbs up" : d.rating === "down" ? "thumbs down" : "no rating"}).
                        Multi-dimensional scores (relevance, accuracy, quality, latency) are only available
                        for queries evaluated by the AI evaluation agent.
                      </span>
                    </div>
                  ) : null}

                  {/* Scenarios */}
                  <div className="mb-3">
                    <div className="text-[9px] uppercase tracking-wider text-[var(--cc-text-muted)] mb-1">
                      Widget Scenarios ({d.widget_count})
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {d.scenarios.map((sc, i) => (
                        <span
                          key={i}
                          className="text-[10px] px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-300 border border-indigo-500/20"
                        >
                          {sc}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* RL Impact Summary */}
                  <div className="pt-3 border-t border-[var(--cc-border)]">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-[9px] uppercase tracking-wider text-[var(--cc-text-muted)]">
                        RL Learning Impact
                      </span>
                      {/* Feedback source pill */}
                      <span className={`text-[9px] px-2 py-0.5 rounded-full font-semibold ${
                        d.feedback_source === "user_direct"
                          ? "bg-blue-500/15 text-blue-400 border border-blue-500/25"
                          : d.feedback_source === "eval_agent"
                            ? "bg-purple-500/15 text-purple-400 border border-purple-500/25"
                            : d.feedback_source === "both"
                              ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/25"
                              : "bg-gray-500/15 text-gray-400 border border-gray-500/25"
                      }`}>
                        {d.feedback_source === "user_direct" ? "User Feedback"
                          : d.feedback_source === "eval_agent" ? "AI Evaluated"
                          : d.feedback_source === "both" ? "User + AI Eval"
                          : "Implicit Only"}
                      </span>
                      {/* Query clarity pill */}
                      {d.query_clarity === "ambiguous_query" && (
                        <span className="text-[9px] px-2 py-0.5 rounded-full font-semibold bg-amber-500/15 text-amber-400 border border-amber-500/25">
                          Ambiguous Query
                        </span>
                      )}
                      {d.query_clarity === "system_mismatch" && (
                        <span className="text-[9px] px-2 py-0.5 rounded-full font-semibold bg-red-500/15 text-red-400 border border-red-500/25">
                          Response Mismatch
                        </span>
                      )}
                    </div>

                    {/* Feedback source explanation */}
                    {d.feedback_source === "user_direct" && !hasEval && (
                      <div className="flex items-start gap-2 text-[10px] leading-relaxed mb-2.5 px-3 py-2 rounded-lg bg-blue-500/5 border border-blue-500/10">
                        <span className="text-blue-400 mt-0.5">ℹ</span>
                        <span className="text-blue-300/80">
                          This query was rated directly by a user (not evaluated by the AI evaluation agent).
                          No multi-dimensional scores (relevance, accuracy, quality, latency) are available —
                          only the user&apos;s thumbs-up/down rating and the computed reward signal.
                        </span>
                      </div>
                    )}

                    {/* Ambiguous query explanation */}
                    {d.query_clarity === "ambiguous_query" && d.correction && (
                      <div className="flex items-start gap-2 text-[10px] leading-relaxed mb-2.5 px-3 py-2 rounded-lg bg-amber-500/5 border border-amber-500/10">
                        <span className="text-amber-400 mt-0.5">⚠</span>
                        <span className="text-amber-300/80">
                          The correction &quot;{d.correction}&quot; indicates the <strong>query was ambiguous</strong>,
                          not that the system responded incorrectly.
                          The system answered &quot;{d.query}&quot; literally — the user wanted a more
                          specific result. The RL system uses this to learn that vague queries in the
                          &quot;{d.primary_characteristic || "general"}&quot; domain may need clarification
                          or more targeted widget selection.
                        </span>
                      </div>
                    )}

                    {/* Signal indicators */}
                    <div className="flex gap-3 mb-2.5">
                      <SignalPill
                        label="Latency"
                        value={d.latency_signal}
                        detail={`${d.processing_time_ms}ms vs ${aggregates?.avg_processing_ms?.toFixed(0) ?? "—"}ms avg`}
                      />
                      <SignalPill
                        label="Widgets"
                        value={d.widget_signal}
                        detail={`${d.widget_count} vs ${aggregates?.avg_widget_count?.toFixed(1) ?? "—"} avg`}
                      />
                      <SignalPill
                        label="Query Type"
                        value={d.diversity_signal}
                        detail={d.primary_characteristic || "unknown"}
                      />
                      <div className="flex flex-col items-center">
                        <span className="text-[10px] font-mono text-[var(--cc-text)]">
                          #{d.buffer_position}
                        </span>
                        <span className="text-[8px] text-[var(--cc-text-muted)]">
                          Buffer Position
                        </span>
                      </div>
                    </div>

                    {/* RL contribution text */}
                    <div className="space-y-1.5">
                      {/* Tier 1: Scorer contribution */}
                      <div className="flex items-start gap-2 text-[11px] leading-relaxed">
                        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 mt-1.5 flex-shrink-0" />
                        <span className="text-[var(--cc-text-muted)]">
                          <span className="text-cyan-400 font-semibold">Tier 1 Scorer:</span>{" "}
                          {d.reward != null ? (
                            <>
                              Explicit reward of{" "}
                              <span className="font-mono" style={{ color: d.reward >= 0 ? C.success : C.error }}>
                                {d.reward >= 0 ? "+" : ""}{d.reward.toFixed(3)}
                              </span>
                              {" "}directly updated scorer weights.
                              {d.query_clarity === "ambiguous_query" ? (
                                <> The negative signal teaches the scorer that broad queries like &quot;{d.query}&quot; may need more targeted widget selection or disambiguation in the &quot;{d.primary_characteristic || "general"}&quot; domain.</>
                              ) : d.rating === "up" ? (
                                <> This positive feedback strengthened the scoring of [{d.scenarios.slice(0, 3).join(", ")}] for similar queries.</>
                              ) : d.query_clarity === "system_mismatch" ? (
                                <> The correction indicates the system selected wrong widgets — scorer weights adjusted to penalize this [{d.scenarios.slice(0, 3).join(", ")}] pattern.</>
                              ) : (
                                <> This negative feedback adjusted scorer weights to reduce the score for [{d.scenarios.slice(0, 3).join(", ")}] in this context.</>
                              )}
                            </>
                          ) : hasEval ? (
                            <>
                              Evaluation score of{" "}
                              <span className="font-mono" style={{ color: (d.eval_overall ?? 0) >= 0.6 ? C.success : C.error }}>
                                {((d.eval_overall ?? 0) * 100).toFixed(1)}%
                              </span>
                              {" "}submitted as feedback.
                              {(d.eval_overall ?? 0) >= 0.6
                                ? ` Scorer learned to assign higher rewards to [{${d.scenarios.slice(0, 3).join(", ")}}] widget pattern.`
                                : " Scorer learned to penalize this widget selection — " +
                                  (d.eval_relevance != null && d.eval_relevance < 0.5
                                    ? `scenario relevance was only ${(d.eval_relevance * 100).toFixed(0)}%.`
                                    : "overall quality below threshold."
                                  )
                              }
                            </>
                          ) : (
                            <>
                              Implicit signal from{" "}
                              <span className="font-mono" style={{ color: d.latency_signal === "fast" ? C.success : d.latency_signal === "slow" ? C.error : C.cyan }}>
                                {d.processing_time_ms.toLocaleString()}ms
                              </span>
                              {" "}latency and {d.widget_count} widgets.
                              {d.latency_signal === "fast"
                                ? " Fast response reinforces this intent→widget mapping in the scorer."
                                : d.latency_signal === "slow"
                                  ? " Slow response signals the scorer to explore alternative widget selections."
                                  : " Normal latency — neutral contribution to scorer weights."
                              }
                            </>
                          )}
                        </span>
                      </div>

                      {/* Tier 2: DPO contribution */}
                      <div className="flex items-start gap-2 text-[11px] leading-relaxed">
                        <span className="w-1.5 h-1.5 rounded-full bg-purple-400 mt-1.5 flex-shrink-0" />
                        <span className="text-[var(--cc-text-muted)]">
                          <span className="text-purple-400 font-semibold">Tier 2 DPO:</span>{" "}
                          {d.reward != null || hasEval ? (
                            <>
                              {d.query_clarity === "ambiguous_query" ? (
                                <>
                                  Used as a <span className="text-amber-400">rejected response</span> in DPO pairs due to query ambiguity —
                                  the model learns that &quot;{d.primary_characteristic || "general"}&quot; queries need more specific widget targeting
                                  rather than broad results.
                                </>
                              ) : (d.reward != null ? d.reward > 0 : (d.eval_overall ?? 0) >= 0.6) ? (
                                <>
                                  Used as a <span className="text-emerald-400">preferred response</span> in DPO training pairs.
                                  The model learns to reproduce [{d.scenarios.slice(0, 3).join(", ")}] for &quot;{d.primary_characteristic || "general"}&quot; queries.
                                </>
                              ) : (
                                <>
                                  Used as a <span className="text-red-400">rejected response</span> in DPO training pairs.
                                  The model learns to avoid this [{d.scenarios.slice(0, 3).join(", ")}] selection for &quot;{d.primary_characteristic || "general"}&quot; queries.
                                </>
                              )}
                              {aggregates?.dpo_pairs_generated ? (
                                <span className="text-[var(--cc-text-muted)]"> ({aggregates.dpo_pairs_generated} total DPO pairs generated.)</span>
                              ) : null}
                            </>
                          ) : (
                            <>
                              Stored in experience buffer as candidate for DPO pair generation.
                              {d.diversity_signal === "rare" ? (
                                <> As a <span className="text-amber-400">rare query type</span> (&quot;{d.primary_characteristic}&quot;), this provides valuable diversity for the training distribution.</>
                              ) : d.diversity_signal === "uncommon" ? (
                                <> This &quot;{d.primary_characteristic}&quot; query type adds useful variation to the training set.</>
                              ) : (
                                <> Common &quot;{d.primary_characteristic}&quot; query — compared against other responses of this type to build preference pairs.</>
                              )}
                            </>
                          )}
                        </span>
                      </div>

                      {/* Correction — shown differently based on query_clarity */}
                      {d.correction && (
                        <div className="flex items-start gap-2 text-[11px] leading-relaxed">
                          <span className="w-1.5 h-1.5 rounded-full bg-amber-400 mt-1.5 flex-shrink-0" />
                          <span className="text-[var(--cc-text-muted)]">
                            <span className="text-amber-400 font-semibold">Correction:</span>{" "}
                            &quot;{d.correction}&quot;
                            {d.query_clarity === "ambiguous_query" ? (
                              <> — the user&apos;s original query was too broad. The RL system learns to prefer more specific widget filtering for this domain.</>
                            ) : (
                              <> — this correction is used as a direct training signal to improve future widget selections.</>
                            )}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-3 pt-3 border-t border-[var(--cc-border)]">
          <button
            onClick={() => setPage(Math.max(0, page - 1))}
            disabled={page === 0}
            className="px-3 py-1 rounded-lg text-[10px] font-semibold bg-[var(--cc-bg)] border border-[var(--cc-border)] text-[var(--cc-text-muted)] hover:border-[var(--cc-accent)] disabled:opacity-30 disabled:hover:border-[var(--cc-border)]"
          >
            ← Previous
          </button>
          <span className="text-[10px] text-[var(--cc-text-muted)]">
            Page {page + 1} of {totalPages}
          </span>
          <button
            onClick={() => setPage(Math.min(totalPages - 1, page + 1))}
            disabled={page >= totalPages - 1}
            className="px-3 py-1 rounded-lg text-[10px] font-semibold bg-[var(--cc-bg)] border border-[var(--cc-border)] text-[var(--cc-text-muted)] hover:border-[var(--cc-accent)] disabled:opacity-30 disabled:hover:border-[var(--cc-border)]"
          >
            Next →
          </button>
        </div>
      )}
    </div>
  );
}
