"use client";

import React from "react";
import {
  AlertTriangle,
  CheckCircle,
  HelpCircle,
  ArrowRight,
  Shield,
  Clock,
  Database,
  Brain,
} from "lucide-react";

interface KnownFact {
  statement: string;
  source: string;
  freshness: string;
  confidence: number;
}

interface UnknownFactor {
  description: string;
  why_unknown: string;
  impact: "low" | "medium" | "high";
  check_action: string;
}

interface NextStepItem {
  action: string;
  automated: boolean;
  priority: "low" | "medium" | "high";
}

interface ConstraintViolation {
  type: string;
  message: string;
  severity: "warning" | "error";
}

interface UncertaintyData {
  overall_confidence: number;
  known_facts: KnownFact[];
  unknown_factors: UnknownFactor[];
  next_steps: NextStepItem[];
  constraint_violations: ConstraintViolation[];
}

const sourceIcons: Record<string, React.ReactNode> = {
  timeseries: <Clock size={12} />,
  rag: <Database size={12} />,
  reasoning: <Brain size={12} />,
  site_memory: <Database size={12} />,
};

const freshnessColors: Record<string, string> = {
  live: "#22c55e",
  "5min": "#84cc16",
  "1hr": "#eab308",
  stale: "#ef4444",
  unknown: "#6b7280",
  computed: "#8b5cf6",
};

const impactColors: Record<string, string> = {
  low: "#6b7280",
  medium: "#eab308",
  high: "#ef4444",
};

const priorityColors: Record<string, string> = {
  low: "#6b7280",
  medium: "#3b82f6",
  high: "#ef4444",
};

function ConfidenceGauge({ confidence }: { confidence: number }) {
  const pct = Math.round(confidence * 100);
  const color =
    pct >= 70 ? "#22c55e" : pct >= 40 ? "#eab308" : "#ef4444";
  const circumference = 2 * Math.PI * 40;
  const offset = circumference * (1 - confidence);

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
      <svg width={48} height={48} viewBox="0 0 100 100">
        <circle
          cx={50}
          cy={50}
          r={40}
          fill="none"
          stroke="#1e293b"
          strokeWidth={8}
        />
        <circle
          cx={50}
          cy={50}
          r={40}
          fill="none"
          stroke={color}
          strokeWidth={8}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          transform="rotate(-90 50 50)"
        />
        <text
          x={50}
          y={55}
          textAnchor="middle"
          fill={color}
          fontSize={22}
          fontWeight="bold"
        >
          {pct}
        </text>
      </svg>
      <div>
        <div style={{ color: "#94a3b8", fontSize: 11 }}>
          Overall Confidence
        </div>
        <div style={{ color, fontWeight: 600, fontSize: 14 }}>
          {pct >= 70 ? "High" : pct >= 40 ? "Moderate" : "Low"}
        </div>
      </div>
    </div>
  );
}

function FreshnessBadge({ freshness }: { freshness: string }) {
  return (
    <span
      style={{
        display: "inline-block",
        padding: "1px 6px",
        borderRadius: 4,
        fontSize: 10,
        fontWeight: 600,
        color: "#fff",
        background: freshnessColors[freshness] || "#6b7280",
      }}
    >
      {freshness}
    </span>
  );
}

function KnownSection({ facts }: { facts: KnownFact[] }) {
  if (!facts.length) return null;
  return (
    <div style={{ marginBottom: 12 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          marginBottom: 6,
          color: "#22c55e",
          fontSize: 12,
          fontWeight: 600,
        }}
      >
        <CheckCircle size={14} />
        Known Facts
      </div>
      {facts.map((f, i) => (
        <div
          key={i}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "4px 8px",
            background: "rgba(34, 197, 94, 0.08)",
            borderRadius: 4,
            marginBottom: 3,
            fontSize: 12,
            color: "#e2e8f0",
          }}
        >
          <span style={{ color: "#94a3b8", flexShrink: 0 }}>
            {sourceIcons[f.source] || <Database size={12} />}
          </span>
          <span style={{ flex: 1 }}>{f.statement}</span>
          <FreshnessBadge freshness={f.freshness} />
        </div>
      ))}
    </div>
  );
}

function UnknownSection({ unknowns }: { unknowns: UnknownFactor[] }) {
  if (!unknowns.length) return null;
  return (
    <div style={{ marginBottom: 12 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          marginBottom: 6,
          color: "#eab308",
          fontSize: 12,
          fontWeight: 600,
        }}
      >
        <HelpCircle size={14} />
        Unknown Factors
      </div>
      {unknowns.map((u, i) => (
        <div
          key={i}
          style={{
            padding: "6px 8px",
            background: "rgba(234, 179, 8, 0.08)",
            borderRadius: 4,
            marginBottom: 3,
            fontSize: 12,
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              color: "#e2e8f0",
            }}
          >
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: "50%",
                background: impactColors[u.impact],
                flexShrink: 0,
              }}
            />
            {u.description}
          </div>
          {u.check_action && (
            <div style={{ color: "#94a3b8", fontSize: 11, marginTop: 2, paddingLeft: 12 }}>
              Check: {u.check_action}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function NextStepsSection({ steps }: { steps: NextStepItem[] }) {
  if (!steps.length) return null;
  return (
    <div style={{ marginBottom: 12 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          marginBottom: 6,
          color: "#3b82f6",
          fontSize: 12,
          fontWeight: 600,
        }}
      >
        <ArrowRight size={14} />
        Recommended Next Steps
      </div>
      {steps.map((s, i) => (
        <div
          key={i}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "4px 8px",
            background: "rgba(59, 130, 246, 0.08)",
            borderRadius: 4,
            marginBottom: 3,
            fontSize: 12,
            color: "#e2e8f0",
          }}
        >
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: "50%",
              background: priorityColors[s.priority],
              flexShrink: 0,
            }}
          />
          <span style={{ flex: 1 }}>{s.action}</span>
          {s.automated && (
            <span
              style={{
                fontSize: 10,
                padding: "1px 4px",
                borderRadius: 3,
                background: "rgba(59, 130, 246, 0.2)",
                color: "#60a5fa",
              }}
            >
              Auto
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

function ViolationsSection({
  violations,
}: {
  violations: ConstraintViolation[];
}) {
  if (!violations.length) return null;
  return (
    <div style={{ marginBottom: 12 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          marginBottom: 6,
          color: "#ef4444",
          fontSize: 12,
          fontWeight: 600,
        }}
      >
        <Shield size={14} />
        Constraint Violations
      </div>
      {violations.map((v, i) => (
        <div
          key={i}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            padding: "4px 8px",
            background:
              v.severity === "error"
                ? "rgba(239, 68, 68, 0.12)"
                : "rgba(234, 179, 8, 0.08)",
            borderRadius: 4,
            marginBottom: 3,
            fontSize: 12,
            color: v.severity === "error" ? "#fca5a5" : "#fde68a",
          }}
        >
          <AlertTriangle size={12} />
          <span>{v.message}</span>
        </div>
      ))}
    </div>
  );
}

export default function ScenarioComponent({
  demoData,
}: {
  demoData: UncertaintyData;
}) {
  const data: UncertaintyData = demoData || {
    overall_confidence: 0.5,
    known_facts: [],
    unknown_factors: [],
    next_steps: [],
    constraint_violations: [],
  };

  return (
    <div
      style={{
        background: "#0f172a",
        borderRadius: 8,
        padding: 16,
        color: "#e2e8f0",
        height: "100%",
        overflow: "auto",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 14,
          borderBottom: "1px solid #1e293b",
          paddingBottom: 10,
        }}
      >
        <div style={{ fontSize: 14, fontWeight: 600, color: "#f1f5f9" }}>
          Uncertainty Assessment
        </div>
        <ConfidenceGauge confidence={data.overall_confidence} />
      </div>

      <KnownSection facts={data.known_facts} />
      <UnknownSection unknowns={data.unknown_factors} />
      <ViolationsSection violations={data.constraint_violations} />
      <NextStepsSection steps={data.next_steps} />
    </div>
  );
}
