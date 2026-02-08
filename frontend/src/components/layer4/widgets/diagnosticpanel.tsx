// @ts-nocheck
/**
 * DiagnosticPanel — Cross-Widget Reasoning Display (Upgrade 2)
 *
 * Renders ranked hypotheses with confidence bars, evidence sections,
 * "what we don't know" section, and recommended next steps checklist.
 */

import React from "react";
import { AlertTriangle, CheckCircle, HelpCircle, ChevronRight, Search } from "lucide-react";

const COLORS = {
  bg: "rgba(15, 23, 42, 0.95)",
  card: "rgba(30, 41, 59, 0.9)",
  border: "rgba(100, 116, 139, 0.3)",
  text: "#e2e8f0",
  textMuted: "#94a3b8",
  accent: "#38bdf8",
  warning: "#f59e0b",
  critical: "#ef4444",
  success: "#22c55e",
  hypothesisBg: "rgba(56, 189, 248, 0.08)",
  unknownBg: "rgba(245, 158, 11, 0.08)",
  checkBg: "rgba(34, 197, 94, 0.08)",
};

interface HypothesisData {
  id?: string;
  statement: string;
  confidence: number;
  supporting_evidence: string[];
  contradicting_evidence: string[];
  check_steps: string[];
  source?: string;
}

interface DiagnosticData {
  hypotheses?: HypothesisData[];
  known_facts?: string[];
  unknown_factors?: string[];
  recommended_checks?: string[];
  reasoning_chain?: string[];
  confidence?: number;
  query_type?: string;
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = pct >= 70 ? COLORS.success : pct >= 40 ? COLORS.warning : COLORS.critical;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 120 }}>
      <div style={{
        flex: 1, height: 6, borderRadius: 3,
        background: "rgba(100,116,139,0.2)",
        overflow: "hidden",
      }}>
        <div style={{
          width: `${pct}%`, height: "100%", borderRadius: 3,
          background: color,
          transition: "width 0.3s ease",
        }} />
      </div>
      <span style={{ color, fontSize: 12, fontWeight: 600, minWidth: 36 }}>
        {pct}%
      </span>
    </div>
  );
}

function HypothesisCard({ hypothesis, rank }: { hypothesis: HypothesisData; rank: number }) {
  return (
    <div style={{
      background: COLORS.hypothesisBg,
      border: `1px solid ${COLORS.border}`,
      borderRadius: 8, padding: "12px 16px",
      marginBottom: 8,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}>
        <div style={{ flex: 1 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
            <span style={{
              background: COLORS.accent, color: "#0f172a",
              borderRadius: 10, width: 20, height: 20,
              display: "inline-flex", alignItems: "center", justifyContent: "center",
              fontSize: 11, fontWeight: 700,
            }}>
              {rank}
            </span>
            <span style={{ color: COLORS.text, fontSize: 13, fontWeight: 500, lineHeight: 1.4 }}>
              {hypothesis.statement}
            </span>
          </div>
        </div>
        <ConfidenceBar value={hypothesis.confidence} />
      </div>

      {/* Supporting Evidence */}
      {hypothesis.supporting_evidence?.length > 0 && (
        <div style={{ marginTop: 8, paddingLeft: 28 }}>
          {hypothesis.supporting_evidence.map((ev, i) => (
            <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 6, marginBottom: 3 }}>
              <CheckCircle size={12} color={COLORS.success} style={{ marginTop: 2, flexShrink: 0 }} />
              <span style={{ color: COLORS.textMuted, fontSize: 11 }}>{ev}</span>
            </div>
          ))}
        </div>
      )}

      {/* Contradicting Evidence */}
      {hypothesis.contradicting_evidence?.length > 0 && (
        <div style={{ marginTop: 6, paddingLeft: 28 }}>
          {hypothesis.contradicting_evidence.map((ev, i) => (
            <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 6, marginBottom: 3 }}>
              <AlertTriangle size={12} color={COLORS.critical} style={{ marginTop: 2, flexShrink: 0 }} />
              <span style={{ color: COLORS.textMuted, fontSize: 11 }}>{ev}</span>
            </div>
          ))}
        </div>
      )}

      {/* Check Steps */}
      {hypothesis.check_steps?.length > 0 && (
        <div style={{ marginTop: 8, paddingLeft: 28 }}>
          <span style={{ color: COLORS.textMuted, fontSize: 10, textTransform: "uppercase", letterSpacing: 1 }}>
            Verify:
          </span>
          {hypothesis.check_steps.map((step, i) => (
            <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 6, marginTop: 3 }}>
              <ChevronRight size={12} color={COLORS.accent} style={{ marginTop: 2, flexShrink: 0 }} />
              <span style={{ color: COLORS.textMuted, fontSize: 11 }}>{step}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function ScenarioComponent({ data }: { data: Record<string, unknown> }) {
  const demo = (data?.demoData || data) as DiagnosticData;

  if (!demo || (!demo.hypotheses?.length && !demo.known_facts?.length)) {
    return (
      <div style={{ padding: 20, color: COLORS.textMuted, textAlign: "center" }}>
        <HelpCircle size={24} style={{ marginBottom: 8 }} />
        <div>No diagnostic data available</div>
      </div>
    );
  }

  return (
    <div style={{
      height: "100%", display: "flex", flexDirection: "column",
      color: COLORS.text, fontSize: 13, overflow: "hidden",
    }}>
      {/* Header */}
      <div style={{
        padding: "12px 16px", borderBottom: `1px solid ${COLORS.border}`,
        display: "flex", alignItems: "center", gap: 8,
      }}>
        <Search size={16} color={COLORS.accent} />
        <span style={{ fontWeight: 600, fontSize: 14 }}>Diagnostic Analysis</span>
        {demo.confidence !== undefined && (
          <span style={{
            marginLeft: "auto", fontSize: 11, color: COLORS.textMuted,
            background: "rgba(100,116,139,0.15)", padding: "2px 8px",
            borderRadius: 10,
          }}>
            {demo.query_type || "diagnostic"} — {Math.round(demo.confidence * 100)}% confidence
          </span>
        )}
      </div>

      {/* Scrollable Content */}
      <div style={{ flex: 1, overflow: "auto", padding: 16 }}>
        {/* Known Facts */}
        {demo.known_facts?.length > 0 && (
          <div style={{ marginBottom: 16 }}>
            <div style={{
              fontSize: 11, fontWeight: 600, textTransform: "uppercase",
              letterSpacing: 1, color: COLORS.textMuted, marginBottom: 8,
            }}>
              Known Facts
            </div>
            {demo.known_facts.map((fact, i) => (
              <div key={i} style={{
                display: "flex", alignItems: "flex-start", gap: 6,
                marginBottom: 4, paddingLeft: 4,
              }}>
                <CheckCircle size={12} color={COLORS.success} style={{ marginTop: 2, flexShrink: 0 }} />
                <span style={{ fontSize: 12, color: COLORS.text }}>{fact}</span>
              </div>
            ))}
          </div>
        )}

        {/* Hypotheses */}
        {demo.hypotheses?.length > 0 && (
          <div style={{ marginBottom: 16 }}>
            <div style={{
              fontSize: 11, fontWeight: 600, textTransform: "uppercase",
              letterSpacing: 1, color: COLORS.textMuted, marginBottom: 8,
            }}>
              Possible Explanations
            </div>
            {demo.hypotheses.map((h, i) => (
              <HypothesisCard key={h.id || i} hypothesis={h} rank={i + 1} />
            ))}
          </div>
        )}

        {/* Unknown Factors */}
        {demo.unknown_factors?.length > 0 && (
          <div style={{
            marginBottom: 16, background: COLORS.unknownBg,
            border: `1px solid rgba(245, 158, 11, 0.2)`,
            borderRadius: 8, padding: "12px 16px",
          }}>
            <div style={{
              fontSize: 11, fontWeight: 600, textTransform: "uppercase",
              letterSpacing: 1, color: COLORS.warning, marginBottom: 8,
              display: "flex", alignItems: "center", gap: 6,
            }}>
              <HelpCircle size={14} />
              What We Don&apos;t Know
            </div>
            {demo.unknown_factors.map((uf, i) => (
              <div key={i} style={{
                fontSize: 12, color: COLORS.textMuted, marginBottom: 4, paddingLeft: 20,
              }}>
                • {uf}
              </div>
            ))}
          </div>
        )}

        {/* Recommended Checks */}
        {demo.recommended_checks?.length > 0 && (
          <div style={{
            background: COLORS.checkBg,
            border: `1px solid rgba(34, 197, 94, 0.2)`,
            borderRadius: 8, padding: "12px 16px",
          }}>
            <div style={{
              fontSize: 11, fontWeight: 600, textTransform: "uppercase",
              letterSpacing: 1, color: COLORS.success, marginBottom: 8,
              display: "flex", alignItems: "center", gap: 6,
            }}>
              <CheckCircle size={14} />
              Recommended Next Steps
            </div>
            {demo.recommended_checks.map((check, i) => (
              <div key={i} style={{
                display: "flex", alignItems: "flex-start", gap: 8,
                marginBottom: 6, paddingLeft: 4,
              }}>
                <input type="checkbox" style={{ marginTop: 3, accentColor: COLORS.success }} />
                <span style={{ fontSize: 12, color: COLORS.text }}>{check}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
