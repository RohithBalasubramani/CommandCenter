/**
 * Debug Panel - Pipeline Visualization
 *
 * Shows the complete Layer 1 + Layer 2 pipeline status:
 * - Voice I/O status (PersonaPlex connection)
 * - RAG pipeline status (embeddings, vector DB, LLM)
 * - Real-time query flow
 * - Response breakdown
 *
 * Toggle with Ctrl+D
 */

"use client";

import { useState, useEffect, useCallback } from "react";
import { config } from "@/lib/config";

// Types for pipeline status
interface PipelineStatus {
  layer1: {
    personaplex: "connected" | "disconnected" | "connecting";
    audioInput: boolean;
    audioOutput: boolean;
  };
  layer2: {
    backend: "online" | "offline" | "unknown";
    ollama: "online" | "offline" | "unknown";
    ragIndex: {
      equipment: number;
      alerts: number;
      maintenance: number;
    };
  };
}

interface QueryLog {
  id: string;
  timestamp: Date;
  transcript: string;
  intent: {
    type: string;
    domains: string[];
    confidence: number;
  } | null;
  ragResults: {
    domain: string;
    success: boolean;
    executionTimeMs: number;
    documentCount: number;
  }[];
  llmResponse: string;
  processingTimeMs: number;
  layout: {
    widgetCount: number;
    widgets: string[];
  };
}

interface DebugPanelProps {
  personaplexStatus?: "connected" | "disconnected" | "connecting";
  onClose?: () => void;
}

export function DebugPanel({ personaplexStatus = "disconnected", onClose }: DebugPanelProps) {
  const [isVisible, setIsVisible] = useState(false);
  const [status, setStatus] = useState<PipelineStatus>({
    layer1: {
      personaplex: personaplexStatus,
      audioInput: false,
      audioOutput: false,
    },
    layer2: {
      backend: "unknown",
      ollama: "unknown",
      ragIndex: {
        equipment: 0,
        alerts: 0,
        maintenance: 0,
      },
    },
  });
  const [queryLogs, setQueryLogs] = useState<QueryLog[]>([]);
  const [testQuery, setTestQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  // Keyboard shortcut (Ctrl+D)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.key === "d") {
        e.preventDefault();
        setIsVisible((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Fetch backend status
  const fetchStatus = useCallback(async () => {
    try {
      // Check backend health
      const backendRes = await fetch(`${config.api.baseUrl}/api/layer2/rag/industrial/health/`, {
        method: "GET",
      }).catch(() => null);

      const backendOnline = backendRes?.ok ?? false;

      // Check Ollama
      const ollamaRes = await fetch("http://localhost:11434/api/tags", {
        method: "GET",
      }).catch(() => null);

      const ollamaOnline = ollamaRes?.ok ?? false;

      // Get RAG stats if backend is online
      let ragIndex = { equipment: 0, alerts: 0, maintenance: 0 };
      if (backendOnline) {
        try {
          const statsRes = await fetch(`${config.api.baseUrl}/api/layer2/rag/industrial/health/`);
          if (statsRes.ok) {
            const data = await statsRes.json();
            ragIndex = {
              equipment: data.equipment_count || 0,
              alerts: data.alerts_count || 0,
              maintenance: data.maintenance_count || 0,
            };
          }
        } catch {
          // Ignore stats fetch errors
        }
      }

      setStatus((prev) => ({
        ...prev,
        layer1: {
          ...prev.layer1,
          personaplex: personaplexStatus,
        },
        layer2: {
          backend: backendOnline ? "online" : "offline",
          ollama: ollamaOnline ? "online" : "offline",
          ragIndex,
        },
      }));
    } catch (err) {
      console.error("Failed to fetch status:", err);
    }
  }, [personaplexStatus]);

  // Poll status every 5 seconds
  useEffect(() => {
    if (isVisible) {
      fetchStatus();
      const interval = setInterval(fetchStatus, 5000);
      return () => clearInterval(interval);
    }
  }, [isVisible, fetchStatus]);

  // Test query function
  const runTestQuery = async () => {
    if (!testQuery.trim()) return;

    setIsLoading(true);
    const startTime = Date.now();

    try {
      const res = await fetch(`${config.api.baseUrl}/api/layer2/orchestrate/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          transcript: testQuery,
          session_id: "debug-session",
          context: {},
        }),
      });

      const data = await res.json();
      const endTime = Date.now();

      const newLog: QueryLog = {
        id: `${Date.now()}`,
        timestamp: new Date(),
        transcript: testQuery,
        intent: data.intent
          ? {
              type: data.intent.type,
              domains: data.intent.domains,
              confidence: data.intent.confidence,
            }
          : null,
        ragResults:
          data.rag_results?.map((r: any) => ({
            domain: r.domain,
            success: r.success,
            executionTimeMs: r.execution_time_ms,
            documentCount: r.data?.devices?.length || r.data?.alerts?.length || 0,
          })) || [],
        llmResponse: data.voice_response || "",
        processingTimeMs: data.processing_time_ms || endTime - startTime,
        layout: {
          widgetCount: data.layout_json?.widgets?.length || 0,
          widgets: data.layout_json?.widgets?.map((w: any) => w.id) || [],
        },
      };

      setQueryLogs((prev) => [newLog, ...prev.slice(0, 9)]);
      setTestQuery("");
    } catch (err) {
      console.error("Query failed:", err);
    } finally {
      setIsLoading(false);
    }
  };

  if (!isVisible) {
    return (
      <button
        onClick={() => setIsVisible(true)}
        className="fixed bottom-4 right-4 bg-gray-800 text-white px-3 py-2 rounded-lg text-sm font-mono opacity-50 hover:opacity-100 transition-opacity z-50"
      >
        Debug (Ctrl+D)
      </button>
    );
  }

  const StatusBadge = ({ status }: { status: string }) => {
    const colors: Record<string, string> = {
      online: "bg-green-500",
      connected: "bg-green-500",
      offline: "bg-red-500",
      disconnected: "bg-red-500",
      connecting: "bg-yellow-500",
      unknown: "bg-gray-500",
    };
    return (
      <span
        className={`inline-block w-2 h-2 rounded-full ${colors[status] || "bg-gray-500"}`}
      />
    );
  };

  return (
    <div className="fixed inset-0 bg-black/80 z-50 overflow-auto">
      <div className="max-w-6xl mx-auto p-4">
        {/* Header */}
        <div className="flex justify-between items-center mb-4">
          <h1 className="text-xl font-bold text-white font-mono">
            Command Center Debug Panel
          </h1>
          <button
            onClick={() => {
              setIsVisible(false);
              onClose?.();
            }}
            className="text-gray-400 hover:text-white text-2xl"
          >
            &times;
          </button>
        </div>

        {/* Pipeline Status */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          {/* Layer 1 Status */}
          <div className="bg-gray-900 rounded-lg p-4 border border-gray-700">
            <h2 className="text-lg font-bold text-cyan-400 mb-3 font-mono">
              Layer 1: Voice I/O
            </h2>
            <div className="space-y-2 text-sm font-mono">
              <div className="flex justify-between items-center">
                <span className="text-gray-400">PersonaPlex</span>
                <span className="flex items-center gap-2">
                  <StatusBadge status={status.layer1.personaplex} />
                  <span className="text-white">{status.layer1.personaplex}</span>
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Model</span>
                <span className="text-white">PersonaPlex-7B (Moshi)</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Protocol</span>
                <span className="text-white">WSS Binary (Opus 24kHz)</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Port</span>
                <span className="text-white">8998</span>
              </div>
            </div>
          </div>

          {/* Layer 2 Status */}
          <div className="bg-gray-900 rounded-lg p-4 border border-gray-700">
            <h2 className="text-lg font-bold text-yellow-400 mb-3 font-mono">
              Layer 2: AI + RAG
            </h2>
            <div className="space-y-2 text-sm font-mono">
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Backend API</span>
                <span className="flex items-center gap-2">
                  <StatusBadge status={status.layer2.backend} />
                  <span className="text-white">{status.layer2.backend}</span>
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Ollama LLM</span>
                <span className="flex items-center gap-2">
                  <StatusBadge status={status.layer2.ollama} />
                  <span className="text-white">{status.layer2.ollama}</span>
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">LLM Model</span>
                <span className="text-white">qwen3:7b</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Embeddings</span>
                <span className="text-white">all-MiniLM-L6-v2</span>
              </div>
            </div>
          </div>
        </div>

        {/* RAG Index Stats */}
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-700 mb-6">
          <h2 className="text-lg font-bold text-green-400 mb-3 font-mono">
            RAG Index (ChromaDB)
          </h2>
          <div className="grid grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-3xl font-bold text-white">
                {status.layer2.ragIndex.equipment}
              </div>
              <div className="text-sm text-gray-400">Equipment</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-white">
                {status.layer2.ragIndex.alerts}
              </div>
              <div className="text-sm text-gray-400">Alerts</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-white">
                {status.layer2.ragIndex.maintenance}
              </div>
              <div className="text-sm text-gray-400">Maintenance</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-white">
                {status.layer2.ragIndex.equipment +
                  status.layer2.ragIndex.alerts +
                  status.layer2.ragIndex.maintenance}
              </div>
              <div className="text-sm text-gray-400">Total Docs</div>
            </div>
          </div>
        </div>

        {/* Test Query */}
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-700 mb-6">
          <h2 className="text-lg font-bold text-purple-400 mb-3 font-mono">
            Test Query
          </h2>
          <div className="flex gap-2">
            <input
              type="text"
              value={testQuery}
              onChange={(e) => setTestQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && runTestQuery()}
              placeholder="e.g., What's the status of the chillers?"
              className="flex-1 bg-gray-800 border border-gray-600 rounded px-3 py-2 text-white font-mono text-sm focus:outline-none focus:border-purple-500"
            />
            <button
              onClick={runTestQuery}
              disabled={isLoading}
              className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded font-mono text-sm disabled:opacity-50"
            >
              {isLoading ? "..." : "Send"}
            </button>
          </div>
          <div className="mt-2 text-xs text-gray-500 font-mono">
            Try: "Show all transformers" | "Any critical alerts?" | "Status of pumps"
          </div>
        </div>

        {/* Query Logs */}
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-700">
          <h2 className="text-lg font-bold text-orange-400 mb-3 font-mono">
            Query Log (Last 10)
          </h2>
          {queryLogs.length === 0 ? (
            <div className="text-gray-500 text-sm font-mono">
              No queries yet. Try sending a test query above.
            </div>
          ) : (
            <div className="space-y-4">
              {queryLogs.map((log) => (
                <div
                  key={log.id}
                  className="bg-gray-800 rounded p-3 border border-gray-700"
                >
                  {/* Query Header */}
                  <div className="flex justify-between items-start mb-2">
                    <div className="text-white font-mono text-sm">
                      "{log.transcript}"
                    </div>
                    <div className="text-xs text-gray-500 font-mono">
                      {log.processingTimeMs}ms
                    </div>
                  </div>

                  {/* Intent */}
                  {log.intent && (
                    <div className="mb-2">
                      <span className="text-xs text-gray-500 font-mono">Intent: </span>
                      <span className="text-xs text-cyan-400 font-mono">
                        {log.intent.type}
                      </span>
                      <span className="text-xs text-gray-500 font-mono"> | Domains: </span>
                      <span className="text-xs text-yellow-400 font-mono">
                        {log.intent.domains.join(", ")}
                      </span>
                      <span className="text-xs text-gray-500 font-mono">
                        {" "}
                        | Confidence: {(log.intent.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  )}

                  {/* RAG Results */}
                  {log.ragResults.length > 0 && (
                    <div className="mb-2">
                      <span className="text-xs text-gray-500 font-mono">RAG: </span>
                      {log.ragResults.map((r, i) => (
                        <span key={i} className="text-xs font-mono mr-2">
                          <span className={r.success ? "text-green-400" : "text-red-400"}>
                            {r.domain}
                          </span>
                          <span className="text-gray-500">
                            ({r.executionTimeMs}ms, {r.documentCount} docs)
                          </span>
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Layout */}
                  <div className="mb-2">
                    <span className="text-xs text-gray-500 font-mono">Layout: </span>
                    <span className="text-xs text-purple-400 font-mono">
                      {log.layout.widgetCount} widgets
                    </span>
                    {log.layout.widgets.length > 0 && (
                      <span className="text-xs text-gray-500 font-mono">
                        {" "}
                        ({log.layout.widgets.join(", ")})
                      </span>
                    )}
                  </div>

                  {/* Response */}
                  <div className="bg-gray-900 rounded p-2 mt-2">
                    <span className="text-xs text-gray-500 font-mono block mb-1">
                      Response:
                    </span>
                    <span className="text-sm text-green-300 font-mono">
                      {log.llmResponse || "(No response)"}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Pipeline Diagram */}
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-700 mt-6">
          <h2 className="text-lg font-bold text-blue-400 mb-3 font-mono">
            Pipeline Flow
          </h2>
          <div className="font-mono text-xs text-gray-400 whitespace-pre">
{`
┌─────────────────────────────────────────────────────────────────────────────┐
│                           COMMAND CENTER PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   LAYER 1: VOICE I/O                                                         │
│   ┌──────────────┐      ┌─────────────────┐      ┌──────────────┐           │
│   │  Microphone  │ ───> │  PersonaPlex-7B │ ───> │   Speaker    │           │
│   │  (Opus 24k)  │      │  (Moshi WSS)    │      │  (Opus 48k)  │           │
│   └──────────────┘      └────────┬────────┘      └──────────────┘           │
│                                  │                                           │
│                           Transcript                                         │
│                                  │                                           │
│   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   │
│                                  │                                           │
│   LAYER 2: AI + RAG              ▼                                           │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │                       ORCHESTRATOR                                │      │
│   │  ┌────────────┐    ┌────────────┐    ┌────────────┐              │      │
│   │  │ 2A: Intent │ -> │ 2B: RAG    │ -> │ 2C: Layout │              │      │
│   │  │  Parser    │    │  Pipeline  │    │  + Response│              │      │
│   │  └────────────┘    └─────┬──────┘    └────────────┘              │      │
│   └──────────────────────────┼───────────────────────────────────────┘      │
│                              │                                               │
│                    ┌─────────┴─────────┐                                    │
│                    │                   │                                    │
│              ┌─────▼─────┐      ┌──────▼──────┐                             │
│              │ ChromaDB  │      │ Ollama LLM  │                             │
│              │ (Vectors) │      │ (qwen3:7b)  │                             │
│              └───────────┘      └─────────────┘                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
`}
          </div>
        </div>
      </div>
    </div>
  );
}

export default DebugPanel;
