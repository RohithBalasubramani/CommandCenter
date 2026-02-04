"use client";

import React, { useEffect, useRef, useState } from "react";
import { MetricsSnapshot, MetricsStats } from "@/types/metrics";

interface MetricsDashboardProps {
  snapshot: MetricsSnapshot;
  latencyStats: MetricsStats;
  isVisible: boolean;
  onClose: () => void;
  onExport: () => void;
}

export default function MetricsDashboard({
  snapshot,
  latencyStats,
  isVisible,
  onClose,
  onExport,
}: MetricsDashboardProps) {
  if (!isVisible) return null;

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) return `${hours}h ${mins}m ${secs}s`;
    if (mins > 0) return `${mins}m ${secs}s`;
    return `${secs}s`;
  };

  const getLatestAudioQuality = () => {
    if (snapshot.audioQuality.length === 0) return null;
    return snapshot.audioQuality[snapshot.audioQuality.length - 1];
  };

  const audioQuality = getLatestAudioQuality();

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-6">
      <div className="bg-black/90 border border-white/20 rounded-lg w-full max-w-6xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
          <div>
            <h2 className="text-lg font-semibold text-white">
              Metrics Dashboard
            </h2>
            <p className="text-xs text-white/50 mt-0.5">
              Real-time performance monitoring
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={onExport}
              className="px-3 py-1.5 text-xs font-medium text-white/80 hover:text-white bg-white/10 hover:bg-white/20 rounded transition-colors"
            >
              Export JSON
            </button>
            <button
              onClick={onClose}
              className="w-8 h-8 flex items-center justify-center text-white/60 hover:text-white hover:bg-white/10 rounded transition-colors"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {/* Session Stats */}
          <div className="mb-6">
            <h3 className="text-sm font-medium text-white/80 mb-3">Session</h3>
            <div className="grid grid-cols-4 gap-3">
              <MetricCard
                label="Uptime"
                value={formatDuration(snapshot.session.uptime)}
                color="blue"
              />
              <MetricCard
                label="Audio Sent"
                value={formatBytes(snapshot.session.totalAudioSent)}
                color="blue"
              />
              <MetricCard
                label="Audio Received"
                value={formatBytes(snapshot.session.totalAudioReceived)}
                color="purple"
              />
              <MetricCard
                label="Chunks Processed"
                value={snapshot.session.totalChunksProcessed.toString()}
                color="green"
              />
            </div>
          </div>

          {/* Latency Stats */}
          <div className="mb-6">
            <h3 className="text-sm font-medium text-white/80 mb-3">
              Latency (ms)
            </h3>
            <div className="grid grid-cols-6 gap-3">
              <MetricCard
                label="Current"
                value={
                  snapshot.latency.length > 0
                    ? snapshot.latency[
                        snapshot.latency.length - 1
                      ].latency.toFixed(0)
                    : "—"
                }
                color="blue"
              />
              <MetricCard
                label="Average"
                value={latencyStats.avg.toFixed(0)}
                color="blue"
              />
              <MetricCard
                label="Min"
                value={latencyStats.min.toFixed(0)}
                color="green"
              />
              <MetricCard
                label="Max"
                value={latencyStats.max.toFixed(0)}
                color="red"
              />
              <MetricCard
                label="P95"
                value={latencyStats.p95.toFixed(0)}
                color="yellow"
              />
              <MetricCard
                label="P99"
                value={latencyStats.p99.toFixed(0)}
                color="orange"
              />
            </div>

            {/* Latency Graph */}
            <div className="mt-4">
              <SparklineGraph
                data={snapshot.latency.map((m) => m.latency)}
                label="Latency over time"
                color="blue"
                maxDataPoints={60}
              />
            </div>
          </div>

          {/* Audio Quality */}
          {audioQuality && (
            <div className="mb-6">
              <h3 className="text-sm font-medium text-white/80 mb-3">
                Audio Quality
              </h3>
              <div className="grid grid-cols-4 gap-3">
                <MetricCard
                  label="Input Level"
                  value={`${(audioQuality.inputLevel * 100).toFixed(0)}%`}
                  color="blue"
                />
                <MetricCard
                  label="Input Peak"
                  value={`${(audioQuality.inputPeak * 100).toFixed(0)}%`}
                  color="blue"
                />
                <MetricCard
                  label="Output Level"
                  value={`${(audioQuality.outputLevel * 100).toFixed(0)}%`}
                  color="purple"
                />
                <MetricCard
                  label="Output Peak"
                  value={`${(audioQuality.outputPeak * 100).toFixed(0)}%`}
                  color="purple"
                />
              </div>

              {/* Audio Level Graphs */}
              <div className="mt-4 grid grid-cols-2 gap-4">
                <SparklineGraph
                  data={snapshot.audioQuality.map((m) => m.inputLevel * 100)}
                  label="Input Level (%)"
                  color="blue"
                  maxDataPoints={60}
                />
                <SparklineGraph
                  data={snapshot.audioQuality.map((m) => m.outputLevel * 100)}
                  label="Output Level (%)"
                  color="purple"
                  maxDataPoints={60}
                />
              </div>
            </div>
          )}

          {/* Errors */}
          <div className="mb-6">
            <h3 className="text-sm font-medium text-white/80 mb-3">Errors</h3>
            <div className="grid grid-cols-3 gap-3">
              <MetricCard
                label="Total Errors"
                value={snapshot.errors.length.toString()}
                color={snapshot.errors.length > 0 ? "red" : "green"}
              />
              <MetricCard
                label="Error Rate"
                value={`${((snapshot.errors.length / snapshot.session.uptime) * 60).toFixed(2)}/min`}
                color={snapshot.errors.length > 0 ? "red" : "green"}
              />
              <MetricCard
                label="Recent Errors"
                value={snapshot.errors.slice(-10).length.toString()}
                color={
                  snapshot.errors.slice(-10).length > 0 ? "orange" : "green"
                }
              />
            </div>

            {/* Recent Errors List */}
            {snapshot.errors.length > 0 && (
              <div className="mt-4 bg-black/40 border border-white/10 rounded p-3 max-h-40 overflow-y-auto">
                <div className="text-xs text-white/60 mb-2">
                  Recent Errors:
                </div>
                {snapshot.errors.slice(-5).reverse().map((err, idx) => (
                  <div
                    key={idx}
                    className="text-xs text-red-400 mb-1 pb-1 border-b border-white/5 last:border-0"
                  >
                    <span className="text-white/40">
                      {new Date(err.timestamp).toLocaleTimeString()}
                    </span>{" "}
                    [{err.type}] {err.message}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* GPU */}
          {snapshot.gpu.length > 0 && (
            <div className="mb-6">
              <h3 className="text-sm font-medium text-white/80 mb-3">GPU</h3>
              <div className="grid grid-cols-2 gap-3">
                <MetricCard
                  label="VRAM Used"
                  value={`${snapshot.gpu[snapshot.gpu.length - 1].vramUsed.toFixed(1)} GB`}
                  color="purple"
                />
                <MetricCard
                  label="Utilization"
                  value={`${snapshot.gpu[snapshot.gpu.length - 1].utilization.toFixed(0)}%`}
                  color="purple"
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ============================================================
// Metric Card Component
// ============================================================

interface MetricCardProps {
  label: string;
  value: string;
  color: "blue" | "purple" | "green" | "yellow" | "orange" | "red";
}

function MetricCard({ label, value, color }: MetricCardProps) {
  const colorClasses = {
    blue: "text-blue-400 border-blue-500/20 bg-blue-500/10",
    purple: "text-purple-400 border-purple-500/20 bg-purple-500/10",
    green: "text-green-400 border-green-500/20 bg-green-500/10",
    yellow: "text-yellow-400 border-yellow-500/20 bg-yellow-500/10",
    orange: "text-orange-400 border-orange-500/20 bg-orange-500/10",
    red: "text-red-400 border-red-500/20 bg-red-500/10",
  };

  return (
    <div
      className={`border rounded p-3 ${colorClasses[color]}`}
    >
      <div className="text-xs text-white/60 mb-1">{label}</div>
      <div className="text-lg font-semibold">{value}</div>
    </div>
  );
}

// ============================================================
// Sparkline Graph Component
// ============================================================

interface SparklineGraphProps {
  data: number[];
  label: string;
  color: "blue" | "purple" | "green";
  maxDataPoints?: number;
}

function SparklineGraph({
  data,
  label,
  color,
  maxDataPoints = 60,
}: SparklineGraphProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (data.length === 0) return;

    // Use last N points
    const points = data.slice(-maxDataPoints);
    const max = Math.max(...points, 1);
    const min = Math.min(...points, 0);
    const range = max - min || 1;

    // Draw graph
    const colorMap = {
      blue: "59, 130, 246",
      purple: "168, 85, 247",
      green: "34, 197, 94",
    };

    ctx.strokeStyle = `rgba(${colorMap[color]}, 0.8)`;
    ctx.lineWidth = 2;
    ctx.beginPath();

    const stepX = canvas.width / (points.length - 1 || 1);
    points.forEach((value, i) => {
      const x = i * stepX;
      const y = canvas.height - ((value - min) / range) * canvas.height;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();

    // Fill area under curve
    ctx.lineTo(canvas.width, canvas.height);
    ctx.lineTo(0, canvas.height);
    ctx.closePath();
    ctx.fillStyle = `rgba(${colorMap[color]}, 0.1)`;
    ctx.fill();
  }, [data, color, maxDataPoints]);

  const colorClasses = {
    blue: "border-blue-500/20 bg-blue-500/5",
    purple: "border-purple-500/20 bg-purple-500/5",
    green: "border-green-500/20 bg-green-500/5",
  };

  return (
    <div className={`border rounded p-3 ${colorClasses[color]}`}>
      <div className="text-xs text-white/60 mb-2">{label}</div>
      <canvas ref={canvasRef} width={800} height={100} className="w-full h-24" />
      {data.length > 0 && (
        <div className="text-xs text-white/40 mt-1">
          {data.length} samples (last {Math.min(data.length, maxDataPoints)}{" "}
          shown)
        </div>
      )}
    </div>
  );
}
