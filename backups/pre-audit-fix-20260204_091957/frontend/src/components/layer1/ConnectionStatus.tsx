"use client";

import React from "react";

export interface ConnectionMetrics {
  wsState: "connecting" | "open" | "closing" | "closed" | "error";
  serverHealthy: boolean;
  latency: number | null; // ms
  gpuVRAM: number | null; // GB
  reconnectAttempts: number;
  uptime: number; // seconds
}

interface ConnectionStatusProps {
  metrics: ConnectionMetrics;
}

export default function ConnectionStatus({ metrics }: ConnectionStatusProps) {
  const getStateColor = () => {
    switch (metrics.wsState) {
      case "open":
        return metrics.serverHealthy ? "text-green-400" : "text-yellow-400";
      case "connecting":
        return "text-yellow-400";
      case "closing":
      case "closed":
        return "text-gray-400";
      case "error":
        return "text-red-400";
      default:
        return "text-gray-400";
    }
  };

  const getStateIcon = () => {
    switch (metrics.wsState) {
      case "open":
        return metrics.serverHealthy ? "●" : "◐";
      case "connecting":
        return "◔";
      case "closing":
      case "closed":
        return "○";
      case "error":
        return "✕";
      default:
        return "○";
    }
  };

  const getStateLabel = () => {
    if (metrics.wsState === "open") {
      return metrics.serverHealthy ? "Connected" : "Connected (Degraded)";
    }
    return metrics.wsState.charAt(0).toUpperCase() + metrics.wsState.slice(1);
  };

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
      return `${hours}h ${mins}m ${secs}s`;
    }
    if (mins > 0) {
      return `${mins}m ${secs}s`;
    }
    return `${secs}s`;
  };

  const getLatencyColor = () => {
    if (!metrics.latency) return "text-white/40";
    if (metrics.latency < 200) return "text-green-400";
    if (metrics.latency < 500) return "text-yellow-400";
    return "text-red-400";
  };

  return (
    <div className="flex items-center gap-4 bg-black/20 backdrop-blur-sm rounded-lg border border-white/10 px-4 py-2">
      {/* Connection State */}
      <div className="flex items-center gap-2">
        <span className={`text-lg ${getStateColor()}`}>{getStateIcon()}</span>
        <div className="flex flex-col">
          <span className={`text-xs font-medium ${getStateColor()}`}>
            {getStateLabel()}
          </span>
          {metrics.reconnectAttempts > 0 && (
            <span className="text-xs text-white/40">
              {metrics.reconnectAttempts} reconnect
              {metrics.reconnectAttempts !== 1 ? "s" : ""}
            </span>
          )}
        </div>
      </div>

      {/* Divider */}
      <div className="h-8 w-px bg-white/10" />

      {/* Latency */}
      <div className="flex flex-col">
        <span className="text-xs text-white/40">Latency</span>
        <span className={`text-xs font-medium ${getLatencyColor()}`}>
          {metrics.latency !== null ? `${metrics.latency.toFixed(0)}ms` : "—"}
        </span>
      </div>

      {/* GPU VRAM */}
      {metrics.gpuVRAM !== null && (
        <>
          <div className="h-8 w-px bg-white/10" />
          <div className="flex flex-col">
            <span className="text-xs text-white/40">GPU VRAM</span>
            <span className="text-xs font-medium text-white/80">
              {metrics.gpuVRAM.toFixed(1)} GB
            </span>
          </div>
        </>
      )}

      {/* Uptime */}
      <div className="h-8 w-px bg-white/10" />
      <div className="flex flex-col">
        <span className="text-xs text-white/40">Uptime</span>
        <span className="text-xs font-medium text-white/80">
          {formatUptime(metrics.uptime)}
        </span>
      </div>

      {/* Server Info */}
      <div className="ml-auto flex items-center gap-1.5">
        <div className="w-1.5 h-1.5 bg-purple-400 rounded-full" />
        <span className="text-xs text-white/60">PersonaPlex-7B</span>
      </div>
    </div>
  );
}
