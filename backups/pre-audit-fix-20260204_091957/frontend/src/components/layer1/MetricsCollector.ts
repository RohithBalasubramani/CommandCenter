// ============================================================
// Command Center — Metrics Collector
// ============================================================

import {
  LatencyMetric,
  AudioQualityMetric,
  ConnectionMetric,
  ErrorMetric,
  GPUMetric,
  SessionStats,
  MetricsSnapshot,
  MetricsStats,
} from "@/types/metrics";

const MAX_SAMPLES = 1000;

export class MetricsCollector {
  private latencyMetrics: LatencyMetric[] = [];
  private audioQualityMetrics: AudioQualityMetric[] = [];
  private connectionMetrics: ConnectionMetric[] = [];
  private errorMetrics: ErrorMetric[] = [];
  private gpuMetrics: GPUMetric[] = [];

  private sessionStartTime: number = 0;
  private totalAudioSent: number = 0;
  private totalAudioReceived: number = 0;
  private totalChunksProcessed: number = 0;

  private pendingAudioChunks: Map<number, number> = new Map(); // chunkId -> sendTime

  constructor() {
    this.sessionStartTime = Date.now();
  }

  // ============================================================
  // Latency Tracking
  // ============================================================

  recordAudioSent(chunkId: number, bytesSent: number): void {
    const now = performance.now();
    this.pendingAudioChunks.set(chunkId, now);
    this.totalAudioSent += bytesSent;
  }

  recordAudioReceived(chunkId: number | null, bytesReceived: number): void {
    const now = performance.now();
    this.totalAudioReceived += bytesReceived;
    this.totalChunksProcessed++;

    // If we can match to a sent chunk, record latency
    if (chunkId !== null && this.pendingAudioChunks.has(chunkId)) {
      const sendTime = this.pendingAudioChunks.get(chunkId)!;
      const latency = now - sendTime;

      this.latencyMetrics.push({
        timestamp: Date.now(),
        sendTime,
        receiveTime: now,
        latency,
      });

      this.pendingAudioChunks.delete(chunkId);
      this.trimArray(this.latencyMetrics, MAX_SAMPLES);
    }
  }

  getCurrentLatency(): number | null {
    if (this.latencyMetrics.length === 0) return null;
    return this.latencyMetrics[this.latencyMetrics.length - 1].latency;
  }

  getAverageLatency(): number {
    if (this.latencyMetrics.length === 0) return 0;
    const sum = this.latencyMetrics.reduce((acc, m) => acc + m.latency, 0);
    return sum / this.latencyMetrics.length;
  }

  // ============================================================
  // Audio Quality Tracking
  // ============================================================

  recordAudioQuality(
    inputLevel: number,
    inputPeak: number,
    outputLevel: number,
    outputPeak: number,
    bitrate: number,
    sampleRate: number
  ): void {
    this.audioQualityMetrics.push({
      timestamp: Date.now(),
      inputLevel,
      inputPeak,
      outputLevel,
      outputPeak,
      bitrate,
      sampleRate,
    });

    this.trimArray(this.audioQualityMetrics, MAX_SAMPLES);
  }

  getAverageInputLevel(): number {
    if (this.audioQualityMetrics.length === 0) return 0;
    const sum = this.audioQualityMetrics.reduce(
      (acc, m) => acc + m.inputLevel,
      0
    );
    return sum / this.audioQualityMetrics.length;
  }

  getAverageOutputLevel(): number {
    if (this.audioQualityMetrics.length === 0) return 0;
    const sum = this.audioQualityMetrics.reduce(
      (acc, m) => acc + m.outputLevel,
      0
    );
    return sum / this.audioQualityMetrics.length;
  }

  // ============================================================
  // Connection Tracking
  // ============================================================

  recordConnectionState(
    wsState: ConnectionMetric["wsState"],
    reconnectAttempts: number,
    messageQueueSize: number
  ): void {
    this.connectionMetrics.push({
      timestamp: Date.now(),
      wsState,
      reconnectAttempts,
      messageQueueSize,
    });

    this.trimArray(this.connectionMetrics, MAX_SAMPLES);
  }

  getCurrentConnectionState(): ConnectionMetric["wsState"] | null {
    if (this.connectionMetrics.length === 0) return null;
    return this.connectionMetrics[this.connectionMetrics.length - 1].wsState;
  }

  getReconnectAttempts(): number {
    if (this.connectionMetrics.length === 0) return 0;
    return this.connectionMetrics[this.connectionMetrics.length - 1]
      .reconnectAttempts;
  }

  // ============================================================
  // Error Tracking
  // ============================================================

  recordError(
    type: ErrorMetric["type"],
    message: string,
    details?: unknown
  ): void {
    this.errorMetrics.push({
      timestamp: Date.now(),
      type,
      message,
      details,
    });

    this.trimArray(this.errorMetrics, MAX_SAMPLES);
  }

  getTotalErrors(): number {
    return this.errorMetrics.length;
  }

  getRecentErrors(count: number): ErrorMetric[] {
    return this.errorMetrics.slice(-count);
  }

  getErrorRate(): number {
    // Errors per minute
    const uptime = this.getUptime();
    if (uptime === 0) return 0;
    return (this.errorMetrics.length / uptime) * 60;
  }

  // ============================================================
  // GPU Tracking
  // ============================================================

  recordGPUUsage(vramUsed: number, utilization: number): void {
    this.gpuMetrics.push({
      timestamp: Date.now(),
      vramUsed,
      utilization,
    });

    this.trimArray(this.gpuMetrics, MAX_SAMPLES);
  }

  getCurrentGPUVRAM(): number | null {
    if (this.gpuMetrics.length === 0) return null;
    return this.gpuMetrics[this.gpuMetrics.length - 1].vramUsed;
  }

  // ============================================================
  // Session Stats
  // ============================================================

  getUptime(): number {
    return (Date.now() - this.sessionStartTime) / 1000; // seconds
  }

  getSessionStats(): SessionStats {
    return {
      startTime: this.sessionStartTime,
      uptime: this.getUptime(),
      totalAudioSent: this.totalAudioSent,
      totalAudioReceived: this.totalAudioReceived,
      totalChunksProcessed: this.totalChunksProcessed,
      totalErrors: this.errorMetrics.length,
      averageLatency: this.getAverageLatency(),
      averageInputLevel: this.getAverageInputLevel(),
      averageOutputLevel: this.getAverageOutputLevel(),
    };
  }

  // ============================================================
  // Full Snapshot
  // ============================================================

  getSnapshot(): MetricsSnapshot {
    return {
      latency: [...this.latencyMetrics],
      audioQuality: [...this.audioQualityMetrics],
      connection: [...this.connectionMetrics],
      errors: [...this.errorMetrics],
      gpu: [...this.gpuMetrics],
      session: this.getSessionStats(),
    };
  }

  // ============================================================
  // Statistics Calculation
  // ============================================================

  calculateStats(values: number[]): MetricsStats {
    if (values.length === 0) {
      return { avg: 0, min: 0, max: 0, p50: 0, p95: 0, p99: 0 };
    }

    const sorted = [...values].sort((a, b) => a - b);
    const sum = sorted.reduce((acc, v) => acc + v, 0);

    const percentile = (p: number) => {
      const index = Math.ceil((p / 100) * sorted.length) - 1;
      return sorted[Math.max(0, index)];
    };

    return {
      avg: sum / sorted.length,
      min: sorted[0],
      max: sorted[sorted.length - 1],
      p50: percentile(50),
      p95: percentile(95),
      p99: percentile(99),
    };
  }

  getLatencyStats(): MetricsStats {
    const latencies = this.latencyMetrics.map((m) => m.latency);
    return this.calculateStats(latencies);
  }

  // ============================================================
  // Reset
  // ============================================================

  reset(): void {
    this.latencyMetrics = [];
    this.audioQualityMetrics = [];
    this.connectionMetrics = [];
    this.errorMetrics = [];
    this.gpuMetrics = [];
    this.pendingAudioChunks.clear();
    this.sessionStartTime = Date.now();
    this.totalAudioSent = 0;
    this.totalAudioReceived = 0;
    this.totalChunksProcessed = 0;
  }

  // ============================================================
  // Utilities
  // ============================================================

  private trimArray<T>(arr: T[], maxLength: number): void {
    if (arr.length > maxLength) {
      arr.splice(0, arr.length - maxLength);
    }
  }

  // Export metrics as JSON for download
  exportJSON(): string {
    return JSON.stringify(this.getSnapshot(), null, 2);
  }
}
