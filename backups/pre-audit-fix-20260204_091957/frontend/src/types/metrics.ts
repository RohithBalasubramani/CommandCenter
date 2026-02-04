// ============================================================
// Command Center — Metrics Types
// ============================================================

export interface LatencyMetric {
  timestamp: number;
  sendTime: number; // When audio chunk was sent
  receiveTime: number; // When first response received
  latency: number; // receiveTime - sendTime (ms)
}

export interface AudioQualityMetric {
  timestamp: number;
  inputLevel: number; // RMS level (0-1)
  inputPeak: number; // Peak level (0-1)
  outputLevel: number; // RMS level (0-1)
  outputPeak: number; // Peak level (0-1)
  bitrate: number; // Opus bitrate (kbps)
  sampleRate: number; // Sample rate (Hz)
}

export interface ConnectionMetric {
  timestamp: number;
  wsState: "connecting" | "open" | "closing" | "closed" | "error";
  reconnectAttempts: number;
  messageQueueSize: number;
}

export interface ErrorMetric {
  timestamp: number;
  type: "audio_encoding" | "websocket" | "server" | "playback" | "other";
  message: string;
  details?: unknown;
}

export interface GPUMetric {
  timestamp: number;
  vramUsed: number; // GB
  utilization: number; // 0-100%
}

export interface SessionStats {
  startTime: number;
  uptime: number; // seconds
  totalAudioSent: number; // bytes
  totalAudioReceived: number; // bytes
  totalChunksProcessed: number;
  totalErrors: number;
  averageLatency: number; // ms
  averageInputLevel: number; // 0-1
  averageOutputLevel: number; // 0-1
}

export interface MetricsSnapshot {
  latency: LatencyMetric[];
  audioQuality: AudioQualityMetric[];
  connection: ConnectionMetric[];
  errors: ErrorMetric[];
  gpu: GPUMetric[];
  session: SessionStats;
}

export interface MetricsStats {
  avg: number;
  min: number;
  max: number;
  p50: number;
  p95: number;
  p99: number;
}
