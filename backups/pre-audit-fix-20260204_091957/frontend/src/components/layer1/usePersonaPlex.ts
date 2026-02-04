"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Recorder from "opus-recorder";
import { PersonaPlexState, Transcript } from "@/types";
import { commandCenterBus } from "@/lib/events";
import { config } from "@/lib/config";
import {
  encodeMessage,
  decodeMessage,
  type WSMessage,
} from "@/lib/personaplex/protocol";
import {
  createDecoderWorker,
  initDecoder,
} from "@/lib/personaplex/decoder";
import { PERSONA_SYSTEM_PROMPT } from "@/lib/personaplex/persona";
import { MetricsCollector } from "./MetricsCollector";
import type { MetricsSnapshot, MetricsStats } from "@/types/metrics";
import type { ConnectionMetrics } from "./ConnectionStatus";
import type { TranscriptMessage } from "./ConversationTranscript";

const LOG_PREFIX = "[Layer1:PersonaPlex]";

function dbg(...args: unknown[]) {
  console.debug(LOG_PREFIX, ...args);
}
function info(...args: unknown[]) {
  console.info(LOG_PREFIX, ...args);
}
function warn(...args: unknown[]) {
  console.warn(LOG_PREFIX, ...args);
}
function err(...args: unknown[]) {
  console.error(LOG_PREFIX, ...args);
}

export interface AudioDevice {
  deviceId: string;
  label: string;
}

interface UsePersonaPlexReturn {
  state: PersonaPlexState;
  transcripts: Transcript[];
  isListening: boolean;
  isSpeaking: boolean;
  startListening: () => void;
  stopListening: () => void;
  error: string | null;
  audioDevices: AudioDevice[];
  selectedDeviceId: string | null;
  setSelectedDeviceId: (id: string) => void;
  refreshDevices: () => Promise<void>;
  // Metrics
  metricsSnapshot: MetricsSnapshot | null;
  latencyStats: MetricsStats;
  connectionMetrics: ConnectionMetrics;
  // Analyzers for audio visualization
  inputAnalyser: AnalyserNode | null;
  outputAnalyser: AnalyserNode | null;
  // Conversation messages
  conversationMessages: TranscriptMessage[];
  // Text injection: send text for PersonaPlex to speak in its voice
  sendText: (text: string) => void;
  // Control: start, endTurn, pause, restart
  sendControl: (action: "start" | "endTurn" | "pause" | "restart") => void;
  // Queue drain callback ref
  onQueueDrained: React.MutableRefObject<(() => void) | null>;
}

/**
 * usePersonaPlex — Hook for Layer 1 Voice I/O.
 *
 * Connects to the NVIDIA PersonaPlex-7B server over its native Moshi
 * binary WebSocket protocol for full-duplex speech-to-speech conversation.
 *
 * Audio pipeline:
 *   Mic → opus-recorder (Opus 24kHz) → WebSocket → PersonaPlex server
 *   PersonaPlex server → WebSocket → Opus decoder worker → AudioWorklet → Speaker
 *
 * Text tokens are accumulated incrementally from the server for transcript display.
 */
export function usePersonaPlex(): UsePersonaPlexReturn {
  const [state, setState] = useState<PersonaPlexState>("idle");
  const [transcripts, setTranscripts] = useState<Transcript[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [audioDevices, setAudioDevices] = useState<AudioDevice[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string | null>(null);

  // Refs for WebSocket, audio pipeline, and Opus encoder/decoder
  const wsRef = useRef<WebSocket | null>(null);
  const recorderRef = useRef<Recorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const decoderWorkerRef = useRef<Worker | null>(null);
  const micDurationRef = useRef(0);
  const lastMessageTimeRef = useRef(0);
  const inactivityTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const onQueueDrainedRef = useRef<(() => void) | null>(null);

  // Metrics and monitoring
  const metricsCollectorRef = useRef<MetricsCollector>(new MetricsCollector());
  const [metricsSnapshot, setMetricsSnapshot] = useState<MetricsSnapshot | null>(null);
  const [latencyStats, setLatencyStats] = useState<MetricsStats>({
    avg: 0,
    min: 0,
    max: 0,
    p50: 0,
    p95: 0,
    p99: 0,
  });
  const [connectionMetrics, setConnectionMetrics] = useState<ConnectionMetrics>({
    wsState: "closed",
    serverHealthy: false,
    latency: null,
    gpuVRAM: 17.6, // PersonaPlex-7B uses ~17.6GB
    reconnectAttempts: 0,
    uptime: 0,
  });
  const reconnectAttemptsRef = useRef(0);
  const audioChunkIdRef = useRef(0);

  // Audio analyzers for visualization
  const inputAnalyserRef = useRef<AnalyserNode | null>(null);
  const outputAnalyserRef = useRef<AnalyserNode | null>(null);

  // Conversation messages
  const [conversationMessages, setConversationMessages] = useState<TranscriptMessage[]>([]);

  // Incremental text accumulation from the server
  const currentAssistantTextRef = useRef("");
  const currentAssistantIdRef = useRef("");
  const turnCountRef = useRef(0);

  // Enumerate audio input devices
  const refreshDevices = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = devices
        .filter((d) => d.kind === "audioinput" && d.deviceId)
        .map((d) => ({
          deviceId: d.deviceId,
          label: d.label || `Microphone ${d.deviceId.slice(0, 8)}`,
        }));
      info(`Found ${audioInputs.length} audio input device(s)`);
      setAudioDevices(audioInputs);

      if (!selectedDeviceId && audioInputs.length > 0) {
        const preferred =
          audioInputs.find(
            (d) =>
              !d.label.toLowerCase().includes("virtual") &&
              !d.label.toLowerCase().includes("oculus"),
          ) || audioInputs[0];
        info(`Auto-selected mic: "${preferred.label}"`);
        setSelectedDeviceId(preferred.deviceId);
      }
    } catch (e) {
      warn("Failed to enumerate audio devices:", e);
    }
  }, [selectedDeviceId]);

  // Broadcast state changes to the event bus
  const updateState = useCallback(
    (newState: PersonaPlexState) => {
      dbg(`State transition → ${newState}`);
      setState(newState);
      commandCenterBus.emit({
        type: "PERSONAPLEX_STATE_CHANGE",
        state: newState,
      });
    },
    [],
  );

  // Add or update a transcript entry and broadcast it
  const addTranscript = useCallback((transcript: Transcript) => {
    setTranscripts((prev) => {
      const existing = prev.findIndex((t) => t.id === transcript.id);
      if (existing >= 0) {
        const updated = [...prev];
        updated[existing] = transcript;
        return updated;
      }
      return [...prev, transcript];
    });

    commandCenterBus.emit({
      type: "TRANSCRIPT_UPDATE",
      transcript,
    });
  }, []);

  /**
   * Set up the audio output pipeline:
   *   Decoder Worker → AudioWorklet (MoshiProcessor) → speakers
   */
  const setupAudioOutput = useCallback(async (audioCtx: AudioContext) => {
    // Check AudioWorklet support
    if (!audioCtx.audioWorklet) {
      throw new Error(
        "AudioWorklet not supported. Please access via http://localhost:3100 (not 127.0.0.1 or IP address) and use a modern browser (Chrome 66+, Firefox 76+, Safari 14.1+)"
      );
    }

    // Load the MoshiProcessor AudioWorklet
    await audioCtx.audioWorklet.addModule("/moshi-processor.js");
    const workletNode = new AudioWorkletNode(audioCtx, "moshi-processor");

    // Create output analyser for visualization
    const outputAnalyser = audioCtx.createAnalyser();
    outputAnalyser.fftSize = 512;
    outputAnalyser.smoothingTimeConstant = 0.8;
    workletNode.connect(outputAnalyser);
    outputAnalyser.connect(audioCtx.destination);
    outputAnalyserRef.current = outputAnalyser;

    workletNodeRef.current = workletNode;

    // Create and init the Opus decoder worker
    const decoder = createDecoderWorker();
    decoderWorkerRef.current = decoder;
    await initDecoder(decoder, audioCtx.sampleRate);

    // Wire decoded PCM from the worker → worklet for playback
    decoder.onmessage = (e: MessageEvent) => {
      if (!e.data) return;
      const pcmFrame: Float32Array = e.data[0];
      workletNode.port.postMessage({
        frame: pcmFrame,
        type: "audio",
        micDuration: micDurationRef.current,
      });
    };

    info("Audio output pipeline ready (decoder → worklet → analyser → speakers)");
  }, []);

  /**
   * Build the PersonaPlex WebSocket URL with query parameters.
   */
  const buildWsUrl = useCallback(() => {
    const base = config.personaPlex.serverUrl;
    // Determine ws:// or wss:// from the configured URL scheme
    const wsProto = base.startsWith("https") ? "wss" : "ws";
    const host = base.replace(/^https?:\/\//, "");
    const params = new URLSearchParams({
      text_temperature: "0.7",
      text_topk: "25",
      audio_temperature: "0.7",
      audio_topk: "250",
      pad_mult: "2",
      text_seed: String(Math.floor(Math.random() * 1_000_000)),
      audio_seed: String(Math.floor(Math.random() * 1_000_000)),
      repetition_penalty_context: "64",
      repetition_penalty: "1.2",
      text_prompt: PERSONA_SYSTEM_PROMPT,
      voice_prompt: config.personaPlex.voice,
    });
    return `${wsProto}://${host}/api/chat?${params.toString()}`;
  }, []);

  /**
   * Handle an incoming binary WebSocket message from PersonaPlex.
   */
  const handleServerMessage = useCallback(
    (data: Uint8Array) => {
      lastMessageTimeRef.current = Date.now();

      let msg: WSMessage;
      try {
        msg = decodeMessage(data);
        dbg(`Received ${msg.type} message (${data.length} bytes)`);
      } catch (e) {
        warn("Failed to decode server message:", e);
        return;
      }

      switch (msg.type) {
        case "handshake":
          info("[WS←] Server handshake received — connection established");
          updateState("listening");
          break;

        case "audio":
          // Feed Opus data to decoder worker
          const audioData = msg.data;
          info(`[WS←] Received audio: ${audioData.length} bytes from PersonaPlex`);

          // Track metrics
          metricsCollectorRef.current.recordAudioReceived(null, audioData.length);

          if (decoderWorkerRef.current) {
            // Check for OggS header
            const hasOggS =
              audioData.length >= 4 &&
              audioData[0] === 0x4f &&
              audioData[1] === 0x67 &&
              audioData[2] === 0x67 &&
              audioData[3] === 0x53;

            if (hasOggS || audioData.length > 0) {
              dbg(`[AUDIO] Decoding ${audioData.length} bytes (hasOggS=${hasOggS})`);
              try {
                decoderWorkerRef.current.postMessage(
                  { command: "decode", pages: audioData },
                  [audioData.buffer],
                );
              } catch (e) {
                err(`[AUDIO] Failed to decode:`, e);
                metricsCollectorRef.current.recordError(
                  "audio_encoding",
                  `Failed to decode audio: ${e instanceof Error ? e.message : String(e)}`
                );
              }
            } else {
              warn(`[AUDIO] Skipping invalid audio data`);
              metricsCollectorRef.current.recordError(
                "audio_encoding",
                "Invalid audio data (no OggS header)"
              );
            }
          } else {
            warn(`[AUDIO] Decoder worker not ready!`);
            metricsCollectorRef.current.recordError(
              "playback",
              "Decoder worker not ready"
            );
          }
          break;

        case "text": {
          // Incremental text token from the server
          const token = msg.data;
          if (!currentAssistantIdRef.current) {
            turnCountRef.current++;
            currentAssistantIdRef.current = `assistant-${turnCountRef.current}-${Date.now()}`;
            currentAssistantTextRef.current = "";
          }
          currentAssistantTextRef.current += token;

          addTranscript({
            id: currentAssistantIdRef.current,
            role: "assistant",
            text: currentAssistantTextRef.current,
            timestamp: Date.now(),
            isFinal: false,
          });

          // Update conversation messages
          setConversationMessages((prev) => {
            const existing = prev.findIndex((m) => m.id === currentAssistantIdRef.current);
            const msg: TranscriptMessage = {
              id: currentAssistantIdRef.current,
              speaker: "ai",
              text: currentAssistantTextRef.current,
              timestamp: Date.now(),
            };
            if (existing >= 0) {
              const updated = [...prev];
              updated[existing] = msg;
              return updated;
            }
            return [...prev, msg];
          });
          break;
        }

        case "control":
          dbg("Control message:", msg.action);
          if (msg.action === "endTurn") {
            // Finalize the current assistant transcript
            if (currentAssistantIdRef.current) {
              addTranscript({
                id: currentAssistantIdRef.current,
                role: "assistant",
                text: currentAssistantTextRef.current,
                timestamp: Date.now(),
                isFinal: true,
              });

              // Finalize conversation message
              setConversationMessages((prev) => {
                const existing = prev.findIndex((m) => m.id === currentAssistantIdRef.current);
                if (existing >= 0) {
                  const updated = [...prev];
                  updated[existing] = {
                    ...updated[existing],
                    text: currentAssistantTextRef.current,
                  };
                  return updated;
                }
                return prev;
              });

              currentAssistantIdRef.current = "";
              currentAssistantTextRef.current = "";
            }
          }
          break;

        case "error":
          err("Server error:", msg.data);
          setError(msg.data);
          metricsCollectorRef.current.recordError("server", msg.data);
          break;

        case "ping":
          // Respond with ping to keep alive
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(encodeMessage({ type: "ping" }));
          }
          break;

        case "queueDrained":
          info("[WS←] Queue drained");
          if (onQueueDrainedRef.current) {
            onQueueDrainedRef.current();
            onQueueDrainedRef.current = null; // one-shot
          }
          break;

        default:
          dbg("Unhandled message type:", msg);
      }
    },
    [updateState, addTranscript],
  );

  /**
   * Start the full-duplex PersonaPlex session:
   *   1. Create AudioContext + output pipeline
   *   2. Connect WebSocket
   *   3. Start opus-recorder for mic input
   */
  const startListening = useCallback(async () => {
    info("startListening() called");

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      info("Session already active");
      return;
    }

    try {
      updateState("connecting");
      setError(null);

      // Track connection state
      metricsCollectorRef.current.recordConnectionState("connecting", reconnectAttemptsRef.current, 0);

      // --- Audio Context ---
      const audioCtx = new AudioContext({ sampleRate: 48000 });
      audioContextRef.current = audioCtx;
      if (audioCtx.state === "suspended") {
        await audioCtx.resume();
      }
      info(`AudioContext: sampleRate=${audioCtx.sampleRate}`);

      // --- Output pipeline (decoder + worklet) ---
      await setupAudioOutput(audioCtx);

      // --- WebSocket ---
      const wsUrl = buildWsUrl();
      const wsBaseUrl = wsUrl.split("?")[0];
      info(`Connecting to PersonaPlex: ${wsBaseUrl}`);

      // Pre-flight check: Try HTTP to see if server is reachable
      const httpUrl = config.personaPlex.serverUrl;
      info(`[DEBUG] Pre-flight HTTP check: ${httpUrl}`);
      const connectStartTime = Date.now();

      try {
        const preflightResponse = await fetch(httpUrl, {
          method: "GET",
          mode: "no-cors", // Allow CORS issues to be bypassed for check
        });
        info(`[DEBUG] Pre-flight HTTP check completed in ${Date.now() - connectStartTime}ms`);
      } catch (preflightError) {
        warn(`[DEBUG] Pre-flight HTTP check failed:`, preflightError);
        warn(`[DEBUG] Server may be unreachable at ${httpUrl}`);
      }

      info(`[DEBUG] Creating WebSocket at ${Date.now() - connectStartTime}ms`);
      info(`[DEBUG] Full WebSocket URL: ${wsUrl.substring(0, 200)}...`);

      let ws: WebSocket;
      try {
        ws = new WebSocket(wsUrl);
        info(`[DEBUG] WebSocket constructor succeeded at ${Date.now() - connectStartTime}ms`);
      } catch (wsConstructorError) {
        err(`[DEBUG] WebSocket constructor FAILED:`, wsConstructorError);
        throw new Error(`WebSocket constructor failed: ${wsConstructorError instanceof Error ? wsConstructorError.message : String(wsConstructorError)}`);
      }

      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      info(`[DEBUG] WebSocket readyState after constructor: ${ws.readyState} (0=CONNECTING, 1=OPEN, 2=CLOSING, 3=CLOSED)`);

      // Wait for socket to open with detailed error logging
      await new Promise<void>((resolve, reject) => {
        let resolved = false;

        ws.onopen = () => {
          if (resolved) return;
          resolved = true;
          const elapsed = Date.now() - connectStartTime;
          info(`[DEBUG] WebSocket OPEN at ${elapsed}ms`);
          info(`[DEBUG] WebSocket readyState: ${ws.readyState}`);
          info(`[DEBUG] WebSocket protocol: ${ws.protocol || "(none)"}`);
          info(`[DEBUG] WebSocket extensions: ${ws.extensions || "(none)"}`);
          resolve();
        };

        ws.onerror = (event: Event) => {
          const elapsed = Date.now() - connectStartTime;
          err(`[DEBUG] WebSocket ERROR at ${elapsed}ms`);
          err(`[DEBUG] Error event type: ${event.type}`);
          err(`[DEBUG] Error event target: ${event.target}`);
          err(`[DEBUG] WebSocket readyState at error: ${ws.readyState}`);
          err(`[DEBUG] WebSocket URL: ${ws.url}`);

          // Log the full event object properties
          const eventProps: Record<string, unknown> = {};
          for (const key in event) {
            try {
              eventProps[key] = (event as any)[key];
            } catch (e) {
              eventProps[key] = "(inaccessible)";
            }
          }
          err(`[DEBUG] Full error event:`, eventProps);

          // Check for common issues
          if (wsUrl.startsWith("wss://")) {
            err(`[DEBUG] SSL/TLS connection - possible certificate issue`);
            err(`[DEBUG] If using self-signed cert, ensure browser trusts it by visiting: ${httpUrl}`);
          }

          if (!resolved) {
            resolved = true;
            reject(new Error(`Connection failed: ${wsBaseUrl} (readyState=${ws.readyState}, elapsed=${elapsed}ms). Check browser console for SSL/certificate errors.`));
          }
        };

        ws.onclose = (event: CloseEvent) => {
          const elapsed = Date.now() - connectStartTime;
          err(`[DEBUG] WebSocket CLOSED during connection at ${elapsed}ms`);
          err(`[DEBUG] Close code: ${event.code}`);
          err(`[DEBUG] Close reason: "${event.reason}"`);
          err(`[DEBUG] Was clean: ${event.wasClean}`);

          // Interpret close codes
          const closeCodeMeaning: Record<number, string> = {
            1000: "Normal closure",
            1001: "Going away (page unload)",
            1002: "Protocol error",
            1003: "Unsupported data",
            1005: "No status received",
            1006: "Abnormal closure (no close frame) - often SSL/network issue",
            1007: "Invalid data",
            1008: "Policy violation",
            1009: "Message too big",
            1010: "Missing extension",
            1011: "Internal server error",
            1015: "TLS handshake failure",
          };
          err(`[DEBUG] Close code meaning: ${closeCodeMeaning[event.code] || "Unknown"}`);

          if (!resolved) {
            resolved = true;
            reject(new Error(`Connection closed: code=${event.code} (${closeCodeMeaning[event.code] || "Unknown"}) reason="${event.reason}"`));
          }
        };

        // Timeout after 10s
        setTimeout(() => {
          if (!resolved) {
            resolved = true;
            const elapsed = Date.now() - connectStartTime;
            err(`[DEBUG] Connection TIMEOUT at ${elapsed}ms`);
            err(`[DEBUG] WebSocket readyState at timeout: ${ws.readyState}`);
            reject(new Error(`Connection timed out after 10s (readyState=${ws.readyState})`));
          }
        }, 10_000);
      });

      // Wire up message handler
      ws.onmessage = (event) => {
        const data = new Uint8Array(event.data);
        handleServerMessage(data);
      };

      // Ensure suppression is ON from the start (belt-and-suspenders with server default)
      ws.send(encodeMessage({ type: "control", action: "pause" }));
      info("[WS→] Initial pause — suppression ON from connection start");

      ws.onclose = (event) => {
        warn(
          `WebSocket CLOSED — code=${event.code} reason="${event.reason}" clean=${event.wasClean}`,
        );
        teardown();
        updateState("disconnected");
      };

      ws.onerror = (event) => {
        err("WebSocket ERROR", event);
        setError(`PersonaPlex connection error`);
      };

      // Inactivity detection — close if no messages for 30s (increased from 10s)
      lastMessageTimeRef.current = Date.now();
      inactivityTimerRef.current = setInterval(() => {
        if (
          wsRef.current?.readyState === WebSocket.OPEN &&
          Date.now() - lastMessageTimeRef.current > 30_000
        ) {
          warn("No messages received for 30s — closing connection");
          wsRef.current.close();
        }
      }, 1000);

      // Ping keepalive — send pings every 5s to keep connection alive
      pingIntervalRef.current = setInterval(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          try {
            wsRef.current.send(encodeMessage({ type: "ping" }));
            dbg("[KEEPALIVE] Sent ping");
          } catch (e) {
            warn("[KEEPALIVE] Failed to send ping:", e);
          }
        }
      }, 5000);

      // --- Mic input via opus-recorder ---
      info("Initializing Opus recorder for mic input...");
      const Recorder = (await import("opus-recorder")).default;

      const audioConstraints: MediaTrackConstraints = {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      };
      if (selectedDeviceId) {
        audioConstraints.deviceId = { exact: selectedDeviceId };
        info(`Using selected device: ${selectedDeviceId}`);
      }

      const recorderOptions = {
        encoderPath: "/assets/encoderWorker.min.js",
        bufferLength: Math.round((960 * audioCtx.sampleRate) / 24000),
        encoderFrameSize: 20,
        encoderSampleRate: 24000,
        maxFramesPerPage: 2,
        numberOfChannels: 1,
        recordingGain: 1,
        resampleQuality: 3,
        encoderComplexity: 0,
        encoderApplication: 2049,
        streamPages: true,
        mediaTrackConstraints: { audio: audioConstraints },
        // sourceNode intentionally omitted - let opus-recorder use getUserMedia
      };

      dbg("Opus recorder options:", recorderOptions);
      const recorder = new Recorder(recorderOptions);
      recorderRef.current = recorder;
      info("Opus recorder created successfully");

      // Create input analyser after recorder starts (we'll connect it in onstart)
      const inputAnalyser = audioCtx.createAnalyser();
      inputAnalyser.fftSize = 512;
      inputAnalyser.smoothingTimeConstant = 0.8;
      inputAnalyserRef.current = inputAnalyser;

      let audioChunkCount = 0;
      recorder.ondataavailable = (chunk: Uint8Array) => {
        audioChunkCount++;
        const chunkId = audioChunkIdRef.current++;
        micDurationRef.current = recorder.encodedSamplePosition / 48000;
        info(`[MIC] Chunk #${audioChunkCount}: ${chunk.length} bytes, encodedPos=${recorder.encodedSamplePosition}`);

        if (ws.readyState === WebSocket.OPEN) {
          const encoded = encodeMessage({ type: "audio", data: chunk });
          ws.send(encoded);
          info(`[WS→] Sent ${encoded.length} bytes (payload: ${chunk.length} bytes) to PersonaPlex`);

          // Track metrics
          metricsCollectorRef.current.recordAudioSent(chunkId, encoded.length);
        } else {
          warn(`[WS] Cannot send audio - WebSocket state: ${ws.readyState}`);
          metricsCollectorRef.current.recordError(
            "websocket",
            `Cannot send audio - WebSocket state: ${ws.readyState}`
          );
        }
      };

      recorder.onstart = async () => {
        info("✓ [MIC] Opus recorder onstart() fired — mic is active!");
        info(`[MIC] Recorder state: encodedSamplePosition=${recorder.encodedSamplePosition}`);

        // Connect microphone stream to input analyser for visualization
        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            audio: audioConstraints,
          });
          const source = audioCtx.createMediaStreamSource(stream);
          source.connect(inputAnalyser);
          info("[MIC] Input analyser connected for visualization");
        } catch (e) {
          warn("[MIC] Failed to connect input analyser:", e);
        }

        // Reset worklet buffer state for fresh session
        workletNodeRef.current?.port.postMessage({ type: "reset" });
        commandCenterBus.emit({ type: "VOICE_INPUT_START" });

        // Track connection state
        metricsCollectorRef.current.recordConnectionState("open", reconnectAttemptsRef.current, 0);
      };

      recorder.onstop = () => {
        info("[MIC] Opus recorder stopped");
      };

      // Add error handler for opus-recorder
      (recorder as any).onerror = (error: Error) => {
        err("[MIC ERROR]", error);
        setError(`Microphone error: ${error.message}`);
      };

      info("[MIC] ⏳ Starting Opus recorder (requesting microphone access)...");
      try {
        await recorder.start();
        info("[MIC] ✓ recorder.start() completed successfully");
        info(`[MIC] Recorder active - waiting for ondataavailable callbacks...`);
      } catch (startError) {
        err("[MIC] Failed to start recorder:", startError);
        throw startError;
      }

      info("🎉 Full-duplex session active — GPU-accelerated PersonaPlex ready");
      updateState("listening");
    } catch (e) {
      const message =
        e instanceof Error ? e.message : "Failed to start PersonaPlex session";
      err("startListening() failed:", e);
      setError(message);
      updateState("error");

      // Track error
      metricsCollectorRef.current.recordError(
        "websocket",
        message,
        e
      );
      metricsCollectorRef.current.recordConnectionState("error", reconnectAttemptsRef.current, 0);
      reconnectAttemptsRef.current++;

      teardown();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    updateState,
    setupAudioOutput,
    buildWsUrl,
    handleServerMessage,
    selectedDeviceId,
    // Note: teardown is intentionally omitted to avoid circular dependency
  ]);

  /** Tear down all audio resources and close the socket. */
  const teardown = useCallback(() => {
    dbg("teardown()");

    if (inactivityTimerRef.current) {
      clearInterval(inactivityTimerRef.current);
      inactivityTimerRef.current = null;
    }

    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }

    if (recorderRef.current) {
      try {
        recorderRef.current.stop();
      } catch {
        /* ignore */
      }
      recorderRef.current = null;
    }

    if (decoderWorkerRef.current) {
      decoderWorkerRef.current.terminate();
      decoderWorkerRef.current = null;
    }

    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }

    // Disconnect analyzers
    if (inputAnalyserRef.current) {
      inputAnalyserRef.current.disconnect();
      inputAnalyserRef.current = null;
    }

    if (outputAnalyserRef.current) {
      outputAnalyserRef.current.disconnect();
      outputAnalyserRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }

    if (wsRef.current) {
      if (
        wsRef.current.readyState === WebSocket.OPEN ||
        wsRef.current.readyState === WebSocket.CONNECTING
      ) {
        wsRef.current.close();
      }
      wsRef.current = null;
    }

    micDurationRef.current = 0;
    currentAssistantTextRef.current = "";
    currentAssistantIdRef.current = "";

    // Track connection closed
    metricsCollectorRef.current.recordConnectionState("closed", reconnectAttemptsRef.current, 0);
  }, []);

  const stopListening = useCallback(() => {
    info("stopListening() called");
    teardown();
    updateState("idle");
    commandCenterBus.emit({ type: "VOICE_INPUT_STOP" });
  }, [teardown, updateState]);

  // Keyboard shortcut handler (Ctrl+Space)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === "Space" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        dbg(`Hotkey Ctrl+Space — current state: ${state}`);
        if (state === "listening" || state === "speaking") {
          stopListening();
        } else if (
          state === "idle" ||
          state === "disconnected" ||
          state === "error"
        ) {
          startListening();
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [state, startListening, stopListening]);

  // Mount: enumerate devices. Unmount: teardown.
  useEffect(() => {
    info("PersonaPlex hook mounted");
    info(
      `  Config: server=${config.personaPlex.serverUrl} model=${config.personaPlex.model} voice=${config.personaPlex.voice}`,
    );
    refreshDevices();

    return () => {
      info("PersonaPlex hook unmounting — cleaning up");
      teardown();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Metrics update loop - refresh every 100ms
  useEffect(() => {
    const interval = setInterval(() => {
      const metrics = metricsCollectorRef.current;

      // Update metrics snapshot
      setMetricsSnapshot(metrics.getSnapshot());
      setLatencyStats(metrics.getLatencyStats());

      // Update connection metrics
      const wsState = wsRef.current?.readyState;
      setConnectionMetrics({
        wsState:
          wsState === WebSocket.CONNECTING
            ? "connecting"
            : wsState === WebSocket.OPEN
              ? "open"
              : wsState === WebSocket.CLOSING
                ? "closing"
                : wsState === WebSocket.CLOSED
                  ? "closed"
                  : "error",
        serverHealthy: wsState === WebSocket.OPEN && Date.now() - lastMessageTimeRef.current < 15000,
        latency: metrics.getCurrentLatency(),
        gpuVRAM: metrics.getCurrentGPUVRAM() || 17.6,
        reconnectAttempts: reconnectAttemptsRef.current,
        uptime: metrics.getUptime(),
      });

      // Track audio quality if analyzers are available
      if (inputAnalyserRef.current && outputAnalyserRef.current) {
        const inputData = new Uint8Array(inputAnalyserRef.current.frequencyBinCount);
        const outputData = new Uint8Array(outputAnalyserRef.current.frequencyBinCount);

        inputAnalyserRef.current.getByteTimeDomainData(inputData);
        outputAnalyserRef.current.getByteTimeDomainData(outputData);

        // Calculate RMS levels
        const calculateRMS = (data: Uint8Array) => {
          let sum = 0;
          let peak = 0;
          for (let i = 0; i < data.length; i++) {
            const normalized = (data[i] - 128) / 128;
            sum += normalized * normalized;
            peak = Math.max(peak, Math.abs(normalized));
          }
          return { rms: Math.sqrt(sum / data.length), peak };
        };

        const inputLevels = calculateRMS(inputData);
        const outputLevels = calculateRMS(outputData);

        metrics.recordAudioQuality(
          inputLevels.rms,
          inputLevels.peak,
          outputLevels.rms,
          outputLevels.peak,
          64, // Opus bitrate (kbps) - configured in opus-recorder
          24000 // Sample rate (Hz)
        );
      }
    }, 100);

    return () => clearInterval(interval);
  }, []);

  /**
   * Send text to PersonaPlex for it to speak in its own voice.
   * Uses the text injection protocol (type 0x02) — the server tokenizes
   * the text and force-feeds tokens into the model, which generates
   * matching audio output.
   */
  const sendText = useCallback((text: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      warn("sendText: WebSocket not open, cannot inject text");
      return;
    }
    if (!text.trim()) return;
    info(`sendText: injecting "${text.substring(0, 80)}..."`);
    wsRef.current.send(encodeMessage({ type: "text", data: text }));
  }, []);

  /**
   * Send a control action to PersonaPlex.
   * "pause" = suppress free generation (force silence)
   * "restart" = resume natural generation
   */
  const sendControl = useCallback((action: "start" | "endTurn" | "pause" | "restart") => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      warn(`sendControl: WebSocket not open, cannot send ${action}`);
      return;
    }
    info(`sendControl: ${action}`);
    wsRef.current.send(encodeMessage({ type: "control", action }));
  }, []);

  return {
    state,
    transcripts,
    isListening: state === "listening" || state === "speaking",
    isSpeaking: state === "speaking",
    startListening,
    stopListening,
    error,
    audioDevices,
    selectedDeviceId,
    setSelectedDeviceId,
    refreshDevices,
    metricsSnapshot,
    latencyStats,
    connectionMetrics,
    inputAnalyser: inputAnalyserRef.current,
    outputAnalyser: outputAnalyserRef.current,
    conversationMessages,
    sendText,
    sendControl,
    onQueueDrained: onQueueDrainedRef,
  };
}
