"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { config } from "@/lib/config";

/**
 * STT model options for the server-based pipeline.
 */
export type STTModel = "parakeet" | "whisper" | "web-speech";

interface STTModelInfo {
  name: string;
  loaded: boolean;
  available: boolean;
  description: string;
  vram_gb?: number;
  runtime_broken?: boolean;
}

export interface STTStats {
  lastLatencyMs: number | null;
  avgLatencyMs: number | null;
  requestCount: number;
  errorCount: number;
  totalAudioSeconds: number;
  wordsTranscribed: number;
}

interface UseSTTReturn {
  // Transcripts
  transcript: string;
  interimTranscript: string;

  // State
  isListening: boolean;
  isSupported: boolean;
  error: string | null;

  // Controls
  start: () => void;
  stop: () => void;

  // Model management
  activeModel: STTModel;
  switchModel: (model: STTModel) => Promise<void>;
  availableModels: STTModelInfo[];
  isServerAvailable: boolean;

  // Stats
  stats: STTStats;
}

/**
 * useSTT — Server-based Speech-to-Text hook.
 *
 * Uses the STT server (Parakeet / Whisper) on port 8890 for transcription.
 * Falls back to browser Web Speech API if the server is unreachable.
 *
 * Audio is recorded via MediaRecorder (browser-native), sent as WAV blobs
 * to the STT server for transcription.
 */
export function useSTT(): UseSTTReturn {
  const [transcript, setTranscript] = useState("");
  const [interimTranscript, setInterimTranscript] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isSupported, setIsSupported] = useState(true);

  const [activeModel, setActiveModel] = useState<STTModel>("parakeet");
  const [availableModels, setAvailableModels] = useState<STTModelInfo[]>([]);
  const [isServerAvailable, setIsServerAvailable] = useState(false);

  // Stats
  const [stats, setStats] = useState<STTStats>({
    lastLatencyMs: null,
    avgLatencyMs: null,
    requestCount: 0,
    errorCount: 0,
    totalAudioSeconds: 0,
    wordsTranscribed: 0,
  });
  const latencySumRef = useRef(0);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const shouldBeListeningRef = useRef(false);
  const sendIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Web Speech API fallback refs
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const usingWebSpeechRef = useRef(false);
  const restartTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const sttUrl = config.stt?.serverUrl || "http://localhost:8890";

  // ------------------------------------------------------------------
  // Health check + model discovery
  // ------------------------------------------------------------------
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${sttUrl}/v1/stt/health`, {
          signal: AbortSignal.timeout(3000),
        });
        if (res.ok) {
          const data = await res.json();
          setIsServerAvailable(true);
          setActiveModel(data.active_model as STTModel);
          setAvailableModels(data.models || []);
          console.info("[STT] Server healthy, active model:", data.active_model);
        } else {
          setIsServerAvailable(false);
        }
      } catch {
        console.warn("[STT] Server unreachable, will use Web Speech API fallback");
        setIsServerAvailable(false);
      }
    };
    checkHealth();
  }, [sttUrl]);

  // ------------------------------------------------------------------
  // Web Speech API fallback setup
  // ------------------------------------------------------------------
  useEffect(() => {
    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      // No Web Speech API either — only server STT available
      if (!isServerAvailable) {
        setIsSupported(false);
      }
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onstart = () => {
      if (usingWebSpeechRef.current) {
        setIsListening(true);
        setError(null);
      }
    };

    recognition.onend = () => {
      if (usingWebSpeechRef.current) {
        setIsListening(false);
        if (shouldBeListeningRef.current) {
          restartTimeoutRef.current = setTimeout(() => {
            if (shouldBeListeningRef.current && recognitionRef.current && usingWebSpeechRef.current) {
              try { recognitionRef.current.start(); } catch { /* ignore */ }
            }
          }, 100);
        }
      }
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      if (usingWebSpeechRef.current) {
        window.dispatchEvent(
          new CustomEvent("speechrecognition-error", { detail: event.error })
        );
        if (event.error !== "no-speech") {
          setError(event.error);
        }
      }
    };

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      if (!usingWebSpeechRef.current) return;
      let finalTranscript = "";
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) {
          finalTranscript += result[0].transcript;
        } else {
          interim += result[0].transcript;
        }
      }
      if (finalTranscript) {
        setTranscript((prev) => (prev ? prev + " " + finalTranscript : finalTranscript));
      }
      setInterimTranscript(interim);
    };

    recognitionRef.current = recognition;

    return () => {
      shouldBeListeningRef.current = false;
      if (restartTimeoutRef.current) clearTimeout(restartTimeoutRef.current);
      try { recognitionRef.current?.stop(); } catch { /* ignore */ }
    };
  }, [isServerAvailable]);

  // ------------------------------------------------------------------
  // Server-based recording: capture audio → send to STT server
  // ------------------------------------------------------------------
  const startServerSTT = useCallback(async () => {
    try {
      console.info("[STT] Requesting microphone access...");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const audioTracks = stream.getAudioTracks();
      console.info("[STT] Microphone granted:", audioTracks.map(t => `${t.label} (${t.readyState})`).join(", "));

      // Browser records as webm/opus — server decodes via ffmpeg
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";
      console.info("[STT] Using MIME type:", mimeType);

      const recorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      recorder.onerror = (e) => {
        console.error("[STT] MediaRecorder error:", e);
      };

      recorder.onstart = () => {
        console.info("[STT] MediaRecorder started (timeslice=500ms)");
      };

      recorder.onstop = () => {
        console.info("[STT] MediaRecorder stopped");
      };

      recorder.start(500); // Collect chunks every 500ms
      setIsListening(true);
      setError(null);

      // Cumulative buffering: send the FULL recording each time.
      // Server returns the full transcript for all audio so far.
      // We REPLACE the transcript state (not append) to avoid duplication.
      let sendCount = 0;

      sendIntervalRef.current = setInterval(async () => {
        if (chunksRef.current.length === 0) {
          return;
        }

        // Send ALL accumulated chunks (cumulative — don't clear)
        const chunkCount = chunksRef.current.length;
        const blob = new Blob(chunksRef.current, { type: mimeType });
        sendCount++;

        const durationEstimate = (chunkCount * 0.5).toFixed(1); // ~0.5s per chunk
        console.info(`[STT] Sending cumulative audio #${sendCount}: ${blob.size} bytes, ~${durationEstimate}s (${chunkCount} chunks)`);

        setInterimTranscript("...");

        try {
          const formData = new FormData();
          formData.append("audio", blob, "audio.webm");

          const t0 = performance.now();
          const res = await fetch(`${sttUrl}/v1/stt`, {
            method: "POST",
            body: formData,
            signal: AbortSignal.timeout(30000),
          });
          const elapsed = Math.round(performance.now() - t0);

          const audioSecs = chunkCount * 0.5;

          if (res.ok) {
            const data = await res.json();
            const fullText = (data.text || "").trim();
            console.info(`[STT] Response #${sendCount} (${elapsed}ms): model=${data.model}, text="${fullText.slice(0, 100)}...", server_ms=${data.duration_ms}`);

            // Update stats
            latencySumRef.current += elapsed;
            const wordCount = fullText ? fullText.split(/\s+/).length : 0;
            setStats(prev => ({
              lastLatencyMs: elapsed,
              avgLatencyMs: Math.round(latencySumRef.current / sendCount),
              requestCount: prev.requestCount + 1,
              errorCount: prev.errorCount,
              totalAudioSeconds: Math.round(audioSecs * 10) / 10,
              wordsTranscribed: wordCount,
            }));

            if (fullText) {
              // REPLACE transcript with server's full result (not append)
              // This avoids duplication since cumulative audio is re-transcribed each time
              setTranscript((prev) => {
                if (prev !== fullText) {
                  console.info(`[STT] Transcript replaced: "${fullText.slice(0, 80)}..."`);
                }
                return fullText;
              });
              setInterimTranscript("");
            } else {
              console.info("[STT] Empty transcription (silence or noise)");
              setInterimTranscript("");
            }
          } else {
            const errText = await res.text().catch(() => "");
            console.warn(`[STT] Server error ${res.status}: ${errText}`);
            setStats(prev => ({ ...prev, requestCount: prev.requestCount + 1, errorCount: prev.errorCount + 1 }));
            setInterimTranscript("");
          }
        } catch (e) {
          console.warn("[STT] Send failed:", e);
          setStats(prev => ({ ...prev, requestCount: prev.requestCount + 1, errorCount: prev.errorCount + 1 }));
          setInterimTranscript("");
        }
      }, 3000); // Send every 3s — gives Whisper enough audio context

    } catch (e) {
      console.error("[STT] Failed to start recording:", e);
      setError("Failed to access microphone");
      setIsListening(false);
    }
  }, [sttUrl]);

  const stopServerSTT = useCallback(() => {
    if (sendIntervalRef.current) {
      clearInterval(sendIntervalRef.current);
      sendIntervalRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    mediaRecorderRef.current = null;
    chunksRef.current = [];
    setInterimTranscript("");
    setIsListening(false);
  }, []);

  // ------------------------------------------------------------------
  // Public start/stop (auto-selects server or Web Speech)
  // ------------------------------------------------------------------
  const start = useCallback(() => {
    console.info(`[STT] start() called — activeModel=${activeModel}, isServerAvailable=${isServerAvailable}`);
    shouldBeListeningRef.current = true;
    setTranscript("");
    setInterimTranscript("");
    setError(null);

    if (activeModel === "web-speech" || !isServerAvailable) {
      // Use Web Speech API
      usingWebSpeechRef.current = true;
      console.info("[STT] Using Web Speech API fallback");
      if (recognitionRef.current) {
        try { recognitionRef.current.start(); } catch { /* ignore */ }
      } else {
        console.error("[STT] Web Speech API not available, no recognition object");
        setError("Web Speech API not available");
        setIsSupported(false);
      }
    } else {
      // Use server STT
      usingWebSpeechRef.current = false;
      console.info("[STT] Using server STT at", sttUrl);
      startServerSTT();
    }
  }, [activeModel, isServerAvailable, startServerSTT, sttUrl]);

  const stop = useCallback(() => {
    shouldBeListeningRef.current = false;
    if (restartTimeoutRef.current) {
      clearTimeout(restartTimeoutRef.current);
      restartTimeoutRef.current = null;
    }

    if (usingWebSpeechRef.current) {
      try { recognitionRef.current?.stop(); } catch { /* ignore */ }
      usingWebSpeechRef.current = false;
    } else {
      stopServerSTT();
    }
    setIsListening(false);
  }, [stopServerSTT]);

  // ------------------------------------------------------------------
  // Model switching
  // ------------------------------------------------------------------
  const switchModel = useCallback(async (model: STTModel) => {
    const wasListening = shouldBeListeningRef.current;
    if (wasListening) stop();

    if (model === "web-speech") {
      setActiveModel("web-speech");
      if (wasListening) {
        // Restart with web speech after a tick
        setTimeout(() => start(), 100);
      }
      return;
    }

    // Switch on server
    try {
      const res = await fetch(`${sttUrl}/v1/stt/switch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model }),
      });
      if (res.ok) {
        const data = await res.json();
        setActiveModel(data.model as STTModel);
        console.info("[STT] Switched to", data.model);
      } else {
        const err = await res.json().catch(() => ({ detail: "Switch failed" }));
        setError(err.detail || "Switch failed");
      }
    } catch (e) {
      setError("Failed to reach STT server for model switch");
    }

    if (wasListening) {
      setTimeout(() => start(), 500);
    }
  }, [sttUrl, stop, start]);

  return {
    transcript,
    interimTranscript,
    isListening,
    isSupported,
    error,
    start,
    stop,
    activeModel,
    switchModel,
    availableModels,
    isServerAvailable,
    stats,
  };
}
