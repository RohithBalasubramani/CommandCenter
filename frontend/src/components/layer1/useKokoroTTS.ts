"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { config } from "@/lib/config";

/**
 * TTS engine options.
 */
export type TTSEngine = "kokoro" | "piper" | "browser";

interface SpeakOptions {
  rate?: number;
  pitch?: number;
  volume?: number;
  voice?: string;
  onEnd?: () => void;
  onError?: (error: string) => void;
}

export interface TTSStats {
  lastLatencyMs: number | null;
  requestCount: number;
  errorCount: number;
}

interface UseKokoroTTSReturn {
  speak: (text: string, options?: SpeakOptions) => void;
  stop: () => void;
  isSpeaking: boolean;
  isSupported: boolean;

  // Engine management
  activeEngine: TTSEngine;
  switchEngine: (engine: TTSEngine) => void;
  isServerAvailable: boolean;

  // Stats
  stats: TTSStats;
}

/**
 * useKokoroTTS — TTS hook supporting Kokoro, Piper, and browser fallback.
 *
 * Kokoro: Docker container on port 8880, OpenAI-compatible /v1/audio/speech
 * Piper: Same server or separate, lightweight CPU TTS
 * Browser: Web Speech API fallback
 */
export function useKokoroTTS(): UseKokoroTTSReturn {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isSupported, setIsSupported] = useState(true);
  const [activeEngine, setActiveEngine] = useState<TTSEngine>("kokoro");
  const [isServerAvailable, setIsServerAvailable] = useState(false);

  // Stats
  const [stats, setStats] = useState<TTSStats>({
    lastLatencyMs: null,
    requestCount: 0,
    errorCount: 0,
  });

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const synthRef = useRef<SpeechSynthesis | null>(null);
  const selectedVoiceRef = useRef<SpeechSynthesisVoice | null>(null);

  const ttsUrl = config.tts?.serverUrl || "http://localhost:8880";

  // ------------------------------------------------------------------
  // Health check
  // ------------------------------------------------------------------
  useEffect(() => {
    const checkHealth = async () => {
      try {
        // Kokoro-FastAPI has OpenAI-compatible endpoints
        const res = await fetch(`${ttsUrl}/v1/models`, {
          signal: AbortSignal.timeout(3000),
        });
        if (res.ok) {
          setIsServerAvailable(true);
          console.info("[TTS] Kokoro server available at", ttsUrl);
        } else {
          setIsServerAvailable(false);
        }
      } catch {
        console.warn("[TTS] Server unreachable, will use browser fallback");
        setIsServerAvailable(false);
      }
    };
    checkHealth();

    // Initialize browser fallback
    if (typeof window !== "undefined" && window.speechSynthesis) {
      synthRef.current = window.speechSynthesis;
      const loadVoices = () => {
        const voices = synthRef.current?.getVoices() || [];
        const preferred =
          voices.find(
            (v) =>
              v.lang.startsWith("en") &&
              (v.name.includes("Natural") ||
                v.name.includes("Neural") ||
                v.name.includes("Premium"))
          ) ||
          voices.find((v) => v.lang.startsWith("en-US")) ||
          voices.find((v) => v.lang.startsWith("en")) ||
          voices[0];
        if (preferred) selectedVoiceRef.current = preferred;
      };
      loadVoices();
      synthRef.current.addEventListener("voiceschanged", loadVoices);
    }

    return () => {
      synthRef.current?.cancel();
    };
  }, [ttsUrl]);

  // ------------------------------------------------------------------
  // Browser TTS fallback
  // ------------------------------------------------------------------
  const speakBrowser = useCallback(
    (text: string, options: SpeakOptions = {}) => {
      if (!synthRef.current) {
        options.onError?.("Browser TTS not supported");
        return;
      }
      synthRef.current.cancel();

      const utterance = new SpeechSynthesisUtterance(text);
      if (selectedVoiceRef.current) utterance.voice = selectedVoiceRef.current;
      utterance.rate = options.rate ?? 1.0;
      utterance.pitch = options.pitch ?? 1.0;
      utterance.volume = options.volume ?? 1.0;

      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => {
        setIsSpeaking(false);
        options.onEnd?.();
      };
      utterance.onerror = (e) => {
        setIsSpeaking(false);
        options.onError?.(e.error);
      };

      synthRef.current.speak(utterance);
    },
    []
  );

  // ------------------------------------------------------------------
  // Server TTS (Kokoro / Piper)
  // ------------------------------------------------------------------
  const speakServer = useCallback(
    async (text: string, options: SpeakOptions = {}) => {
      console.info(`[TTS] speakServer: "${text.slice(0, 60)}..." engine=${activeEngine}`);

      // Cancel any in-flight request
      if (abortRef.current) {
        abortRef.current.abort();
      }
      // Detach old audio element's handlers BEFORE clearing src,
      // otherwise setting src="" fires onerror which triggers browser fallback
      if (audioRef.current) {
        audioRef.current.onended = null;
        audioRef.current.onerror = null;
        audioRef.current.pause();
        audioRef.current.src = "";
        audioRef.current = null;
      }
      // Also cancel any browser TTS that might be speaking
      synthRef.current?.cancel();

      const controller = new AbortController();
      abortRef.current = controller;
      setIsSpeaking(true);

      try {
        const t0 = performance.now();
        const res = await fetch(`${ttsUrl}/v1/audio/speech`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: activeEngine === "piper" ? "piper" : "kokoro",
            input: text,
            voice: options.voice || "af_heart",
            speed: options.rate ?? 1.0,
            response_format: "mp3",
          }),
          signal: controller.signal,
        });

        const fetchElapsed = Math.round(performance.now() - t0);

        if (!res.ok) {
          const errBody = await res.text().catch(() => "");
          throw new Error(`TTS server returned ${res.status}: ${errBody.slice(0, 200)}`);
        }

        const blob = await res.blob();
        console.info(`[TTS] Got audio blob: ${blob.size} bytes, type=${blob.type} (${fetchElapsed}ms)`);

        // Update stats
        setStats(prev => ({
          lastLatencyMs: fetchElapsed,
          requestCount: prev.requestCount + 1,
          errorCount: prev.errorCount,
        }));

        const url = URL.createObjectURL(blob);

        const audio = new Audio(url);
        audioRef.current = audio;

        audio.volume = options.volume ?? 1.0;

        audio.onended = () => {
          console.info("[TTS] Audio playback ended");
          setIsSpeaking(false);
          URL.revokeObjectURL(url);
          options.onEnd?.();
        };

        audio.onerror = (e) => {
          // Only handle errors for the CURRENT audio element
          if (audioRef.current !== audio) return;
          const mediaError = audio.error;
          const errorDetail = mediaError
            ? `code=${mediaError.code} message="${mediaError.message}"`
            : "unknown";
          console.error(`[TTS] Audio playback error: ${errorDetail}`, e);
          setIsSpeaking(false);
          URL.revokeObjectURL(url);
          options.onError?.(errorDetail);
        };

        await audio.play();
        console.info("[TTS] Audio play() started");
      } catch (e: any) {
        if (e.name === "AbortError") return; // Intentional cancel
        console.error("[TTS] Server speak failed:", e);
        setIsSpeaking(false);
        setStats(prev => ({ ...prev, requestCount: prev.requestCount + 1, errorCount: prev.errorCount + 1 }));
        options.onError?.(e.message || "Server TTS failed");
      }
    },
    [ttsUrl, activeEngine]
  );

  // ------------------------------------------------------------------
  // Public speak/stop
  // ------------------------------------------------------------------
  const speak = useCallback(
    (text: string, options: SpeakOptions = {}) => {
      if (!text.trim()) return;

      if (activeEngine === "browser" || !isServerAvailable) {
        speakBrowser(text, options);
      } else {
        speakServer(text, options);
      }
    },
    [activeEngine, isServerAvailable, speakBrowser, speakServer]
  );

  const stop = useCallback(() => {
    // Stop server audio
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = "";
      audioRef.current = null;
    }

    // Stop browser TTS
    synthRef.current?.cancel();

    setIsSpeaking(false);
  }, []);

  // ------------------------------------------------------------------
  // Engine switching
  // ------------------------------------------------------------------
  const switchEngine = useCallback((engine: TTSEngine) => {
    stop();
    setActiveEngine(engine);
    console.info("[TTS] Switched to engine:", engine);
  }, [stop]);

  return {
    speak,
    stop,
    isSpeaking,
    isSupported,
    activeEngine,
    switchEngine,
    isServerAvailable,
    stats,
  };
}
