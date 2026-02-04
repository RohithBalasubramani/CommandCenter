"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Voice Activity Detection (VAD) hook using Silero VAD.
 *
 * Provides accurate speech start/end detection for natural conversation flow.
 * This is more reliable than detecting "2 identical transcripts" for silence.
 *
 * @see https://github.com/ricky0123/vad
 */

export interface VADCallbacks {
  onSpeechStart?: () => void;
  onSpeechEnd?: (audio: Float32Array) => void;
  onVADMisfire?: () => void;
}

export interface UseVADReturn {
  /** Whether VAD is currently listening */
  isListening: boolean;
  /** Whether speech is currently detected */
  isSpeaking: boolean;
  /** Start VAD processing */
  start: () => Promise<void>;
  /** Stop VAD processing */
  stop: () => void;
  /** Whether VAD is supported and loaded */
  isSupported: boolean;
  /** Error message if any */
  error: string | null;
}

interface VADOptions {
  /** Minimum speech duration (ms) to trigger onSpeechEnd. Default: 250 */
  minSpeechMs?: number;
  /** Threshold for speech detection (0-1). Higher = less sensitive. Default: 0.5 */
  positiveSpeechThreshold?: number;
  /** Threshold for non-speech detection (0-1). Lower = more sensitive. Default: 0.35 */
  negativeSpeechThreshold?: number;
  /** After silence detected, wait this many ms before confirming speech end. Default: 320 */
  redemptionMs?: number;
  /** Pre-speech padding (ms) to include in audio. Default: 480 */
  preSpeechPadMs?: number;
}

const DEFAULT_OPTIONS: VADOptions = {
  minSpeechMs: 250,  // Minimum 250ms speech to trigger
  positiveSpeechThreshold: 0.5,
  negativeSpeechThreshold: 0.35,
  redemptionMs: 320,  // Wait 320ms of silence before confirming speech end
  preSpeechPadMs: 480,  // Include 480ms of audio before speech start
};

export function useVAD(
  callbacks: VADCallbacks = {},
  options: VADOptions = {}
): UseVADReturn {
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isSupported, setIsSupported] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const vadRef = useRef<any>(null);
  const callbacksRef = useRef(callbacks);
  callbacksRef.current = callbacks;

  const mergedOptions = { ...DEFAULT_OPTIONS, ...options };

  // Check for browser support
  useEffect(() => {
    if (typeof window === "undefined") {
      setIsSupported(false);
      return;
    }

    // Check for required APIs
    const hasMediaDevices = !!(navigator.mediaDevices?.getUserMedia);
    const hasAudioContext = !!(window.AudioContext || (window as any).webkitAudioContext);

    if (!hasMediaDevices || !hasAudioContext) {
      setIsSupported(false);
      setError("Browser does not support audio capture");
    }
  }, []);

  const start = useCallback(async () => {
    if (!isSupported) {
      setError("VAD not supported in this browser");
      return;
    }

    if (vadRef.current) {
      console.warn("[VAD] Already running");
      return;
    }

    setError(null);

    try {
      // Dynamic import to avoid SSR issues
      const { MicVAD } = await import("@ricky0123/vad-web");

      console.info("[VAD] Initializing with options:", mergedOptions);

      const vad = await MicVAD.new({
        positiveSpeechThreshold: mergedOptions.positiveSpeechThreshold,
        negativeSpeechThreshold: mergedOptions.negativeSpeechThreshold,
        redemptionMs: mergedOptions.redemptionMs,
        preSpeechPadMs: mergedOptions.preSpeechPadMs,
        minSpeechMs: mergedOptions.minSpeechMs,

        onSpeechStart: () => {
          console.info("[VAD] Speech started");
          setIsSpeaking(true);
          callbacksRef.current.onSpeechStart?.();
        },

        onSpeechEnd: (audio: Float32Array) => {
          console.info(`[VAD] Speech ended (${audio.length} samples, ${(audio.length / 16000).toFixed(2)}s)`);
          setIsSpeaking(false);
          callbacksRef.current.onSpeechEnd?.(audio);
        },

        onVADMisfire: () => {
          console.info("[VAD] Misfire (speech too short)");
          setIsSpeaking(false);
          callbacksRef.current.onVADMisfire?.();
        },
      });

      vadRef.current = vad;
      vad.start();
      setIsListening(true);
      console.info("[VAD] Started successfully");

    } catch (e: any) {
      console.error("[VAD] Failed to start:", e);
      setError(e.message || "Failed to initialize VAD");
      setIsListening(false);
    }
  }, [isSupported, mergedOptions]);

  const stop = useCallback(() => {
    if (vadRef.current) {
      console.info("[VAD] Stopping...");
      vadRef.current.pause();
      vadRef.current.destroy();
      vadRef.current = null;
    }
    setIsListening(false);
    setIsSpeaking(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (vadRef.current) {
        vadRef.current.pause();
        vadRef.current.destroy();
        vadRef.current = null;
      }
    };
  }, []);

  return {
    isListening,
    isSpeaking,
    start,
    stop,
    isSupported,
    error,
  };
}
