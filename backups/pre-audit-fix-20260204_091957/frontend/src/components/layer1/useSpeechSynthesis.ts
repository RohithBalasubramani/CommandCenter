"use client";

import { useCallback, useEffect, useRef, useState } from "react";

interface UseSpeechSynthesisReturn {
  speak: (text: string, options?: SpeakOptions) => void;
  stop: () => void;
  isSpeaking: boolean;
  isSupported: boolean;
  voices: SpeechSynthesisVoice[];
  selectedVoice: SpeechSynthesisVoice | null;
  setVoice: (voice: SpeechSynthesisVoice) => void;
}

interface SpeakOptions {
  rate?: number; // 0.1 - 10, default 1
  pitch?: number; // 0 - 2, default 1
  volume?: number; // 0 - 1, default 1
  onEnd?: () => void;
  onError?: (error: string) => void;
}

/**
 * Hook for browser-native speech synthesis (Web Speech API TTS).
 *
 * Used to speak Layer 2 responses and fillers.
 * This is a temporary solution until a proper TTS service is integrated.
 */
export function useSpeechSynthesis(): UseSpeechSynthesisReturn {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isSupported, setIsSupported] = useState(true);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [selectedVoice, setSelectedVoice] =
    useState<SpeechSynthesisVoice | null>(null);

  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
  const synthRef = useRef<SpeechSynthesis | null>(null);

  // Initialize speech synthesis
  useEffect(() => {
    if (typeof window === "undefined" || !window.speechSynthesis) {
      setIsSupported(false);
      console.warn("[SpeechSynthesis] Not supported in this browser");
      return;
    }

    synthRef.current = window.speechSynthesis;

    // Load available voices
    const loadVoices = () => {
      const availableVoices = synthRef.current?.getVoices() || [];
      setVoices(availableVoices);

      // Select a good default voice (prefer English, natural-sounding)
      if (!selectedVoice && availableVoices.length > 0) {
        const preferredVoice =
          // Try to find a good English voice
          availableVoices.find(
            (v) =>
              v.lang.startsWith("en") &&
              (v.name.includes("Natural") ||
                v.name.includes("Neural") ||
                v.name.includes("Premium"))
          ) ||
          // Fallback to any English voice
          availableVoices.find((v) => v.lang.startsWith("en-US")) ||
          availableVoices.find((v) => v.lang.startsWith("en")) ||
          // Last resort: first available
          availableVoices[0];

        if (preferredVoice) {
          setSelectedVoice(preferredVoice);
          console.info(
            "[SpeechSynthesis] Selected voice:",
            preferredVoice.name
          );
        }
      }
    };

    // Voices may load asynchronously
    loadVoices();
    synthRef.current.addEventListener("voiceschanged", loadVoices);

    return () => {
      synthRef.current?.removeEventListener("voiceschanged", loadVoices);
      // Cancel any ongoing speech
      synthRef.current?.cancel();
    };
  }, [selectedVoice]);

  const speak = useCallback(
    (text: string, options: SpeakOptions = {}) => {
      if (!synthRef.current || !isSupported) {
        console.warn("[SpeechSynthesis] Not supported");
        options.onError?.("Speech synthesis not supported");
        return;
      }

      // Cancel any ongoing speech
      synthRef.current.cancel();

      const utterance = new SpeechSynthesisUtterance(text);
      utteranceRef.current = utterance;

      // Apply options
      if (selectedVoice) {
        utterance.voice = selectedVoice;
      }
      utterance.rate = options.rate ?? 1.0;
      utterance.pitch = options.pitch ?? 1.0;
      utterance.volume = options.volume ?? 1.0;

      // Event handlers
      utterance.onstart = () => {
        console.info("[SpeechSynthesis] Speaking:", text.substring(0, 50));
        setIsSpeaking(true);
      };

      utterance.onend = () => {
        console.info("[SpeechSynthesis] Finished speaking");
        setIsSpeaking(false);
        options.onEnd?.();
      };

      utterance.onerror = (event) => {
        console.error("[SpeechSynthesis] Error:", event.error);
        setIsSpeaking(false);
        options.onError?.(event.error);
      };

      // Speak
      synthRef.current.speak(utterance);
    },
    [isSupported, selectedVoice]
  );

  const stop = useCallback(() => {
    if (synthRef.current) {
      synthRef.current.cancel();
      setIsSpeaking(false);
    }
  }, []);

  const setVoice = useCallback((voice: SpeechSynthesisVoice) => {
    setSelectedVoice(voice);
  }, []);

  return {
    speak,
    stop,
    isSpeaking,
    isSupported,
    voices,
    selectedVoice,
    setVoice,
  };
}
