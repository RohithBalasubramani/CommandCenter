"use client";

import { useCallback, useEffect, useRef, useState } from "react";

interface UseSpeechRecognitionReturn {
  transcript: string;
  interimTranscript: string;
  isListening: boolean;
  start: () => void;
  stop: () => void;
  isSupported: boolean;
  error: string | null;
}

/**
 * Hook for browser-native speech recognition (Web Speech API).
 * Used to transcribe user speech for display in the conversation.
 *
 * Includes auto-restart logic to handle browser's tendency to stop
 * speech recognition after periods of silence.
 */
export function useSpeechRecognition(): UseSpeechRecognitionReturn {
  const [transcript, setTranscript] = useState("");
  const [interimTranscript, setInterimTranscript] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isSupported, setIsSupported] = useState(true);

  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const shouldBeListeningRef = useRef(false); // Track if we WANT to be listening
  const restartTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    // Check for browser support
    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setIsSupported(false);
      console.warn("[SpeechRecognition] Not supported in this browser");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onstart = () => {
      console.info("[SpeechRecognition] Started");
      setIsListening(true);
      setError(null);
    };

    recognition.onend = () => {
      console.info("[SpeechRecognition] Ended");
      setIsListening(false);

      // Auto-restart if we should still be listening
      // This handles browser's tendency to stop after silence
      if (shouldBeListeningRef.current) {
        console.info("[SpeechRecognition] Auto-restarting after unexpected stop...");
        // Small delay to avoid rapid restart loops
        restartTimeoutRef.current = setTimeout(() => {
          if (shouldBeListeningRef.current && recognitionRef.current) {
            try {
              recognitionRef.current.start();
            } catch (e) {
              // May fail if already started or other issue
              console.warn("[SpeechRecognition] Auto-restart failed:", e);
            }
          }
        }, 100);
      }
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      console.error("[SpeechRecognition] Error:", event.error);
      // Emit custom event so VoiceInterface can track no-speech count
      window.dispatchEvent(
        new CustomEvent("speechrecognition-error", { detail: event.error })
      );
      // Don't report "no-speech" as an error - it's normal
      if (event.error !== "no-speech") {
        setError(event.error);
      }
    };

    recognition.onresult = (event: SpeechRecognitionEvent) => {
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
      if (restartTimeoutRef.current) {
        clearTimeout(restartTimeoutRef.current);
        restartTimeoutRef.current = null;
      }
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop();
        } catch {
          // Ignore
        }
      }
    };
  }, []);

  const start = useCallback(() => {
    if (!recognitionRef.current || !isSupported) return;

    // Mark that we want to be listening (for auto-restart)
    shouldBeListeningRef.current = true;

    // Reset transcripts on new start
    setTranscript("");
    setInterimTranscript("");
    setError(null);

    try {
      recognitionRef.current.start();
    } catch (e) {
      console.error("[SpeechRecognition] Failed to start:", e);
      setError("Failed to start speech recognition");
      shouldBeListeningRef.current = false;
    }
  }, [isSupported]);

  const stop = useCallback(() => {
    // Mark that we want to stop (prevent auto-restart)
    shouldBeListeningRef.current = false;

    // Clear any pending restart
    if (restartTimeoutRef.current) {
      clearTimeout(restartTimeoutRef.current);
      restartTimeoutRef.current = null;
    }

    if (!recognitionRef.current) return;

    try {
      recognitionRef.current.stop();
    } catch {
      // Ignore
    }
  }, []);

  return {
    transcript,
    interimTranscript,
    isListening,
    start,
    stop,
    isSupported,
    error,
  };
}
