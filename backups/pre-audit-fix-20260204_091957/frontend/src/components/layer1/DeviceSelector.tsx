"use client";

import React, { useEffect, useState } from "react";

interface AudioDeviceInfo {
  deviceId: string;
  label: string;
  kind: "audioinput" | "audiooutput";
}

interface DeviceSelectorProps {
  selectedInputId: string | null;
  selectedOutputId: string | null;
  onInputChange: (deviceId: string) => void;
  onOutputChange: (deviceId: string) => void;
  disabled?: boolean;
}

export default function DeviceSelector({
  selectedInputId,
  selectedOutputId,
  onInputChange,
  onOutputChange,
  disabled = false,
}: DeviceSelectorProps) {
  const [inputDevices, setInputDevices] = useState<AudioDeviceInfo[]>([]);
  const [outputDevices, setOutputDevices] = useState<AudioDeviceInfo[]>([]);
  const [isSupported, setIsSupported] = useState(true);

  useEffect(() => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
      setIsSupported(false);
      return;
    }

    const loadDevices = async () => {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();

        const inputs = devices
          .filter((d) => d.kind === "audioinput")
          .map((d) => ({
            deviceId: d.deviceId,
            label: d.label || `Microphone ${d.deviceId.slice(0, 8)}`,
            kind: "audioinput" as const,
          }));

        const outputs = devices
          .filter((d) => d.kind === "audiooutput")
          .map((d) => ({
            deviceId: d.deviceId,
            label: d.label || `Speaker ${d.deviceId.slice(0, 8)}`,
            kind: "audiooutput" as const,
          }));

        setInputDevices(inputs);
        setOutputDevices(outputs);
      } catch (err) {
        console.error("[DeviceSelector] Failed to enumerate devices:", err);
      }
    };

    loadDevices();

    // Listen for device changes (plug/unplug)
    const handleDeviceChange = () => {
      loadDevices();
    };

    navigator.mediaDevices.addEventListener("devicechange", handleDeviceChange);

    return () => {
      navigator.mediaDevices.removeEventListener(
        "devicechange",
        handleDeviceChange
      );
    };
  }, []);

  if (!isSupported) {
    return (
      <div className="bg-black/20 backdrop-blur-sm rounded-lg border border-white/10 px-4 py-3">
        <div className="text-xs text-yellow-400">
          Device selection not supported in this browser
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3 bg-black/20 backdrop-blur-sm rounded-lg border border-white/10 px-4 py-3">
      {/* Microphone Selector */}
      <div className="flex-1">
        <label className="block text-xs text-white/60 mb-1.5">
          Microphone
        </label>
        <select
          value={selectedInputId || ""}
          onChange={(e) => onInputChange(e.target.value)}
          disabled={disabled || inputDevices.length === 0}
          className="w-full bg-black/40 border border-white/20 rounded px-2 py-1.5 text-sm text-white/90 disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-1 focus:ring-blue-500"
        >
          {inputDevices.length === 0 ? (
            <option>No microphones found</option>
          ) : (
            inputDevices.map((device) => (
              <option key={device.deviceId} value={device.deviceId}>
                {device.label}
              </option>
            ))
          )}
        </select>
      </div>

      {/* Speaker Selector */}
      <div className="flex-1">
        <label className="block text-xs text-white/60 mb-1.5">Speaker</label>
        <select
          value={selectedOutputId || ""}
          onChange={(e) => onOutputChange(e.target.value)}
          disabled={disabled || outputDevices.length === 0}
          className="w-full bg-black/40 border border-white/20 rounded px-2 py-1.5 text-sm text-white/90 disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-1 focus:ring-purple-500"
        >
          {outputDevices.length === 0 ? (
            <option>No speakers found</option>
          ) : (
            outputDevices.map((device) => (
              <option key={device.deviceId} value={device.deviceId}>
                {device.label}
              </option>
            ))
          )}
        </select>
      </div>
    </div>
  );
}
