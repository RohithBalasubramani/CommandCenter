"use client";

import React, { useEffect, useRef, useState } from "react";

interface AudioVisualizerProps {
  analyser: AnalyserNode | null;
  type: "input" | "output";
  isActive: boolean;
}

export default function AudioVisualizer({
  analyser,
  type,
  isActive,
}: AudioVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const [level, setLevel] = useState(0);
  const [peak, setPeak] = useState(0);

  useEffect(() => {
    if (!analyser || !canvasRef.current || !isActive) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      animationRef.current = requestAnimationFrame(draw);

      analyser.getByteTimeDomainData(dataArray);

      // Calculate RMS (root mean square) level
      let sum = 0;
      let peakValue = 0;
      for (let i = 0; i < bufferLength; i++) {
        const normalized = (dataArray[i] - 128) / 128;
        sum += normalized * normalized;
        peakValue = Math.max(peakValue, Math.abs(normalized));
      }
      const rms = Math.sqrt(sum / bufferLength);

      setLevel(rms);
      setPeak(peakValue);

      // Clear canvas
      ctx.fillStyle = "rgba(0, 0, 0, 0.1)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw waveform
      ctx.lineWidth = 2;
      const color = type === "input" ? "59, 130, 246" : "168, 85, 247"; // blue or purple
      ctx.strokeStyle = `rgba(${color}, 0.8)`;

      ctx.beginPath();
      const sliceWidth = canvas.width / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }

        x += sliceWidth;
      }

      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
    };

    draw();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [analyser, isActive, type]);

  const getLevelColor = () => {
    if (level < 0.1) return "bg-green-500";
    if (level < 0.5) return "bg-yellow-500";
    if (level < 0.8) return "bg-orange-500";
    return "bg-red-500";
  };

  const label = type === "input" ? "Microphone" : "Speaker";
  const colorClass = type === "input" ? "text-blue-400" : "text-purple-400";

  return (
    <div className="flex flex-col gap-2 bg-black/20 backdrop-blur-sm rounded-lg border border-white/10 p-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className={`text-xs font-medium ${colorClass}`}>{label}</span>
        {isActive && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-white/40">
              Level: {(level * 100).toFixed(0)}%
            </span>
            <span className="text-xs text-white/40">
              Peak: {(peak * 100).toFixed(0)}%
            </span>
          </div>
        )}
      </div>

      {/* Waveform */}
      <canvas
        ref={canvasRef}
        width={800}
        height={80}
        className="w-full h-20 rounded bg-black/40"
      />

      {/* Level meter */}
      <div className="flex items-center gap-2">
        <div className="flex-1 h-2 bg-black/40 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-100 ${getLevelColor()}`}
            style={{ width: `${Math.min(level * 100, 100)}%` }}
          />
        </div>
        {peak > 0.95 && (
          <span className="text-xs text-red-400 font-medium">CLIPPING!</span>
        )}
      </div>
    </div>
  );
}
