"use client";

import Canvas from "@/components/canvas/Canvas";
import VoiceInterfaceV2 from "@/components/layer1/VoiceInterfaceV2";
import TranscriptPanel from "@/components/layer1/TranscriptPanel";
import StatusBar from "@/components/status-bar/StatusBar";
import { DebugPanel } from "@/components/debug";

/**
 * Command Center — Main Page
 *
 * 100vh x 100vw zero-scroll canvas.
 * Layer 1 (Voice I/O) is the primary content for now.
 * Blob (Layer 3) will control widget placement once Layer 2 feeds layout JSON.
 *
 * Debug Panel: Press Ctrl+D to toggle pipeline visualization
 */
export default function CommandCenterPage() {
  return (
    <>
      <Canvas statusBar={<StatusBar />}>
        {/* Layer 1: Voice I/O — Full canvas for now */}
        <div className="h-full w-full flex flex-col">
          {/* Main voice interface */}
          <VoiceInterfaceV2 />

          {/* Transcript overlay — shows what flows to Layer 2 */}
          <TranscriptPanel />
        </div>
      </Canvas>

      {/* Debug Panel — Toggle with Ctrl+D */}
      <DebugPanel />
    </>
  );
}
