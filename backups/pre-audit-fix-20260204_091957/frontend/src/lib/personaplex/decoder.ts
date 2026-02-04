/**
 * PersonaPlex Opus Decoder Worker Manager
 *
 * Manages a Web Worker that decodes Opus/Ogg audio from the PersonaPlex server.
 * The decoder worker uses WASM for real-time Opus decoding.
 */

// Minimal valid Ogg BOS page with OpusHead header (mono, 48kHz)
// This triggers the decoder's internal init() to create buffers
function createWarmupBosPage(): Uint8Array {
  // OpusHead: "OpusHead" + version(1) + channels(1) + preskip(2) + samplerate(4) + gain(2) + mapping(1)
  const opusHead = new Uint8Array([
    0x4f, 0x70, 0x75, 0x73, 0x48, 0x65, 0x61, 0x64, // "OpusHead"
    0x01,       // Version 1
    0x01,       // 1 channel (mono)
    0x38, 0x01, // Pre-skip: 312 samples (little-endian)
    0x80, 0xbb, 0x00, 0x00, // Sample rate: 48000 Hz (little-endian)
    0x00, 0x00, // Output gain: 0
    0x00,       // Channel mapping: 0 (mono/stereo)
  ]);

  // Ogg page header
  const pageHeader = new Uint8Array([
    0x4f, 0x67, 0x67, 0x53, // "OggS" magic
    0x00,       // Version 0
    0x02,       // BOS flag (Beginning of Stream)
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Granule position: 0
    0x01, 0x00, 0x00, 0x00, // Stream serial: 1
    0x00, 0x00, 0x00, 0x00, // Page sequence: 0
    0x00, 0x00, 0x00, 0x00, // CRC
    0x01,       // 1 segment
    0x13,       // Segment size: 19 bytes (OpusHead)
  ]);

  const bosPage = new Uint8Array(pageHeader.length + opusHead.length);
  bosPage.set(pageHeader, 0);
  bosPage.set(opusHead, pageHeader.length);
  return bosPage;
}

function sendInitCommand(worker: Worker, audioContextSampleRate: number): void {
  worker.postMessage({
    command: "init",
    bufferLength: (960 * audioContextSampleRate) / 24000,
    decoderSampleRate: 24000,
    outputBufferSampleRate: audioContextSampleRate,
    resampleQuality: 0,
  });

  // After a short delay, send warmup BOS page to trigger decoder's internal init
  setTimeout(() => {
    worker.postMessage({
      command: "decode",
      pages: createWarmupBosPage(),
    });
  }, 100);
}

/** Create a fresh decoder Web Worker. */
export function createDecoderWorker(): Worker {
  const worker = new Worker("/assets/decoderWorker.min.js");
  worker.onerror = (event) => {
    console.error("[PersonaPlex] Decoder worker error:", event.message);
  };
  return worker;
}

/** Initialize a decoder worker and resolve when ready. */
export function initDecoder(
  worker: Worker,
  audioContextSampleRate: number,
): Promise<void> {
  return new Promise((resolve) => {
    sendInitCommand(worker, audioContextSampleRate);
    // Give worker time to load WASM and process init
    setTimeout(resolve, 1000);
  });
}
