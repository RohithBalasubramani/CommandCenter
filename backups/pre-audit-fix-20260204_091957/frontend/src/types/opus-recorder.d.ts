declare module "opus-recorder" {
  interface RecorderOptions {
    encoderPath?: string;
    bufferLength?: number;
    encoderFrameSize?: number;
    encoderSampleRate?: number;
    maxFramesPerPage?: number;
    numberOfChannels?: number;
    recordingGain?: number;
    resampleQuality?: number;
    encoderComplexity?: number;
    encoderApplication?: number;
    streamPages?: boolean;
    mediaTrackConstraints?: MediaStreamConstraints;
    sourceNode?: AudioNode;
  }

  class Recorder {
    constructor(options?: RecorderOptions);
    encodedSamplePosition: number;
    ondataavailable: ((data: Uint8Array) => void) | null;
    onstart: (() => void) | null;
    onstop: (() => void) | null;
    start(): Promise<void>;
    stop(): void;
    pause(): void;
    resume(): void;
    close(): void;
  }

  export default Recorder;
}
