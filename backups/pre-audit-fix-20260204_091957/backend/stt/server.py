"""
STT Server — Parakeet (primary) + Faster-Whisper (fallback)

FastAPI server on port 8890 providing speech-to-text via:
  - NVIDIA Parakeet (NeMo ASR) — best accuracy for industrial terms
  - Faster-Whisper — lighter fallback when Parakeet unavailable

Endpoints:
  POST /v1/stt          — transcribe audio blob → { text }
  GET  /v1/stt/health   — status + active model info
  POST /v1/stt/switch   — switch between parakeet/whisper
  GET  /v1/stt/models   — list available models

Hot-swap: selected model loaded in GPU, other cached on disk.
"""

import asyncio
import io
import logging
import time
from contextlib import asynccontextmanager
from enum import Enum
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger("stt")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [STT] %(message)s")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

class ModelName(str, Enum):
    PARAKEET = "parakeet"
    WHISPER = "whisper"


class ModelInfo(BaseModel):
    name: str
    loaded: bool
    available: bool
    description: str
    runtime_broken: bool = False


class TranscribeResponse(BaseModel):
    text: str
    model: str
    duration_ms: float


class SwitchRequest(BaseModel):
    model: ModelName


class HealthResponse(BaseModel):
    status: str
    active_model: str
    models: list[ModelInfo]


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

class ParakeetASR:
    """NVIDIA Parakeet ASR via NeMo toolkit."""

    def __init__(self):
        self.model = None
        self.available = False
        self.runtime_broken = False  # Set True if model loads but fails at runtime

    def load(self):
        try:
            import nemo.collections.asr as nemo_asr  # type: ignore
            from omegaconf import OmegaConf  # type: ignore
            logger.info("Loading Parakeet model (this may download ~5GB on first run)...")
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name="nvidia/parakeet-tdt-0.6b-v2"
            )
            # Fix: NeMo 2.6 greedy_batch decoder uses CUDA graphs that are
            # broken with PyTorch 2.10 (cu_call return value mismatch).
            # Switch to non-batch greedy decoding which avoids CUDA graphs.
            try:
                decoding_cfg = OmegaConf.create({
                    'strategy': 'greedy',
                    'model_type': 'tdt',
                    'durations': [0, 1, 2, 3, 4],
                    'greedy': {'max_symbols': 10},
                })
                self.model.change_decoding_strategy(decoding_cfg)
                logger.info("Parakeet: switched to greedy decoding (CUDA graph workaround)")
            except Exception as e:
                logger.warning(f"Parakeet: could not change decoding strategy: {e}")
            self.available = True
            logger.info("Parakeet loaded successfully")
        except Exception as e:
            logger.warning(f"Parakeet unavailable: {e}")
            self.available = False

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
            # Free GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        logger.info("Parakeet unloaded")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if self.model is None:
            raise RuntimeError("Parakeet not loaded")

        # Parakeet expects 16kHz mono float32
        if sample_rate != 16000:
            try:
                import librosa  # type: ignore
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            except ImportError:
                ratio = sample_rate / 16000
                indices = np.round(np.arange(0, len(audio), ratio)).astype(int)
                indices = indices[indices < len(audio)]
                audio = audio[indices]

        # Write to temp file (NeMo expects file paths)
        import tempfile
        import os
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, audio, 16000)

            logger.info(f"Parakeet: transcribing from {tmp_path} ({len(audio)} samples)")

            # NeMo 2.6+ API: transcribe(audio=[path], ...)
            result = None
            try:
                result = self.model.transcribe(audio=[tmp_path])
                logger.info(f"Parakeet: transcribe(audio=) returned type={type(result)}")
            except TypeError:
                # Fallback for older NeMo: paths2audio_files keyword
                try:
                    result = self.model.transcribe(paths2audio_files=[tmp_path])
                    logger.info(f"Parakeet: transcribe(paths2audio_files=) returned type={type(result)}")
                except TypeError as e:
                    logger.error(f"Parakeet: all calling conventions failed: {e}")
                    raise

            # Parse result — NeMo returns various formats depending on version
            if result is None:
                return ""

            # Hypothesis object (newer NeMo)
            if hasattr(result, 'text'):
                return str(result.text)

            # List of strings
            if isinstance(result, list) and len(result) > 0:
                item = result[0]
                if isinstance(item, str):
                    return item
                # Hypothesis object in list
                if hasattr(item, 'text'):
                    return str(item.text)
                # Tuple (text, ...) format
                if isinstance(item, (list, tuple)) and len(item) > 0:
                    return str(item[0])
                return str(item)

            return str(result)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


class WhisperASR:
    """Faster-Whisper ASR."""

    def __init__(self):
        self.model = None
        self.available = False
        self.runtime_broken = False  # Set True if model loads but fails at runtime

    def load(self):
        try:
            from faster_whisper import WhisperModel  # type: ignore
            # Use medium for fast startup; switch to large-v3 via /v1/stt/switch
            model_size = "medium"
            logger.info(f"Loading Faster-Whisper ({model_size})...")
            self.model = WhisperModel(
                model_size,
                device="cuda",
                compute_type="float16",
            )
            self.available = True
            logger.info("Faster-Whisper loaded successfully")
        except Exception as e:
            logger.warning(f"Faster-Whisper unavailable: {e}")
            self.available = False

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        logger.info("Whisper unloaded")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if self.model is None:
            raise RuntimeError("Whisper not loaded")

        # Whisper expects 16kHz mono float32
        if sample_rate != 16000:
            try:
                import librosa  # type: ignore
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            except ImportError:
                ratio = sample_rate / 16000
                indices = np.round(np.arange(0, len(audio), ratio)).astype(int)
                indices = indices[indices < len(audio)]
                audio = audio[indices]

        duration_s = len(audio) / 16000
        # Disable VAD for short clips (<5s) — it aggressively filters them as silence
        use_vad = duration_s >= 5.0
        logger.info(f"Whisper transcribing {duration_s:.1f}s audio (vad={use_vad})")

        segments, _info = self.model.transcribe(
            audio,
            beam_size=5,
            language="en",
            vad_filter=use_vad,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        logger.info(f"Whisper result: '{text[:100]}'")
        return text


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

parakeet = ParakeetASR()
whisper = WhisperASR()
active_model: ModelName = ModelName.PARAKEET
_lock = asyncio.Lock()
_transcribe_semaphore = asyncio.Semaphore(1)  # Max 1 concurrent transcription


def get_active_asr():
    if active_model == ModelName.PARAKEET:
        return parakeet
    return whisper


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load both models so we have runtime fallback
    logger.info("STT Server starting — loading models...")
    global active_model

    # Load Whisper first (fast, reliable fallback)
    whisper.load()

    # Then try Parakeet (primary, but may have API issues)
    parakeet.load()

    if parakeet.available:
        active_model = ModelName.PARAKEET
        logger.info("Active model: Parakeet (Whisper ready as fallback)")
    elif whisper.available:
        active_model = ModelName.WHISPER
        logger.info("Active model: Whisper (Parakeet unavailable)")
    else:
        logger.error("No STT models available!")

    yield
    # Shutdown
    parakeet.unload()
    whisper.unload()


app = FastAPI(title="STT Server", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _decode_audio_bytes(content: bytes, filename: str) -> tuple[np.ndarray, int]:
    """Decode audio bytes to numpy array, handling webm/opus via ffmpeg."""
    # Try soundfile first (handles WAV, FLAC, OGG/Vorbis)
    try:
        buf = io.BytesIO(content)
        data, sample_rate = sf.read(buf, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        logger.info(f"Decoded with soundfile: {len(data)} samples @ {sample_rate}Hz")
        return data, sample_rate
    except Exception as sf_err:
        logger.info(f"soundfile failed ({sf_err}), trying ffmpeg...")

    # Fallback: use ffmpeg to convert any format → WAV PCM
    import subprocess
    import tempfile

    suffix = ".webm" if "webm" in (filename or "") else ".ogg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(content)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path + ".wav"
    try:
        cmd = [
            "ffmpeg", "-y", "-i", tmp_in_path,
            "-ar", "16000", "-ac", "1", "-f", "wav",
            tmp_out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"ffmpeg failed (rc={result.returncode}): {stderr}")

        data, sample_rate = sf.read(tmp_out_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        logger.info(f"Decoded with ffmpeg: {len(data)} samples @ {sample_rate}Hz")
        return data, sample_rate
    finally:
        import os
        for p in [tmp_in_path, tmp_out_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


@app.post("/v1/stt", response_model=TranscribeResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe an audio file to text. Max 1 concurrent request — rejects if busy."""
    global active_model

    # Concurrency guard: reject if another transcription is in progress.
    # This prevents CUDA OOM from multiple large audio files queuing up.
    if _transcribe_semaphore.locked():
        logger.warning("STT busy — rejecting concurrent request")
        raise HTTPException(status_code=429, detail="STT busy, try again shortly")

    async with _transcribe_semaphore:
        return await _do_transcribe(audio)


async def _do_transcribe(audio: UploadFile) -> TranscribeResponse:
    """Internal: actual transcription logic."""
    global active_model
    start = time.monotonic()

    # Read audio bytes
    content = await audio.read()
    logger.info(f"Received audio: filename={audio.filename}, content_type={audio.content_type}, size={len(content)} bytes")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Decode audio (handles webm/opus via ffmpeg fallback)
    try:
        data, sample_rate = _decode_audio_bytes(content, audio.filename or "")
    except Exception as e:
        logger.error(f"Audio decode failed: {e}")
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {e}")

    logger.info(f"Audio decoded: {len(data)} samples, {sample_rate}Hz, duration={len(data)/sample_rate:.2f}s")

    # Transcribe (with runtime fallback + auto-switch)
    used_model = active_model.value
    async with _lock:
        asr = get_active_asr()
        if not asr.available or asr.model is None:
            logger.error("No STT model loaded!")
            raise HTTPException(status_code=503, detail="No STT model loaded")

        logger.info(f"Transcribing with {active_model.value}...")
        text = None
        try:
            text = await asyncio.to_thread(asr.transcribe, data, sample_rate)
        except Exception as e:
            logger.error(f"Transcription with {active_model.value} failed: {e}")
            # Mark the failed model as runtime-broken
            failed_asr = get_active_asr()
            failed_asr.runtime_broken = True
            logger.warning(f"Marked {active_model.value} as runtime-broken")

            # Runtime fallback: try the other model
            fallback = whisper if active_model == ModelName.PARAKEET else parakeet
            fallback_name = "whisper" if active_model == ModelName.PARAKEET else "parakeet"

            # Reload fallback if it was unloaded
            if fallback.model is None and fallback.available is False and not fallback.runtime_broken:
                logger.info(f"Reloading {fallback_name} for fallback...")
                fallback.load()

            if fallback.available and fallback.model is not None and not fallback.runtime_broken:
                logger.info(f"Falling back to {fallback_name}...")
                try:
                    text = await asyncio.to_thread(fallback.transcribe, data, sample_rate)
                    used_model = fallback_name
                    # Auto-switch: make the fallback the active model
                    logger.warning(f"Auto-switching active model from {active_model.value} to {fallback_name} (primary failed at runtime)")
                    active_model = ModelName(fallback_name)
                except Exception as e2:
                    logger.error(f"Fallback {fallback_name} also failed: {e2}")
                    fallback.runtime_broken = True
                    raise HTTPException(status_code=500, detail=f"Both models failed: {e} / {e2}")
            else:
                raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    if text is None:
        text = ""

    elapsed = (time.monotonic() - start) * 1000
    logger.info(f"Transcribed ({used_model}): '{text[:120]}' in {elapsed:.0f}ms")

    return TranscribeResponse(text=text.strip(), model=used_model, duration_ms=round(elapsed, 1))


@app.get("/v1/stt/health", response_model=HealthResponse)
async def health():
    """Health check with active model info."""
    models = [
        ModelInfo(
            name="parakeet",
            loaded=parakeet.model is not None,
            available=parakeet.available and not parakeet.runtime_broken,
            description="NVIDIA Parakeet TDT 0.6B — best accuracy for industrial terms",
            runtime_broken=parakeet.runtime_broken,
        ),
        ModelInfo(
            name="whisper",
            loaded=whisper.model is not None,
            available=whisper.available and not whisper.runtime_broken,
            description="Faster-Whisper large-v3 — general-purpose fallback",
            runtime_broken=whisper.runtime_broken,
        ),
    ]
    return HealthResponse(
        status="ok" if get_active_asr().available else "degraded",
        active_model=active_model.value,
        models=models,
    )


@app.post("/v1/stt/switch")
async def switch_model(req: SwitchRequest):
    """Switch the active STT model (hot-swap)."""
    global active_model

    if req.model == active_model:
        return {"status": "already_active", "model": active_model.value}

    async with _lock:
        target = parakeet if req.model == ModelName.PARAKEET else whisper

        # Reject switching to a model that's known to be broken at runtime
        if target.runtime_broken:
            raise HTTPException(
                status_code=503,
                detail=f"{req.model.value} failed at runtime and cannot be used. "
                       f"This model loads but crashes during transcription.",
            )

        if not target.available and target.model is None:
            logger.info(f"Loading {req.model.value}...")
            target.load()

        if not target.available:
            raise HTTPException(
                status_code=503,
                detail=f"{req.model.value} is not available on this system",
            )

        # Unload the current model to free VRAM
        old = parakeet if active_model == ModelName.PARAKEET else whisper
        old.unload()
        active_model = req.model

    logger.info(f"Switched to {active_model.value}")
    return {"status": "switched", "model": active_model.value}


@app.get("/v1/stt/models")
async def list_models():
    """List available STT models."""
    return {
        "active": active_model.value,
        "models": [
            {
                "name": "parakeet",
                "loaded": parakeet.model is not None,
                "available": parakeet.available,
                "vram_gb": 5,
                "description": "NVIDIA Parakeet TDT 0.6B",
            },
            {
                "name": "whisper",
                "loaded": whisper.model is not None,
                "available": whisper.available,
                "vram_gb": 3,
                "description": "Faster-Whisper large-v3",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8890, log_level="info")
