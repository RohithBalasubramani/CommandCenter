"""
GGUF Export for Ollama Deployment

Converts fine-tuned models to GGUF format and registers with Ollama.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import GGUF_CONFIG, MODEL_CONFIG, CHECKPOINTS_DIR

logger = logging.getLogger(__name__)


@dataclass
class ExportResult:
    """Result from an export operation."""
    success: bool
    gguf_path: Optional[str] = None
    ollama_model_name: Optional[str] = None
    error_message: Optional[str] = None
    file_size_mb: Optional[float] = None


def merge_lora_weights(
    base_model_path: str,
    lora_path: str,
    output_path: str,
) -> str:
    """
    Merge LoRA weights back into the base model.

    This creates a full model that can be converted to GGUF.

    Args:
        base_model_path: HuggingFace model ID or path
        lora_path: Path to LoRA checkpoint
        output_path: Path to save merged model

    Returns:
        Path to merged model
    """
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("Install transformers and peft: pip install transformers peft")

    logger.info(f"Loading base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="cpu",  # Use CPU to save GPU memory
        trust_remote_code=True,
    )

    logger.info(f"Loading LoRA weights: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)

    logger.info("Merging weights...")
    merged_model = model.merge_and_unload()

    logger.info(f"Saving merged model to: {output_path}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_path)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)

    return output_path


def find_llama_cpp() -> Optional[str]:
    """Find llama.cpp installation."""
    # Check common locations
    possible_paths = [
        Path.home() / "llama.cpp",
        Path("/opt/llama.cpp"),
        Path("/usr/local/llama.cpp"),
        Path.cwd() / "llama.cpp",
    ]

    for path in possible_paths:
        convert_script = path / "convert_hf_to_gguf.py"
        if convert_script.exists():
            return str(path)

    # Check if llama-cpp-python is installed with conversion support
    try:
        import llama_cpp
        return "llama_cpp"  # Signal to use llama-cpp-python
    except ImportError:
        pass

    return None


def convert_to_gguf(
    model_path: str,
    output_path: str,
    quantization: str = None,
    llama_cpp_path: Optional[str] = None,
) -> str:
    """
    Convert HuggingFace model to GGUF format.

    Args:
        model_path: Path to HuggingFace model
        output_path: Output path for GGUF file
        quantization: Quantization level (q4_k_m, q5_k_m, q8_0, etc.)
        llama_cpp_path: Path to llama.cpp installation

    Returns:
        Path to GGUF file
    """
    if quantization is None:
        quantization = GGUF_CONFIG.get("quantization", "q4_k_m")

    if llama_cpp_path is None:
        llama_cpp_path = find_llama_cpp()

    if llama_cpp_path is None:
        raise RuntimeError(
            "llama.cpp not found. Install it with:\n"
            "  git clone https://github.com/ggerganov/llama.cpp\n"
            "  cd llama.cpp && make\n"
            "Or install llama-cpp-python: pip install llama-cpp-python"
        )

    # Ensure output path has .gguf extension
    if not output_path.endswith(".gguf"):
        output_path = f"{output_path}.gguf"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting to GGUF with quantization: {quantization}")

    if llama_cpp_path == "llama_cpp":
        # Use llama-cpp-python conversion (if available)
        raise NotImplementedError(
            "Direct conversion via llama-cpp-python not yet supported. "
            "Please install llama.cpp manually."
        )
    else:
        # Use llama.cpp conversion script
        convert_script = Path(llama_cpp_path) / "convert_hf_to_gguf.py"
        if not convert_script.exists():
            raise FileNotFoundError(f"Conversion script not found: {convert_script}")

        # First convert to f16 GGUF
        f16_path = output_path.replace(".gguf", "-f16.gguf")
        cmd = [
            "python", str(convert_script),
            model_path,
            "--outfile", f16_path,
            "--outtype", "f16",
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Conversion failed: {result.stderr}")

        # Then quantize
        if quantization != "f16":
            quantize_bin = Path(llama_cpp_path) / "llama-quantize"
            if not quantize_bin.exists():
                quantize_bin = Path(llama_cpp_path) / "quantize"

            if quantize_bin.exists():
                cmd = [str(quantize_bin), f16_path, output_path, quantization]
                logger.info(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    raise RuntimeError(f"Quantization failed: {result.stderr}")

                # Remove f16 intermediate file
                Path(f16_path).unlink(missing_ok=True)
            else:
                logger.warning("Quantize binary not found, using f16 format")
                shutil.move(f16_path, output_path)
        else:
            shutil.move(f16_path, output_path)

    logger.info(f"GGUF file created: {output_path}")
    return output_path


def create_ollama_modelfile(
    gguf_path: str,
    model_name: str = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    num_ctx: int = 4096,
) -> str:
    """
    Generate an Ollama Modelfile.

    Args:
        gguf_path: Path to GGUF file
        model_name: Name for the Ollama model
        system_prompt: Custom system prompt
        temperature: Temperature setting
        num_ctx: Context window size

    Returns:
        Modelfile content
    """
    if model_name is None:
        model_name = MODEL_CONFIG.get("output_model_name", "cc-widget-selector")

    if system_prompt is None:
        system_prompt = (
            "You are Command Center's widget selection assistant. "
            "Given a user query and available widgets, select the most appropriate "
            "widgets and fixtures to display relevant information. "
            "Be precise and concise in your selections."
        )

    modelfile = f"""FROM {gguf_path}

PARAMETER temperature {temperature}
PARAMETER num_ctx {num_ctx}
PARAMETER num_predict 512
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"

SYSTEM {system_prompt}
"""
    return modelfile


def register_with_ollama(
    modelfile_content: str,
    model_name: str,
    modelfile_path: Optional[str] = None,
) -> bool:
    """
    Register a model with Ollama.

    Args:
        modelfile_content: Content of the Modelfile
        model_name: Name for the Ollama model
        modelfile_path: Optional path to save Modelfile

    Returns:
        True if successful
    """
    # Check if Ollama is available
    result = subprocess.run(["which", "ollama"], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError("Ollama not found. Install from: https://ollama.ai")

    # Write Modelfile
    if modelfile_path is None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".modelfile", delete=False) as f:
            f.write(modelfile_content)
            modelfile_path = f.name
    else:
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)

    try:
        logger.info(f"Registering model with Ollama: {model_name}")
        cmd = ["ollama", "create", model_name, "-f", modelfile_path]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Ollama create failed: {result.stderr}")

        logger.info(f"Model registered: {model_name}")
        return True

    finally:
        # Clean up temp file
        if modelfile_path and Path(modelfile_path).exists():
            Path(modelfile_path).unlink(missing_ok=True)


def export_to_ollama(
    checkpoint_path: str,
    model_name: str = None,
    base_model: str = None,
    quantization: str = None,
    output_dir: Optional[str] = None,
    register: bool = True,
) -> ExportResult:
    """
    Full export pipeline: merge LoRA → convert GGUF → register Ollama.

    Args:
        checkpoint_path: Path to LoRA checkpoint
        model_name: Name for the Ollama model
        base_model: Base model path/ID
        quantization: GGUF quantization level
        output_dir: Output directory for files
        register: Whether to register with Ollama

    Returns:
        ExportResult with paths and status
    """
    if model_name is None:
        model_name = MODEL_CONFIG.get("output_model_name", "cc-widget-selector")

    if base_model is None:
        base_model = MODEL_CONFIG.get("base_model")

    if quantization is None:
        quantization = GGUF_CONFIG.get("quantization", "q4_k_m")

    if output_dir is None:
        output_dir = str(CHECKPOINTS_DIR / "export")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Merge LoRA weights
        logger.info("Step 1/3: Merging LoRA weights...")
        merged_path = str(output_dir / "merged_model")
        merge_lora_weights(base_model, checkpoint_path, merged_path)

        # Step 2: Convert to GGUF
        logger.info("Step 2/3: Converting to GGUF...")
        gguf_path = str(output_dir / f"{model_name}.gguf")
        gguf_path = convert_to_gguf(merged_path, gguf_path, quantization)

        # Get file size
        file_size_mb = Path(gguf_path).stat().st_size / (1024 * 1024)

        # Step 3: Register with Ollama
        if register:
            logger.info("Step 3/3: Registering with Ollama...")
            modelfile = create_ollama_modelfile(gguf_path, model_name)
            register_with_ollama(modelfile, model_name)

        # Clean up merged model to save space
        shutil.rmtree(merged_path, ignore_errors=True)

        logger.info(f"Export complete! Model: {model_name}, Size: {file_size_mb:.1f}MB")

        return ExportResult(
            success=True,
            gguf_path=gguf_path,
            ollama_model_name=model_name if register else None,
            file_size_mb=file_size_mb,
        )

    except Exception as e:
        logger.error(f"Export failed: {e}")
        return ExportResult(
            success=False,
            error_message=str(e),
        )


def list_ollama_models() -> list[str]:
    """List available Ollama models."""
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if result.returncode != 0:
        return []

    models = []
    for line in result.stdout.strip().split("\n")[1:]:  # Skip header
        if line.strip():
            models.append(line.split()[0])
    return models


def delete_ollama_model(model_name: str) -> bool:
    """Delete an Ollama model."""
    result = subprocess.run(["ollama", "rm", model_name], capture_output=True, text=True)
    return result.returncode == 0


def get_export_status() -> dict:
    """Get current export status and available models."""
    export_dir = CHECKPOINTS_DIR / "export"

    status = {
        "export_dir": str(export_dir),
        "gguf_files": [],
        "ollama_models": [],
    }

    if export_dir.exists():
        status["gguf_files"] = [
            {
                "name": f.name,
                "size_mb": f.stat().st_size / (1024 * 1024),
                "path": str(f),
            }
            for f in export_dir.glob("*.gguf")
        ]

    status["ollama_models"] = list_ollama_models()

    return status
