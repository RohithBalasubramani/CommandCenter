#!/usr/bin/env python3
"""
Merge LoRA adapters with base model and export to GGUF.

Pipeline:
1. Load base LLaMA 3.1 8B model
2. Load LoRA adapters from SFT checkpoint
3. Merge adapters into base model
4. Save merged model
5. Export to GGUF using llama.cpp
6. Quantize to q4_k_m
"""

import os
import sys
import subprocess
from pathlib import Path
from unsloth import FastLanguageModel
import torch

# Configuration
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
LORA_CHECKPOINT = "data/models/sft_checkpoints/claude-bc-20260208_055059/final"
MERGED_OUTPUT = "data/models/merged_sft"
EXPORT_DIR = "data/exports"
LLAMA_CPP_PATH = Path.home() / "llama.cpp"

def print_header(title: str):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def check_prerequisites():
    """Check that all required paths exist."""
    print_header("Checking Prerequisites")

    if not Path(LORA_CHECKPOINT).exists():
        print(f"❌ LoRA checkpoint not found: {LORA_CHECKPOINT}")
        sys.exit(1)
    print(f"✅ Found LoRA checkpoint: {LORA_CHECKPOINT}")

    if not LLAMA_CPP_PATH.exists():
        print(f"❌ llama.cpp not found: {LLAMA_CPP_PATH}")
        sys.exit(1)
    print(f"✅ Found llama.cpp: {LLAMA_CPP_PATH}")

    # Check for adapter_model.safetensors
    adapter_path = Path(LORA_CHECKPOINT) / "adapter_model.safetensors"
    if not adapter_path.exists():
        print(f"❌ Adapter file not found: {adapter_path}")
        sys.exit(1)
    print(f"✅ Found adapter file: {adapter_path}")
    print(f"   Size: {adapter_path.stat().st_size / (1024*1024):.1f} MB")

def merge_lora_with_base():
    """Merge LoRA adapters with base model."""
    print_header("Step 1/4: Merging LoRA with Base Model")

    print(f"Loading base model: {BASE_MODEL}")
    print("(This may take a few minutes on first run...)")

    # Load base model WITHOUT quantization for proper merging
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=2048,
        dtype=torch.bfloat16,  # Use bfloat16 for merging
        load_in_4bit=False,  # NO quantization for merging!
    )
    print("✅ Base model loaded")

    print(f"\nLoading LoRA adapters from: {LORA_CHECKPOINT}")
    # Apply LoRA configuration
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Same rank used in training
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Load the trained adapter weights
    from peft import PeftModel, set_peft_model_state_dict
    import safetensors.torch as safetensors

    # Load adapter state dict
    adapter_path = Path(LORA_CHECKPOINT) / "adapter_model.safetensors"
    adapter_weights = safetensors.load_file(str(adapter_path))

    # Set the adapter weights
    set_peft_model_state_dict(model, adapter_weights)
    print("✅ LoRA adapters loaded")

    print("\nMerging adapters into base model...")
    # Merge and unload adapters
    model = model.merge_and_unload()
    print("✅ Merge complete")

    # Save merged model with full config
    print(f"\nSaving merged model to: {MERGED_OUTPUT}")
    os.makedirs(MERGED_OUTPUT, exist_ok=True)

    # Save model with config
    model.save_pretrained(
        MERGED_OUTPUT,
        safe_serialization=True,
        max_shard_size="5GB"
    )

    # Save tokenizer
    tokenizer.save_pretrained(MERGED_OUTPUT)

    # Manually copy config.json from base model if needed
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(BASE_MODEL)
    config.save_pretrained(MERGED_OUTPUT)

    print("✅ Merged model saved")

    # Verify config.json exists
    config_path = Path(MERGED_OUTPUT) / "config.json"
    if config_path.exists():
        print(f"✅ config.json created: {config_path}")
        print(f"   Size: {config_path.stat().st_size} bytes")
    else:
        print(f"❌ config.json not found after merge!")
        sys.exit(1)

    return model, tokenizer

def convert_to_gguf():
    """Convert merged model to GGUF f16."""
    print_header("Step 2/4: Converting to GGUF f16")

    os.makedirs(EXPORT_DIR, exist_ok=True)
    output_file = f"{EXPORT_DIR}/cc-claude-agent-f16.gguf"

    cmd = [
        "python",
        str(LLAMA_CPP_PATH / "convert_hf_to_gguf.py"),
        MERGED_OUTPUT,
        "--outfile", output_file,
        "--outtype", "f16"
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print("❌ GGUF conversion failed!")
        sys.exit(1)

    print(f"✅ F16 GGUF created: {output_file}")

    # Show file size
    f16_path = Path(output_file)
    if f16_path.exists():
        size_gb = f16_path.stat().st_size / (1024**3)
        print(f"   Size: {size_gb:.2f} GB")

    return output_file

def quantize_to_q4km(f16_file: str):
    """Quantize f16 GGUF to q4_k_m."""
    print_header("Step 3/4: Quantizing to q4_k_m")

    output_file = f"{EXPORT_DIR}/cc-claude-agent.gguf"
    quantize_bin = LLAMA_CPP_PATH / "build" / "bin" / "llama-quantize"

    if not quantize_bin.exists():
        print(f"❌ Quantize binary not found: {quantize_bin}")
        sys.exit(1)

    cmd = [
        str(quantize_bin),
        f16_file,
        output_file,
        "q4_k_m"
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print("❌ Quantization failed!")
        sys.exit(1)

    print(f"✅ Quantized GGUF created: {output_file}")

    # Show file size
    q4_path = Path(output_file)
    if q4_path.exists():
        size_gb = q4_path.stat().st_size / (1024**3)
        print(f"   Size: {size_gb:.2f} GB")

    # Cleanup f16 file
    print(f"\nRemoving intermediate f16 file: {f16_file}")
    Path(f16_file).unlink()
    print("✅ Cleanup complete")

    return output_file

def main():
    print_header("🚀 LoRA Merge & GGUF Export Pipeline")

    try:
        # Step 0: Check prerequisites
        check_prerequisites()

        # Step 1: Merge LoRA with base model
        model, tokenizer = merge_lora_with_base()

        # Clear GPU memory
        del model
        del tokenizer
        torch.cuda.empty_cache()

        # Step 2: Convert to GGUF f16
        f16_file = convert_to_gguf()

        # Step 3: Quantize to q4_k_m
        final_file = quantize_to_q4km(f16_file)

        # Step 4: Summary
        print_header("✅ Export Complete!")
        print(f"Final model: {final_file}")

        final_path = Path(final_file)
        if final_path.exists():
            size_gb = final_path.stat().st_size / (1024**3)
            print(f"Size: {size_gb:.2f} GB")

        print("\nNext steps:")
        print("  1. Deploy to Ollama: cd /home/rohith/desktop/CommandCenter/claude-rl-agent && ./auto_deploy.sh")
        print("  2. Test model: ollama run cc-claude-agent 'Show me all pressure sensors'")
        print("  3. Run comparisons: ./run_comparisons.sh 10")
        print("  4. Monitor progress: ./dashboard.sh")

    except KeyboardInterrupt:
        print("\n\n❌ Export interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Export failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
