# Reinforcement Learning Training Guide

This guide explains how to train and deploy custom models for Command Center's widget and fixture selection using Direct Preference Optimization (DPO).

## Overview

Command Center uses DPO (Direct Preference Optimization) to learn from user feedback and improve widget selection over time. DPO is a simpler, more stable alternative to traditional RLHF that doesn't require a separate reward model.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Command Center RL Pipeline                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   Feedback   │───▶│  Training    │───▶│  Fine-tuned  │              │
│  │  Collection  │    │   Pipeline   │    │    Model     │              │
│  │  (ratings)   │    │  (DPO/TRL)   │    │   (LoRA)     │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  Widget      │    │  GGUF        │    │  Ollama      │              │
│  │  Ratings     │    │  Export      │    │  Deployment  │              │
│  │  (up/down)   │    │  (llama.cpp) │    │  (swap model)│              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### Hardware Requirements

- **Training**: 16GB+ VRAM GPU (RTX 3090/4090, A10, etc.)
- **Inference**: 8GB VRAM (same as base Ollama model)
- **Storage**: ~10GB for checkpoints and GGUF files

### Software Requirements

Install the RL dependencies:

```bash
cd backend
pip install -r requirements-rl.txt
```

For GGUF conversion, install llama.cpp:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make
```

## Collecting Training Data

### Method 1: Exhaustive Simulation UI

1. Run the exhaustive simulation to generate all widget variations
2. Rate widgets using the UI (thumbs up/down)
3. Ratings are automatically saved to the database

```bash
# Start the simulation
cd scripts/simulation
python run_exhaustive.py

# Open the rating UI
open http://localhost:3000/widgets
```

### Method 2: Production Feedback

Enable online learning to collect feedback from production:

```bash
# In .env
ENABLE_ONLINE_LEARNING=true
```

Ratings from the production app are automatically added to the training buffer.

## Training a Model

### Basic Training

```bash
cd backend
python manage.py train_dpo
```

### Advanced Options

```bash
# Use a specific configuration
python manage.py train_dpo --config small_gpu

# Customize training parameters
python manage.py train_dpo --epochs 5 --batch-size 2 --learning-rate 1e-5

# Train from JSON files
python manage.py train_dpo \
  --source file \
  --data-file ../scripts/simulation/results/exhaustive_data.json \
  --ratings-file ../scripts/simulation/results/ratings.json

# Export to GGUF after training
python manage.py train_dpo --export --register-ollama

# Dry run (build dataset without training)
python manage.py train_dpo --dry-run
```

### Configuration Presets

| Preset | GPU VRAM | Description |
|--------|----------|-------------|
| `default` | 16GB+ | Balanced performance and quality |
| `small_gpu` | 8-12GB | Reduced batch size, 3B model option |
| `high_quality` | 24GB+ | More epochs, larger LoRA rank |

### Hyperparameters

Key hyperparameters in `backend/rl/config.py`:

```python
DPO_CONFIG = {
    "lora_r": 16,           # LoRA rank (higher = more capacity)
    "lora_alpha": 32,       # LoRA scaling factor
    "learning_rate": 5e-5,  # Learning rate
    "beta": 0.1,            # DPO temperature (lower = stronger preferences)
    "num_epochs": 3,        # Training epochs
    "batch_size": 4,        # Batch size per GPU
}
```

## Exporting to Ollama

### Manual Export

```bash
# Export checkpoint to GGUF
python manage.py export_model \
  --checkpoint ./rl_checkpoints/final \
  --model-name cc-widget-selector \
  --quantization q4_k_m \
  --register
```

### Quantization Options

| Level | Size | Quality | Use Case |
|-------|------|---------|----------|
| `q3_k_m` | ~3GB | Good | Memory constrained |
| `q4_k_m` | ~4GB | Better | Recommended |
| `q5_k_m` | ~5GB | High | Quality priority |
| `q8_0` | ~8GB | Excellent | Best quality |

### Using the Fine-tuned Model

After export, update your `.env`:

```bash
OLLAMA_MODEL_FAST=cc-widget-selector
```

Then restart the backend.

## Online Learning

Online learning automatically collects feedback and triggers retraining when enough samples are collected.

### Configuration

```python
# In backend/rl/config.py
ONLINE_LEARNING_CONFIG = {
    "min_samples_to_retrain": 100,  # Minimum samples before retraining
    "max_buffer_size": 10000,       # Maximum samples in memory
    "retrain_interval_hours": 24,   # Maximum time between retraining
    "auto_export": True,            # Auto-export to GGUF
    "auto_deploy": False,           # Auto-swap Ollama model (dangerous!)
}
```

### Enable Online Learning

```bash
# In .env
ENABLE_ONLINE_LEARNING=true
```

### Check Status

```python
from rl.online_learner import get_online_learner

learner = get_online_learner()
print(learner.get_status())
# {'buffer_size': 45, 'min_samples': 100, 'should_retrain': False, ...}
```

## Evaluating Models

### Run Evaluation

```bash
python manage.py evaluate_model \
  --checkpoint ./rl_checkpoints/final \
  --compare-base \
  --output results.json
```

### Metrics

- **Accuracy**: % of times the model prefers the human-chosen response
- **Comparison**: Improvement over base model

### A/B Testing

For production A/B testing:

1. Deploy both models to Ollama with different names
2. Configure routing in `rag_pipeline.py`
3. Track metrics per model

## Troubleshooting

### CUDA Out of Memory

Try the `small_gpu` configuration:

```bash
python manage.py train_dpo --config small_gpu
```

Or reduce batch size:

```bash
python manage.py train_dpo --batch-size 1
```

### Training Loss Not Decreasing

- Check that you have enough positive/negative pairs
- Try increasing `beta` (e.g., 0.2)
- Ensure data quality (remove low-confidence ratings)

### GGUF Conversion Fails

Ensure llama.cpp is properly installed:

```bash
cd llama.cpp
make clean && make
```

### Model Quality Degradation

Always keep the baseline model and A/B test:

```bash
# Evaluate before deploying
python manage.py evaluate_model --checkpoint ./rl_checkpoints/final --compare-base
```

## File Structure

```
backend/rl/
├── __init__.py
├── config.py              # Training hyperparameters
├── data_formatter.py      # Convert feedback to DPO format
├── dataset_builder.py     # Build HuggingFace datasets
├── trainer.py             # DPOTrainer wrapper
├── export.py              # GGUF conversion
├── online_learner.py      # Continuous learning
├── tests.py               # Unit tests
└── management/commands/
    ├── train_dpo.py       # Training command
    ├── export_model.py    # Export command
    └── evaluate_model.py  # Evaluation command
```

## API Reference

### Data Formatter

```python
from rl.data_formatter import build_widget_dpo_pairs, build_fixture_dpo_pairs

# Build DPO pairs from rated entries
widget_pairs = build_widget_dpo_pairs(entries, all_scenarios)
fixture_pairs = build_fixture_dpo_pairs(entries, fixture_descriptions)
```

### Dataset Builder

```python
from rl.dataset_builder import prepare_training_dataset

# Build dataset from database
dataset = prepare_training_dataset(source="db", min_samples=50)

# Build dataset from files
dataset = prepare_training_dataset(
    source="file",
    data_path="path/to/exhaustive_data.json",
    ratings_path="path/to/ratings.json",
)
```

### Trainer

```python
from rl.trainer import CommandCenterDPOTrainer

trainer = CommandCenterDPOTrainer(config_name="default")
trainer.load_base_model()
result = trainer.train(train_dataset, eval_dataset)
```

### Export

```python
from rl.export import export_to_ollama

result = export_to_ollama(
    checkpoint_path="./rl_checkpoints/final",
    model_name="cc-widget-selector",
    quantization="q4_k_m",
    register=True,
)
```

### Online Learner

```python
from rl.online_learner import get_online_learner, init_online_learner

# Initialize
learner = init_online_learner(min_samples=100)

# Add feedback
learner.add_feedback({
    "entry_id": "abc123",
    "rating": "up",
    "tags": ["good"],
    "notes": "Perfect widget choice",
})

# Check status
status = learner.get_status()

# Trigger retraining manually
if learner.should_retrain():
    learner.trigger_retrain()
```
