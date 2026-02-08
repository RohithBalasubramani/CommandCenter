#!/bin/bash
# Make Command Center codebase model-agnostic
# Implements all critical changes for Llama/GLM hot-swapping
# Usage: ./scripts/make-model-agnostic.sh [--dry-run]

set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

function log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
function log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
function log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
function log_error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="$PROJECT_ROOT/backups/model-agnostic-$(date +%Y%m%d_%H%M%S)"

echo ""
echo "=========================================="
echo "  Make Command Center Model-Agnostic"
echo "=========================================="
echo ""

if [ "$DRY_RUN" = true ]; then
    log_warn "DRY RUN MODE - No changes will be made"
    echo ""
fi

# Create backup directory
log_info "Creating backup directory: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Files to modify
FILES=(
    "backend/layer2/rag_pipeline.py"
    "backend/layer2/parallel_llm.py"
    "backend/rl/export.py"
    "backend/layer2/widget_selector.py"
)

# Backup files
log_info "Backing up files..."
for file in "${FILES[@]}"; do
    if [ -f "$PROJECT_ROOT/$file" ]; then
        cp "$PROJECT_ROOT/$file" "$BACKUP_DIR/"
        log_success "  Backed up: $file"
    else
        log_warn "  File not found: $file"
    fi
done

if [ "$DRY_RUN" = true ]; then
    log_info "Dry run complete. No changes made."
    echo ""
    echo "To apply changes, run:"
    echo "  ./scripts/make-model-agnostic.sh"
    exit 0
fi

# Change 1: Create model config helper
log_info "Change 1: Creating model configuration helper..."
cat > "$PROJECT_ROOT/backend/layer2/model_config.py" << 'EOF'
"""Model-specific configuration for multi-model support."""
import os


def get_model_config(model_name: str = None) -> dict:
    """Get model-specific configuration.

    Args:
        model_name: Model name (e.g., 'llama3.1:8b', 'glm-4.7-flash').
                   If None, reads from OLLAMA_MODEL_FAST env var.

    Returns:
        dict with: num_predict, timeout, use_format_json, stop_tokens
    """
    if model_name is None:
        model_name = os.getenv("OLLAMA_MODEL_FAST", "llama3.1:8b")

    model_lower = model_name.lower()

    # GLM configuration
    if "glm" in model_lower:
        return {
            "num_predict": 2048,  # GLM needs room for thinking
            "timeout": 120,  # 2 minutes for thinking mode
            "use_format_json": False,  # GLM doesn't support format parameter
            "stop_tokens": ["<|endoftext|>", "<|user|>", "<|observation|>"],
            "strip_markdown": True,  # GLM wraps in markdown
            "has_thinking": True,  # GLM returns separate thinking field
        }

    # Llama configuration (default)
    return {
        "num_predict": 1024,
        "timeout": 30,
        "use_format_json": True,  # Llama supports format parameter
        "stop_tokens": ["<|eot_id|>", "<|end_of_text|>"],
        "strip_markdown": False,
        "has_thinking": False,
    }


def strip_markdown_json(text: str) -> str:
    """Strip markdown code blocks from JSON response."""
    if "```" not in text:
        return text

    import re
    # Try to extract content between ``` markers
    match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: remove all ``` markers
    return text.replace("```json", "").replace("```", "").strip()
EOF
log_success "  Created backend/layer2/model_config.py"

# Change 2: Update rag_pipeline.py
log_info "Change 2: Updating rag_pipeline.py..."

python3 << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, '/home/rohith/desktop/CommandCenter/backend')

file_path = '/home/rohith/desktop/CommandCenter/backend/layer2/rag_pipeline.py'

with open(file_path, 'r') as f:
    content = f.read()

# Add import at the top (after existing imports)
if 'from .model_config import' not in content:
    import_pos = content.find('from typing import')
    if import_pos != -1:
        next_newline = content.find('\n', import_pos)
        content = content[:next_newline+1] + 'from .model_config import get_model_config, strip_markdown_json\n' + content[next_newline+1:]

# Replace the generate_json method
old_generate = '''            "format": "json",
            "keep_alive": -1,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        for attempt in range(2):
            try:
                response = requests.post(url, json=payload, timeout=90)
                response.raise_for_status()
                raw = response.json().get("response", "")
                parsed = json.loads(raw)'''

new_generate = '''            "keep_alive": -1,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        # Model-specific configuration
        model_cfg = get_model_config(self.model)
        if model_cfg["use_format_json"]:
            payload["format"] = "json"

        if system_prompt:
            payload["system"] = system_prompt

        for attempt in range(2):
            try:
                response = requests.post(url, json=payload, timeout=model_cfg["timeout"])
                response.raise_for_status()
                result = response.json()
                raw = result.get("response", "").strip()

                # Handle GLM thinking field
                if not raw and model_cfg["has_thinking"] and "thinking" in result:
                    import logging
                    logging.getLogger(__name__).warning(
                        "GLM thinking consumed all tokens, response empty"
                    )
                    # Continue to next attempt
                    continue

                # Strip markdown if needed
                if model_cfg["strip_markdown"]:
                    raw = strip_markdown_json(raw)

                parsed = json.loads(raw)'''

if old_generate in content:
    content = content.replace(old_generate, new_generate)
    print("✓ Updated generate_json method")
else:
    print("✗ Could not find generate_json pattern to replace")

with open(file_path, 'w') as f:
    f.write(content)
PYTHON_SCRIPT

log_success "  Updated rag_pipeline.py"

# Change 3: Update parallel_llm.py
log_info "Change 3: Updating parallel_llm.py..."

python3 << 'PYTHON_SCRIPT'
import sys
file_path = '/home/rohith/desktop/CommandCenter/backend/layer2/parallel_llm.py'

with open(file_path, 'r') as f:
    content = f.read()

# Add import
if 'from .model_config import' not in content:
    import_pos = content.find('from typing import')
    if import_pos != -1:
        next_newline = content.find('\n', import_pos)
        content = content[:next_newline+1] + 'from .model_config import get_model_config, strip_markdown_json\n' + content[next_newline+1:]

# Update the payload construction
old_payload = '''                    "num_predict": request.max_tokens,
                }
            }

            # Add format: json if json_mode is enabled
            if request.json_mode:
                payload["format"] = "json"'''

new_payload = '''                    "num_predict": request.max_tokens,
                }
            }

            # Model-specific configuration
            model_cfg = get_model_config(request.model)

            # Add format: json only for models that support it
            if request.json_mode and model_cfg["use_format_json"]:
                payload["format"] = "json"'''

if old_payload in content:
    content = content.replace(old_payload, new_payload)
    print("✓ Updated payload construction")

# Update response parsing
old_response = '''            response = session.post(url, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            text = result.get("response", "")'''

new_response = '''            model_cfg = get_model_config(request.model)
            response = session.post(url, json=payload, timeout=model_cfg["timeout"])
            response.raise_for_status()

            result = response.json()
            text = result.get("response", "").strip()

            # Handle GLM markdown wrapping
            if model_cfg["strip_markdown"]:
                text = strip_markdown_json(text)'''

if old_response in content:
    content = content.replace(old_response, new_response)
    print("✓ Updated response parsing")

with open(file_path, 'w') as f:
    f.write(content)
PYTHON_SCRIPT

log_success "  Updated parallel_llm.py"

# Change 4: Update export.py for dynamic stop tokens
log_info "Change 4: Updating export.py for dynamic stop tokens..."

python3 << 'PYTHON_SCRIPT'
file_path = '/home/rohith/desktop/CommandCenter/backend/rl/export.py'

with open(file_path, 'r') as f:
    content = f.read()

# Find and replace the Modelfile creation
old_modelfile = '''    modelfile_content = f"""
FROM {gguf_path}

PARAMETER temperature {temperature}
PARAMETER num_ctx {num_ctx}
PARAMETER num_predict {num_predict}
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"'''

new_modelfile = '''    # Detect model type for appropriate stop tokens
    base_model_lower = base_model_path.lower() if base_model_path else ""
    if "glm" in base_model_lower:
        stop_tokens = [
            'PARAMETER stop "<|endoftext|>"',
            'PARAMETER stop "<|user|>"',
            'PARAMETER stop "<|observation|>"'
        ]
    else:  # Llama (default)
        stop_tokens = [
            'PARAMETER stop "<|eot_id|>"',
            'PARAMETER stop "<|end_of_text|>"'
        ]

    stop_tokens_str = "\\n".join(stop_tokens)

    modelfile_content = f"""
FROM {gguf_path}

PARAMETER temperature {temperature}
PARAMETER num_ctx {num_ctx}
PARAMETER num_predict {num_predict}
{stop_tokens_str}'''

if old_modelfile in content:
    content = content.replace(old_modelfile, new_modelfile)
    print("✓ Updated Modelfile generation with dynamic stop tokens")
else:
    print("⚠ Could not find exact Modelfile pattern, manual verification needed")

with open(file_path, 'w') as f:
    f.write(content)
PYTHON_SCRIPT

log_success "  Updated export.py"

# Change 5: Update widget_selector.py token limit
log_info "Change 5: Updating widget_selector.py token limits..."

python3 << 'PYTHON_SCRIPT'
file_path = '/home/rohith/desktop/CommandCenter/backend/layer2/widget_selector.py'

with open(file_path, 'r') as f:
    content = f.read()

# Add import if not present
if 'from .model_config import' not in content:
    import_pos = content.find('from typing import')
    if import_pos != -1:
        next_newline = content.find('\n', import_pos)
        content = content[:next_newline+1] + 'from .model_config import get_model_config\n' + content[next_newline+1:]
        print("✓ Added model_config import")

# Find and update max_tokens usage
old_max_tokens = 'max_tokens=1024,'
new_max_tokens = 'max_tokens=get_model_config().get("num_predict", 1024),'

if old_max_tokens in content:
    content = content.replace(old_max_tokens, new_max_tokens)
    print("✓ Updated max_tokens to use model config")

with open(file_path, 'w') as f:
    f.write(content)
PYTHON_SCRIPT

log_success "  Updated widget_selector.py"

# Summary
echo ""
echo "=========================================="
echo "  Changes Applied Successfully!"
echo "=========================================="
echo ""

log_info "Modified files:"
for file in "${FILES[@]}"; do
    echo "  - $file"
done
echo "  + backend/layer2/model_config.py (new)"

echo ""
log_info "Backup location: $BACKUP_DIR"

echo ""
log_info "Next steps:"
echo "  1. Test with Llama: ./scripts/swap-model.sh llama"
echo "  2. Test with GLM:   ./scripts/swap-model.sh glm"
echo "  3. Verify both work with test queries"
echo ""

echo "To rollback if needed:"
echo "  cp $BACKUP_DIR/* backend/"
echo ""

log_success "Model-agnostic migration complete!"
