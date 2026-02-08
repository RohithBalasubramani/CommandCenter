# Code Changes Required for Hot-Swap Model Support

## Executive Summary

For seamless hot-swapping between Llama and GLM, the Command Center codebase needs to be **model-agnostic**. Currently, it's optimized for Llama. Here are the required changes:

---

## 🔴 Critical Changes (Breaks GLM without these)

### 1. Remove `"format": "json"` from Ollama API calls
**Affected files:**
- `backend/layer2/rag_pipeline.py` line 612
- `backend/layer2/parallel_llm.py` line 211

**Issue:** GLM-4.7-Flash returns empty responses when `"format": "json"` is specified

**Fix:** Make it conditional based on model:
```python
# Only use format=json for Llama models
if "llama" in model.lower():
    payload["format"] = "json"
```

---

### 2. Increase `num_predict` token limit
**Affected files:**
- `backend/layer2/rag_pipeline.py` (multiple locations)
- `backend/layer2/parallel_llm.py` line 203
- `backend/layer2/widget_selector.py` line 484

**Issue:** GLM uses ~512 tokens for internal thinking, needs 2048+ total to avoid truncation

**Fix:** Use model-specific defaults:
```python
# Model-specific token limits
TOKEN_LIMITS = {
    "llama": 1024,
    "glm": 2048,
    "default": 1024
}

model_name = os.getenv("OLLAMA_MODEL_FAST", "llama3.1:8b")
num_predict = TOKEN_LIMITS.get(
    next((k for k in TOKEN_LIMITS if k in model_name.lower()), "default"),
    TOKEN_LIMITS["default"]
)
```

---

### 3. Handle markdown-wrapped responses
**Affected files:**
- `backend/layer2/rag_pipeline.py` line 626-627
- `backend/layer2/parallel_llm.py` (response parsing)

**Issue:** GLM wraps JSON in markdown code blocks: ` ```json\n{...}\n``` `

**Fix:** Add markdown stripping before JSON parsing:
```python
raw = response.json().get("response", "")

# Strip markdown code blocks if present
if "```" in raw:
    import re
    match = re.search(r'```(?:json)?\s*\n(.*?)\n```', raw, re.DOTALL)
    if match:
        raw = match.group(1).strip()
    else:
        raw = raw.replace("```json", "").replace("```", "").strip()

parsed = json.loads(raw)
```

---

### 4. Handle GLM's `thinking` field
**Affected files:**
- `backend/layer2/rag_pipeline.py` (response extraction)
- `backend/layer2/parallel_llm.py` (response extraction)

**Issue:** GLM returns `{"thinking": "...", "response": "..."}` - response may be empty if thinking uses all tokens

**Fix:** Check both fields:
```python
result = response.json()
raw = result.get("response", "").strip()

# If response is empty but thinking exists, GLM hit token limit
if not raw and "thinking" in result:
    logger.warning(f"GLM thinking consumed all tokens, increase num_predict")
    # Optionally: extract answer from thinking field
```

---

### 5. Update GGUF export stop tokens
**Affected file:**
- `backend/rl/export.py` lines 258-259

**Issue:** Hardcoded Llama stop tokens in Modelfile generation

**Fix:** Make stop tokens model-specific:
```python
# Detect model type from checkpoint
if "glm" in base_model_name.lower():
    stop_tokens = [
        'PARAMETER stop "<|endoftext|>"',
        'PARAMETER stop "<|user|>"',
        'PARAMETER stop "<|observation|>"'
    ]
else:  # Llama
    stop_tokens = [
        'PARAMETER stop "<|eot_id|>"',
        'PARAMETER stop "<|end_of_text|>"'
    ]

modelfile_content = f"""
FROM {gguf_path}
PARAMETER temperature {temperature}
{chr(10).join(stop_tokens)}
...
"""
```

---

## 🟡 Important Changes (Affects quality)

### 6. Model-specific prompt prefixes
**Affected files:**
- `backend/layer2/rag_pipeline.py` (multiple locations)

**Issue:** GLM has `/nothink` control token to disable verbose reasoning, but it causes empty responses with certain prompt structures

**Recommendation:** Don't use `/nothink` - instead increase `num_predict` and strip thinking from response

---

### 7. Timeout adjustments
**Affected files:**
- `backend/layer2/rag_pipeline.py` line 624 (timeout=90)
- `backend/layer2/parallel_llm.py` (request timeouts)

**Issue:** GLM takes 5-8 seconds per query (vs Llama's sub-second)

**Fix:** Model-specific timeouts:
```python
TIMEOUTS = {
    "llama": 30,
    "glm": 120,
    "default": 60
}
```

---

### 8. Cache key differentiation
**Affected files:**
- `backend/layer2/rag_pipeline.py` (cache logic)

**Issue:** Same prompt might get different results from different models - cache needs model awareness

**Fix:** Include model name in cache key:
```python
cache_key = f"{model_name}:{prompt_hash}"
```

---

## 🟢 Optional Enhancements

### 9. Dynamic model selection based on query complexity
**New file:** `backend/layer2/model_selector.py`

**Feature:** Route simple queries to Llama (fast), complex queries to GLM (accurate)

```python
def select_model(query: str, intent: dict) -> str:
    """Select best model based on query complexity."""
    complexity_score = calculate_complexity(query, intent)

    if complexity_score > 0.7:
        # Complex query: multi-equipment, time ranges, comparisons
        return os.getenv("OLLAMA_MODEL_QUALITY")  # GLM
    else:
        # Simple query: single equipment, single metric
        return os.getenv("OLLAMA_MODEL_FAST")  # Llama or GLM
```

---

### 10. Fallback chain
**Enhancement:** If GLM fails or times out, automatically retry with Llama

```python
try:
    response = call_model(primary_model, prompt)
except TimeoutError:
    logger.warning(f"{primary_model} timed out, falling back to Llama")
    response = call_model("llama3.1:8b", prompt)
```

---

## 📊 Summary: Change Impact

| File | Lines Changed | Risk | Effort |
|------|---------------|------|--------|
| `backend/layer2/rag_pipeline.py` | ~50 | High | 2-3 hours |
| `backend/layer2/parallel_llm.py` | ~30 | Medium | 1-2 hours |
| `backend/rl/export.py` | ~20 | Low | 30 min |
| `backend/layer2/widget_selector.py` | ~10 | Low | 15 min |
| **Total** | **~110 lines** | **Medium** | **4-6 hours** |

---

## ✅ Testing Checklist

After making changes, test both models:

```bash
# Switch to GLM
./scripts/swap-model.sh glm

# Test widget selection
curl -X POST http://127.0.0.1:8100/api/layer2/orchestrate/ \
  -H "Content-Type: application/json" \
  -d '{"transcript":"show pump 1 vibration"}'

# Test voice response
curl -X POST http://127.0.0.1:8100/api/layer2/orchestrate/ \
  -H "Content-Type: application/json" \
  -d '{"transcript":"explain the vibration trend"}'

# Switch to Llama
./scripts/swap-model.sh llama

# Repeat tests
# ...

# Run E2E tests
cd frontend && npm run test:e2e
```

---

## 🎯 Recommended Implementation Order

1. **Phase 1 (Critical - must do):** Changes 1-5 (format, tokens, markdown, thinking, stop tokens)
2. **Phase 2 (Important):** Changes 6-8 (timeouts, cache)
3. **Phase 3 (Optional):** Changes 9-10 (dynamic selection, fallback)

**Estimated total effort:** 1 day for Phase 1, 0.5 days for Phase 2

---

## 🚨 Without These Changes

If you hot-swap without code changes:
- ❌ GLM will return empty responses (format issue)
- ❌ GLM will truncate responses (token limit)
- ❌ JSON parsing will fail (markdown wrapping)
- ❌ Exported LoRA models will have wrong stop tokens
- ⚠️ Responses will be slow but you won't notice why

**Bottom line:** Hot-swap script gets you 20% of the way there. Code changes are required for the other 80%.
