# Reconciliation Pipeline

Production-ready pipeline for transforming unreliable LLM outputs into validated widget data.

## Overview

```
LLM Output → CLASSIFY → REWRITE → RESOLVE → NORMALIZE → VALIDATE → Render
                │           │          │           │           │
                │           │          │           │           └── HARD GATE: Must pass
                │           │          │           └── Domain-aware (unit conversion)
                │           │          └── LLM-based (bounded attempts)
                │           └── Syntactic (lossless, reversible)
                └── Formal classification
```

## Invariants (Non-Negotiable)

1. **Final output MUST pass `validate_widget_data()` or be explicit refusal**
2. **No silent semantic guessing** - all assumptions are explicit
3. **All transforms are lossless, reversible, self-declaring**
4. **Full provenance and audit trail for all decisions**

## Mismatch Classes

| Class | Definition | Action |
|-------|------------|--------|
| `NONE` | No mismatch | Pass through |
| `STRUCTURAL_EQUIVALENCE` | Different structure, same meaning | Rewrite allowed |
| `REPRESENTATIONAL_EQUIVALENCE` | Different representation, same value | Rewrite allowed |
| `UNKNOWN_AMBIGUOUS` | Cannot determine without context | Resolver attempt |
| `SEMANTIC_DIFFERENCE` | Different meaning | Refuse/Escalate |
| `SECURITY_VIOLATION` | Injection detected | Immediate reject |

## Usage

```python
from layer2.reconciliation import ReconciliationPipeline

pipeline = ReconciliationPipeline()
result = pipeline.process(scenario="kpi", data=llm_output)

if result.success:
    validated_data = result.data
    # Use validated_data for rendering
else:
    refusal = result.refusal
    # Handle structured refusal
    print(f"Reason: {refusal.reason}")
    print(f"Recommendations: {refusal.recommendations}")
```

## Allowed Transformations

### Structural Equivalence
- `{"demoData": X}` → `X` (unwrap container)
- `[X]` → `X` (unwrap singleton)

### Representational Equivalence
- `"42"` → `42` (string to int)
- `"3.14"` → `3.14` (string to float)
- `"true"` → `true` (string to bool)
- `"null"` → `null` (string to null)
- `"  text  "` → `"text"` (whitespace normalize)
- `"2024-01-15"` → `"2024-01-15T00:00:00"` (ISO date normalize)

## Forbidden Operations

- Guessing metric identity without explicit evidence
- Assuming time frames without provenance
- Silent coercion or value clamping
- Semantic inference from labels alone
- Cross-dimension unit conversion (power → energy)

## Resolver Policy

When encountering `UNKNOWN_AMBIGUOUS` mismatches:

1. **Attempt 1** (basic): Simple schema-driven prompt
2. **Attempt 2** (detailed): Add examples and constraints
3. **Attempt 3** (canonical): Strict format only

Each attempt:
- Requires JSON-only output from LLM
- Validates response syntactically
- Checks confidence thresholds (≥0.9 for semantic claims)
- Records all assumptions explicitly

## Confidence Thresholds

| Claim Type | Required Confidence |
|------------|---------------------|
| Pure format conversion | Any (0.0-1.0) |
| Unit extraction | ≥ 0.7 |
| Semantic claim (metric_id) | ≥ 0.9 |
| Frame claim (time base) | ≥ 0.9 |

## Adding New Rewrite Rules

1. Create a class extending `RewriteRule` in `rewriter.py`
2. Implement required methods:
   - `rule_id` → Unique identifier
   - `applies_to_class` → Which mismatch class
   - `can_apply(value, field_path, mismatch)` → Applicability check
   - `apply(value)` → Forward transformation
   - `inverse(value)` → Reverse transformation
   - `description` → Human-readable description
3. Add to `REWRITE_RULES` list

Example:
```python
class MyNewRule(RewriteRule):
    @property
    def rule_id(self) -> str:
        return "my_new_rule"

    @property
    def applies_to_class(self) -> MismatchClass:
        return MismatchClass.REPRESENTATIONAL_EQUIVALENCE

    def can_apply(self, value, field_path, mismatch) -> bool:
        return isinstance(value, str) and value.startswith("prefix:")

    def apply(self, value) -> str:
        return value[7:]  # Remove "prefix:"

    def inverse(self, value) -> str:
        return f"prefix:{value}"

    @property
    def description(self) -> str:
        return "Remove 'prefix:' from string"
```

## Audit Trail

All decisions are logged with:
- Event ID (UUID)
- Timestamp (ISO 8601)
- Decision type (TRANSFORM, RESOLVE, REFUSE, ESCALATE)
- Mismatch classification
- Input/output hashes
- Full provenance chain
- Attempt count
- Error messages

Configure audit sinks:
```python
from layer2.reconciliation.audit import configure_audit

configure_audit(
    file_path="/var/log/reconciliation/audit.jsonl",
    enable_logging=True,
)
```

## CI Gating

Required tests:
```bash
# Run all reconciliation tests
pytest backend/layer2/reconciliation/tests/ -v

# Verify validation gate is not bypassed
pytest backend/layer2/reconciliation/tests/test_pipeline.py::TestValidationGate -v

# Run hostile validation tests (must remain at 64/64)
python tests/hostile_validation_test.py
```

## Escalation Playbook

### When to Escalate

1. **Semantic difference detected** - Different metric identities
2. **Low confidence on semantic claims** - LLM confidence < 0.9
3. **Conflicting time frames** - Monthly vs daily aggregations
4. **Max resolution attempts exhausted** - 3 attempts failed

### How to Present

```json
{
  "type": "RECONCILIATION_ESCALATION",
  "severity": "HIGH",
  "reason": "LLM semantic claim requires verification",
  "scenario": "kpi",
  "missing_fields": ["metric_id"],
  "recommendations": [
    "Verify the metric identity",
    "Provide explicit metric_id in source data"
  ],
  "action_required": "Manual review and data correction required"
}
```

### Resolution Steps

1. Review the original LLM output
2. Check provenance chain for transformation attempts
3. Verify metric identity with domain expert
4. Update source data with explicit metadata
5. Re-run pipeline with corrected input
