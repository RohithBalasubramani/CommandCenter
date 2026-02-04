"""
LLM Prompts for Resolution.

Contains exact prompt templates used by the resolver.
All prompts are:
- Schema-driven (provide explicit schema)
- Constrained (JSON-only output)
- Explicit about required fields and assumptions
"""

# ═══════════════════════════════════════════════════════════════════════════════
# JSON SCHEMA FOR LLM OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

RESOLVE_CANDIDATE_SCHEMA = """{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["value", "confidence"],
  "properties": {
    "value": {
      "description": "The resolved value in the correct type"
    },
    "unit": {
      "type": ["string", "null"],
      "description": "Unit of measurement if applicable (e.g., 'kW', 'MWh', '°C')"
    },
    "metric_id": {
      "type": ["string", "null"],
      "description": "Metric identifier if known with HIGH confidence. Leave null if uncertain."
    },
    "frame": {
      "type": ["string", "null"],
      "enum": ["instant", "hourly", "daily", "weekly", "monthly", "yearly", null],
      "description": "Time frame/aggregation. Leave null if unknown."
    },
    "assumptions": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["field", "assumed_value", "basis"],
        "properties": {
          "field": {"type": "string"},
          "assumed_value": {},
          "basis": {"type": "string"},
          "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        }
      },
      "description": "List of assumptions made. MUST include ALL inferred information."
    },
    "confidence": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
      "description": "Confidence in the resolution. Must be < 0.9 for semantic claims."
    },
    "reasoning": {
      "type": "string",
      "description": "Brief explanation of the resolution logic."
    }
  }
}"""


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

BASIC_RESOLUTION_PROMPT = """You are a data normalization assistant. Your task is to convert a value to the expected format.

FIELD: {field_path}
CURRENT VALUE: {current_value}
EXPECTED TYPE: {expected_type}
WIDGET SCENARIO: {scenario}

RULES (MUST FOLLOW):
1. You MUST return valid JSON matching the schema below
2. You MUST NOT invent or guess metric identity (metric_id)
3. You MUST include any assumptions in the assumptions array
4. You MUST set confidence < 0.9 if making any semantic inference
5. You MUST preserve all information - no lossy transformations

REQUIRED OUTPUT SCHEMA:
{schema}

EXAMPLES:
- Input: "500 kW" → Output: {{"value": 500, "unit": "kW", "confidence": 0.95, "assumptions": [], "reasoning": "Extracted numeric value and unit"}}
- Input: "42%" → Output: {{"value": 42, "unit": "%", "confidence": 0.95, "assumptions": [], "reasoning": "Extracted percentage"}}
- Input: "monthly average" → Output: {{"value": null, "frame": "monthly", "confidence": 0.7, "assumptions": [{{"field": "aggregation", "assumed_value": "average", "basis": "text says 'average'"}}], "reasoning": "Incomplete - value missing"}}

OUTPUT JSON ONLY (no markdown, no explanation outside JSON):"""


DETAILED_RESOLUTION_PROMPT = """You are a strict data normalization assistant. A previous attempt failed. Be more precise.

FIELD: {field_path}
CURRENT VALUE: {current_value}
EXPECTED TYPE: {expected_type}
WIDGET SCENARIO: {scenario}

PREVIOUS ERROR: {previous_error}

STRICT RULES:
1. Return ONLY valid JSON - no markdown code blocks
2. The "value" field MUST match the expected type ({expected_type})
3. DO NOT set metric_id unless you are 100% certain from explicit labels
4. ALL assumptions MUST be listed in the assumptions array
5. Confidence MUST reflect actual certainty:
   - 0.9-1.0: Only for pure format conversion (string "42" → number 42)
   - 0.7-0.9: For unit extraction where unit is explicit
   - 0.5-0.7: For any inferred information
   - Below 0.5: Return null and explain in reasoning

OUTPUT SCHEMA:
{schema}

CORRECT EXAMPLES:
1. String to number:
   Input: "42"
   Output: {{"value": 42, "unit": null, "metric_id": null, "frame": null, "assumptions": [], "confidence": 0.98, "reasoning": "Direct string-to-int conversion"}}

2. Number with unit:
   Input: "500 kW"
   Output: {{"value": 500, "unit": "kW", "metric_id": null, "frame": null, "assumptions": [], "confidence": 0.95, "reasoning": "Extracted value and explicit unit"}}

3. Ambiguous case:
   Input: "Power reading"
   Output: {{"value": null, "unit": null, "metric_id": null, "frame": null, "assumptions": [{{"field": "value", "assumed_value": null, "basis": "No numeric value present"}}], "confidence": 0.3, "reasoning": "Cannot extract numeric value from text"}}

INCORRECT EXAMPLES (DO NOT DO):
- Setting metric_id to "power_consumption" without explicit evidence
- Setting confidence to 0.95 when guessing the unit
- Omitting assumptions when making inferences

OUTPUT JSON ONLY:"""


CANONICAL_RESOLUTION_PROMPT = """FINAL ATTEMPT: Strict canonical conversion only.

FIELD: {field_path}
VALUE: {current_value}
TARGET TYPE: {expected_type}

PREVIOUS ATTEMPTS FAILED. This is the last try.

ONLY ALLOWED TRANSFORMATIONS:
1. String "42" → number 42 (confidence: 0.99)
2. String "3.14" → float 3.14 (confidence: 0.99)
3. String "true"/"false" → boolean (confidence: 0.99)
4. Extract number from "X unit" pattern (confidence: 0.95)

FORBIDDEN:
- Guessing metric identity
- Inferring time frames
- Assuming units when not explicit
- ANY semantic interpretation

If you cannot perform a safe, lossless transformation, return:
{{"value": null, "confidence": 0.0, "assumptions": [], "reasoning": "Cannot safely transform"}}

OUTPUT SCHEMA:
{schema}

JSON ONLY:"""


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_resolution_prompt(
    field_path: str,
    current_value: any,
    expected_type: str,
    scenario: str,
    level: str = "basic",
    previous_error: str = None,
) -> str:
    """
    Build a resolution prompt for the LLM.

    Args:
        field_path: Path to the field being resolved
        current_value: Current value of the field
        expected_type: Expected type (string, number, boolean, etc.)
        scenario: Widget scenario name
        level: Prompt level (basic, detailed, canonical)
        previous_error: Error from previous attempt (for retry prompts)

    Returns:
        Complete prompt string
    """
    if level == "canonical":
        template = CANONICAL_RESOLUTION_PROMPT
    elif level == "detailed" or previous_error:
        template = DETAILED_RESOLUTION_PROMPT
    else:
        template = BASIC_RESOLUTION_PROMPT

    return template.format(
        field_path=field_path,
        current_value=repr(current_value),
        expected_type=expected_type,
        scenario=scenario,
        schema=RESOLVE_CANDIDATE_SCHEMA,
        previous_error=previous_error or "N/A",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ESCALATION PROMPT (for human review)
# ═══════════════════════════════════════════════════════════════════════════════

ESCALATION_SUMMARY_TEMPLATE = """
RECONCILIATION ESCALATION REQUIRED

Scenario: {scenario}
Field: {field_path}
Original Value: {original_value}

Attempted Resolutions: {attempt_count}

Last LLM Response:
{last_response}

Reason for Escalation:
{escalation_reason}

Recommended Actions:
{recommendations}

To Resolve:
1. Verify the correct metric identity
2. Provide explicit unit and time frame
3. Update upstream data source with complete metadata
"""


def build_escalation_summary(
    scenario: str,
    field_path: str,
    original_value: any,
    attempts: list,
    escalation_reason: str,
    recommendations: list[str],
) -> str:
    """Build a human-readable escalation summary."""
    last_response = ""
    if attempts:
        last_attempt = attempts[-1]
        if hasattr(last_attempt, 'llm_response_raw'):
            last_response = last_attempt.llm_response_raw[:500]

    return ESCALATION_SUMMARY_TEMPLATE.format(
        scenario=scenario,
        field_path=field_path,
        original_value=repr(original_value),
        attempt_count=len(attempts),
        last_response=last_response or "No response",
        escalation_reason=escalation_reason,
        recommendations="\n".join(f"  - {r}" for r in recommendations),
    )
