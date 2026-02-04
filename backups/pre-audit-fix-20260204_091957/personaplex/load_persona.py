#!/usr/bin/env python3
"""
Persona Loader — Loads persona and context JSON files and generates system prompt.
"""

import json
import os
from pathlib import Path

def load_json(file_path: str) -> dict:
    """Load a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_system_prompt(persona_path: str, context_path: str) -> str:
    """Generate a system prompt from persona and context JSON files."""
    # Load persona and context
    persona = load_json(persona_path)
    context = load_json(context_path)

    # Build traits string
    traits = "; ".join(persona["personality"]["traits"])

    # Build context summary
    facility_name = context["facility"]["name"]
    facility_type = context["facility"]["type"]

    lines_status = []
    for line in context["operations"]["production_lines"]:
        status = line["status"]
        name = line["name"]
        lines_status.append(f"{name}: {status}")

    ops_summary = f"Production Lines: {', '.join(lines_status)}"

    # Equipment summary
    equipment_status = []
    for equip in context["equipment"]["critical_assets"]:
        equipment_status.append(f"{equip['name']}: {equip['status']}")

    equipment_summary = f"Equipment: {', '.join(equipment_status)}"

    # KPIs
    oee = context["performance_kpis"]["oee"]
    kpi_summary = f"Current OEE: {oee['overall']}% (target: {oee['target']}%)"

    # Build full context
    industrial_context = f"""
Facility: {facility_name} ({facility_type})
{ops_summary}
{equipment_summary}
{kpi_summary}

Recent Issues:
""" + "\n".join([f"- {issue['issue']}" for issue in context.get("supply_chain", {}).get("supplier_issues", [])])

    industrial_context += "\n\nUpcoming Events:\n" + "\n".join([
        f"- {event['event']} ({event['date']})" for event in context.get("upcoming_events", [])
    ])

    # Format the prompt
    prompt = f"""You are {persona['name']}, a {persona['role']}.

Personality: {traits}
Tone: {persona['personality']['tone']}
Engagement: {persona['personality']['engagement_style']}

Conversation Guidelines:
- {persona['conversation_style']['opening_behavior']}
- {persona['conversation_style']['contextual_awareness']}
- {persona['conversation_style']['suggestion_pattern']}
- {persona['conversation_style']['clarification_approach']}
- {persona['conversation_style']['information_density']}

Current Operational Context:
{industrial_context}

Remember: Be proactive, ask about their industry and needs, and reference the operational context in your responses. Keep responses concise and actionable."""

    return prompt

def main():
    """Main function to generate and print the system prompt."""
    # Get paths relative to script location
    script_dir = Path(__file__).parent
    persona_path = script_dir / "personas" / "industrial_assistant.json"
    context_path = script_dir / "context" / "dummy_industrial_context.json"

    # Generate prompt
    prompt = generate_system_prompt(str(persona_path), str(context_path))

    # Print the prompt
    print(prompt)

if __name__ == "__main__":
    main()
