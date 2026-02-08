#!/usr/bin/env python3
"""Simple side-by-side comparison of Llama vs GLM for JSON widget selection."""

import json
import time
import requests
from datetime import datetime

OLLAMA_URL = "http://localhost:11434"

# Simplified widget selection prompt
PROMPT_TEMPLATE = """You are a dashboard widget selector for industrial operations.

Given the user query, select 1-3 appropriate widgets from: trend, kpi, distribution, comparison, alerts, equipment-panel.

User query: "{query}"

Return valid JSON in this exact format:
{{
  "widgets": [
    {{
      "type": "trend",
      "size": "medium",
      "why": "Shows time-series data for the requested metric"
    }}
  ]
}}"""

TEST_QUERIES = [
    "Show me pump 1 vibration",
    "What's the power consumption trend?",
    "Compare efficiency across all pumps",
    "Display temperature for motor 3",
    "Show equipment health status",
]

def call_model(model: str, query: str):
    """Call Ollama model and return (response_dict, time_ms, error)."""
    prompt = PROMPT_TEMPLATE.format(query=query)

    # NOTE: /nothink causes empty responses with structured prompts
    # GLM-4.7-Flash will return markdown-wrapped JSON, which we'll strip below

    start = time.time()

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 2048}  # GLM needs room for thinking+response
            },
            timeout=30
        )

        elapsed_ms = (time.time() - start) * 1000

        if resp.status_code != 200:
            return None, elapsed_ms, f"HTTP {resp.status_code}"

        text = resp.json().get("response", "").strip()

        # DEBUG: Show what we got
        if not text:
            return None, elapsed_ms, "Empty response from API"

        # Strip markdown code blocks if present
        if "```" in text:
            # Extract content between ``` markers
            import re
            match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
            if match:
                text = match.group(1).strip()
            else:
                # Fallback: remove all ``` markers
                text = text.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(text)
        return parsed, elapsed_ms, None

    except json.JSONDecodeError as e:
        elapsed_ms = (time.time() - start) * 1000
        return None, elapsed_ms, f"Invalid JSON: {str(e)[:50]}"
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        return None, elapsed_ms, str(e)[:50]

def main():
    print("\n" + "="*80)
    print("WIDGET SELECTION COMPARISON: Llama 3.1 8B vs GLM-4.7-Flash")
    print("="*80 + "\n")

    llama_wins = 0
    glm_wins = 0
    ties = 0

    for query in TEST_QUERIES:
        print(f"\nQuery: {query}")
        print("-" * 80)

        # Test Llama
        llama_resp, llama_time, llama_err = call_model("llama3.1:8b", query)
        llama_valid = llama_resp is not None
        llama_widgets = len(llama_resp.get("widgets", [])) if llama_valid else 0

        if llama_valid:
            print(f"[Llama] ✓ Valid ({llama_widgets} widgets, {llama_time:.0f}ms)")
        else:
            print(f"[Llama] ✗ {llama_err} ({llama_time:.0f}ms)")

        # Test GLM
        glm_resp, glm_time, glm_err = call_model("glm-4.7-flash", query)
        glm_valid = glm_resp is not None
        glm_widgets = len(glm_resp.get("widgets", [])) if glm_valid else 0

        if glm_valid:
            print(f"[GLM]   ✓ Valid ({glm_widgets} widgets, {glm_time:.0f}ms)")
        else:
            print(f"[GLM]   ✗ {glm_err} ({glm_time:.0f}ms)")

        # Determine winner
        if llama_valid and glm_valid:
            if glm_time < llama_time * 0.9:
                glm_wins += 1
                print(f"Winner: GLM (faster by {llama_time - glm_time:.0f}ms)")
            elif llama_time < glm_time * 0.9:
                llama_wins += 1
                print(f"Winner: Llama (faster by {glm_time - llama_time:.0f}ms)")
            else:
                ties += 1
                print("Winner: Tie (similar speed)")
        elif llama_valid:
            llama_wins += 1
            print("Winner: Llama (only valid response)")
        elif glm_valid:
            glm_wins += 1
            print("Winner: GLM (only valid response)")
        else:
            print("Winner: None (both failed)")

    print("\n" + "="*80)
    print("FINAL SCORE")
    print("="*80)
    print(f"Llama 3.1 8B:    {llama_wins} wins")
    print(f"GLM-4.7-Flash:   {glm_wins} wins")
    print(f"Ties:            {ties}")
    print("="*80)

    if glm_wins > llama_wins:
        print("\n🏆 GLM-4.7-Flash wins overall!")
    elif llama_wins > glm_wins:
        print("\n🏆 Llama 3.1 8B wins overall!")
    else:
        print("\n🤝 It's a tie!")

if __name__ == "__main__":
    main()
