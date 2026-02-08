#!/usr/bin/env python3
"""Accuracy-focused comparison: Which model selects better widgets?"""

import json
import time
import requests
from datetime import datetime

OLLAMA_URL = "http://localhost:11434"

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
    ("Show me pump 1 vibration", ["trend"], "Single equipment, single metric - needs time-series"),
    ("Compare efficiency across all pumps", ["comparison", "trend-multi-line"], "Multi-equipment comparison"),
    ("Show equipment health status", ["equipment-panel", "alerts"], "Overall status monitoring"),
    ("What was the peak load yesterday?", ["kpi", "trend"], "Historical peak value query"),
    ("Display vibration for pumps 1, 2, and 3", ["comparison", "trend-multi-line"], "Multi-equipment, same metric"),
    ("Show power consumption trends for the last week", ["trend", "trends-cumulative"], "Time-series with accumulation"),
    ("Alert me if any equipment is failing", ["alerts"], "Alert-focused query"),
    ("Dashboard overview of facility operations", ["kpi", "alerts", "equipment-panel"], "High-level overview"),
]

def call_model(model: str, query: str):
    """Call model and return full response."""
    prompt = PROMPT_TEMPLATE.format(query=query)

    start = time.time()

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 2048}
            },
            timeout=60
        )

        elapsed_ms = (time.time() - start) * 1000

        if resp.status_code != 200:
            return None, elapsed_ms, f"HTTP {resp.status_code}"

        text = resp.json().get("response", "").strip()

        if not text:
            return None, elapsed_ms, "Empty response"

        # Strip markdown
        if "```" in text:
            import re
            match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
            if match:
                text = match.group(1).strip()
            else:
                text = text.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(text)
        return parsed, elapsed_ms, None

    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        return None, elapsed_ms, str(e)[:100]

def score_widget_selection(widgets, expected_types, query):
    """Score a widget selection on accuracy."""
    if not widgets or "widgets" not in widgets:
        return 0, "No widgets returned"

    selected = widgets["widgets"]
    if not selected:
        return 0, "Empty widgets array"

    # Count how many expected widget types are present
    selected_types = [w.get("type", "") for w in selected]
    matches = sum(1 for exp in expected_types if any(exp in sel for sel in selected_types))

    # Score = (matches / expected) * 100, capped at 100
    score = min(100, (matches / len(expected_types)) * 100) if expected_types else 50

    # Bonus for good reasoning
    reasons = [w.get("why", "") for w in selected]
    avg_reason_length = sum(len(r) for r in reasons) / len(reasons) if reasons else 0

    if avg_reason_length > 30:  # Detailed reasoning
        score += 10

    # Penalty for too many widgets (>3 requested)
    if len(selected) > 3:
        score -= 20

    return score, f"{len(selected)} widgets: {', '.join(selected_types)}"

def main():
    print("\n" + "="*80)
    print("ACCURACY-FOCUSED COMPARISON: Widget Selection Quality")
    print("="*80 + "\n")

    results = []

    for query, expected_types, rationale in TEST_QUERIES:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"Expected: {', '.join(expected_types)}")
        print(f"Rationale: {rationale}")
        print(f"{'='*80}")

        # Test Llama
        llama_resp, llama_time, llama_err = call_model("llama3.1:8b", query)
        if llama_err:
            llama_score = 0
            llama_detail = f"Error: {llama_err}"
            print(f"\n[Llama] ✗ {llama_err}")
        else:
            llama_score, llama_detail = score_widget_selection(llama_resp, expected_types, query)
            print(f"\n[Llama] Score: {llama_score:.0f}/100")
            print(f"        {llama_detail}")
            for w in llama_resp.get("widgets", [])[:3]:
                print(f"        - {w.get('type')}: {w.get('why', 'No reason')[:60]}")

        # Test GLM
        glm_resp, glm_time, glm_err = call_model("glm-4.7-flash", query)
        if glm_err:
            glm_score = 0
            glm_detail = f"Error: {glm_err}"
            print(f"\n[GLM]   ✗ {glm_err}")
        else:
            glm_score, glm_detail = score_widget_selection(glm_resp, expected_types, query)
            print(f"\n[GLM]   Score: {glm_score:.0f}/100")
            print(f"        {glm_detail}")
            for w in glm_resp.get("widgets", [])[:3]:
                print(f"        - {w.get('type')}: {w.get('why', 'No reason')[:60]}")

        # Determine winner
        if glm_score > llama_score:
            winner = "GLM"
            print(f"\n🏆 Winner: GLM (+{glm_score - llama_score:.0f} points)")
        elif llama_score > glm_score:
            winner = "Llama"
            print(f"\n🏆 Winner: Llama (+{llama_score - glm_score:.0f} points)")
        else:
            winner = "Tie"
            print(f"\n🤝 Tie")

        results.append({
            "query": query,
            "expected": expected_types,
            "llama_score": llama_score,
            "glm_score": glm_score,
            "winner": winner
        })

    # Final summary
    print("\n" + "="*80)
    print("FINAL ACCURACY SCORES")
    print("="*80)

    llama_avg = sum(r["llama_score"] for r in results) / len(results)
    glm_avg = sum(r["glm_score"] for r in results) / len(results)

    llama_wins = sum(1 for r in results if r["winner"] == "Llama")
    glm_wins = sum(1 for r in results if r["winner"] == "GLM")
    ties = sum(1 for r in results if r["winner"] == "Tie")

    print(f"\nAverage Accuracy Score:")
    print(f"  Llama 3.1 8B:    {llama_avg:.1f}/100")
    print(f"  GLM-4.7-Flash:   {glm_avg:.1f}/100")

    print(f"\nWins:")
    print(f"  Llama: {llama_wins}")
    print(f"  GLM:   {glm_wins}")
    print(f"  Ties:  {ties}")

    print("\n" + "="*80)

    if glm_avg > llama_avg + 5:
        print(f"\n🏆 GLM-4.7-Flash is MORE ACCURATE (+{glm_avg - llama_avg:.1f} points)")
    elif llama_avg > glm_avg + 5:
        print(f"\n🏆 Llama 3.1 8B is MORE ACCURATE (+{llama_avg - glm_avg:.1f} points)")
    else:
        print(f"\n🤝 Both models have similar accuracy (difference: {abs(glm_avg - llama_avg):.1f} points)")

    # Save detailed report
    output_file = f"/home/rohith/desktop/CommandCenter/rl_training_data/accuracy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "llama_avg_score": llama_avg,
            "glm_avg_score": glm_avg,
            "results": results
        }, f, indent=2)

    print(f"\nDetailed report: {output_file}")

if __name__ == "__main__":
    main()
