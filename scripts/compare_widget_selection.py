#!/usr/bin/env python3
"""
Live comparison: Llama 3.1 8B vs GLM-4.7-Flash for widget selection.

Tests widget selection quality, JSON validity, and appropriateness without touching production code.
"""

import json
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

# Test queries covering different scenarios
TEST_QUERIES = [
    # Equipment monitoring
    "Show me pump 1 vibration",
    "What's the status of chiller 2?",
    "Display motor 3 temperature trends",

    # Performance analysis
    "Compare efficiency across all pumps",
    "Show power consumption trends for the last week",

    # Multi-equipment
    "Show me all equipment health metrics",
    "Display vibration for pumps 1, 2, and 3",

    # Time-based
    "What was the peak load yesterday?",
    "Show historical temperature data",

    # Complex scenarios
    "Analyze pump performance and identify issues",
    "Dashboard overview of facility operations",
]

@dataclass
class ComparisonResult:
    query: str
    timestamp: str

    # Llama results
    llama_response: Optional[dict]
    llama_time_ms: float
    llama_valid_json: bool
    llama_num_widgets: int
    llama_error: Optional[str]

    # GLM results
    glm_response: Optional[dict]
    glm_time_ms: float
    glm_valid_json: bool
    glm_num_widgets: int
    glm_error: Optional[str]

    # Comparison
    winner: Optional[str]  # "llama", "glm", "tie", or None
    notes: str


class WidgetSelectorComparator:
    """Side-by-side comparison of widget selection between models."""

    def __init__(
        self,
        llama_model: str = "llama3.1:8b",
        glm_model: str = "glm-4.7-flash",
        ollama_base_url: str = "http://localhost:11434"
    ):
        self.llama_model = llama_model
        self.glm_model = glm_model
        self.ollama_base_url = ollama_base_url
        self.results: list[ComparisonResult] = []

        # Load widget selection prompt from backend
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Load widget selection prompt from backend code."""
        backend_path = Path(__file__).parent.parent / "backend" / "layer2" / "widget_selector.py"

        if not backend_path.exists():
            # Fallback minimal prompt
            return """You are a dashboard widget selector. Given a user query, select appropriate widgets.
Return JSON with this structure:
{
  "widgets": [
    {
      "scenario": "trend",
      "widget_size": "medium",
      "widget_count": 1,
      "why": "explanation"
    }
  ]
}

User query: {query}"""

        # Extract SELECT_PROMPT from widget_selector.py
        with open(backend_path) as f:
            content = f.read()

        # Find SELECT_PROMPT definition
        start = content.find('SELECT_PROMPT = """')
        if start == -1:
            start = content.find("SELECT_PROMPT = '''")

        if start != -1:
            start += len('SELECT_PROMPT = """')
            end = content.find('"""', start)
            if end == -1:
                end = content.find("'''", start)
            prompt = content[start:end].strip()
            return prompt

        return ""

    def _call_ollama(self, model: str, prompt: str, timeout: float = 30.0) -> tuple[Optional[dict], float, Optional[str]]:
        """Call Ollama and return (response, time_ms, error)."""
        start = time.time()

        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    # NOTE: Don't use "format": "json" - GLM-4.7-Flash doesn't support it
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 1024,
                    }
                },
                timeout=timeout
            )

            elapsed_ms = (time.time() - start) * 1000

            if response.status_code != 200:
                return None, elapsed_ms, f"HTTP {response.status_code}"

            result = response.json()
            text = result.get("response", "").strip()

            # Parse JSON from response
            # Strip markdown code blocks if present
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
                if text.startswith("json"):
                    text = text[4:].strip()

            parsed = json.loads(text)
            return parsed, elapsed_ms, None

        except requests.Timeout:
            elapsed_ms = (time.time() - start) * 1000
            return None, elapsed_ms, "Timeout"
        except json.JSONDecodeError as e:
            elapsed_ms = (time.time() - start) * 1000
            return None, elapsed_ms, f"Invalid JSON: {str(e)}"
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            return None, elapsed_ms, str(e)

    def compare_query(self, query: str) -> ComparisonResult:
        """Compare both models on a single query."""
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")

        prompt = self.prompt_template.replace("{query}", query) if "{query}" in self.prompt_template else f"{self.prompt_template}\n\nUser query: {query}"

        # Add /nothink prefix for GLM if needed
        glm_prompt = "/nothink\n" + prompt

        # Test Llama
        print(f"[Llama] Calling {self.llama_model}...")
        llama_response, llama_time, llama_error = self._call_ollama(self.llama_model, prompt)
        llama_valid = llama_response is not None and isinstance(llama_response, dict)
        llama_num_widgets = len(llama_response.get("widgets", [])) if llama_valid else 0

        if llama_valid:
            print(f"[Llama] ✓ Valid JSON, {llama_num_widgets} widgets, {llama_time:.0f}ms")
        else:
            print(f"[Llama] ✗ {llama_error}, {llama_time:.0f}ms")

        # Test GLM
        print(f"[GLM] Calling {self.glm_model}...")
        glm_response, glm_time, glm_error = self._call_ollama(self.glm_model, glm_prompt)
        glm_valid = glm_response is not None and isinstance(glm_response, dict)
        glm_num_widgets = len(glm_response.get("widgets", [])) if glm_valid else 0

        if glm_valid:
            print(f"[GLM] ✓ Valid JSON, {glm_num_widgets} widgets, {glm_time:.0f}ms")
        else:
            print(f"[GLM] ✗ {glm_error}, {glm_time:.0f}ms")

        # Determine winner
        winner = None
        notes = ""

        if llama_valid and glm_valid:
            if llama_num_widgets > 0 and glm_num_widgets > 0:
                # Both valid, check speed
                if abs(llama_time - glm_time) < 100:
                    winner = "tie"
                    notes = "Both valid, similar speed"
                elif glm_time < llama_time:
                    winner = "glm"
                    notes = f"GLM faster by {llama_time - glm_time:.0f}ms"
                else:
                    winner = "llama"
                    notes = f"Llama faster by {glm_time - llama_time:.0f}ms"
            elif llama_num_widgets > 0:
                winner = "llama"
                notes = "Llama returned widgets, GLM empty"
            elif glm_num_widgets > 0:
                winner = "glm"
                notes = "GLM returned widgets, Llama empty"
            else:
                winner = None
                notes = "Both returned empty widgets"
        elif llama_valid:
            winner = "llama"
            notes = f"Only Llama valid (GLM error: {glm_error})"
        elif glm_valid:
            winner = "glm"
            notes = f"Only GLM valid (Llama error: {llama_error})"
        else:
            winner = None
            notes = f"Both failed (Llama: {llama_error}, GLM: {glm_error})"

        print(f"[Result] {winner.upper() if winner else 'NO WINNER'}: {notes}")

        result = ComparisonResult(
            query=query,
            timestamp=datetime.now().isoformat(),
            llama_response=llama_response,
            llama_time_ms=llama_time,
            llama_valid_json=llama_valid,
            llama_num_widgets=llama_num_widgets,
            llama_error=llama_error,
            glm_response=glm_response,
            glm_time_ms=glm_time,
            glm_valid_json=glm_valid,
            glm_num_widgets=glm_num_widgets,
            glm_error=glm_error,
            winner=winner,
            notes=notes
        )

        self.results.append(result)
        return result

    def run_comparison(self, queries: list[str] = None):
        """Run full comparison across all test queries."""
        queries = queries or TEST_QUERIES

        print("\n" + "="*80)
        print("WIDGET SELECTION COMPARISON: Llama 3.1 8B vs GLM-4.7-Flash")
        print("="*80)
        print(f"Llama model: {self.llama_model}")
        print(f"GLM model: {self.glm_model}")
        print(f"Test queries: {len(queries)}")
        print("="*80)

        for query in queries:
            self.compare_query(query)
            time.sleep(0.5)  # Brief pause between queries

        self._print_summary()
        self._save_report()

    def _print_summary(self):
        """Print comparison summary."""
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)

        llama_wins = sum(1 for r in self.results if r.winner == "llama")
        glm_wins = sum(1 for r in self.results if r.winner == "glm")
        ties = sum(1 for r in self.results if r.winner == "tie")
        no_winner = sum(1 for r in self.results if r.winner is None)

        llama_valid = sum(1 for r in self.results if r.llama_valid_json)
        glm_valid = sum(1 for r in self.results if r.glm_valid_json)

        llama_avg_time = sum(r.llama_time_ms for r in self.results) / len(self.results)
        glm_avg_time = sum(r.glm_time_ms for r in self.results) / len(self.results)

        print(f"Total queries: {len(self.results)}")
        print(f"\nWins:")
        print(f"  Llama: {llama_wins}")
        print(f"  GLM:   {glm_wins}")
        print(f"  Tie:   {ties}")
        print(f"  None:  {no_winner}")
        print(f"\nJSON validity:")
        print(f"  Llama: {llama_valid}/{len(self.results)} ({llama_valid/len(self.results)*100:.1f}%)")
        print(f"  GLM:   {glm_valid}/{len(self.results)} ({glm_valid/len(self.results)*100:.1f}%)")
        print(f"\nAverage response time:")
        print(f"  Llama: {llama_avg_time:.0f}ms")
        print(f"  GLM:   {glm_avg_time:.0f}ms")

        if glm_wins + ties > llama_wins:
            print(f"\n🏆 GLM-4.7-Flash shows better or equal performance on {glm_wins + ties}/{len(self.results)} queries")
        elif llama_wins > glm_wins + ties:
            print(f"\n🏆 Llama 3.1 8B shows better performance on {llama_wins}/{len(self.results)} queries")
        else:
            print(f"\n🤝 Both models perform equally")

        print("="*80)

    def _save_report(self):
        """Save detailed comparison report."""
        output_dir = Path(__file__).parent.parent / "rl_training_data"
        output_dir.mkdir(exist_ok=True)

        report_file = output_dir / f"widget_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report = {
            "timestamp": datetime.now().isoformat(),
            "llama_model": self.llama_model,
            "glm_model": self.glm_model,
            "results": [asdict(r) for r in self.results]
        }

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nDetailed report saved: {report_file}")


def main():
    """Run widget selection comparison."""
    comparator = WidgetSelectorComparator()
    comparator.run_comparison()


if __name__ == "__main__":
    main()
