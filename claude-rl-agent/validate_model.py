#!/usr/bin/env python3
"""
Comprehensive Model Validation Suite

Tests the deployed Ollama model across multiple dimensions:
1. Basic functionality (simple queries)
2. Command Center domain knowledge (equipment, sensors, maintenance)
3. Tool understanding (Bash, Read, Grep, etc.)
4. Reasoning quality (assumptions, validations, counterfactuals)
5. Response format (markdown, code blocks, structured output)
"""

import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

@dataclass
class ValidationResult:
    """Result of a single validation test."""
    test_name: str
    category: str
    prompt: str
    response: str
    passed: bool
    score: float  # 0.0 to 1.0
    notes: str
    response_time_ms: int

class ModelValidator:
    """Validates deployed Ollama model quality."""

    def __init__(self, model_name: str = "cc-claude-agent:latest"):
        self.model_name = model_name
        self.results: List[ValidationResult] = []

    def query_model(self, prompt: str, timeout: int = 30) -> Tuple[str, int]:
        """Query Ollama model and return response with timing."""
        start_time = time.time()

        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            if result.returncode == 0:
                return result.stdout.strip(), elapsed_ms
            else:
                return f"ERROR: {result.stderr}", elapsed_ms

        except subprocess.TimeoutExpired:
            elapsed_ms = timeout * 1000
            return "ERROR: Query timeout", elapsed_ms
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return f"ERROR: {str(e)}", elapsed_ms

    def validate_basic_functionality(self):
        """Test basic model functionality."""
        print("\n" + "="*70)
        print("  Test Category: Basic Functionality")
        print("="*70)

        tests = [
            {
                "name": "Simple greeting",
                "prompt": "Hello! What are you?",
                "check": lambda r: len(r) > 20 and ("assistant" in r.lower() or "model" in r.lower() or "help" in r.lower())
            },
            {
                "name": "Simple math",
                "prompt": "What is 2 + 2?",
                "check": lambda r: "4" in r
            },
            {
                "name": "Basic reasoning",
                "prompt": "If I have 3 apples and give away 1, how many do I have?",
                "check": lambda r: "2" in r
            }
        ]

        for test in tests:
            print(f"\nRunning: {test['name']}")
            response, elapsed = self.query_model(test['prompt'])

            passed = test['check'](response)
            score = 1.0 if passed else 0.0

            result = ValidationResult(
                test_name=test['name'],
                category="basic_functionality",
                prompt=test['prompt'],
                response=response[:200] + "..." if len(response) > 200 else response,
                passed=passed,
                score=score,
                notes="",
                response_time_ms=elapsed
            )

            self.results.append(result)
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} ({elapsed}ms)")

    def validate_command_center_domain(self):
        """Test Command Center domain knowledge."""
        print("\n" + "="*70)
        print("  Test Category: Command Center Domain Knowledge")
        print("="*70)

        tests = [
            {
                "name": "Pressure sensor query",
                "prompt": "Show me all pressure sensors in the system",
                "keywords": ["pressure", "sensor", "psi", "gauge", "read", "grep", "bash"],
                "min_keywords": 2
            },
            {
                "name": "Anomaly detection",
                "prompt": "How would you detect anomalies in temperature readings?",
                "keywords": ["temperature", "anomaly", "threshold", "average", "deviation", "alert"],
                "min_keywords": 3
            },
            {
                "name": "Maintenance scheduling",
                "prompt": "What tables would you check for maintenance schedules?",
                "keywords": ["maintenance", "schedule", "equipment", "table", "database"],
                "min_keywords": 3
            },
            {
                "name": "Equipment status",
                "prompt": "How would you check if equipment E-101 is operational?",
                "keywords": ["equipment", "operational", "status", "query", "database"],
                "min_keywords": 2
            }
        ]

        for test in tests:
            print(f"\nRunning: {test['name']}")
            response, elapsed = self.query_model(test['prompt'])

            # Score based on keyword presence
            keywords_found = sum(1 for kw in test['keywords'] if kw.lower() in response.lower())
            passed = keywords_found >= test['min_keywords']
            score = min(1.0, keywords_found / len(test['keywords']))

            result = ValidationResult(
                test_name=test['name'],
                category="domain_knowledge",
                prompt=test['prompt'],
                response=response[:300] + "..." if len(response) > 300 else response,
                passed=passed,
                score=score,
                notes=f"Found {keywords_found}/{len(test['keywords'])} keywords",
                response_time_ms=elapsed
            )

            self.results.append(result)
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} ({elapsed}ms) - {keywords_found}/{len(test['keywords'])} keywords")

    def validate_tool_understanding(self):
        """Test understanding of Claude Code tools."""
        print("\n" + "="*70)
        print("  Test Category: Tool Understanding")
        print("="*70)

        tests = [
            {
                "name": "Bash tool for listing files",
                "prompt": "How would you list all Python files in the current directory?",
                "keywords": ["bash", "ls", "*.py", "glob", "find"],
                "min_keywords": 1
            },
            {
                "name": "Read tool for file content",
                "prompt": "How would you read the contents of config.py?",
                "keywords": ["read", "file", "config.py"],
                "min_keywords": 1
            },
            {
                "name": "Grep for searching",
                "prompt": "How would you search for the word 'sensor' in all Python files?",
                "keywords": ["grep", "search", "sensor", "*.py"],
                "min_keywords": 2
            }
        ]

        for test in tests:
            print(f"\nRunning: {test['name']}")
            response, elapsed = self.query_model(test['prompt'])

            keywords_found = sum(1 for kw in test['keywords'] if kw.lower() in response.lower())
            passed = keywords_found >= test['min_keywords']
            score = min(1.0, keywords_found / len(test['keywords']))

            result = ValidationResult(
                test_name=test['name'],
                category="tool_understanding",
                prompt=test['prompt'],
                response=response[:300] + "..." if len(response) > 300 else response,
                passed=passed,
                score=score,
                notes=f"Found {keywords_found}/{len(test['keywords'])} tool references",
                response_time_ms=elapsed
            )

            self.results.append(result)
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} ({elapsed}ms) - {keywords_found}/{len(test['keywords'])} keywords")

    def validate_reasoning_quality(self):
        """Test reasoning quality (assumptions, validations, counterfactuals)."""
        print("\n" + "="*70)
        print("  Test Category: Reasoning Quality")
        print("="*70)

        tests = [
            {
                "name": "Assumption clarity",
                "prompt": "If I ask you to optimize a query, what assumptions would you make?",
                "keywords": ["assume", "assuming", "likely", "typically", "usually", "probably"],
                "min_keywords": 2
            },
            {
                "name": "Validation checks",
                "prompt": "How would you validate that a sensor reading is correct?",
                "keywords": ["check", "verify", "validate", "ensure", "confirm", "test"],
                "min_keywords": 2
            },
            {
                "name": "Alternative approaches",
                "prompt": "What are different ways to detect equipment failures?",
                "keywords": ["alternative", "another", "could", "option", "approach", "method"],
                "min_keywords": 2
            }
        ]

        for test in tests:
            print(f"\nRunning: {test['name']}")
            response, elapsed = self.query_model(test['prompt'])

            keywords_found = sum(1 for kw in test['keywords'] if kw.lower() in response.lower())
            passed = keywords_found >= test['min_keywords']
            score = min(1.0, keywords_found / len(test['keywords']))

            result = ValidationResult(
                test_name=test['name'],
                category="reasoning_quality",
                prompt=test['prompt'],
                response=response[:300] + "..." if len(response) > 300 else response,
                passed=passed,
                score=score,
                notes=f"Reasoning indicators: {keywords_found}/{len(test['keywords'])}",
                response_time_ms=elapsed
            )

            self.results.append(result)
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} ({elapsed}ms)")

    def validate_response_format(self):
        """Test response formatting (markdown, code blocks)."""
        print("\n" + "="*70)
        print("  Test Category: Response Formatting")
        print("="*70)

        tests = [
            {
                "name": "Code block generation",
                "prompt": "Show me a Python example of reading a CSV file",
                "check": lambda r: "```" in r or "import" in r or "csv" in r.lower(),
                "description": "Should contain code block or Python code"
            },
            {
                "name": "Structured response",
                "prompt": "List 3 ways to improve database performance",
                "check": lambda r: any(marker in r for marker in ["1.", "2.", "3.", "-", "*"]),
                "description": "Should contain list markers"
            }
        ]

        for test in tests:
            print(f"\nRunning: {test['name']}")
            response, elapsed = self.query_model(test['prompt'])

            passed = test['check'](response)
            score = 1.0 if passed else 0.0

            result = ValidationResult(
                test_name=test['name'],
                category="response_format",
                prompt=test['prompt'],
                response=response[:300] + "..." if len(response) > 300 else response,
                passed=passed,
                score=score,
                notes=test['description'],
                response_time_ms=elapsed
            )

            self.results.append(result)
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} ({elapsed}ms)")

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "="*70)
        print("  VALIDATION SUMMARY")
        print("="*70)

        # Overall stats
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        avg_score = sum(r.score for r in self.results) / total_tests if total_tests > 0 else 0
        avg_time = sum(r.response_time_ms for r in self.results) / total_tests if total_tests > 0 else 0

        print(f"\nOverall Results:")
        print(f"  Tests Passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"  Average Score: {avg_score:.2f}/1.00")
        print(f"  Average Response Time: {avg_time:.0f}ms")

        # Category breakdown
        print(f"\nBy Category:")
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)

        for category, results in categories.items():
            cat_passed = sum(1 for r in results if r.passed)
            cat_score = sum(r.score for r in results) / len(results)
            cat_time = sum(r.response_time_ms for r in results) / len(results)

            print(f"  {category}:")
            print(f"    Passed: {cat_passed}/{len(results)} ({cat_passed/len(results)*100:.1f}%)")
            print(f"    Score: {cat_score:.2f}/1.00")
            print(f"    Avg Time: {cat_time:.0f}ms")

    def save_results(self, output_file: str = "data/validation_results.json"):
        """Save results to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_dict = {
            "model_name": self.model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_tests": len(self.results),
                "passed_tests": sum(1 for r in self.results if r.passed),
                "average_score": sum(r.score for r in self.results) / len(self.results) if self.results else 0,
                "average_response_time_ms": sum(r.response_time_ms for r in self.results) / len(self.results) if self.results else 0
            },
            "results": [asdict(r) for r in self.results]
        }

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"\n✅ Results saved to: {output_path}")

def main():
    print("="*70)
    print("  🧪 Model Validation Suite")
    print("="*70)

    validator = ModelValidator("cc-claude-agent:latest")

    # Check if model exists
    print("\nChecking if model is available...")
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True
    )

    if "cc-claude-agent" not in result.stdout:
        print("❌ Model 'cc-claude-agent:latest' not found in Ollama!")
        print("   Please deploy the model first: ./auto_deploy.sh")
        return 1

    print("✅ Model found: cc-claude-agent:latest")

    # Run all validation tests
    validator.validate_basic_functionality()
    validator.validate_command_center_domain()
    validator.validate_tool_understanding()
    validator.validate_reasoning_quality()
    validator.validate_response_format()

    # Print summary and save results
    validator.print_summary()
    validator.save_results()

    # Return exit code based on pass rate
    pass_rate = sum(1 for r in validator.results if r.passed) / len(validator.results)
    if pass_rate >= 0.7:
        print(f"\n✅ Validation PASSED (pass rate: {pass_rate*100:.1f}%)")
        return 0
    else:
        print(f"\n⚠️  Validation FAILED (pass rate: {pass_rate*100:.1f}% < 70%)")
        return 1

if __name__ == "__main__":
    exit(main())
