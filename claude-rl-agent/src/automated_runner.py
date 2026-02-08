#!/usr/bin/env python3
"""
Automated Parallel Runner

Runs the SAME prompt through both Claude and LLaMA:
1. Build production prompt (same one Ollama gets)
2. Send to Claude → get thinking + answer
3. Send to LLaMA → get answer
4. Save trace (thinking + answer as paired samples for SFT)
"""

import subprocess
import json
import time
from pathlib import Path
from typing import List, Tuple
import logging

try:
    from .claude_teacher import ClaudeTeacher, ClaudeThinking, build_full_prompt
    from .v4_trace import V4Trace, V4TraceStore
except ImportError:
    from claude_teacher import ClaudeTeacher, ClaudeThinking, build_full_prompt
    from v4_trace import V4Trace, V4TraceStore

logger = logging.getLogger(__name__)


class AutomatedRunner:
    """Runs prompts through Claude and LLaMA, saves traces for SFT training."""

    def __init__(
        self,
        claude_cli_path: str = "claude",
        llama_model: str = "cc-claude-agent:latest",
        storage_dir: str = "/home/rohith/desktop/CommandCenter/claude-rl-agent/data"
    ):
        self.teacher = ClaudeTeacher(cli_path=claude_cli_path)
        self.llama_model = llama_model
        self.trace_store = V4TraceStore(storage_dir)

    def run_claude(self, prompt: str, system_prompt: str = None) -> Tuple[ClaudeThinking, float]:
        """Send prompt to Claude, get thinking + answer."""
        result = self.teacher.think(prompt, system_prompt=system_prompt)
        logger.info(
            f"Claude: {result.time_s:.1f}s | "
            f"thinking: {len(result.thinking_text)} chars | "
            f"answer: {len(result.answer_text)} chars"
        )
        return result, result.time_s

    def run_llama(self, prompt: str, system_prompt: str = None) -> Tuple[str, float]:
        """Send same prompt to LLaMA via Ollama."""
        start = time.time()
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            result = subprocess.run(
                ["ollama", "run", self.llama_model, full_prompt],
                capture_output=True, text=True, timeout=300,
            )
            duration = time.time() - start
            logger.info(f"LLaMA: {duration:.1f}s")
            return result.stdout, duration
        except subprocess.TimeoutExpired:
            return "[ERROR: Timeout]", 300.0
        except Exception as e:
            return f"[ERROR: {e}]", 0.0

    def run_parallel_comparison(self, raw_query: str) -> dict:
        """Run same prompt through Claude and LLaMA, save trace.

        Builds the production prompt once, sends it to both.
        Saves the trace (thinking + answer as paired SFT samples).
        """
        # Build the EXACT same prompt Ollama gets
        system_prompt, user_prompt, prompt_version = build_full_prompt(query=raw_query)

        print(f"\nQuery: {raw_query[:60]}...")
        print(f"Prompt: {len(user_prompt)} chars (version: {prompt_version})")

        # Same prompt → both models
        claude_thinking, claude_time = self.run_claude(user_prompt, system_prompt=system_prompt)
        llama_response, llama_time = self.run_llama(user_prompt, system_prompt=system_prompt)

        # Save trace
        trace = V4Trace.from_thinking(
            prompt=user_prompt,
            claude=claude_thinking,
            agent_response=llama_response,
            agent_time_s=llama_time,
            prompt_version=prompt_version,
        )
        self.trace_store.save(trace)

        # Print summary
        mode = "THINKING" if claude_thinking.has_thinking else "DIRECT"
        print(f"  Claude [{mode}]: {claude_time:.1f}s | "
              f"thinking: {len(claude_thinking.thinking_text)} chars | "
              f"answer: {len(claude_thinking.answer_text)} chars")
        print(f"  LLaMA: {llama_time:.1f}s | {len(llama_response)} chars")

        return {
            "trace_id": trace.trace_id,
            "raw_query": raw_query,
            "prompt_version": prompt_version,
            "has_thinking": claude_thinking.has_thinking,
            "claude_time": claude_time,
            "llama_time": llama_time,
        }

    def run_batch(self, queries: List[str], max_prompts: int = None):
        """Run batch of queries through both models, save traces."""
        if max_prompts:
            queries = queries[:max_prompts]

        print(f"\nRunning {len(queries)} queries through Claude + LLaMA\n")

        results = []
        for i, query in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}]", end=" ")
            try:
                result = self.run_parallel_comparison(query)
                results.append(result)
            except Exception as e:
                logger.error(f"Error: {e}")

        # Summary
        stats = self.trace_store.stats()
        thinking = sum(1 for r in results if r.get("has_thinking"))
        print(f"\nDone: {len(results)} traces ({thinking} with thinking)")
        print(f"Total in store: {stats.get('count', 0)}")
        print(f"\nNext: python automated_runner.py --build-dataset")

        return results


# Command Center specific prompt generator
def generate_command_center_prompts(count: int = 50) -> List[str]:
    """Generate Command Center database prompts."""

    prompts = [
        # Equipment queries
        "What's the average power consumption of chiller_001 in the last 24 hours?",
        "Show me the current status of all 6 DG sets",
        "Compare efficiency of transformer trf_001 vs trf_002 today",
        "List all equipment in fault state right now",
        "What's the total energy consumption for July 2024?",

        # Anomaly detection
        "Find temperature anomalies in chiller_003 for last week",
        "Detect power spikes in transformer trf_002 yesterday",
        "Show motor_005 vibration trends for the past month",
        "Identify unusual runtime patterns for DG sets",

        # Maintenance
        "When was the last maintenance for chiller_001?",
        "Show maintenance history for all transformers",
        "Which equipment needs maintenance in the next 30 days?",
        "Calculate MTBF for motor_001 to motor_015",

        # Real-time monitoring
        "What's the current power factor across all LT panels?",
        "Display real-time cooling tower temperatures",
        "Show UPS battery status for all 8 units",
        "What alerts have been triggered in the last hour?",

        # Data aggregation
        "Calculate hourly averages for chiller_001 on June 15, 2024",
        "Show daily peak demand for all transformers in Q2 2024",
        "Aggregate monthly energy by building",
        "Create a weekly summary for all AHUs",

        # Optimization
        "What's the optimal chiller staging to minimize cost?",
        "When should we switch from grid to DG?",
        "Show the load curve for all transformers combined",
        "Calculate ROI of different chiller setpoints",

        # Schema queries
        "Show me the schema of lt_mcc_001 table",
        "List all 357 equipment tables",
        "What columns are in the alerts table?",
        "Show relationships between buildings and equipment",

        # Predictive maintenance
        "Predict next chiller filter change based on pressure drop trend",
        "When will motor_003 bearings need replacement?",
        "Forecast transformer oil testing schedule",
        "Show equipment health scores across all tables",
    ]

    # Repeat to reach desired count
    while len(prompts) < count:
        prompts.extend(prompts[:count - len(prompts)])

    return prompts[:count]


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run prompts through Claude + LLaMA, capture traces for SFT")
    parser.add_argument("--prompt", help="Single query to run")
    parser.add_argument("--batch", type=int, help="Run batch of N prompts")
    parser.add_argument("--llama-model", default="cc-claude-agent:latest", help="Ollama model name")
    parser.add_argument("--build-dataset", action="store_true", help="Build SFT dataset from traces")
    parser.add_argument("--stats", action="store_true", help="Show trace statistics")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    runner = AutomatedRunner(llama_model=args.llama_model)

    if args.build_dataset:
        dataset_path = runner.trace_store.build_sft_dataset()
        stats = runner.trace_store.stats()
        print(f"Traces: {stats.get('count', 0)} ({stats.get('thinking_traces', 0)} with thinking)")
        print(f"Dataset: {dataset_path}")
        print(f"\nTrain: python sft_trainer.py --dataset {dataset_path}")

    elif args.stats:
        stats = runner.trace_store.stats()
        if stats.get('count', 0) == 0:
            print("No traces yet. Run with --prompt or --batch first.")
        else:
            for k, v in stats.items():
                print(f"  {k}: {v}")

    elif args.prompt:
        runner.run_parallel_comparison(args.prompt)

    elif args.batch:
        queries = generate_command_center_prompts(args.batch)
        runner.run_batch(queries, max_prompts=args.batch)

    else:
        print("Usage:")
        print("  python automated_runner.py --prompt 'your query'")
        print("  python automated_runner.py --batch 50")
        print("  python automated_runner.py --build-dataset")
        print("  python automated_runner.py --stats")


if __name__ == "__main__":
    main()
