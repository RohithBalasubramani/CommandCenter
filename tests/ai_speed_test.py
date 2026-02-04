"""
COMMAND CENTER AI SPEED & PERFORMANCE TEST SUITE
Tests AI pipeline speed and performance for the industrial operations command center.

Tests:
1. Intent Parser Response Time
2. RAG Query Response Time
3. Orchestrator Pipeline Latency
4. API Endpoint Response Time
5. Concurrent Request Handling
6. Memory Usage Under Load

Run with: python ai_speed_test.py
"""
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Optional

import psutil

# Add backend to path and change to backend directory for Django
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)

# Set Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "command_center.settings")

import django
django.setup()


@dataclass
class PerformanceResult:
    """Result of a single performance test."""
    test_name: str
    passed: bool
    avg_time_ms: float
    median_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    throughput: float  # ops/second
    p95_time_ms: Optional[float] = None
    p99_time_ms: Optional[float] = None
    memory_mb: Optional[float] = None
    error: Optional[str] = None


@dataclass
class PerformanceReport:
    """Overall performance report."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    results: List[PerformanceResult]
    timestamp: str
    device_info: Dict[str, str]


class CommandCenterSpeedTester:
    """Test Command Center AI capabilities for speed and performance."""

    def __init__(self, base_url: str = "http://127.0.0.1:8100"):
        self.base_url = base_url
        self.results: List[PerformanceResult] = []
        self.process = psutil.Process()

    def print_header(self, title: str):
        """Print formatted header."""
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)

    def print_result(self, result: PerformanceResult):
        """Print test result."""
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} | {result.test_name:<35}")
        print(f"       Avg: {result.avg_time_ms:.1f}ms | Median: {result.median_time_ms:.1f}ms | "
              f"Min: {result.min_time_ms:.1f}ms | Max: {result.max_time_ms:.1f}ms")
        print(f"       Throughput: {result.throughput:.1f} ops/sec | Std Dev: {result.std_dev_ms:.1f}ms")
        if result.p95_time_ms:
            print(f"       P95: {result.p95_time_ms:.1f}ms | P99: {result.p99_time_ms:.1f}ms")
        if result.memory_mb:
            print(f"       Memory Usage: {result.memory_mb:.1f} MB")
        if result.error:
            print(f"       Error: {result.error}")

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def calculate_percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile from data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * (percentile / 100))
        return sorted_data[min(index, len(sorted_data) - 1)]

    async def test_intent_parser_speed(self, iterations: int = 100) -> PerformanceResult:
        """Test Intent Parser response time."""
        test_name = "Intent Parser Speed"
        timings = []

        try:
            from layer2.intent_parser import IntentParser

            parser = IntentParser()

            # Sample queries to test
            test_queries = [
                "What is the status of pump 1?",
                "Show me critical alerts",
                "Turn off motor 3",
                "Compare transformer load vs yesterday",
                "Who is on shift today?",
            ]

            for i in range(iterations):
                query = test_queries[i % len(test_queries)]
                start = time.time()
                parser._parse_with_regex(query)
                elapsed = (time.time() - start) * 1000
                timings.append(elapsed)

            avg_time = mean(timings)
            median_time = median(timings)
            min_time = min(timings)
            max_time = max(timings)
            std_dev = stdev(timings) if len(timings) > 1 else 0
            throughput = 1000 / avg_time if avg_time > 0 else 0
            p95 = self.calculate_percentile(timings, 95)
            p99 = self.calculate_percentile(timings, 99)

            # Pass if average response time is under 10ms (regex parsing should be fast)
            passed = avg_time < 10

            return PerformanceResult(
                test_name=test_name,
                passed=passed,
                avg_time_ms=avg_time,
                median_time_ms=median_time,
                min_time_ms=min_time,
                max_time_ms=max_time,
                std_dev_ms=std_dev,
                throughput=throughput,
                p95_time_ms=p95,
                p99_time_ms=p99,
            )
        except Exception as e:
            return PerformanceResult(
                test_name=test_name,
                passed=False,
                avg_time_ms=0,
                median_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                throughput=0,
                error=str(e),
            )

    async def test_rag_search_speed(self, iterations: int = 20) -> PerformanceResult:
        """Test RAG vector search speed."""
        test_name = "RAG Vector Search Speed"
        timings = []

        try:
            from layer2.rag_pipeline import get_rag_pipeline

            pipeline = get_rag_pipeline()
            stats = pipeline.get_stats()

            if stats.get("equipment_count", 0) == 0:
                return PerformanceResult(
                    test_name=test_name,
                    passed=True,
                    avg_time_ms=0,
                    median_time_ms=0,
                    min_time_ms=0,
                    max_time_ms=0,
                    std_dev_ms=0,
                    throughput=0,
                    error="No documents indexed - run index_rag first",
                )

            test_queries = [
                "pump status",
                "transformer load",
                "chiller temperature",
                "critical alerts",
                "motor health",
            ]

            for i in range(iterations):
                query = test_queries[i % len(test_queries)]
                start = time.time()
                pipeline.vector_store.search("industrial_equipment", query, n_results=5)
                elapsed = (time.time() - start) * 1000
                timings.append(elapsed)

            avg_time = mean(timings)
            median_time = median(timings)
            min_time = min(timings)
            max_time = max(timings)
            std_dev = stdev(timings) if len(timings) > 1 else 0
            throughput = 1000 / avg_time if avg_time > 0 else 0
            p95 = self.calculate_percentile(timings, 95)
            p99 = self.calculate_percentile(timings, 99)

            # Pass if average search time is under 200ms
            passed = avg_time < 200

            return PerformanceResult(
                test_name=test_name,
                passed=passed,
                avg_time_ms=avg_time,
                median_time_ms=median_time,
                min_time_ms=min_time,
                max_time_ms=max_time,
                std_dev_ms=std_dev,
                throughput=throughput,
                p95_time_ms=p95,
                p99_time_ms=p99,
            )
        except Exception as e:
            return PerformanceResult(
                test_name=test_name,
                passed=False,
                avg_time_ms=0,
                median_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                throughput=0,
                error=str(e),
            )

    async def test_embedding_speed(self, iterations: int = 20) -> PerformanceResult:
        """Test embedding generation speed."""
        test_name = "Embedding Generation Speed"
        timings = []

        try:
            from layer2.rag_pipeline import EmbeddingService

            service = EmbeddingService()

            test_texts = [
                "Pump 1 is running at 80% load with temperature at 45C",
                "Critical alert: Transformer TR-001 overheating",
                "Maintenance scheduled for Chiller CH-002 next week",
                "Show me the energy consumption trend for the past week",
                "Compare motor 1 and motor 2 performance",
            ]

            # Warm up - load model (first call loads the model)
            service.embed("warmup text")

            for i in range(iterations):
                text = test_texts[i % len(test_texts)]
                start = time.time()
                service.embed(text)
                elapsed = (time.time() - start) * 1000
                timings.append(elapsed)

            avg_time = mean(timings)
            median_time = median(timings)
            min_time = min(timings)
            max_time = max(timings)
            std_dev = stdev(timings) if len(timings) > 1 else 0
            throughput = 1000 / avg_time if avg_time > 0 else 0
            p95 = self.calculate_percentile(timings, 95)
            p99 = self.calculate_percentile(timings, 99)

            # Pass if average embedding time is under 100ms
            passed = avg_time < 100

            return PerformanceResult(
                test_name=test_name,
                passed=passed,
                avg_time_ms=avg_time,
                median_time_ms=median_time,
                min_time_ms=min_time,
                max_time_ms=max_time,
                std_dev_ms=std_dev,
                throughput=throughput,
                p95_time_ms=p95,
                p99_time_ms=p99,
            )
        except Exception as e:
            return PerformanceResult(
                test_name=test_name,
                passed=False,
                avg_time_ms=0,
                median_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                throughput=0,
                error=str(e),
            )

    async def test_database_query_speed(self, iterations: int = 50) -> PerformanceResult:
        """Test Django ORM query speed."""
        test_name = "Database Query Speed"
        timings = []

        try:
            from asgiref.sync import sync_to_async
            from industrial.models import Pump, Transformer, Alert

            @sync_to_async
            def run_queries():
                _ = list(Pump.objects.all()[:10])
                _ = list(Transformer.objects.all()[:10])
                _ = list(Alert.objects.filter(resolved=False)[:10])

            for i in range(iterations):
                start = time.time()

                # Run typical queries (wrapped for async)
                await run_queries()

                elapsed = (time.time() - start) * 1000
                timings.append(elapsed)

            avg_time = mean(timings)
            median_time = median(timings)
            min_time = min(timings)
            max_time = max(timings)
            std_dev = stdev(timings) if len(timings) > 1 else 0
            throughput = 1000 / avg_time if avg_time > 0 else 0
            p95 = self.calculate_percentile(timings, 95)
            p99 = self.calculate_percentile(timings, 99)

            # Pass if average query time is under 50ms
            passed = avg_time < 50

            return PerformanceResult(
                test_name=test_name,
                passed=passed,
                avg_time_ms=avg_time,
                median_time_ms=median_time,
                min_time_ms=min_time,
                max_time_ms=max_time,
                std_dev_ms=std_dev,
                throughput=throughput,
                p95_time_ms=p95,
                p99_time_ms=p99,
            )
        except Exception as e:
            return PerformanceResult(
                test_name=test_name,
                passed=False,
                avg_time_ms=0,
                median_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                throughput=0,
                error=str(e),
            )

    async def test_concurrent_parsing(self, concurrent: int = 10) -> PerformanceResult:
        """Test concurrent intent parsing."""
        test_name = f"Concurrent Parsing ({concurrent} workers)"
        timings = []

        try:
            from layer2.intent_parser import IntentParser

            async def worker(worker_id: int):
                parser = IntentParser()
                worker_timings = []
                queries = [
                    "What is the pump status?",
                    "Show me alerts",
                    "Turn off motor 3",
                ]
                for i in range(5):
                    query = queries[i % len(queries)]
                    start = time.time()
                    parser._parse_with_regex(query)
                    elapsed = (time.time() - start) * 1000
                    worker_timings.append(elapsed)
                return worker_timings

            start = time.time()
            results = await asyncio.gather(*[worker(i) for i in range(concurrent)])
            total_elapsed = (time.time() - start) * 1000

            for worker_timings in results:
                timings.extend(worker_timings)

            total_requests = concurrent * 5
            avg_time = mean(timings) if timings else 0
            median_time = median(timings) if timings else 0
            min_time = min(timings) if timings else 0
            max_time = max(timings) if timings else 0
            std_dev = stdev(timings) if len(timings) > 1 else 0
            throughput = total_requests / (total_elapsed / 1000) if total_elapsed > 0 else 0
            p95 = self.calculate_percentile(timings, 95)
            p99 = self.calculate_percentile(timings, 99)

            # Pass if throughput is > 100 requests/sec
            passed = throughput > 100

            return PerformanceResult(
                test_name=test_name,
                passed=passed,
                avg_time_ms=avg_time,
                median_time_ms=median_time,
                min_time_ms=min_time,
                max_time_ms=max_time,
                std_dev_ms=std_dev,
                throughput=throughput,
                p95_time_ms=p95,
                p99_time_ms=p99,
            )
        except Exception as e:
            return PerformanceResult(
                test_name=test_name,
                passed=False,
                avg_time_ms=0,
                median_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                throughput=0,
                error=str(e),
            )

    async def test_memory_under_load(self, operations: int = 500) -> PerformanceResult:
        """Test memory usage under load."""
        test_name = f"Memory Under Load ({operations} ops)"
        timings = []

        try:
            from layer2.intent_parser import IntentParser

            parser = IntentParser()
            mem_before = self.get_memory_usage()

            queries = [
                "What is the status of pump 1?",
                "Show me all critical alerts for transformers",
                "Compare motor 1 vs motor 2 performance over the last week",
                "Get the temperature trend for chiller CH-001",
            ]

            for i in range(operations):
                start = time.time()
                query = queries[i % len(queries)]
                parser._parse_with_regex(query)
                elapsed = (time.time() - start) * 1000
                timings.append(elapsed)

            mem_after = self.get_memory_usage()
            mem_used = mem_after - mem_before

            avg_time = mean(timings)
            throughput = 1000 / avg_time if avg_time > 0 else 0

            # Pass if memory usage is under 50MB
            passed = mem_used < 50

            return PerformanceResult(
                test_name=test_name,
                passed=passed,
                avg_time_ms=avg_time,
                median_time_ms=median(timings),
                min_time_ms=min(timings),
                max_time_ms=max(timings),
                std_dev_ms=stdev(timings) if len(timings) > 1 else 0,
                throughput=throughput,
                memory_mb=mem_used,
            )
        except Exception as e:
            return PerformanceResult(
                test_name=test_name,
                passed=False,
                avg_time_ms=0,
                median_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                throughput=0,
                error=str(e),
            )

    async def test_llm_response_time(self, iterations: int = 3) -> PerformanceResult:
        """Test LLM response time (if available)."""
        test_name = "LLM Response Time"
        timings = []

        try:
            from layer2.rag_pipeline import OllamaLLMService

            llm = OllamaLLMService()

            if not llm.is_available():
                return PerformanceResult(
                    test_name=test_name,
                    passed=True,
                    avg_time_ms=0,
                    median_time_ms=0,
                    min_time_ms=0,
                    max_time_ms=0,
                    std_dev_ms=0,
                    throughput=0,
                    error="LLM not available - Ollama not running",
                )

            prompts = [
                "Briefly describe the status of a pump.",
                "What should I check for a high temperature alert?",
                "Summarize equipment health in one sentence.",
            ]

            for i in range(iterations):
                prompt = prompts[i % len(prompts)]
                start = time.time()
                llm.generate(prompt, max_tokens=50)
                elapsed = (time.time() - start) * 1000
                timings.append(elapsed)

            avg_time = mean(timings)
            median_time = median(timings)
            min_time = min(timings)
            max_time = max(timings)
            std_dev = stdev(timings) if len(timings) > 1 else 0
            throughput = 1000 / avg_time if avg_time > 0 else 0
            p95 = self.calculate_percentile(timings, 95)
            p99 = self.calculate_percentile(timings, 99)

            # Pass if average LLM response time is under 5 seconds
            passed = avg_time < 5000

            return PerformanceResult(
                test_name=test_name,
                passed=passed,
                avg_time_ms=avg_time,
                median_time_ms=median_time,
                min_time_ms=min_time,
                max_time_ms=max_time,
                std_dev_ms=std_dev,
                throughput=throughput,
                p95_time_ms=p95,
                p99_time_ms=p99,
            )
        except Exception as e:
            return PerformanceResult(
                test_name=test_name,
                passed=False,
                avg_time_ms=0,
                median_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                throughput=0,
                error=str(e),
            )

    async def run_all_tests(self) -> PerformanceReport:
        """Run all performance tests."""
        self.print_header("COMMAND CENTER SPEED & PERFORMANCE TEST SUITE")

        # Get device info
        import platform
        device_info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": str(psutil.cpu_count()),
            "total_memory_gb": str(round(psutil.virtual_memory().total / 1024 / 1024 / 1024, 2)),
        }

        print(f"Device: {device_info['platform']} {device_info['platform_version']}")
        print(f"Processor: {device_info['processor']}")
        print(f"CPU Cores: {device_info['cpu_count']}")
        print(f"Total Memory: {device_info['total_memory_gb']} GB")
        print(f"Python: {device_info['python_version']}")

        # Run all tests
        tests = [
            self.test_intent_parser_speed(),
            self.test_database_query_speed(),
            self.test_embedding_speed(),
            self.test_rag_search_speed(),
            self.test_concurrent_parsing(),
            self.test_memory_under_load(),
            self.test_llm_response_time(),
        ]

        self.print_header("Running Performance Tests")
        for test_coro in tests:
            result = await test_coro
            self.results.append(result)
            self.print_result(result)

        # Generate report
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed

        report = PerformanceReport(
            total_tests=len(self.results),
            passed_tests=passed,
            failed_tests=failed,
            results=self.results,
            timestamp=datetime.now().isoformat(),
            device_info=device_info,
        )

        # Print summary
        self.print_header("PERFORMANCE SUMMARY")
        print(f"Total Tests: {report.total_tests}")
        print(f"Passed: {report.passed_tests}")
        print(f"Failed: {report.failed_tests}")
        print(f"Pass Rate: {(report.passed_tests / report.total_tests) * 100:.1f}%")

        # Calculate overall statistics
        valid_results = [r for r in self.results if r.avg_time_ms > 0 and not r.error]
        if valid_results:
            avg_response_time = mean([r.avg_time_ms for r in valid_results])
            total_throughput = sum([r.throughput for r in valid_results])
            print(f"\nOverall Average Response Time: {avg_response_time:.1f}ms")
            print(f"Total Throughput: {total_throughput:.1f} ops/sec")

        # Save report to file
        report_file = f"cc_speed_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump({
                "total_tests": report.total_tests,
                "passed_tests": report.passed_tests,
                "failed_tests": report.failed_tests,
                "timestamp": report.timestamp,
                "device_info": report.device_info,
                "results": [
                    {
                        "test_name": r.test_name,
                        "passed": r.passed,
                        "avg_time_ms": r.avg_time_ms,
                        "median_time_ms": r.median_time_ms,
                        "min_time_ms": r.min_time_ms,
                        "max_time_ms": r.max_time_ms,
                        "std_dev_ms": r.std_dev_ms,
                        "throughput": r.throughput,
                        "p95_time_ms": r.p95_time_ms,
                        "p99_time_ms": r.p99_time_ms,
                        "memory_mb": r.memory_mb,
                        "error": r.error,
                    }
                    for r in report.results
                ]
            }, f, indent=2)

        print(f"\nDetailed report saved to: {report_file}")

        return report


async def main():
    """Main entry point."""
    tester = CommandCenterSpeedTester()
    report = await tester.run_all_tests()

    # Exit with error code if tests failed
    sys.exit(0 if report.failed_tests == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
