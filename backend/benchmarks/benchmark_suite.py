#!/usr/bin/env python3
"""
Command Center Benchmark Suite

Measures performance across all layers:
- STT Server latency and throughput
- RAG search accuracy and latency
- TTS Server latency
- End-to-end orchestrator pipeline

Run with:
    cd backend && python -m benchmarks.benchmark_suite

Outputs JSON report to benchmarks/results/
"""

import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "command_center.settings")

import django
django.setup()

import requests
import numpy as np


# ============================================================
# Configuration
# ============================================================

STT_SERVER_URL = os.getenv("STT_SERVER_URL", "http://localhost:8890")
TTS_SERVER_URL = os.getenv("TTS_SERVER_URL", "http://localhost:8880")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8100")

# Test queries for RAG accuracy
RAG_TEST_QUERIES = [
    # (query, expected_equipment_types, expected_in_response)
    ("What's the status of the pumps?", ["pump"], ["pump", "running"]),
    ("Show transformer temperatures", ["transformer"], ["transformer", "temperature"]),
    ("What alerts are active?", [], ["alert"]),
    ("How are the chillers performing?", ["chiller"], ["chiller"]),
    ("What's the energy consumption?", ["energy_meter"], ["energy", "consumption"]),
    ("Status of all motors", ["motor"], ["motor"]),
    ("TX-001 status", ["transformer"], ["TX-001"]),
    ("Pump 1 temperature", ["pump"], ["pump", "temperature"]),
    ("Generator load", ["diesel_generator"], ["generator", "load"]),
    ("AHU status", ["ahu"], ["AHU", "air"]),
]

# Test transcripts for end-to-end
E2E_TEST_TRANSCRIPTS = [
    "What's the status of the pumps?",
    "Show me transformer temperatures",
    "What alerts are active?",
    "How are the chillers performing?",
    "Compare pump 1 and pump 2",
    "Hello",
    "What's the weather like?",  # out of scope
]

# TTS test sentences (varying lengths)
TTS_TEST_SENTENCES = [
    "Hello.",
    "The pump is running normally.",
    "There are 3 active alerts. Transformer TX-001 has high temperature. Pump 2 shows elevated pressure.",
    "Based on the latest data, all 5 transformers are operating within normal parameters. The average load is 72% and oil temperatures are stable.",
]


# ============================================================
# Data Classes
# ============================================================

@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""
    name: str
    latency_ms: float
    success: bool
    details: dict = None

    def to_dict(self):
        return asdict(self)


@dataclass
class BenchmarkSummary:
    """Summary statistics for a benchmark category."""
    name: str
    count: int
    success_rate: float
    latency_min_ms: float
    latency_max_ms: float
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    details: dict = None

    def to_dict(self):
        return asdict(self)


@dataclass
class FullBenchmarkReport:
    """Complete benchmark report."""
    timestamp: str
    system_info: dict
    stt_summary: Optional[BenchmarkSummary]
    rag_summary: Optional[BenchmarkSummary]
    tts_summary: Optional[BenchmarkSummary]
    e2e_summary: Optional[BenchmarkSummary]
    raw_results: dict

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "stt_summary": self.stt_summary.to_dict() if self.stt_summary else None,
            "rag_summary": self.rag_summary.to_dict() if self.rag_summary else None,
            "tts_summary": self.tts_summary.to_dict() if self.tts_summary else None,
            "e2e_summary": self.e2e_summary.to_dict() if self.e2e_summary else None,
            "raw_results": self.raw_results,
        }


# ============================================================
# Benchmark Utilities
# ============================================================

def calculate_summary(name: str, results: list[BenchmarkResult]) -> BenchmarkSummary:
    """Calculate summary statistics from benchmark results."""
    if not results:
        return BenchmarkSummary(
            name=name, count=0, success_rate=0,
            latency_min_ms=0, latency_max_ms=0, latency_mean_ms=0,
            latency_median_ms=0, latency_p95_ms=0, latency_p99_ms=0
        )

    successful = [r for r in results if r.success]
    latencies = [r.latency_ms for r in successful]

    if not latencies:
        return BenchmarkSummary(
            name=name, count=len(results), success_rate=0,
            latency_min_ms=0, latency_max_ms=0, latency_mean_ms=0,
            latency_median_ms=0, latency_p95_ms=0, latency_p99_ms=0
        )

    return BenchmarkSummary(
        name=name,
        count=len(results),
        success_rate=len(successful) / len(results),
        latency_min_ms=round(min(latencies), 2),
        latency_max_ms=round(max(latencies), 2),
        latency_mean_ms=round(statistics.mean(latencies), 2),
        latency_median_ms=round(statistics.median(latencies), 2),
        latency_p95_ms=round(np.percentile(latencies, 95), 2),
        latency_p99_ms=round(np.percentile(latencies, 99), 2),
    )


def check_service_health(url: str, endpoint: str = "/") -> bool:
    """Check if a service is available."""
    try:
        resp = requests.get(f"{url}{endpoint}", timeout=3)
        return resp.status_code in (200, 404)  # 404 means server is up
    except Exception:
        return False


def generate_test_audio(duration_seconds: float = 3.0, sample_rate: int = 16000) -> bytes:
    """Generate test audio (silence + small noise for realism)."""
    import io
    import soundfile as sf

    samples = int(duration_seconds * sample_rate)
    # Small random noise to simulate real audio
    audio = np.random.randn(samples).astype(np.float32) * 0.01

    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format='WAV')
    buf.seek(0)
    return buf.read()


# ============================================================
# STT Benchmarks
# ============================================================

def benchmark_stt_server(iterations: int = 5) -> list[BenchmarkResult]:
    """Benchmark STT server latency."""
    print("\n=== STT Server Benchmark ===")

    if not check_service_health(STT_SERVER_URL, "/v1/stt/health"):
        print(f"STT server not available at {STT_SERVER_URL}")
        return []

    results = []

    # Generate test audio files of different durations
    test_audios = [
        ("1s", generate_test_audio(1.0)),
        ("3s", generate_test_audio(3.0)),
        ("5s", generate_test_audio(5.0)),
    ]

    for duration_name, audio_data in test_audios:
        for i in range(iterations):
            try:
                t0 = time.perf_counter()
                resp = requests.post(
                    f"{STT_SERVER_URL}/v1/stt",
                    files={"audio": ("test.wav", audio_data, "audio/wav")},
                    timeout=30
                )
                latency = (time.perf_counter() - t0) * 1000

                success = resp.status_code == 200
                details = {"duration": duration_name, "iteration": i + 1}

                if success:
                    data = resp.json()
                    details["model"] = data.get("model", "unknown")
                    details["server_ms"] = data.get("duration_ms", 0)
                    print(f"  {duration_name} iter {i+1}: {latency:.0f}ms (server: {details['server_ms']:.0f}ms)")
                else:
                    print(f"  {duration_name} iter {i+1}: FAILED ({resp.status_code})")

                results.append(BenchmarkResult(
                    name=f"stt_{duration_name}",
                    latency_ms=latency,
                    success=success,
                    details=details
                ))

            except Exception as e:
                print(f"  {duration_name} iter {i+1}: ERROR ({e})")
                results.append(BenchmarkResult(
                    name=f"stt_{duration_name}",
                    latency_ms=0,
                    success=False,
                    details={"error": str(e)}
                ))

    return results


# ============================================================
# RAG Benchmarks
# ============================================================

def benchmark_rag_accuracy() -> list[BenchmarkResult]:
    """Benchmark RAG search accuracy and latency."""
    print("\n=== RAG Accuracy Benchmark ===")

    from layer2.rag_pipeline import get_rag_pipeline

    try:
        pipeline = get_rag_pipeline()
    except Exception as e:
        print(f"Failed to initialize RAG pipeline: {e}")
        return []

    # Check if index exists
    stats = pipeline.get_stats()
    if stats.get("equipment_count", 0) == 0:
        print("RAG index is empty. Run 'python manage.py index_rag' first.")
        return []

    print(f"  Index stats: {stats['equipment_count']} equipment, {stats['alerts_count']} alerts")

    results = []

    for query, expected_types, expected_terms in RAG_TEST_QUERIES:
        try:
            t0 = time.perf_counter()
            response = pipeline.query(query, n_results=5)
            latency = (time.perf_counter() - t0) * 1000

            # Check accuracy
            retrieved_types = set()
            for doc in response.retrieved_docs:
                eq_type = doc.metadata.get("equipment_type", "")
                if eq_type:
                    retrieved_types.add(eq_type)

            # Check if expected types were retrieved
            type_match = all(t in retrieved_types for t in expected_types) if expected_types else True

            # Check if expected terms appear in context
            context_lower = response.context.lower()
            term_match = all(term.lower() in context_lower for term in expected_terms)

            success = type_match and term_match

            print(f"  '{query[:40]}...' → {latency:.0f}ms, types={type_match}, terms={term_match}")

            results.append(BenchmarkResult(
                name="rag_search",
                latency_ms=latency,
                success=success,
                details={
                    "query": query,
                    "retrieved_types": list(retrieved_types),
                    "expected_types": expected_types,
                    "type_match": type_match,
                    "term_match": term_match,
                    "doc_count": len(response.retrieved_docs),
                }
            ))

        except Exception as e:
            print(f"  '{query[:40]}...' → ERROR ({e})")
            results.append(BenchmarkResult(
                name="rag_search",
                latency_ms=0,
                success=False,
                details={"query": query, "error": str(e)}
            ))

    return results


def benchmark_rag_latency(iterations: int = 10) -> list[BenchmarkResult]:
    """Benchmark RAG search latency with repeated queries."""
    print("\n=== RAG Latency Benchmark ===")

    from layer2.rag_pipeline import get_rag_pipeline

    try:
        pipeline = get_rag_pipeline()
    except Exception as e:
        print(f"Failed to initialize RAG pipeline: {e}")
        return []

    results = []
    test_query = "What's the status of the pumps?"

    for i in range(iterations):
        try:
            t0 = time.perf_counter()
            response = pipeline.query(test_query, n_results=5)
            latency = (time.perf_counter() - t0) * 1000

            print(f"  Iteration {i+1}: {latency:.0f}ms ({len(response.retrieved_docs)} docs)")

            results.append(BenchmarkResult(
                name="rag_latency",
                latency_ms=latency,
                success=True,
                details={"iteration": i + 1}
            ))

        except Exception as e:
            print(f"  Iteration {i+1}: ERROR ({e})")
            results.append(BenchmarkResult(
                name="rag_latency",
                latency_ms=0,
                success=False,
                details={"error": str(e)}
            ))

    return results


# ============================================================
# TTS Benchmarks
# ============================================================

def benchmark_tts_server(iterations: int = 3) -> list[BenchmarkResult]:
    """Benchmark TTS server latency."""
    print("\n=== TTS Server Benchmark ===")

    if not check_service_health(TTS_SERVER_URL, "/v1/models"):
        print(f"TTS server not available at {TTS_SERVER_URL}")
        return []

    results = []

    for sentence in TTS_TEST_SENTENCES:
        word_count = len(sentence.split())

        for i in range(iterations):
            try:
                t0 = time.perf_counter()
                resp = requests.post(
                    f"{TTS_SERVER_URL}/v1/audio/speech",
                    json={
                        "model": "kokoro",
                        "input": sentence,
                        "voice": "af_heart",
                        "response_format": "mp3",
                    },
                    timeout=60
                )
                latency = (time.perf_counter() - t0) * 1000

                success = resp.status_code == 200
                details = {
                    "word_count": word_count,
                    "char_count": len(sentence),
                    "iteration": i + 1,
                }

                if success:
                    details["audio_bytes"] = len(resp.content)
                    print(f"  {word_count}w iter {i+1}: {latency:.0f}ms ({len(resp.content)} bytes)")
                else:
                    print(f"  {word_count}w iter {i+1}: FAILED ({resp.status_code})")

                results.append(BenchmarkResult(
                    name=f"tts_{word_count}w",
                    latency_ms=latency,
                    success=success,
                    details=details
                ))

            except Exception as e:
                print(f"  {word_count}w iter {i+1}: ERROR ({e})")
                results.append(BenchmarkResult(
                    name=f"tts_{word_count}w",
                    latency_ms=0,
                    success=False,
                    details={"error": str(e)}
                ))

    return results


# ============================================================
# End-to-End Benchmarks
# ============================================================

def benchmark_e2e_orchestrator(iterations: int = 2) -> list[BenchmarkResult]:
    """Benchmark end-to-end orchestrator pipeline."""
    print("\n=== E2E Orchestrator Benchmark ===")

    from layer2.orchestrator import Layer2Orchestrator

    results = []
    orchestrator = Layer2Orchestrator()

    for transcript in E2E_TEST_TRANSCRIPTS:
        for i in range(iterations):
            try:
                t0 = time.perf_counter()
                response = orchestrator.process_transcript(
                    transcript,
                    user_id="benchmark_user"
                )
                latency = (time.perf_counter() - t0) * 1000

                success = bool(response.voice_response) and len(response.voice_response) > 0

                details = {
                    "transcript": transcript,
                    "iteration": i + 1,
                    "intent_type": response.intent.type if response.intent else None,
                    "processing_time_ms": response.processing_time_ms,
                    "widget_count": len(response.layout_json.get("widgets", [])) if response.layout_json else 0,
                }

                # Add timing breakdown if available
                if response.timings:
                    details["timings"] = response.timings.to_dict()

                print(f"  '{transcript[:30]}...' iter {i+1}: {latency:.0f}ms (intent={details['intent_type']})")

                results.append(BenchmarkResult(
                    name="e2e_orchestrator",
                    latency_ms=latency,
                    success=success,
                    details=details
                ))

            except Exception as e:
                print(f"  '{transcript[:30]}...' iter {i+1}: ERROR ({e})")
                results.append(BenchmarkResult(
                    name="e2e_orchestrator",
                    latency_ms=0,
                    success=False,
                    details={"transcript": transcript, "error": str(e)}
                ))

    return results


def benchmark_e2e_api(iterations: int = 2) -> list[BenchmarkResult]:
    """Benchmark end-to-end via HTTP API."""
    print("\n=== E2E API Benchmark ===")

    if not check_service_health(BACKEND_URL, "/api/layer2/rag/industrial/health/"):
        print(f"Backend not available at {BACKEND_URL}")
        return []

    results = []

    for transcript in E2E_TEST_TRANSCRIPTS[:3]:  # Fewer for API test
        for i in range(iterations):
            try:
                t0 = time.perf_counter()
                resp = requests.post(
                    f"{BACKEND_URL}/api/layer2/orchestrate/",
                    json={"transcript": transcript},
                    timeout=120
                )
                latency = (time.perf_counter() - t0) * 1000

                success = resp.status_code == 200
                details = {
                    "transcript": transcript,
                    "iteration": i + 1,
                }

                if success:
                    data = resp.json()
                    details["processing_time_ms"] = data.get("processing_time_ms", 0)
                    details["intent_type"] = data.get("intent", {}).get("type")
                    print(f"  '{transcript[:30]}...' iter {i+1}: {latency:.0f}ms (server: {details['processing_time_ms']:.0f}ms)")
                else:
                    print(f"  '{transcript[:30]}...' iter {i+1}: FAILED ({resp.status_code})")

                results.append(BenchmarkResult(
                    name="e2e_api",
                    latency_ms=latency,
                    success=success,
                    details=details
                ))

            except Exception as e:
                print(f"  '{transcript[:30]}...' iter {i+1}: ERROR ({e})")
                results.append(BenchmarkResult(
                    name="e2e_api",
                    latency_ms=0,
                    success=False,
                    details={"error": str(e)}
                ))

    return results


# ============================================================
# Main Benchmark Runner
# ============================================================

def get_system_info() -> dict:
    """Collect system information."""
    import platform

    info = {
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
    }

    # Try to get GPU info
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except ImportError:
        pass

    return info


def run_full_benchmark(output_dir: str = None) -> FullBenchmarkReport:
    """Run all benchmarks and generate report."""
    print("=" * 60)
    print("COMMAND CENTER BENCHMARK SUITE")
    print("=" * 60)

    timestamp = datetime.now().isoformat()
    system_info = get_system_info()

    print(f"\nTimestamp: {timestamp}")
    print(f"System: {system_info}")

    # Collect all results
    all_results = {}

    # STT Benchmarks
    stt_results = benchmark_stt_server(iterations=3)
    all_results["stt"] = [r.to_dict() for r in stt_results]
    stt_summary = calculate_summary("STT Server", stt_results) if stt_results else None

    # RAG Benchmarks
    rag_accuracy_results = benchmark_rag_accuracy()
    rag_latency_results = benchmark_rag_latency(iterations=5)
    all_rag_results = rag_accuracy_results + rag_latency_results
    all_results["rag"] = [r.to_dict() for r in all_rag_results]
    rag_summary = calculate_summary("RAG Pipeline", all_rag_results) if all_rag_results else None

    # Add accuracy rate to RAG summary
    if rag_summary and rag_accuracy_results:
        accuracy_rate = sum(1 for r in rag_accuracy_results if r.success) / len(rag_accuracy_results)
        rag_summary.details = {"accuracy_rate": round(accuracy_rate, 3)}

    # TTS Benchmarks
    tts_results = benchmark_tts_server(iterations=2)
    all_results["tts"] = [r.to_dict() for r in tts_results]
    tts_summary = calculate_summary("TTS Server", tts_results) if tts_results else None

    # E2E Benchmarks
    e2e_orch_results = benchmark_e2e_orchestrator(iterations=2)
    e2e_api_results = benchmark_e2e_api(iterations=1)
    all_e2e_results = e2e_orch_results + e2e_api_results
    all_results["e2e"] = [r.to_dict() for r in all_e2e_results]
    e2e_summary = calculate_summary("E2E Pipeline", all_e2e_results) if all_e2e_results else None

    # Create report
    report = FullBenchmarkReport(
        timestamp=timestamp,
        system_info=system_info,
        stt_summary=stt_summary,
        rag_summary=rag_summary,
        tts_summary=tts_summary,
        e2e_summary=e2e_summary,
        raw_results=all_results,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    for name, summary in [
        ("STT", stt_summary),
        ("RAG", rag_summary),
        ("TTS", tts_summary),
        ("E2E", e2e_summary),
    ]:
        if summary:
            print(f"\n{name}:")
            print(f"  Success Rate: {summary.success_rate * 100:.1f}%")
            print(f"  Latency (mean): {summary.latency_mean_ms:.0f}ms")
            print(f"  Latency (p95):  {summary.latency_p95_ms:.0f}ms")
            print(f"  Latency (p99):  {summary.latency_p99_ms:.0f}ms")
            if summary.details:
                for k, v in summary.details.items():
                    print(f"  {k}: {v}")

    # Save report
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_file = output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(report_file, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    print(f"\nReport saved to: {report_file}")

    return report


if __name__ == "__main__":
    run_full_benchmark()
