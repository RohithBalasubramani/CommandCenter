"""
Pipeline Executor — Parallel execution of orchestrator stages

Maximizes throughput by running independent stages concurrently:
1. Intent parsing (fast model)
2. Widget selection (fast model)
3. Data fetching (multiple sources in parallel)
4. Voice response generation (quality model)

The executor manages model utilization by:
- Running multiple intents in parallel during batch processing
- Prefetching data while LLM is generating
- Overlapping quality model generation with widget data loading
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor
import os

from .parallel_llm import (
    get_parallel_llm,
    Priority,
    generate_fast,
    generate_quality,
    generate_json_fast,
    process_rl_batch,
)

logger = logging.getLogger(__name__)

# Configuration
MAX_PARALLEL_QUERIES = int(os.getenv("MAX_PARALLEL_QUERIES", "4"))
PREFETCH_ENABLED = os.getenv("PREFETCH_ENABLED", "true").lower() == "true"


@dataclass
class PipelineStage:
    """Represents a stage in the pipeline."""
    name: str
    duration_ms: int
    result: Any = None
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    query_id: str
    total_duration_ms: int
    stages: list[PipelineStage]
    parallel_efficiency: float  # How much time was saved by parallelization


class PipelineExecutor:
    """
    Executes orchestrator pipeline stages with maximum parallelism.

    Key optimizations:
    1. Run intent parsing and data prefetch concurrently
    2. Batch multiple user queries together when possible
    3. Use fast model for parsing, quality model for responses
    4. Overlap I/O with LLM computation
    """

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=MAX_PARALLEL_QUERIES * 2)
        self._active_queries = 0
        self._total_queries = 0
        self._total_time_saved_ms = 0

    async def execute_parallel(
        self,
        transcript: str,
        context: dict,
        intent_parser,
        widget_selector,
        data_collector,
        response_generator,
    ) -> dict:
        """
        Execute the full pipeline with parallel stages.

        Args:
            transcript: User query
            context: Conversation context
            intent_parser: IntentParser instance
            widget_selector: WidgetSelector instance
            data_collector: DataCollector/prefetcher instance
            response_generator: VoiceResponseGenerator instance

        Returns:
            Complete orchestrator result
        """
        start_time = time.time()
        stages: list[PipelineStage] = []

        # Stage 1 & 2: Intent parsing + Data prefetch (PARALLEL)
        t1 = time.time()

        intent_task = asyncio.create_task(
            self._parse_intent_async(intent_parser, transcript, context)
        )

        # Start prefetching likely data while parsing
        prefetch_task = None
        if PREFETCH_ENABLED and data_collector:
            prefetch_task = asyncio.create_task(
                self._prefetch_data_async(data_collector, transcript)
            )

        # Wait for intent (required for next stage)
        intent = await intent_task
        stages.append(PipelineStage(
            name="intent_parsing",
            duration_ms=int((time.time() - t1) * 1000),
            result=intent,
        ))

        # Stage 3: Widget selection (uses intent)
        t2 = time.time()
        widgets = await self._select_widgets_async(
            widget_selector, intent, context
        )
        stages.append(PipelineStage(
            name="widget_selection",
            duration_ms=int((time.time() - t2) * 1000),
            result=widgets,
        ))

        # Stage 4 & 5: Data collection + Response generation (PARALLEL)
        t3 = time.time()

        # Get prefetched data or fetch now
        if prefetch_task:
            prefetched_data = await prefetch_task
        else:
            prefetched_data = {}

        # Fetch remaining data needed for widgets
        data_task = asyncio.create_task(
            self._collect_widget_data_async(
                data_collector, widgets, prefetched_data
            )
        )

        # Start generating voice response (can run while data loads)
        response_task = asyncio.create_task(
            self._generate_response_async(
                response_generator, transcript, intent, widgets
            )
        )

        # Wait for both
        widget_data, voice_response = await asyncio.gather(
            data_task, response_task
        )

        stages.append(PipelineStage(
            name="data_collection",
            duration_ms=int((time.time() - t3) * 1000),
            result=widget_data,
        ))
        stages.append(PipelineStage(
            name="response_generation",
            duration_ms=int((time.time() - t3) * 1000),
            result=voice_response,
        ))

        # Calculate efficiency
        total_ms = int((time.time() - start_time) * 1000)
        sequential_ms = sum(s.duration_ms for s in stages)
        time_saved = max(0, sequential_ms - total_ms)
        efficiency = time_saved / max(sequential_ms, 1)

        self._total_queries += 1
        self._total_time_saved_ms += time_saved

        logger.info(
            f"Pipeline completed: {total_ms}ms (saved {time_saved}ms, "
            f"efficiency={efficiency:.1%})"
        )

        return {
            "intent": intent,
            "widgets": widgets,
            "widget_data": widget_data,
            "voice_response": voice_response,
            "pipeline_stats": {
                "total_ms": total_ms,
                "time_saved_ms": time_saved,
                "efficiency": efficiency,
                "stages": [
                    {"name": s.name, "ms": s.duration_ms}
                    for s in stages
                ],
            },
        }

    async def _parse_intent_async(self, intent_parser, transcript: str, context: dict) -> dict:
        """Parse intent using parallel LLM."""
        if hasattr(intent_parser, "parse_async"):
            return await intent_parser.parse_async(transcript, context)

        # Fallback to sync in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            intent_parser.parse,
            transcript,
            context,
        )

    async def _select_widgets_async(self, widget_selector, intent: dict, context: dict) -> dict:
        """Select widgets using parallel LLM."""
        if hasattr(widget_selector, "select_async"):
            return await widget_selector.select_async(intent, context)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            widget_selector.select,
            intent,
            context,
        )

    async def _prefetch_data_async(self, data_collector, transcript: str) -> dict:
        """Prefetch likely data based on transcript keywords."""
        if hasattr(data_collector, "prefetch_async"):
            return await data_collector.prefetch_async(transcript)

        loop = asyncio.get_event_loop()
        if hasattr(data_collector, "prefetch"):
            return await loop.run_in_executor(
                self._executor,
                data_collector.prefetch,
                transcript,
            )
        return {}

    async def _collect_widget_data_async(
        self, data_collector, widgets: dict, prefetched: dict
    ) -> dict:
        """Collect data for all widgets in parallel."""
        if not widgets or not data_collector:
            return prefetched

        widget_list = widgets.get("widgets", [])
        if not widget_list:
            return prefetched

        # Fetch data for each widget in parallel
        async def fetch_widget_data(widget):
            scenario = widget.get("scenario", "")
            if scenario in prefetched:
                return scenario, prefetched[scenario]

            if hasattr(data_collector, "collect_async"):
                data = await data_collector.collect_async(scenario)
            else:
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(
                    self._executor,
                    data_collector.collect,
                    scenario,
                )
            return scenario, data

        tasks = [fetch_widget_data(w) for w in widget_list[:10]]  # Limit
        results = await asyncio.gather(*tasks, return_exceptions=True)

        data = dict(prefetched)
        for result in results:
            if isinstance(result, tuple):
                scenario, widget_data = result
                data[scenario] = widget_data

        return data

    async def _generate_response_async(
        self, response_generator, transcript: str, intent: dict, widgets: dict
    ) -> str:
        """Generate voice response using quality model."""
        if hasattr(response_generator, "generate_async"):
            return await response_generator.generate_async(
                transcript, intent, widgets
            )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            response_generator.generate,
            transcript,
            intent,
            widgets,
        )

    def get_stats(self) -> dict:
        """Get executor statistics."""
        return {
            "total_queries": self._total_queries,
            "total_time_saved_ms": self._total_time_saved_ms,
            "avg_time_saved_ms": (
                self._total_time_saved_ms / max(self._total_queries, 1)
            ),
            "prefetch_enabled": PREFETCH_ENABLED,
            "max_parallel": MAX_PARALLEL_QUERIES,
        }


# ============================================================
# Batch Query Processor
# ============================================================

class BatchQueryProcessor:
    """
    Process multiple queries together for maximum throughput.

    Useful for:
    - Processing voice commands that come in quick succession
    - Batch testing and evaluation
    - RL training data generation
    """

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self._batch_queue = asyncio.Queue()
        self._running = False

    async def process_batch(
        self,
        transcripts: list[str],
        session_id: str = "batch",
    ) -> list[dict]:
        """
        Process multiple transcripts in parallel.

        Args:
            transcripts: List of user queries
            session_id: Session ID for context

        Returns:
            List of orchestrator results
        """
        start_time = time.time()

        # Process all in parallel
        tasks = [
            self._process_single(transcript, f"{session_id}-{i}")
            for i, transcript in enumerate(transcripts)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle errors
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch query {i} failed: {result}")
                processed_results.append({
                    "error": str(result),
                    "transcript": transcripts[i],
                })
            else:
                processed_results.append(result)

        total_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"Batch processed {len(transcripts)} queries in {total_ms}ms "
            f"({total_ms / len(transcripts):.0f}ms/query)"
        )

        return processed_results

    async def _process_single(self, transcript: str, session_id: str) -> dict:
        """Process a single query."""
        if hasattr(self.orchestrator, "process_async"):
            return await self.orchestrator.process_async(
                transcript, session_id
            )

        # Fallback to sync
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.orchestrator.process_transcript,
            transcript,
            session_id,
        )


# ============================================================
# Singleton instances
# ============================================================

_pipeline_executor: Optional[PipelineExecutor] = None
_batch_processor: Optional[BatchQueryProcessor] = None


def get_pipeline_executor() -> PipelineExecutor:
    """Get the global pipeline executor."""
    global _pipeline_executor
    if _pipeline_executor is None:
        _pipeline_executor = PipelineExecutor()
    return _pipeline_executor


def get_batch_processor(orchestrator) -> BatchQueryProcessor:
    """Get batch processor for an orchestrator."""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchQueryProcessor(orchestrator)
    return _batch_processor
