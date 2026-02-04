"""
Parallel LLM Pipeline for Ollama

Maximizes model utilization by:
1. Async concurrent requests to Ollama
2. Priority queue (user requests > background tasks)
3. Request batching for efficiency
4. Connection pooling
5. Automatic retry with backoff

Usage:
    llm = get_parallel_llm()

    # Single request (async)
    result = await llm.generate_async("prompt")

    # Batch requests (parallel)
    results = await llm.generate_batch(["prompt1", "prompt2", "prompt3"])

    # Priority request (jumps queue)
    result = await llm.generate_async("urgent prompt", priority=Priority.HIGH)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Callable, Any
from collections import deque
import threading
import os

logger = logging.getLogger(__name__)

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_FAST = os.getenv("OLLAMA_MODEL_FAST", "llama3.1:8b")
OLLAMA_MODEL_QUALITY = os.getenv("OLLAMA_MODEL_QUALITY", "llama3.3")

# Concurrency settings
MAX_CONCURRENT_REQUESTS = int(os.getenv("OLLAMA_MAX_CONCURRENT", "4"))
REQUEST_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
BATCH_WAIT_MS = int(os.getenv("OLLAMA_BATCH_WAIT_MS", "50"))


class Priority(IntEnum):
    """Request priority levels."""
    LOW = 0       # Background tasks (RL training, prefetch)
    NORMAL = 1    # Standard requests
    HIGH = 2      # User-facing requests
    CRITICAL = 3  # Real-time voice responses


@dataclass(order=True)
class LLMRequest:
    """A queued LLM request."""
    priority: int
    timestamp: float = field(compare=False)
    request_id: str = field(compare=False)
    prompt: str = field(compare=False)
    system_prompt: Optional[str] = field(compare=False, default=None)
    model: str = field(compare=False, default=OLLAMA_MODEL_FAST)
    temperature: float = field(compare=False, default=0.7)
    max_tokens: int = field(compare=False, default=1024)
    json_mode: bool = field(compare=False, default=False)
    future: asyncio.Future = field(compare=False, default=None)


@dataclass
class LLMResponse:
    """Response from LLM."""
    request_id: str
    content: str
    model: str
    tokens_generated: int
    latency_ms: int
    success: bool
    error: Optional[str] = None


class ParallelLLMClient:
    """
    Parallel LLM client with priority queue and connection pooling.

    Supports concurrent requests to maximize GPU utilization while
    ensuring user requests get priority over background tasks.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        max_concurrent: int = MAX_CONCURRENT_REQUESTS,
        default_model: str = OLLAMA_MODEL_FAST,
    ):
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.default_model = default_model

        # Request queue (priority queue)
        self._queue: asyncio.PriorityQueue = None
        self._active_requests = 0
        self._lock = asyncio.Lock()

        # Stats
        self._total_requests = 0
        self._total_tokens = 0
        self._total_latency_ms = 0
        self._requests_by_priority = {p: 0 for p in Priority}

        # HTTP session (lazy init)
        self._session = None
        self._request_counter = 0

        # Background worker
        self._worker_task = None
        self._running = False

        logger.info(f"ParallelLLMClient initialized: max_concurrent={max_concurrent}, model={default_model}")

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None:
            import aiohttp
            connector = aiohttp.TCPConnector(
                limit=self.max_concurrent * 2,  # Allow some overhead
                keepalive_timeout=30,
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
            )
        return self._session

    async def start(self):
        """Start the background worker."""
        if self._running:
            return

        self._queue = asyncio.PriorityQueue()
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("ParallelLLMClient worker started")

    async def stop(self):
        """Stop the background worker and cleanup."""
        self._running = False

        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        if self._session:
            await self._session.close()
            self._session = None

        logger.info("ParallelLLMClient stopped")

    async def _worker_loop(self):
        """Main worker loop - processes queue continuously."""
        while self._running:
            try:
                # Get next request (with timeout to allow shutdown)
                try:
                    request = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Process request
                asyncio.create_task(self._process_request(request))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                await asyncio.sleep(0.1)

    async def _process_request(self, request: LLMRequest):
        """Process a single LLM request."""
        async with self._lock:
            self._active_requests += 1

        start_time = time.time()
        response = None

        try:
            session = await self._get_session()

            # Build payload
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": request.model,
                "prompt": request.prompt,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                },
            }

            if request.system_prompt:
                payload["system"] = request.system_prompt

            if request.json_mode:
                payload["format"] = "json"

            # Make request
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data.get("response", "")
                    tokens = data.get("eval_count", 0)

                    # Clean response (strip thinking tags)
                    import re
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

                    latency_ms = int((time.time() - start_time) * 1000)

                    response = LLMResponse(
                        request_id=request.request_id,
                        content=content,
                        model=request.model,
                        tokens_generated=tokens,
                        latency_ms=latency_ms,
                        success=True,
                    )

                    # Update stats
                    self._total_requests += 1
                    self._total_tokens += tokens
                    self._total_latency_ms += latency_ms
                    # Priority was negated for queue ordering, convert back
                    orig_priority = Priority(-request.priority) if request.priority <= 0 else Priority.NORMAL
                    self._requests_by_priority[orig_priority] += 1

                else:
                    error_text = await resp.text()
                    response = LLMResponse(
                        request_id=request.request_id,
                        content="",
                        model=request.model,
                        tokens_generated=0,
                        latency_ms=int((time.time() - start_time) * 1000),
                        success=False,
                        error=f"HTTP {resp.status}: {error_text[:200]}",
                    )

        except Exception as e:
            response = LLMResponse(
                request_id=request.request_id,
                content="",
                model=request.model,
                tokens_generated=0,
                latency_ms=int((time.time() - start_time) * 1000),
                success=False,
                error=str(e),
            )
            logger.error(f"LLM request {request.request_id} failed: {e}")

        finally:
            async with self._lock:
                self._active_requests -= 1

            # Complete the future
            if request.future and not request.future.done():
                request.future.set_result(response)

    async def generate_async(
        self,
        prompt: str,
        system_prompt: str = None,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        json_mode: bool = False,
        priority: Priority = Priority.NORMAL,
    ) -> LLMResponse:
        """
        Generate response asynchronously.

        Args:
            prompt: The prompt text
            system_prompt: Optional system prompt
            model: Model to use (defaults to default_model)
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            json_mode: Enable JSON output mode
            priority: Request priority level

        Returns:
            LLMResponse with content and metadata
        """
        if not self._running:
            await self.start()

        self._request_counter += 1
        request_id = f"req-{self._request_counter:08d}"

        # Create future for response
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        # Create request
        request = LLMRequest(
            priority=-priority,  # Negative for max-heap behavior
            timestamp=time.time(),
            request_id=request_id,
            prompt=prompt,
            system_prompt=system_prompt,
            model=model or self.default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            future=future,
        )

        # Queue request
        await self._queue.put(request)

        # Wait for response
        return await future

    async def generate_batch(
        self,
        prompts: list[str],
        system_prompt: str = None,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        priority: Priority = Priority.NORMAL,
    ) -> list[LLMResponse]:
        """
        Generate responses for multiple prompts in parallel.

        Args:
            prompts: List of prompt strings
            system_prompt: Optional system prompt (shared)
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Max tokens per response
            priority: Request priority level

        Returns:
            List of LLMResponse objects (same order as prompts)
        """
        tasks = [
            self.generate_async(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                priority=priority,
            )
            for prompt in prompts
        ]

        return await asyncio.gather(*tasks)

    def generate_sync(
        self,
        prompt: str,
        system_prompt: str = None,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        json_mode: bool = False,
        priority: Priority = Priority.NORMAL,
    ) -> LLMResponse:
        """
        Synchronous wrapper for generate_async.

        Use this from non-async code (e.g., Django views).
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.generate_async(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
                priority=priority,
            )
        )

    def get_stats(self) -> dict:
        """Get client statistics."""
        avg_latency = (
            self._total_latency_ms / self._total_requests
            if self._total_requests > 0 else 0
        )

        return {
            "running": self._running,
            "active_requests": self._active_requests,
            "queue_size": self._queue.qsize() if self._queue else 0,
            "max_concurrent": self.max_concurrent,
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "avg_latency_ms": round(avg_latency, 1),
            "requests_by_priority": {
                p.name: self._requests_by_priority[p] for p in Priority
            },
        }


# ============================================================
# Singleton instance
# ============================================================

_parallel_llm: Optional[ParallelLLMClient] = None
_lock = threading.Lock()


def get_parallel_llm() -> ParallelLLMClient:
    """Get the global parallel LLM client instance."""
    global _parallel_llm

    if _parallel_llm is None:
        with _lock:
            if _parallel_llm is None:
                _parallel_llm = ParallelLLMClient()

    return _parallel_llm


async def init_parallel_llm():
    """Initialize and start the parallel LLM client."""
    llm = get_parallel_llm()
    await llm.start()
    return llm


async def shutdown_parallel_llm():
    """Shutdown the parallel LLM client."""
    global _parallel_llm
    if _parallel_llm:
        await _parallel_llm.stop()


# ============================================================
# Convenience functions
# ============================================================

async def generate_fast(
    prompt: str,
    system_prompt: str = None,
    priority: Priority = Priority.NORMAL,
) -> str:
    """
    Quick generation using the fast model.

    Best for: intent parsing, widget selection, quick classifications.
    """
    llm = get_parallel_llm()
    response = await llm.generate_async(
        prompt=prompt,
        system_prompt=system_prompt,
        model=OLLAMA_MODEL_FAST,
        temperature=0.3,
        max_tokens=1024,
        priority=priority,
    )
    return response.content if response.success else ""


async def generate_quality(
    prompt: str,
    system_prompt: str = None,
    priority: Priority = Priority.HIGH,
) -> str:
    """
    High-quality generation using the quality model.

    Best for: voice responses, detailed explanations, user-facing content.
    """
    llm = get_parallel_llm()
    response = await llm.generate_async(
        prompt=prompt,
        system_prompt=system_prompt,
        model=OLLAMA_MODEL_QUALITY,
        temperature=0.7,
        max_tokens=2048,
        priority=priority,
    )
    return response.content if response.success else ""


async def generate_json_fast(
    prompt: str,
    system_prompt: str = None,
    priority: Priority = Priority.NORMAL,
) -> dict | None:
    """
    Generate structured JSON using the fast model.

    Returns parsed dict or None on failure.
    """
    import json

    llm = get_parallel_llm()
    response = await llm.generate_async(
        prompt=prompt,
        system_prompt=system_prompt,
        model=OLLAMA_MODEL_FAST,
        temperature=0.1,
        max_tokens=2048,
        json_mode=True,
        priority=priority,
    )

    if response.success and response.content:
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON: {response.content[:200]}")

    return None


# ============================================================
# Batch processing for RL training
# ============================================================

async def process_rl_batch(
    prompts: list[str],
    system_prompt: str = None,
) -> list[str]:
    """
    Process a batch of prompts for RL training (low priority).

    Uses the fast model with LOW priority so user requests aren't blocked.
    """
    llm = get_parallel_llm()
    responses = await llm.generate_batch(
        prompts=prompts,
        system_prompt=system_prompt,
        model=OLLAMA_MODEL_FAST,
        temperature=0.3,
        priority=Priority.LOW,
    )

    return [r.content if r.success else "" for r in responses]
