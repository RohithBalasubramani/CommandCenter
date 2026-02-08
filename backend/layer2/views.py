import logging
import os
import uuid as _uuid

logger = logging.getLogger(__name__)

from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view, throttle_classes
from rest_framework.response import Response
from rest_framework.throttling import AnonRateThrottle
from dataclasses import asdict

_FEEDBACK_API_KEY = os.environ.get("FEEDBACK_API_KEY", "")


class FeedbackThrottle(AnonRateThrottle):
    rate = "20/minute"

from .models import RAGPipeline, RAGQuery, RAGResult
from .serializers import (
    RAGPipelineSerializer,
    RAGQuerySerializer,
    RAGResultSerializer,
)
from .orchestrator import get_orchestrator, Intent, RAGResult as OrchestratorRAGResult


class RAGPipelineViewSet(viewsets.ModelViewSet):
    """
    Layer 2 — RAG Pipeline registry.

    Endpoints:
      GET    /api/layer2/pipelines/           — List all pipelines
      POST   /api/layer2/pipelines/           — Register a new pipeline
      GET    /api/layer2/pipelines/{id}/       — Get pipeline details
      PATCH  /api/layer2/pipelines/{id}/       — Update pipeline (enable/disable)
      DELETE /api/layer2/pipelines/{id}/       — Remove pipeline
    """
    queryset = RAGPipeline.objects.all()
    serializer_class = RAGPipelineSerializer


class RAGQueryViewSet(viewsets.ModelViewSet):
    """
    Layer 2 — RAG query log.

    Endpoints:
      GET    /api/layer2/queries/             — List all queries
      POST   /api/layer2/queries/             — Log a new query
      GET    /api/layer2/queries/{id}/         — Get query with result
    """
    queryset = RAGQuery.objects.select_related("result").all()
    serializer_class = RAGQuerySerializer

    def get_queryset(self):
        qs = super().get_queryset()
        domain = self.request.query_params.get("domain")
        if domain:
            qs = qs.filter(pipeline__domain=domain)
        return qs


@api_view(["POST"])
def industrial_rag_query(request):
    """
    Layer 2B — Industrial RAG endpoint.

    Receives a query from the frontend orchestrator and returns
    domain-specific data from the industrial data sources.

    POST /api/layer2/rag/industrial/
    Body: { "query": "...", "context": {} }
    """
    query_text = request.data.get("query", "")
    # context = request.data.get("context", {})

    # STUB ENDPOINT — explicitly marked per F6 audit requirement.
    # safe_to_answer: false → AI MUST refuse to use this data.
    return Response(
        {
            "domain": "industrial",
            "raw_data": {
                "message": f"Industrial RAG stub for query: {query_text}",
                "metrics": [],
                "devices": [],
                "alerts": [],
            },
            "status": "stub",
            "reason": "integration_pending",
            "safe_to_answer": False,
            "_data_source": "api.industrial_rag",
            "_integration_status": "stub",
            "_authoritative": False,
        }
    )


@api_view(["GET"])
def industrial_rag_health(request):
    """
    Health check for Industrial RAG pipeline.

    Returns status and index statistics.
    """
    try:
        from .rag_pipeline import get_rag_pipeline

        pipeline = get_rag_pipeline()
        stats = pipeline.get_stats()

        return Response({
            "status": "ok",
            "domain": "industrial",
            "equipment_count": stats.get("equipment_count", 0),
            "alerts_count": stats.get("alerts_count", 0),
            "maintenance_count": stats.get("maintenance_count", 0),
            "llm_available": stats.get("llm_available", False),
            "llm_model": stats.get("llm_model", "unknown"),
        })
    except Exception as e:
        return Response({
            "status": "error",
            "domain": "industrial",
            "error": str(e),
            "equipment_count": 0,
            "alerts_count": 0,
            "maintenance_count": 0,
            "llm_available": False,
            "llm_model": "unknown",
        })


@api_view(["POST"])
def orchestrate(request):
    """
    Layer 2 Main Orchestration Endpoint.

    Receives transcript from Layer 1, processes through:
    - 2A: Intent parsing
    - 2B: Parallel RAG queries
    - 2C: Response generation + Layout decisions

    POST /api/layer2/orchestrate/
    Body: {
        "transcript": "What's the status of the pumps?",
        "session_id": "uuid",
        "context": {}
    }

    Returns: {
        "voice_response": "3 of 4 pumps running...",
        "filler_text": "Let me check that for you.",
        "layout_json": { "widgets": [...] },
        "context_update": {...},
        "intent": {...},
        "processing_time_ms": 150
    }
    """
    transcript = request.data.get("transcript", "")
    session_id = request.data.get("session_id")
    context = request.data.get("context", {})
    user_id = request.data.get("user_id", "default_user")

    if not transcript:
        return Response(
            {"error": "transcript is required"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Get orchestrator and process
    orchestrator = get_orchestrator()
    result = orchestrator.process_transcript(transcript, context, user_id)

    # Convert dataclasses to dicts for JSON serialization
    response_data = {
        "query_id": result.query_id,  # For RL feedback tracking
        "voice_response": result.voice_response,
        "filler_text": result.filler_text,
        "layout_json": result.layout_json,
        "context_update": result.context_update,
        "intent": {
            "type": result.intent.type,
            "domains": result.intent.domains,
            "entities": result.intent.entities,
            "confidence": result.intent.confidence,
        } if result.intent else None,
        "rag_results": [
            {
                "domain": r.domain,
                "success": r.success,
                "data": r.data,
                "error": r.error,
                "execution_time_ms": r.execution_time_ms,
            }
            for r in result.rag_results
        ],
        "processing_time_ms": result.processing_time_ms,
        # F2/F5: Mandatory provenance — derived_from, data sources, demo warnings
        "provenance": result.provenance,
    }

    return Response(response_data)


@api_view(["POST"])
def get_filler(request):
    """
    Get filler text for Layer 1 to speak while waiting.

    POST /api/layer2/filler/
    Body: { "intent_type": "query", "domains": ["industrial"] }
    """
    intent_type = request.data.get("intent_type", "query")
    domains = request.data.get("domains", ["industrial"])

    orchestrator = get_orchestrator()

    # Create a minimal intent for filler generation
    intent = Intent(
        type=intent_type,
        domains=domains,
        entities={},
        confidence=0.5,
        raw_text="",
    )

    filler = orchestrator._generate_filler(intent)

    return Response({"filler_text": filler})


@api_view(["POST"])
def proactive_trigger(request):
    """
    Get proactive question/trigger for Layer 1.

    Called when system context changes (alerts, shift start, etc.)
    Layer 1 can use this to initiate conversation proactively.

    POST /api/layer2/proactive/
    Body: {
        "system_context": {
            "active_alerts": 2,
            "shift_start": true,
            "anomalies": [...]
        }
    }
    """
    system_context = request.data.get("system_context", {})

    orchestrator = get_orchestrator()
    trigger = orchestrator.get_proactive_trigger(system_context)

    return Response({
        "has_trigger": trigger is not None,
        "trigger_text": trigger,
    })


@api_view(["POST"])
@throttle_classes([FeedbackThrottle])
def submit_feedback(request):
    """
    Submit feedback for continuous RL.

    Called by frontend when user provides explicit feedback.

    POST /api/layer2/feedback/
    Body: {
        "query_id": "uuid-from-response",
        "rating": "up" | "down",
        "interactions": [
            {"widget_index": 0, "action": "expand", "duration_ms": 5000},
            ...
        ],
        "correction": "I meant pump-002"  # Optional correction text
    }
    """
    # Lightweight API key check (skip if FEEDBACK_API_KEY not configured)
    if _FEEDBACK_API_KEY:
        provided_key = request.headers.get("X-Feedback-Key", "")
        if provided_key != _FEEDBACK_API_KEY:
            return Response(
                {"error": "Invalid or missing X-Feedback-Key header"},
                status=status.HTTP_403_FORBIDDEN,
            )

    query_id = request.data.get("query_id")
    rating = request.data.get("rating")
    interactions = request.data.get("interactions", [])
    correction = request.data.get("correction")

    # Extract rich evaluation fields from Claude Sonnet 4.5 auto-evaluator
    evaluation_confidence = request.data.get("evaluation_confidence")
    evaluation_reasoning = request.data.get("evaluation_reasoning")
    query_understanding = request.data.get("query_understanding")
    per_widget_feedback = request.data.get("per_widget_feedback", [])
    missing_widgets = request.data.get("missing_widgets", [])
    suggested_improvements = request.data.get("suggested_improvements", [])

    if not query_id:
        return Response(
            {"error": "query_id is required"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Validate query_id is a UUID
    try:
        _uuid.UUID(query_id)
    except (ValueError, TypeError):
        return Response(
            {"error": "query_id must be a valid UUID"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Validate rating
    if rating is not None and rating not in ("up", "down"):
        return Response(
            {"error": "rating must be 'up' or 'down'"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Validate correction length
    if correction and (not isinstance(correction, str) or len(correction) > 500):
        return Response(
            {"error": "correction must be a string with max 500 characters"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Validate interactions
    if not isinstance(interactions, list) or len(interactions) > 50:
        return Response(
            {"error": "interactions must be a list with max 50 items"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        from rl.continuous import get_rl_system

        rl = get_rl_system()
        success = rl.update_feedback(
            query_id=query_id,
            rating=rating,
            interactions=interactions,
            correction=correction,
            # Rich evaluation fields
            evaluation_confidence=evaluation_confidence,
            evaluation_reasoning=evaluation_reasoning,
            query_understanding=query_understanding,
            per_widget_feedback=per_widget_feedback,
            missing_widgets=missing_widgets,
            suggested_improvements=suggested_improvements,
        )

        if success:
            # Also persist to WidgetRating for durability
            try:
                from django.utils import timezone
                from feedback.models import WidgetRating

                WidgetRating.objects.create(
                    entry_id=query_id,
                    rating=rating or "up",
                    tags=[],
                    notes=correction or "",
                    rated_at=timezone.now(),
                )
            except Exception as e:
                # Non-critical - RL buffer already updated
                pass

            return Response({"status": "ok", "updated": True})
        else:
            return Response(
                {"status": "not_found", "updated": False},
                status=status.HTTP_404_NOT_FOUND,
            )

    except ImportError:
        return Response(
            {"error": "RL module not available"},
            status=status.HTTP_501_NOT_IMPLEMENTED,
        )
    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["POST"])
def approve_lora_training(request):
    """
    Admin endpoint to approve pending Tier 2 LoRA training.

    POST /api/layer2/approve-training/
    """
    from rl.config import TRAINING_DATA_DIR

    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    approval_file = TRAINING_DATA_DIR / "approve_lora_training"
    approval_file.touch()

    return Response({"status": "approved", "file": str(approval_file)})


@api_view(["GET"])
def rl_status(request):
    """
    Get status of the continuous RL system.

    GET /api/layer2/rl-status/
    """
    try:
        from rl.continuous import get_rl_system

        rl = get_rl_system()
        return Response(rl.get_status())

    except ImportError:
        return Response({
            "running": False,
            "error": "RL module not available",
        })


@api_view(["GET"])
def rl_history(request):
    """
    Aggregate historical RL training data from persisted files.

    GET /api/layer2/rl-history/
    Returns reward timelines, feedback distribution, latency buckets,
    training loss curves, and evaluation summaries.
    """
    import json as _json
    from pathlib import Path
    from collections import Counter

    from rl.config import PROJECT_DIR, TRAINING_DATA_DIR, CHECKPOINTS_DIR
    from rl.continuous import get_rl_system

    limit = min(int(request.query_params.get("limit", 500)), 2000)

    result = {
        "reward_timeline": [],
        "feedback_distribution": {"up": 0, "down": 0, "none": 0},
        "latency_buckets": [],
        "intent_distribution": {},
        "scenario_frequency": {},
        "processing_time_trend": [],
        "training_loss_curve": [],
        "evaluation_summary": {},
        "query_details": [],
    }

    # ── 1. Experience buffer (read from live in-memory buffer) ────
    try:
        rl = get_rl_system()
        live_experiences = rl.buffer.get_recent(limit)
        experiences = [exp.to_dict() for exp in live_experiences]
    except Exception:
        # Fallback to disk if RL system not available
        experiences = []
        buffer_path = TRAINING_DATA_DIR / "experience_buffer.json"
        if buffer_path.exists():
            try:
                raw = _json.loads(buffer_path.read_text())
                experiences = raw.get("experiences", [])[-limit:]
            except Exception:
                pass

    if experiences:
        try:
            ratings = Counter()
            intents = Counter()
            scenarios = Counter()
            lat_buckets = [0, 0, 0, 0, 0]  # <1s, 1-3s, 3-6s, 6-10s, >10s

            for exp in experiences:
                ts = exp.get("timestamp", "")
                reward = exp.get("computed_reward")
                rating = exp.get("user_rating")  # "up" / "down" / None
                ms = exp.get("processing_time_ms", 0)

                # Reward timeline
                if reward is not None:
                    result["reward_timeline"].append({
                        "timestamp": ts,
                        "reward": round(reward, 4),
                        "rating": rating or "none",
                    })

                # Feedback distribution
                ratings[rating or "none"] += 1

                # Latency buckets
                s = ms / 1000.0
                if s < 1:
                    lat_buckets[0] += 1
                elif s < 3:
                    lat_buckets[1] += 1
                elif s < 6:
                    lat_buckets[2] += 1
                elif s < 10:
                    lat_buckets[3] += 1
                else:
                    lat_buckets[4] += 1

                # Processing time trend (sample every 5th to keep payload small)
                if len(result["processing_time_trend"]) < 200:
                    result["processing_time_trend"].append({
                        "timestamp": ts,
                        "ms": ms,
                    })

                # Intent distribution
                intent_type = (exp.get("parsed_intent") or {}).get("type", "unknown")
                intents[intent_type] += 1

                # Scenario frequency
                widgets = ((exp.get("widget_plan") or {}).get("widgets") or [])
                for w in widgets:
                    sc = w.get("scenario", "unknown")
                    scenarios[sc] += 1

            result["feedback_distribution"] = {
                "up": ratings.get("up", 0),
                "down": ratings.get("down", 0),
                "none": ratings.get("none", 0) + ratings.get(None, 0),
            }
            result["latency_buckets"] = [
                {"range": "<1s", "count": lat_buckets[0]},
                {"range": "1-3s", "count": lat_buckets[1]},
                {"range": "3-6s", "count": lat_buckets[2]},
                {"range": "6-10s", "count": lat_buckets[3]},
                {"range": ">10s", "count": lat_buckets[4]},
            ]
            result["intent_distribution"] = dict(intents.most_common(10))
            result["scenario_frequency"] = dict(scenarios.most_common(15))

        except Exception as e:
            logger.warning(f"rl_history: failed to read experience buffer: {e}")

    # ── 2. Trainer state (loss curves) ────────────────────────────
    try:
        log_entries = []
        for version_dir in sorted(CHECKPOINTS_DIR.glob("lora_v*")):
            for ckpt in sorted(version_dir.glob("checkpoint-*/trainer_state.json")):
                try:
                    state = _json.loads(ckpt.read_text())
                    log_entries.extend(state.get("log_history", []))
                except _json.JSONDecodeError:
                    logger.debug(f"rl_history: skipping corrupt {ckpt}")

        # Deduplicate by step, keep latest
        seen = set()
        unique = []
        for entry in log_entries:
            s = entry.get("step")
            if s is not None and s not in seen:
                seen.add(s)
                unique.append({
                    "step": s,
                    "loss": entry.get("loss"),
                    "accuracy": entry.get("rewards/accuracies"),
                    "margins": entry.get("rewards/margins"),
                    "grad_norm": entry.get("grad_norm"),
                    "lr": entry.get("learning_rate"),
                })
        result["training_loss_curve"] = sorted(unique, key=lambda e: e["step"])
    except Exception as e:
        logger.warning(f"rl_history: failed to read trainer state: {e}")

    # ── 3. Evaluation report ──────────────────────────────────────
    eval_path = (
        PROJECT_DIR / "ai rl agent" / "ai-rl-agent" / "evidence" / "rl"
        / "api-evaluation-report.json"
    )
    if eval_path.exists():
        try:
            eval_data = _json.loads(eval_path.read_text())
            evals = eval_data.get("evaluations", [])
            if evals:
                n = len(evals)
                result["evaluation_summary"] = {
                    "count": n,
                    "avg_overall": round(sum(e["overallScore"] for e in evals) / n, 4),
                    "avg_scenario_relevance": round(
                        sum(e.get("scenarioRelevance", 0) for e in evals) / n, 4
                    ),
                    "avg_data_accuracy": round(
                        sum(e.get("dataAccuracy", 0) for e in evals) / n, 4
                    ),
                    "avg_response_quality": round(
                        sum(e.get("responseQuality", 0) for e in evals) / n, 4
                    ),
                    "avg_latency_score": round(
                        sum(e.get("latencyScore", 0) for e in evals) / n, 4
                    ),
                    "scores": [
                        {
                            "score": round(e["overallScore"], 4),
                            "query": e.get("query", "")[:60],
                        }
                        for e in evals
                    ],
                }
        except Exception as e:
            logger.warning(f"rl_history: failed to read evaluation report: {e}")

    # ── 4. Per-query details (merge experience + evaluation) ──────
    try:
        # Build evaluation lookup by query text
        eval_by_query = {}
        eval_path2 = (
            PROJECT_DIR / "ai rl agent" / "ai-rl-agent" / "evidence" / "rl"
            / "api-evaluation-report.json"
        )
        if eval_path2.exists():
            try:
                eval_data2 = _json.loads(eval_path2.read_text())
                for ev in eval_data2.get("evaluations", []):
                    q = ev.get("query", "").strip()
                    if q:
                        eval_by_query[q] = ev
            except Exception:
                pass

        # Read experiences for detail rows (reuse in-memory buffer)
        experiences2 = experiences  # Already fetched from live buffer above

        if experiences2:
            # Compute aggregates for relative comparison
            all_times = [
                e.get("processing_time_ms", 0)
                for e in experiences2 if e.get("processing_time_ms", 0) > 0
            ]
            avg_ms = sum(all_times) / len(all_times) if all_times else 5000
            all_widgets = [
                len((e.get("widget_plan") or {}).get("widgets") or [])
                for e in experiences2
            ]
            avg_widgets = sum(all_widgets) / len(all_widgets) if all_widgets else 5

            # Count how many experiences share each characteristic
            char_counts = Counter()
            for e in experiences2:
                pc = (e.get("parsed_intent") or {}).get("primary_characteristic", "")
                if pc:
                    char_counts[pc] += 1

            # Scorer + DPO info from live state
            try:
                from rl.continuous import get_rl_manager
                mgr = get_rl_manager()
                scorer_steps = mgr.scorer.training_steps if mgr and mgr.scorer else 0
                dpo_pairs = mgr.trainer.total_pairs if mgr and mgr.trainer else 0
            except Exception:
                scorer_steps = 0
                dpo_pairs = 0

            result["query_aggregates"] = {
                "avg_processing_ms": round(avg_ms, 1),
                "avg_widget_count": round(avg_widgets, 2),
                "total_experiences": len(experiences2),
                "scorer_steps": scorer_steps,
                "dpo_pairs_generated": dpo_pairs,
                "characteristic_counts": dict(char_counts.most_common(20)),
            }

            for idx, exp in enumerate(experiences2):
                transcript = exp.get("transcript", "")
                wp = exp.get("widget_plan") or {}
                widgets = wp.get("widgets") or []
                pi = exp.get("parsed_intent") or {}
                ms = exp.get("processing_time_ms", 0)

                # Match evaluation data by query text
                ev = eval_by_query.get(transcript.strip(), {})

                scenarios = [w.get("scenario", "?") for w in widgets]
                pc = pi.get("primary_characteristic", "")

                # Compute implicit reward signal components
                latency_signal = "fast" if ms < avg_ms * 0.7 else (
                    "slow" if ms > avg_ms * 1.5 else "normal"
                )
                widget_signal = "rich" if len(widgets) > avg_widgets + 1 else (
                    "sparse" if len(widgets) < max(1, avg_widgets - 2) else "normal"
                )
                # How common is this query type — rarer types provide more
                # learning diversity
                type_frequency = char_counts.get(pc, 0)
                diversity_signal = (
                    "rare" if type_frequency <= 3 else
                    "uncommon" if type_frequency <= 10 else "common"
                )

                # Determine feedback source
                user_rating = exp.get("user_rating")
                has_eval = ev.get("overallScore") is not None
                if user_rating and has_eval:
                    feedback_source = "both"
                elif user_rating:
                    feedback_source = "user_direct"
                elif has_eval:
                    feedback_source = "eval_agent"
                else:
                    feedback_source = "implicit_only"

                # Detect query clarity from correction text
                correction = exp.get("correction_text") or ""
                query_clarity = "clear"
                if correction:
                    cl = correction.lower()
                    ambiguity_markers = [
                        "specifically", "particular", "specific",
                        "not all", "just the", "only the", "exact",
                        "i wanted", "i meant", "i was asking",
                        "wrong one", "the one", "certain",
                    ]
                    if any(m in cl for m in ambiguity_markers):
                        query_clarity = "ambiguous_query"
                    else:
                        query_clarity = "system_mismatch"

                detail = {
                    "query_id": exp.get("query_id", ""),
                    "timestamp": exp.get("timestamp", ""),
                    "query": transcript[:120],
                    "rating": user_rating,  # up/down/None
                    "reward": exp.get("computed_reward"),
                    "processing_time_ms": ms,
                    "widget_count": len(widgets),
                    "scenarios": scenarios[:8],
                    "intent_type": pi.get("type", "unknown"),
                    "domains": pi.get("domains", []),
                    "primary_characteristic": pc,
                    "confidence": pi.get("confidence"),
                    "select_method": wp.get("select_method", ""),
                    "heading": wp.get("heading", "")[:80],
                    # Evaluation dimensions (from RL agent report)
                    "eval_overall": ev.get("overallScore"),
                    "eval_relevance": ev.get("scenarioRelevance"),
                    "eval_accuracy": ev.get("dataAccuracy"),
                    "eval_quality": ev.get("responseQuality"),
                    "eval_latency": ev.get("latencyScore"),
                    "eval_rating": ev.get("rating"),  # up/down from eval
                    "correction": correction or None,
                    # Feedback context
                    "feedback_source": feedback_source,
                    "query_clarity": query_clarity,
                    # RL contribution signals
                    "latency_signal": latency_signal,
                    "widget_signal": widget_signal,
                    "diversity_signal": diversity_signal,
                    "buffer_position": idx + 1,
                }
                result["query_details"].append(detail)

    except Exception as e:
        logger.warning(f"rl_history: failed to build query_details: {e}")

    return Response(result)


# ── In-memory trigger queue for webhook & role_change triggers ──────────────
import threading

class _TriggerStore:
    """Thread-safe store for external triggers (webhook, role_change).

    Triggers are pushed via POST /api/layer2/triggers/webhook/ and drained
    on each GET /api/layer2/triggers/ poll.  Max 100 per kind to prevent
    unbounded growth.
    """
    _MAX_PER_KIND = 100

    def __init__(self):
        self._lock = threading.Lock()
        self._queues: dict[str, list[dict]] = {}

    def push(self, kind: str, trigger: dict):
        with self._lock:
            q = self._queues.setdefault(kind, [])
            q.append(trigger)
            if len(q) > self._MAX_PER_KIND:
                q[:] = q[-self._MAX_PER_KIND:]

    def drain(self, kind: str) -> list[dict]:
        with self._lock:
            items = self._queues.pop(kind, [])
        return items

_webhook_trigger_store = _TriggerStore()


@api_view(["GET"])
def system_triggers(request):
    """
    Poll for system triggers (per README blueprint):
      alert_fired, threshold_breach, scheduled_event, role_change, time_of_day, webhook

    GET /api/layer2/triggers/
    Query params:
      since — ISO timestamp (only triggers after this time)

    Returns:
      { "triggers": [...], "timestamp": "..." }
    """
    import time as _time
    from django.utils import timezone

    since_str = request.query_params.get("since")
    now = timezone.now()
    triggers = []

    # --- alert_fired: check for recent unacknowledged alerts ---
    try:
        from industrial.models import Alert
        alert_qs = Alert.objects.filter(acknowledged=False).order_by("-timestamp")
        if since_str:
            from django.utils.dateparse import parse_datetime
            since_dt = parse_datetime(since_str)
            if since_dt:
                alert_qs = alert_qs.filter(timestamp__gt=since_dt)

        for alert in alert_qs[:5]:
            triggers.append({
                "kind": "alert_fired",
                "source": str(alert.source) if hasattr(alert, "source") else alert.device_name if hasattr(alert, "device_name") else "unknown",
                "message": alert.message if hasattr(alert, "message") else str(alert),
                "timestamp": alert.timestamp.isoformat() if hasattr(alert, "timestamp") else now.isoformat(),
                "payload": {
                    "severity": alert.severity if hasattr(alert, "severity") else "unknown",
                    "alert_id": str(alert.pk),
                },
            })
    except Exception:
        pass  # industrial app may not be migrated

    # --- threshold_breach: check for readings above thresholds ---
    try:
        from industrial.models import SensorReading
        breach_qs = SensorReading.objects.filter(
            is_anomaly=True
        ).order_by("-timestamp")[:3]
        for reading in breach_qs:
            triggers.append({
                "kind": "threshold_breach",
                "source": reading.device_name if hasattr(reading, "device_name") else "sensor",
                "message": f"Threshold breach: {reading.value} {reading.unit}" if hasattr(reading, "value") else "Threshold breach detected",
                "timestamp": reading.timestamp.isoformat() if hasattr(reading, "timestamp") else now.isoformat(),
                "payload": {"reading_id": str(reading.pk)},
            })
    except Exception:
        pass

    # --- scheduled_event: upcoming maintenance within next 24h ---
    try:
        from industrial.models import MaintenanceRecord
        upcoming_qs = MaintenanceRecord.objects.filter(
            scheduled_date__gte=now,
            scheduled_date__lte=now + timezone.timedelta(hours=24),
            completed_at__isnull=True,
        ).order_by("scheduled_date")[:3]
        for maint in upcoming_qs:
            triggers.append({
                "kind": "scheduled_event",
                "source": maint.equipment_name if hasattr(maint, "equipment_name") else "maintenance",
                "message": f"Upcoming: {maint.maintenance_type} maintenance for {maint.equipment_name}",
                "timestamp": maint.scheduled_date.isoformat() if maint.scheduled_date else now.isoformat(),
                "payload": {
                    "record_id": str(maint.pk),
                    "maintenance_type": maint.maintenance_type,
                    "equipment": maint.equipment_name,
                },
            })
    except Exception:
        pass  # industrial app may not have MaintenanceRecord table

    # --- role_change: drain queued role-change triggers ---
    for rc in _webhook_trigger_store.drain("role_change"):
        triggers.append(rc)

    # --- time_of_day: shift-based triggers ---
    hour = now.hour
    if hour == 6:
        triggers.append({
            "kind": "time_of_day",
            "source": "system",
            "message": "Morning shift starting — loading shift overview",
            "timestamp": now.isoformat(),
            "payload": {"shift": "morning"},
        })
    elif hour == 14:
        triggers.append({
            "kind": "time_of_day",
            "source": "system",
            "message": "Afternoon shift starting — loading shift handover",
            "timestamp": now.isoformat(),
            "payload": {"shift": "afternoon"},
        })
    elif hour == 22:
        triggers.append({
            "kind": "time_of_day",
            "source": "system",
            "message": "Night shift starting — loading night monitoring",
            "timestamp": now.isoformat(),
            "payload": {"shift": "night"},
        })

    # --- webhook: drain queued external webhook triggers ---
    for wh in _webhook_trigger_store.drain("webhook"):
        triggers.append(wh)

    return Response({
        "triggers": triggers,
        "timestamp": now.isoformat(),
    })


@api_view(["POST"])
def webhook_trigger(request):
    """
    Accept external webhook triggers (per README blueprint).

    POST /api/layer2/triggers/webhook/
    Body: {
        "kind": "webhook" | "role_change",
        "source": "external-system-name",
        "message": "Human-readable description",
        "payload": { ... optional extra data ... }
    }

    Triggers are queued and returned on the next GET /api/layer2/triggers/ poll.
    """
    from django.utils import timezone

    kind = request.data.get("kind", "webhook")
    if kind not in ("webhook", "role_change"):
        return Response({"error": f"Invalid trigger kind: {kind}. Must be 'webhook' or 'role_change'."}, status=400)

    source = request.data.get("source", "external")
    message = request.data.get("message", "External trigger received")
    payload = request.data.get("payload", {})

    now = timezone.now()
    trigger = {
        "kind": kind,
        "source": source,
        "message": message,
        "timestamp": now.isoformat(),
        "payload": payload,
    }
    _webhook_trigger_store.push(kind, trigger)

    return Response({"status": "queued", "trigger": trigger}, status=201)


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM GROUNDING API (Phase 1-5 Audit)
# ═══════════════════════════════════════════════════════════════════════════════


@api_view(["GET"])
def system_registry_view(request):
    """
    Get the complete system registry — all data sources, schemas, domains.

    GET /api/layer2/grounding/registry/
    """
    from layer2.system_registry import get_system_registry
    registry = get_system_registry()
    return Response(registry.to_dict())


@api_view(["GET"])
def schema_introspect_view(request):
    """
    Run live schema introspection — actual table counts, column info.

    GET /api/layer2/grounding/introspect/
    """
    from layer2.schema_introspector import SchemaIntrospector
    introspector = SchemaIntrospector()
    result = introspector.introspect_all()
    return Response(result)


@api_view(["GET"])
def find_table_view(request):
    """
    Answer: "Which table contains this data?"

    GET /api/layer2/grounding/find-table/?q=transformer+load
    """
    query = request.query_params.get("q", "")
    if not query:
        return Response(
            {"error": "q parameter is required"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    from layer2.schema_introspector import SchemaIntrospector
    introspector = SchemaIntrospector()
    matches = introspector.find_table_for_data(query)
    return Response({"query": query, "matches": matches})


@api_view(["GET"])
def grounding_audit_view(request):
    """
    Get recent grounding audit entries and defect summary.

    GET /api/layer2/grounding/audit/
    """
    from layer2.grounding_audit import get_grounding_auditor
    auditor = get_grounding_auditor()
    return Response({
        "recent_entries": auditor.get_recent_entries(limit=20),
        "defect_summary": auditor.get_defect_summary(),
    })


@api_view(["POST"])
def traversal_action_view(request):
    """
    Execute an explicit traversal action.

    POST /api/layer2/grounding/traverse/
    Body: {
        "action": "list_databases" | "describe_table" | "preview_rows" |
                  "check_entity_exists" | "get_metric_reading" | "get_alert_state",
        "args": { ... action-specific arguments ... }
    }
    """
    from layer2.traversal import TraversalEngine

    action_name = request.data.get("action", "")
    args = request.data.get("args", {})

    engine = TraversalEngine()

    action_map = {
        "list_databases": lambda: engine.list_databases(),
        "describe_table": lambda: engine.describe_table(args.get("table_name", "")),
        "preview_rows": lambda: engine.preview_rows(
            args.get("table_name", ""), args.get("limit", 5)
        ),
        "check_entity_exists": lambda: engine.check_entity_exists(
            args.get("entity_name", "")
        ),
        "get_metric_reading": lambda: engine.get_metric_reading(
            args.get("entity_name", ""), args.get("metric", "")
        ),
        "get_alert_state": lambda: engine.get_alert_state(
            args.get("entity_name")
        ),
    }

    if action_name not in action_map:
        return Response(
            {"error": f"Unknown action: {action_name}", "valid_actions": list(action_map.keys())},
            status=status.HTTP_400_BAD_REQUEST,
        )

    step = action_map[action_name]()
    return Response({
        "action": step.action,
        "args": step.args,
        "success": step.success,
        "result": step.result,
        "source_id": step.source_id,
        "duration_ms": step.duration_ms,
        "error": step.error,
    })
