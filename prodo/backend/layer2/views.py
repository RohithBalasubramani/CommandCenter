from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view
from rest_framework.response import Response
from dataclasses import asdict

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

    # Stub: In production, this would query LoggerDeploy / PostgreSQL
    # for metrics, devices, device_status tables
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

    if not transcript:
        return Response(
            {"error": "transcript is required"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Get orchestrator and process
    orchestrator = get_orchestrator()
    result = orchestrator.process_transcript(transcript, context)

    # Convert dataclasses to dicts for JSON serialization
    response_data = {
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
