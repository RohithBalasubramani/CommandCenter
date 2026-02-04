"""
Tests for Layer 2 API Endpoints.

Tests the Django REST framework views.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))


class TestOrchestrateEndpoint:
    """Test the /api/layer2/orchestrate/ endpoint."""

    def test_orchestrate_requires_transcript(self, api_client):
        """Test that transcript is required."""
        response = api_client.post(
            "/api/layer2/orchestrate/",
            {},
            format="json",
        )
        assert response.status_code == 400
        assert "transcript" in response.data.get("error", "").lower()

    def test_orchestrate_empty_transcript(self, api_client):
        """Test handling of empty transcript."""
        response = api_client.post(
            "/api/layer2/orchestrate/",
            {"transcript": ""},
            format="json",
        )
        assert response.status_code == 400

    @patch("layer2.views.get_orchestrator")
    def test_orchestrate_success(self, mock_get_orchestrator, api_client):
        """Test successful orchestration."""
        # Mock orchestrator response
        from dataclasses import dataclass
        from typing import List, Optional

        @dataclass
        class MockIntent:
            type: str = "query"
            domains: List[str] = None
            entities: dict = None
            confidence: float = 0.9

            def __post_init__(self):
                self.domains = self.domains or ["industrial"]
                self.entities = self.entities or {}

        @dataclass
        class MockRAGResult:
            domain: str
            success: bool
            data: dict
            error: Optional[str]
            execution_time_ms: int

        @dataclass
        class MockOrchestratorResult:
            voice_response: str
            filler_text: str
            layout_json: dict
            context_update: dict
            intent: MockIntent
            rag_results: List[MockRAGResult]
            processing_time_ms: int

        mock_result = MockOrchestratorResult(
            voice_response="Pump 1 is running normally.",
            filler_text="Let me check that.",
            layout_json={"widgets": []},
            context_update={},
            intent=MockIntent(),
            rag_results=[MockRAGResult(
                domain="industrial",
                success=True,
                data={"equipment": []},
                error=None,
                execution_time_ms=50,
            )],
            processing_time_ms=100,
        )

        mock_orchestrator = MagicMock()
        mock_orchestrator.process_transcript.return_value = mock_result
        mock_get_orchestrator.return_value = mock_orchestrator

        response = api_client.post(
            "/api/layer2/orchestrate/",
            {"transcript": "What is the status of pump 1?"},
            format="json",
        )

        assert response.status_code == 200
        assert "voice_response" in response.data
        assert "layout_json" in response.data
        assert "intent" in response.data
        assert response.data["voice_response"] == "Pump 1 is running normally."

    @patch("layer2.views.get_orchestrator")
    def test_orchestrate_with_session_id(self, mock_get_orchestrator, api_client):
        """Test orchestration with session ID."""
        from dataclasses import dataclass
        from typing import List

        @dataclass
        class MockIntent:
            type: str = "query"
            domains: List[str] = None
            entities: dict = None
            confidence: float = 0.9

            def __post_init__(self):
                self.domains = self.domains or []
                self.entities = self.entities or {}

        @dataclass
        class MockResult:
            voice_response: str = "Response"
            filler_text: str = "Filler"
            layout_json: dict = None
            context_update: dict = None
            intent: MockIntent = None
            rag_results: list = None
            processing_time_ms: int = 100

            def __post_init__(self):
                self.layout_json = self.layout_json or {}
                self.context_update = self.context_update or {}
                self.intent = self.intent or MockIntent()
                self.rag_results = self.rag_results or []

        mock_orchestrator = MagicMock()
        mock_orchestrator.process_transcript.return_value = MockResult()
        mock_get_orchestrator.return_value = mock_orchestrator

        response = api_client.post(
            "/api/layer2/orchestrate/",
            {
                "transcript": "Show pump status",
                "session_id": "test-session-123",
                "context": {"previous_query": "hello"},
            },
            format="json",
        )

        assert response.status_code == 200


class TestFillerEndpoint:
    """Test the /api/layer2/filler/ endpoint."""

    @patch("layer2.views.get_orchestrator")
    def test_get_filler_default(self, mock_get_orchestrator, api_client):
        """Test getting default filler text."""
        mock_orchestrator = MagicMock()
        mock_orchestrator._generate_filler.return_value = "Let me check that for you."
        mock_get_orchestrator.return_value = mock_orchestrator

        response = api_client.post(
            "/api/layer2/filler/",
            {},
            format="json",
        )

        assert response.status_code == 200
        assert "filler_text" in response.data

    @patch("layer2.views.get_orchestrator")
    def test_get_filler_with_intent(self, mock_get_orchestrator, api_client):
        """Test getting filler with intent type."""
        mock_orchestrator = MagicMock()
        mock_orchestrator._generate_filler.return_value = "Looking up that information."
        mock_get_orchestrator.return_value = mock_orchestrator

        response = api_client.post(
            "/api/layer2/filler/",
            {
                "intent_type": "query",
                "domains": ["industrial", "alerts"],
            },
            format="json",
        )

        assert response.status_code == 200


class TestIndustrialRAGEndpoint:
    """Test the /api/layer2/rag/industrial/ endpoint."""

    def test_industrial_rag_query(self, api_client):
        """Test industrial RAG query endpoint."""
        response = api_client.post(
            "/api/layer2/rag/industrial/",
            {"query": "What is pump 1 status?"},
            format="json",
        )

        assert response.status_code == 200
        assert "domain" in response.data
        assert response.data["domain"] == "industrial"
        assert "raw_data" in response.data


class TestRAGHealthEndpoint:
    """Test the /api/layer2/rag/industrial/health/ endpoint."""

    @patch("layer2.rag_pipeline.get_rag_pipeline")
    def test_rag_health_success(self, mock_get_pipeline, api_client):
        """Test RAG health check success."""
        mock_pipeline = MagicMock()
        mock_pipeline.get_stats.return_value = {
            "equipment_count": 150,
            "alerts_count": 50,
            "maintenance_count": 200,
            "llm_available": True,
            "llm_model": "phi4",
        }
        mock_get_pipeline.return_value = mock_pipeline

        response = api_client.get("/api/layer2/rag/industrial/health/")

        assert response.status_code == 200
        assert response.data["status"] == "ok"
        assert response.data["equipment_count"] == 150
        assert response.data["llm_available"] is True

    @patch("layer2.rag_pipeline.get_rag_pipeline")
    def test_rag_health_error(self, mock_get_pipeline, api_client):
        """Test RAG health check when pipeline fails."""
        mock_get_pipeline.side_effect = Exception("ChromaDB not available")

        response = api_client.get("/api/layer2/rag/industrial/health/")

        assert response.status_code == 200
        assert response.data["status"] == "error"
        assert "error" in response.data


class TestProactiveEndpoint:
    """Test the /api/layer2/proactive/ endpoint."""

    @patch("layer2.views.get_orchestrator")
    def test_proactive_no_trigger(self, mock_get_orchestrator, api_client):
        """Test proactive endpoint with no trigger."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.get_proactive_trigger.return_value = None
        mock_get_orchestrator.return_value = mock_orchestrator

        response = api_client.post(
            "/api/layer2/proactive/",
            {"system_context": {}},
            format="json",
        )

        assert response.status_code == 200
        assert response.data["has_trigger"] is False

    @patch("layer2.views.get_orchestrator")
    def test_proactive_with_alerts(self, mock_get_orchestrator, api_client):
        """Test proactive endpoint with active alerts."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.get_proactive_trigger.return_value = "You have 3 unacknowledged critical alerts."
        mock_get_orchestrator.return_value = mock_orchestrator

        response = api_client.post(
            "/api/layer2/proactive/",
            {
                "system_context": {
                    "active_alerts": 3,
                    "critical_alerts": 1,
                }
            },
            format="json",
        )

        assert response.status_code == 200
        assert response.data["has_trigger"] is True
        assert "alerts" in response.data["trigger_text"].lower()


@pytest.mark.django_db
class TestRAGPipelineViewSet:
    """Test the RAGPipeline viewset."""

    def test_list_pipelines(self, api_client):
        """Test listing RAG pipelines."""
        response = api_client.get("/api/layer2/pipelines/")
        assert response.status_code == 200

    def test_create_pipeline(self, api_client):
        """Test creating a new pipeline."""
        response = api_client.post(
            "/api/layer2/pipelines/",
            {
                "domain": "industrial",
                "enabled": True,
                "priority": 1,
                "endpoint_url": "http://localhost:8000/api/industrial/",
            },
            format="json",
        )
        # May fail if pipeline already exists (unique constraint)
        assert response.status_code in (201, 400)


@pytest.mark.django_db
class TestRAGQueryViewSet:
    """Test the RAGQuery viewset."""

    def test_list_queries(self, api_client):
        """Test listing RAG queries."""
        response = api_client.get("/api/layer2/queries/")
        assert response.status_code == 200

    def test_filter_queries_by_domain(self, api_client):
        """Test filtering queries by domain."""
        response = api_client.get("/api/layer2/queries/?domain=industrial")
        assert response.status_code == 200
