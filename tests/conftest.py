"""
Pytest configuration and fixtures for Command Center tests.
"""
import os
import sys
from pathlib import Path

# Add backend to path for imports
BACKEND_DIR = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# Set Django settings before any Django imports
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "command_center.settings")

import django
django.setup()

import pytest
from unittest.mock import MagicMock, patch


# ============================================================
# Django Fixtures
# ============================================================

@pytest.fixture
def api_client():
    """Django REST framework test client."""
    from rest_framework.test import APIClient
    return APIClient()


@pytest.fixture
def sample_transcript():
    """Sample voice transcript for testing."""
    return "What is the status of pump 1?"


@pytest.fixture
def sample_industrial_query():
    """Sample industrial domain query."""
    return "Show me the current load on transformer TR-001"


@pytest.fixture
def sample_greeting():
    """Sample greeting transcript."""
    return "Hello, good morning"


@pytest.fixture
def sample_out_of_scope():
    """Sample out-of-scope query."""
    return "What is the capital of France?"


# ============================================================
# Mock Fixtures
# ============================================================

@pytest.fixture
def mock_ollama():
    """Mock Ollama LLM service."""
    with patch("layer2.rag_pipeline.requests") as mock_requests:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Test LLM response"
        }
        mock_requests.post.return_value = mock_response
        mock_requests.get.return_value = mock_response
        yield mock_requests


@pytest.fixture
def mock_chromadb():
    """Mock ChromaDB for tests without vector store."""
    with patch("layer2.rag_pipeline.chromadb") as mock_chroma:
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 100
        mock_collection.query.return_value = {
            "ids": [["eq_1", "eq_2"]],
            "documents": [["Doc 1 content", "Doc 2 content"]],
            "metadatas": [[{"equipment_type": "pump"}, {"equipment_type": "motor"}]],
            "distances": [[0.1, 0.2]],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.PersistentClient.return_value = mock_client
        yield mock_chroma


@pytest.fixture
def mock_embedding_model():
    """Mock sentence transformer embedding model."""
    with patch("layer2.rag_pipeline.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        # Return a fake 768-dim embedding
        import numpy as np
        mock_model.encode.return_value = np.random.rand(768)
        mock_st.return_value = mock_model
        yield mock_st


# ============================================================
# Test Data Fixtures
# ============================================================

@pytest.fixture
def sample_equipment_data():
    """Sample equipment data for testing."""
    return {
        "equipment_id": "PUMP-001",
        "name": "Chilled Water Pump 1",
        "location": "Plant Room A",
        "building": "Building 1",
        "status": "running",
        "health_score": 95,
        "criticality": "high",
        "description": "Primary chilled water circulation pump",
    }


@pytest.fixture
def sample_alert_data():
    """Sample alert data for testing."""
    return {
        "equipment_id": "PUMP-001",
        "equipment_name": "Chilled Water Pump 1",
        "severity": "warning",
        "alert_type": "threshold",
        "message": "Temperature exceeded threshold",
        "parameter": "temperature",
        "value": 85.5,
        "unit": "C",
        "threshold": 80.0,
    }


@pytest.fixture
def sample_intent_queries():
    """Collection of sample queries for intent parsing tests."""
    return {
        "industrial_query": "What is the status of pump 1?",
        "energy_query": "Show me the power consumption for today",
        "comparison_query": "Compare transformer load vs yesterday",
        "trend_query": "Show the temperature trend over the last week",
        "alert_query": "Are there any critical alerts?",
        "maintenance_query": "When was the last maintenance on chiller 2?",
        "action_control": "Turn off pump 3",
        "action_reminder": "Remind me to check motor 5 in 2 hours",
        "greeting": "Hello, good morning",
        "conversation": "Thank you for the help",
        "out_of_scope": "What is the weather like?",
        "hvac_query": "What is the AHU 1 supply temperature?",
        "ups_query": "What is the UPS battery status?",
    }
