"""
Pytest fixtures for reconciliation tests.
"""
import pytest
import sys
import os
from pathlib import Path

# Setup Django
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "command_center.settings")

import django
django.setup()

from layer2.reconciliation.types import WidgetSchema, FieldSchema
from layer2.reconciliation.resolver import MockLLMCaller
from layer2.reconciliation.audit import MemoryAuditSink, get_audit_manager
from layer2.reconciliation.validator_integration import reset_validation_flag
import json


@pytest.fixture
def kpi_schema():
    """Sample KPI widget schema."""
    return WidgetSchema(
        scenario="kpi",
        fields=[
            FieldSchema(name="label", type="string", required=True),
            FieldSchema(name="value", type="number", required=True, unit_dimension="power"),
            FieldSchema(name="unit", type="string", required=True),
            FieldSchema(name="state", type="string", required=False, valid_values=["normal", "warning", "critical"]),
        ],
        required_fields=["label", "value", "unit"],
    )


@pytest.fixture
def comparison_schema():
    """Sample comparison widget schema."""
    return WidgetSchema(
        scenario="comparison",
        fields=[
            FieldSchema(name="label", type="string", required=True),
            FieldSchema(name="valueA", type="number", required=True),
            FieldSchema(name="valueB", type="number", required=True),
            FieldSchema(name="unit", type="string", required=True),
            FieldSchema(name="labelA", type="string", required=False),
            FieldSchema(name="labelB", type="string", required=False),
        ],
        required_fields=["label", "valueA", "valueB", "unit"],
    )


@pytest.fixture
def mock_llm():
    """Mock LLM caller with predefined responses."""
    mock = MockLLMCaller()

    # Add common responses
    mock.add_response(
        "500 kW",
        json.dumps({
            "value": 500,
            "unit": "kW",
            "metric_id": None,
            "frame": None,
            "assumptions": [],
            "confidence": 0.95,
            "reasoning": "Extracted numeric value and unit",
        }),
    )

    mock.add_response(
        "42",
        json.dumps({
            "value": 42,
            "unit": None,
            "metric_id": None,
            "frame": None,
            "assumptions": [],
            "confidence": 0.98,
            "reasoning": "Direct string-to-int conversion",
        }),
    )

    return mock


@pytest.fixture
def memory_audit():
    """Memory-based audit sink for testing."""
    sink = MemoryAuditSink()
    manager = get_audit_manager()
    manager.add_sink(sink)
    yield sink
    manager.remove_sink(sink)
    sink.clear()


@pytest.fixture(autouse=True)
def reset_validation():
    """Reset validation flag before each test."""
    reset_validation_flag()


@pytest.fixture
def valid_kpi_data():
    """Valid KPI data that should pass validation."""
    return {
        "demoData": {
            "label": "Power Output",
            "value": 500,
            "unit": "kW",
            "state": "normal",
        }
    }


@pytest.fixture
def invalid_kpi_data_string_value():
    """KPI data with string value (fixable)."""
    return {
        "demoData": {
            "label": "Power Output",
            "value": "500",  # String instead of number
            "unit": "kW",
        }
    }


@pytest.fixture
def ambiguous_kpi_data():
    """KPI data with unit in value (needs resolution)."""
    return {
        "demoData": {
            "label": "Power Output",
            "value": "500 kW",  # Value with unit mixed in
            "unit": "",
        }
    }


@pytest.fixture
def semantic_conflict_data():
    """Data with semantic conflict (cannot reconcile)."""
    return {
        "demoData": {
            "label": "Power Output",
            "value": 500,
            "unit": "kW",
            # This would need metric_id verification
        },
        "metadata": {
            "metric_id": "installed_capacity",  # vs expected "actual_output"
            "frame": "monthly",  # vs expected "instant"
        }
    }


@pytest.fixture
def xss_injection_data():
    """Data with XSS injection attempt."""
    return {
        "demoData": {
            "label": "<script>alert('xss')</script>",
            "value": 500,
            "unit": "kW",
        }
    }


@pytest.fixture
def sql_injection_data():
    """Data with SQL injection attempt."""
    return {
        "demoData": {
            "label": "Power'; DROP TABLE users; --",
            "value": 500,
            "unit": "kW",
        }
    }
