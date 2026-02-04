"""
Tests for Django Models.

Tests both Layer 2 models and Industrial equipment models.
"""
import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))


class TestRAGPipelineModel:
    """Test the RAGPipeline model."""

    @pytest.mark.django_db
    def test_create_pipeline(self):
        """Test creating a RAG pipeline."""
        from layer2.models import RAGPipeline

        pipeline = RAGPipeline.objects.create(
            domain="industrial",
            enabled=True,
            priority=1,
            endpoint_url="http://localhost:8000/api/industrial/",
        )

        assert pipeline.id is not None
        assert pipeline.domain == "industrial"
        assert pipeline.enabled is True

    @pytest.mark.django_db
    def test_pipeline_str_representation(self):
        """Test pipeline string representation."""
        from layer2.models import RAGPipeline

        pipeline = RAGPipeline.objects.create(
            domain="alerts",
            enabled=True,
            priority=2,
            endpoint_url="http://localhost:8000/api/alerts/",
        )

        assert "alerts" in str(pipeline)
        assert "enabled" in str(pipeline)

    @pytest.mark.django_db
    def test_pipeline_domain_choices(self):
        """Test that domain choices are valid."""
        from layer2.models import RAGPipeline

        valid_domains = ["industrial", "supply", "people", "tasks", "alerts"]
        for domain in valid_domains:
            # Clean up any existing pipeline with this domain
            RAGPipeline.objects.filter(domain=domain).delete()

            pipeline = RAGPipeline.objects.create(
                domain=domain,
                enabled=True,
                priority=1,
                endpoint_url=f"http://localhost:8000/api/{domain}/",
            )
            assert pipeline.domain == domain

    @pytest.mark.django_db
    def test_pipeline_ordering(self):
        """Test that pipelines are ordered by priority."""
        from layer2.models import RAGPipeline

        # Clear existing
        RAGPipeline.objects.all().delete()

        RAGPipeline.objects.create(
            domain="alerts", enabled=True, priority=3,
            endpoint_url="http://localhost/alerts",
        )
        RAGPipeline.objects.create(
            domain="industrial", enabled=True, priority=1,
            endpoint_url="http://localhost/industrial",
        )
        RAGPipeline.objects.create(
            domain="supply", enabled=True, priority=2,
            endpoint_url="http://localhost/supply",
        )

        pipelines = list(RAGPipeline.objects.all())
        assert pipelines[0].priority <= pipelines[1].priority


class TestRAGQueryModel:
    """Test the RAGQuery model."""

    @pytest.mark.django_db
    def test_create_query(self):
        """Test creating a RAG query."""
        from layer2.models import RAGPipeline, RAGQuery
        import uuid

        # Create pipeline first
        pipeline = RAGPipeline.objects.create(
            domain="industrial",
            enabled=True,
            priority=1,
            endpoint_url="http://localhost/api",
        )

        query = RAGQuery.objects.create(
            pipeline=pipeline,
            transcript_id=uuid.uuid4(),
            query_text="What is the pump status?",
            intent=["query", "industrial"],
        )

        assert query.id is not None
        assert "pump" in query.query_text

    @pytest.mark.django_db
    def test_query_str_representation(self):
        """Test query string representation."""
        from layer2.models import RAGPipeline, RAGQuery
        import uuid

        pipeline = RAGPipeline.objects.create(
            domain="industrial",
            enabled=True,
            priority=1,
            endpoint_url="http://localhost/api",
        )

        query = RAGQuery.objects.create(
            pipeline=pipeline,
            transcript_id=uuid.uuid4(),
            query_text="What is the transformer load?",
            intent=[],
        )

        assert "industrial" in str(query)
        assert "transformer" in str(query)


class TestRAGResultModel:
    """Test the RAGResult model."""

    @pytest.mark.django_db
    def test_create_result(self):
        """Test creating a RAG result."""
        from layer2.models import RAGPipeline, RAGQuery, RAGResult
        import uuid

        pipeline = RAGPipeline.objects.create(
            domain="industrial",
            enabled=True,
            priority=1,
            endpoint_url="http://localhost/api",
        )

        query = RAGQuery.objects.create(
            pipeline=pipeline,
            transcript_id=uuid.uuid4(),
            query_text="Test query",
            intent=[],
        )

        result = RAGResult.objects.create(
            query=query,
            raw_data={"equipment": [], "alerts": []},
            execution_time_ms=150,
        )

        assert result.id is not None
        assert result.execution_time_ms == 150
        assert result.error is None

    @pytest.mark.django_db
    def test_result_with_error(self):
        """Test creating a result with error."""
        from layer2.models import RAGPipeline, RAGQuery, RAGResult
        import uuid

        pipeline = RAGPipeline.objects.create(
            domain="industrial",
            enabled=True,
            priority=1,
            endpoint_url="http://localhost/api",
        )

        query = RAGQuery.objects.create(
            pipeline=pipeline,
            transcript_id=uuid.uuid4(),
            query_text="Test query",
            intent=[],
        )

        result = RAGResult.objects.create(
            query=query,
            raw_data={},
            error="Connection timeout",
            execution_time_ms=5000,
        )

        assert result.error == "Connection timeout"
        assert "error" in str(result)


class TestUserMemoryModel:
    """Test the UserMemory model."""

    @pytest.mark.django_db
    def test_create_user_memory(self):
        """Test creating a user memory entry."""
        from layer2.models import UserMemory

        memory = UserMemory.objects.create(
            user_id="test_user_001",
            query="Show me pump status",
            primary_characteristic="health_status",
            domains=["industrial"],
            entities_mentioned=["pump_1"],
            scenarios_used=["kpi", "alerts"],
        )

        assert memory.id is not None
        assert memory.user_id == "test_user_001"
        assert "industrial" in memory.domains

    @pytest.mark.django_db
    def test_user_memory_ordering(self):
        """Test that memories are ordered by created_at descending."""
        from layer2.models import UserMemory
        import time

        UserMemory.objects.filter(user_id="test_order").delete()

        UserMemory.objects.create(
            user_id="test_order",
            query="First query",
        )
        time.sleep(0.01)  # Small delay to ensure different timestamps
        UserMemory.objects.create(
            user_id="test_order",
            query="Second query",
        )

        memories = list(UserMemory.objects.filter(user_id="test_order"))
        if len(memories) >= 2:
            # Most recent should be first
            assert memories[0].query == "Second query"


class TestIndustrialModels:
    """Test industrial equipment models."""

    @pytest.mark.django_db
    def test_create_transformer(self):
        """Test creating a transformer."""
        from industrial.models import Transformer

        transformer = Transformer.objects.create(
            equipment_id="TR-001",
            name="Main Distribution Transformer",
            description="1000 kVA oil-filled transformer",
            location="Substation A",
            building="Building 1",
            status="running",
            criticality="critical",
            health_score=95,
            transformer_type="distribution",
            capacity_kva=1000,
            primary_voltage=11000,
            secondary_voltage=433,
            vector_group="Dyn11",
            load_percent=75.5,
        )

        assert transformer.id is not None
        assert transformer.capacity_kva == 1000
        assert transformer.load_percent == 75.5

    @pytest.mark.django_db
    def test_create_pump(self):
        """Test creating a pump."""
        from industrial.models import Pump

        pump = Pump.objects.create(
            equipment_id="PUMP-001",
            name="Chilled Water Pump 1",
            description="Primary circulation pump",
            location="Plant Room A",
            status="running",
            criticality="high",
            health_score=90,
            pump_type="chw_primary",
            flow_rate=100,
            head=30,
            motor_kw=15,
            flow_rate_actual=85.5,
        )

        assert pump.id is not None
        assert pump.flow_rate_actual == 85.5

    @pytest.mark.django_db
    def test_create_alert(self):
        """Test creating an alert."""
        from industrial.models import Alert

        alert = Alert.objects.create(
            equipment_id="PUMP-001",
            equipment_name="Chilled Water Pump 1",
            severity="warning",
            alert_type="threshold",
            message="Temperature exceeded threshold",
            parameter="temperature",
            value=85.5,
            unit="C",
            threshold=80.0,
        )

        assert alert.id is not None
        assert alert.severity == "warning"
        assert alert.acknowledged is False
        assert alert.resolved is False

    @pytest.mark.django_db
    def test_equipment_status_choices(self):
        """Test equipment status choices."""
        from industrial.models import BaseEquipment

        valid_statuses = ["running", "stopped", "maintenance", "fault", "standby", "offline"]
        assert all(
            status in [choice[0] for choice in BaseEquipment.Status.choices]
            for status in valid_statuses
        )

    @pytest.mark.django_db
    def test_equipment_criticality_choices(self):
        """Test equipment criticality choices."""
        from industrial.models import BaseEquipment

        valid_criticalities = ["critical", "high", "medium", "low"]
        assert all(
            crit in [choice[0] for choice in BaseEquipment.Criticality.choices]
            for crit in valid_criticalities
        )

    @pytest.mark.django_db
    def test_create_diesel_generator(self):
        """Test creating a diesel generator."""
        from industrial.models import DieselGenerator

        dg = DieselGenerator.objects.create(
            equipment_id="DG-001",
            name="Standby Generator 1",
            description="500 kVA standby generator",
            location="Generator Room",
            status="standby",
            criticality="critical",
            health_score=98,
            capacity_kva=500,
            capacity_kw=400,
            voltage=433,
            frequency=50,
            power_factor=0.8,
            fuel_level_percent=85.0,
        )

        assert dg.id is not None
        assert dg.fuel_level_percent == 85.0

    @pytest.mark.django_db
    def test_create_chiller(self):
        """Test creating a chiller."""
        from industrial.models import Chiller

        chiller = Chiller.objects.create(
            equipment_id="CH-001",
            name="Chiller 1",
            description="500 TR water-cooled chiller",
            location="Chiller Plant",
            status="running",
            criticality="high",
            health_score=92,
            chiller_type="water_cooled",
            capacity_tr=500,
            capacity_kw=1758,  # 500 TR ≈ 1758 kW
            cop_rating=5.5,
            chilled_water_supply_temp=7.0,
            chilled_water_return_temp=12.0,
        )

        assert chiller.id is not None
        assert chiller.cop_rating == 5.5

    @pytest.mark.django_db
    def test_create_energy_meter(self):
        """Test creating an energy meter."""
        from industrial.models import EnergyMeter

        meter = EnergyMeter.objects.create(
            equipment_id="EM-001",
            name="Main Incomer Meter",
            description="Main building energy meter",
            location="Substation A",
            status="running",
            criticality="high",
            health_score=100,
            meter_type="main",
            ct_ratio="1000/5",
            pt_ratio="11000/110",
            power_kw=750.5,
            power_factor=0.95,
            total_kwh=125000.0,
        )

        assert meter.id is not None
        assert meter.power_kw == 750.5
        assert meter.total_kwh == 125000.0

    @pytest.mark.django_db
    def test_create_maintenance_record(self):
        """Test creating a maintenance record."""
        from industrial.models import MaintenanceRecord

        record = MaintenanceRecord.objects.create(
            equipment_id="PUMP-001",
            equipment_name="Chilled Water Pump 1",
            maintenance_type="preventive",
            description="Quarterly preventive maintenance",
            work_done="Bearing inspection, oil change, alignment check",
            parts_replaced="Oil filter, seal kit",
            technician="John Smith",
            cost=500.00,
        )

        assert record.id is not None
        assert record.maintenance_type == "preventive"
        assert record.cost == 500.00

    @pytest.mark.django_db
    def test_equipment_string_representation(self):
        """Test equipment string representation."""
        from industrial.models import Pump

        pump = Pump.objects.create(
            equipment_id="PUMP-TEST",
            name="Test Pump",
            location="Test Location",
            pump_type="chw_primary",
            flow_rate=50,
            head=20,
            motor_kw=10,
        )

        str_repr = str(pump)
        assert "PUMP-TEST" in str_repr
        assert "Test Pump" in str_repr
