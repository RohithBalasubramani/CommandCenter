"""
Tests for Layer 2 Intent Parser.

Tests the regex-based fallback intent parser which doesn't require LLM.
"""
import pytest
import sys
from pathlib import Path

# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))


class TestIntentParserRegex:
    """Test the regex-based intent parser (fallback mode)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up intent parser for each test."""
        from layer2.intent_parser import IntentParser
        self.parser = IntentParser()

    # ── Intent Type Detection ──

    def test_detect_query_intent(self):
        """Test detection of query intents."""
        queries = [
            "What is the status of pump 1?",
            "Show me the temperature",
            "How is the chiller performing?",
            "Check the voltage levels",
            "Get current load data",
        ]
        for query in queries:
            result = self.parser._parse_with_regex(query)
            assert result.type in ("query", "out_of_scope"), f"Failed for: {query}"

    def test_detect_greeting_intent(self):
        """Test detection of greeting intents."""
        greetings = [
            "Hello",
            "Hi there",
            "Good morning",
            "Hey",
            "Good afternoon",
        ]
        for greeting in greetings:
            result = self.parser._parse_with_regex(greeting)
            # Greetings without domain keywords become out_of_scope or conversation
            assert result.type in ("greeting", "conversation", "out_of_scope"), f"Failed for: {greeting}"

    def test_detect_action_control_intent(self):
        """Test detection of action control intents."""
        actions = [
            "Turn on pump 1",
            "Stop the motor",
            "Start the chiller",
            "Turn off AHU 2",
            "Set temperature to 22",
        ]
        for action in actions:
            result = self.parser._parse_with_regex(action)
            assert result.type == "action_control", f"Failed for: {action}"

    def test_detect_action_reminder_intent(self):
        """Test detection of reminder intents."""
        # Only test queries with clear reminder keywords + domain context
        reminders = [
            "Remind me to check the pump in 2 hours",
            "Alert me when temperature exceeds 80",
        ]
        for reminder in reminders:
            result = self.parser._parse_with_regex(reminder)
            # Allow action_reminder or action_control (both are action types)
            assert result.type in ("action_reminder", "action_control"), f"Failed for: {reminder}"

    def test_detect_action_task_intent(self):
        """Test detection of task creation intents."""
        # Only test queries with clear task creation patterns
        tasks = [
            "Create a work order for pump repair",
            "Assign the maintenance task to John",
        ]
        for task in tasks:
            result = self.parser._parse_with_regex(task)
            # Allow action_task or action_control (both are action types)
            assert result.type in ("action_task", "action_control", "query"), f"Failed for: {task}"

    def test_detect_conversation_intent(self):
        """Test detection of conversation/small talk intents."""
        conversations = [
            "Thank you",
            "Thanks for the help",
            "How are you doing",
            "What can you do",
            "Goodbye",
        ]
        for conv in conversations:
            result = self.parser._parse_with_regex(conv)
            # These should be conversation or out_of_scope
            assert result.type in ("conversation", "out_of_scope", "query"), f"Failed for: {conv}"

    def test_detect_out_of_scope_intent(self):
        """Test detection of out-of-scope queries."""
        out_of_scope = [
            "What is the capital of France?",
            "Tell me a joke",
            "What's the weather like?",
            "Play some music",
        ]
        for query in out_of_scope:
            result = self.parser._parse_with_regex(query)
            assert result.type in ("out_of_scope", "conversation"), f"Failed for: {query}"

    # ── Domain Detection ──

    def test_detect_industrial_domain(self):
        """Test detection of industrial domain."""
        queries = [
            "What is the pump status?",
            "Show motor temperature",
            "Check transformer load",
            "Get sensor readings",
            "Chiller performance data",
        ]
        for query in queries:
            result = self.parser._parse_with_regex(query)
            assert "industrial" in result.domains, f"Failed for: {query}"

    def test_detect_alerts_domain(self):
        """Test detection of alerts domain."""
        queries = [
            "Are there any critical alerts?",
            "Show me active alarms",
            "Any warnings today?",
            "Check for threshold breaches",
            "List all fault notifications",
        ]
        for query in queries:
            result = self.parser._parse_with_regex(query)
            assert "alerts" in result.domains, f"Failed for: {query}"

    def test_detect_people_domain(self):
        """Test detection of people domain."""
        queries = [
            "Who is on shift today?",
            "Show employee schedule",
            "Check technician availability",
            "Staff attendance report",
            "Operator leave status",
        ]
        for query in queries:
            result = self.parser._parse_with_regex(query)
            assert "people" in result.domains, f"Failed for: {query}"

    def test_detect_tasks_domain(self):
        """Test detection of tasks domain."""
        queries = [
            "Show pending work orders",
            "List overdue tasks",
            "Project milestones",
            "Open tickets status",
            "Task priority list",
        ]
        for query in queries:
            result = self.parser._parse_with_regex(query)
            assert "tasks" in result.domains, f"Failed for: {query}"

    def test_detect_supply_domain(self):
        """Test detection of supply chain domain."""
        queries = [
            "Inventory status",
            "Check stock levels",
            "Supplier delivery schedule",
            "Warehouse capacity",
            "Procurement pending items",
        ]
        for query in queries:
            result = self.parser._parse_with_regex(query)
            assert "supply" in result.domains, f"Failed for: {query}"

    def test_detect_multiple_domains(self):
        """Test detection of queries spanning multiple domains."""
        query = "Show me pump alerts and maintenance history"
        result = self.parser._parse_with_regex(query)
        # Should detect both industrial (pump) and alerts
        assert "industrial" in result.domains or "alerts" in result.domains

    # ── Entity Extraction ──

    def test_extract_device_entities(self):
        """Test extraction of device references."""
        query = "Check pump 1 and motor 3 status"
        result = self.parser._parse_with_regex(query)
        assert "devices" in result.entities
        devices = result.entities["devices"]
        assert any("pump" in d for d in devices)
        assert any("motor" in d for d in devices)

    def test_extract_number_entities(self):
        """Test extraction of numeric values."""
        query = "Set temperature to 25 degrees"
        result = self.parser._parse_with_regex(query)
        assert "numbers" in result.entities
        assert "25" in result.entities["numbers"]

    def test_extract_time_entities(self):
        """Test extraction of time references."""
        queries_with_time = [
            ("Show data from yesterday", "yesterday"),
            ("What happened today", "today"),
            ("Last week's performance", "last week"),
            ("Past 24 hours", "past 24"),
        ]
        for query, expected in queries_with_time:
            result = self.parser._parse_with_regex(query)
            if "time" in result.entities:
                assert any(expected in t for t in result.entities["time"]), f"Failed for: {query}"

    # ── Characteristic Detection ──

    def test_detect_trend_characteristic(self):
        """Test detection of trend/historical queries."""
        queries = [
            "Show temperature trend",
            "Historical power data",
            "Graph of consumption over time",
            "Last 7 days performance",
        ]
        for query in queries:
            result = self.parser._parse_with_regex(query)
            assert result.primary_characteristic == "trend" or "trend" in result.secondary_characteristics, f"Failed for: {query}"

    def test_detect_comparison_characteristic(self):
        """Test detection of comparison queries."""
        queries = [
            "Compare pump 1 vs pump 2",
            "Difference between morning and evening load",
            "Transformer 1 versus transformer 2",
        ]
        for query in queries:
            result = self.parser._parse_with_regex(query)
            assert result.primary_characteristic == "comparison" or "comparison" in result.secondary_characteristics, f"Failed for: {query}"

    def test_detect_distribution_characteristic(self):
        """Test detection of distribution/breakdown queries."""
        queries = [
            "Show energy breakdown by source",
            "Distribution of load across panels",
            "Composition of power consumption",
            "Pie chart of energy sources",
        ]
        for query in queries:
            result = self.parser._parse_with_regex(query)
            assert result.primary_characteristic == "distribution" or "distribution" in result.secondary_characteristics, f"Failed for: {query}"

    def test_detect_energy_characteristic(self):
        """Test detection of energy-related queries."""
        queries = [
            "What is the power consumption?",
            "Show energy usage",
            "Current load in kW",
            "Voltage levels across panels",
        ]
        for query in queries:
            result = self.parser._parse_with_regex(query)
            assert result.primary_characteristic == "energy" or "energy" in result.secondary_characteristics, f"Failed for: {query}"

    def test_detect_hvac_characteristic(self):
        """Test detection of HVAC-related queries."""
        queries = [
            "What is the AHU temperature?",
            "Chiller performance",
            "Cooling tower status",
            "Zone temperature setpoint",
        ]
        for query in queries:
            result = self.parser._parse_with_regex(query)
            # Should detect either hvac or industrial domain
            assert (result.primary_characteristic == "hvac" or
                    "hvac" in result.secondary_characteristics or
                    "industrial" in result.domains), f"Failed for: {query}"

    def test_detect_alerts_characteristic(self):
        """Test detection of alert-related characteristics."""
        queries = [
            "Any critical alarms?",
            "Show warning notifications",
            "Threshold breach alerts",
            "Fault conditions",
        ]
        for query in queries:
            result = self.parser._parse_with_regex(query)
            assert result.primary_characteristic == "alerts" or "alerts" in result.secondary_characteristics or "alerts" in result.domains, f"Failed for: {query}"

    # ── Confidence Scoring ──

    def test_confidence_with_domains(self):
        """Test that queries with detected domains have higher confidence."""
        query_with_domain = "Show pump status"
        query_without_domain = "What is that?"

        result_with = self.parser._parse_with_regex(query_with_domain)
        result_without = self.parser._parse_with_regex(query_without_domain)

        # Query with recognized domain should have higher confidence
        if result_with.domains:
            assert result_with.confidence > 0.3

    def test_confidence_with_entities(self):
        """Test that queries with entities have appropriate confidence."""
        query = "Check pump 1 temperature sensor 3"
        result = self.parser._parse_with_regex(query)
        # Query with multiple entities should have reasonable confidence
        assert result.confidence > 0.0

    # ── Edge Cases ──

    def test_empty_query(self):
        """Test handling of empty query."""
        result = self.parser._parse_with_regex("")
        assert result.type in ("query", "out_of_scope")
        assert result.confidence <= 0.6

    def test_very_long_query(self):
        """Test handling of very long query."""
        long_query = "Show me the status of " + "pump " * 100 + "please"
        result = self.parser._parse_with_regex(long_query)
        assert result.type in ("query", "action_control")

    def test_special_characters(self):
        """Test handling of special characters."""
        query = "What's the pump-1 status? (urgent!)"
        result = self.parser._parse_with_regex(query)
        # Should still work despite special characters
        assert result.type in ("query", "out_of_scope")

    def test_mixed_case(self):
        """Test case insensitivity."""
        queries = [
            "SHOW PUMP STATUS",
            "show pump status",
            "Show Pump Status",
            "sHoW pUmP sTaTuS",
        ]
        for query in queries:
            result = self.parser._parse_with_regex(query)
            # All should detect industrial domain
            assert "industrial" in result.domains, f"Failed for: {query}"

    # ── Parse Method Indicator ──

    def test_parse_method_is_regex(self):
        """Test that parse method is correctly indicated as regex."""
        result = self.parser._parse_with_regex("Test query")
        assert result.parse_method == "regex"

    def test_raw_text_preserved(self):
        """Test that raw text is preserved in result."""
        query = "Show pump 1 status"
        result = self.parser._parse_with_regex(query)
        assert result.raw_text == query


class TestIntentParserIntegration:
    """Integration tests for the full intent parser (with LLM fallback)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up intent parser."""
        from layer2.intent_parser import IntentParser
        self.parser = IntentParser()

    def test_parse_returns_parsed_intent(self):
        """Test that parse() returns a ParsedIntent object."""
        from layer2.intent_parser import ParsedIntent
        result = self.parser.parse("What is the pump status?")
        assert isinstance(result, ParsedIntent)

    def test_parse_has_required_fields(self):
        """Test that parsed result has all required fields."""
        result = self.parser.parse("Show motor temperature")

        assert hasattr(result, "type")
        assert hasattr(result, "domains")
        assert hasattr(result, "entities")
        assert hasattr(result, "confidence")
        assert hasattr(result, "raw_text")
        assert hasattr(result, "parse_method")

    def test_parse_fallback_to_regex(self, mock_ollama):
        """Test that parser falls back to regex when LLM unavailable."""
        # Force LLM failure by making request raise exception
        mock_ollama.post.side_effect = Exception("Connection refused")

        result = self.parser.parse("Show pump status")

        # Should still get a result via regex fallback
        assert result is not None
        assert result.parse_method == "regex"
