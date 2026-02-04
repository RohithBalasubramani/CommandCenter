"""
HOSTILE DATA VALIDATION TEST

Purpose: Prove or falsify that data validation is ENFORCED, not just DEFINED.

Methodology:
- Inject malformed data at every entry point
- Assume developers rely on implicit trust
- Find silent acceptance, coercion, or defaulting
- Distinguish between "schema exists" and "schema is enforced"
"""
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

# Setup Django
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "command_center.settings")

import django
django.setup()


class Severity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Outcome(Enum):
    ACCEPTED = "ACCEPTED"  # Silently accepted invalid data
    COERCED = "COERCED"    # Transformed invalid data without error
    REJECTED = "REJECTED"  # Properly rejected with error
    CRASHED = "CRASHED"    # Unhandled exception


@dataclass
class HostileTestResult:
    test_name: str
    input_data: Dict[str, Any]
    expected_outcome: Outcome
    actual_outcome: Outcome
    severity: Severity
    passed: bool
    explanation: str
    danger_in_production: str


class HostileDataValidator:
    """Adversarial tester for data validation enforcement."""

    def __init__(self):
        self.results: List[HostileTestResult] = []
        self.load_schemas()

    def load_schemas(self):
        """Load schema definitions to understand what SHOULD be enforced."""
        try:
            from layer2.widget_schemas import WIDGET_SCHEMAS
            self.schemas = WIDGET_SCHEMAS
            self.has_schemas = True
        except ImportError:
            self.schemas = {}
            self.has_schemas = False

    def find_validation_function(self) -> Tuple[bool, str]:
        """Determine if validation function exists and is callable."""
        # Check for explicit validation function
        try:
            from layer2.widget_schemas import validate_widget_data
            return True, "validate_widget_data exists"
        except ImportError:
            pass

        # Check for validation in data collector
        try:
            from layer2.data_collector import validate_data
            return True, "validate_data exists in data_collector"
        except ImportError:
            pass

        # Check for validation in orchestrator
        try:
            from layer2.orchestrator import Orchestrator
            orch = Orchestrator.__dict__
            if 'validate' in str(orch) or '_validate' in str(orch):
                return True, "Orchestrator has validate method"
        except:
            pass

        return False, "NO VALIDATION FUNCTION FOUND"

    def test_type_mismatch_array_vs_object(self) -> HostileTestResult:
        """Test: Pass array where object expected."""
        test_name = "Type Mismatch: Array vs Object"
        input_data = {
            "scenario": "kpi",
            "demoData": ["this", "is", "an", "array"]  # Should be object
        }

        # Try to pass this through any validation
        outcome = self._attempt_validation("kpi", input_data)

        return HostileTestResult(
            test_name=test_name,
            input_data=input_data,
            expected_outcome=Outcome.REJECTED,
            actual_outcome=outcome,
            severity=Severity.HIGH,
            passed=outcome == Outcome.REJECTED,
            explanation="KPI widget expects object with label/value/unit, received array",
            danger_in_production="Widget will crash or render garbage data to users"
        )

    def test_type_mismatch_string_vs_number(self) -> HostileTestResult:
        """Test: Pass string where number expected."""
        test_name = "Type Mismatch: String vs Number"
        input_data = {
            "scenario": "kpi",
            "demoData": {
                "label": "Power",
                "value": "not-a-number",  # Should be numeric
                "unit": "kW"
            }
        }

        outcome = self._attempt_validation("kpi", input_data)

        return HostileTestResult(
            test_name=test_name,
            input_data=input_data,
            expected_outcome=Outcome.REJECTED,
            actual_outcome=outcome,
            severity=Severity.HIGH,
            passed=outcome == Outcome.REJECTED,
            explanation="KPI value should be numeric, received string 'not-a-number'",
            danger_in_production="Display shows 'not-a-number' to operators making decisions"
        )

    def test_missing_required_fields(self) -> HostileTestResult:
        """Test: Omit required fields."""
        test_name = "Missing Required Fields"
        input_data = {
            "scenario": "kpi",
            "demoData": {
                "label": "Power"
                # Missing: value, unit (required per schema)
            }
        }

        outcome = self._attempt_validation("kpi", input_data)

        return HostileTestResult(
            test_name=test_name,
            input_data=input_data,
            expected_outcome=Outcome.REJECTED,
            actual_outcome=outcome,
            severity=Severity.HIGH,
            passed=outcome == Outcome.REJECTED,
            explanation="KPI requires label, value, unit - only label provided",
            danger_in_production="Widget renders incomplete data, operators see blank/undefined"
        )

    def test_negative_percentage(self) -> HostileTestResult:
        """Test: Negative percentage value."""
        test_name = "Invalid Range: Negative Percentage"
        input_data = {
            "scenario": "kpi",
            "demoData": {
                "label": "Efficiency",
                "value": -50,  # Impossible: negative percentage
                "unit": "%"
            }
        }

        outcome = self._attempt_validation("kpi", input_data)

        return HostileTestResult(
            test_name=test_name,
            input_data=input_data,
            expected_outcome=Outcome.REJECTED,
            actual_outcome=outcome,
            severity=Severity.HIGH,
            passed=outcome == Outcome.REJECTED,
            explanation="Percentage cannot be negative",
            danger_in_production="Operators see impossible -50% efficiency, lose trust in system"
        )

    def test_percentage_over_100(self) -> HostileTestResult:
        """Test: Percentage over 100%."""
        test_name = "Invalid Range: Percentage > 100"
        input_data = {
            "scenario": "kpi",
            "demoData": {
                "label": "Load",
                "value": 250,  # Suspicious: 250% load
                "unit": "%"
            }
        }

        outcome = self._attempt_validation("kpi", input_data)

        # Note: This one is tricky - some percentages CAN exceed 100%
        # But it should at least warn or flag
        return HostileTestResult(
            test_name=test_name,
            input_data=input_data,
            expected_outcome=Outcome.REJECTED,  # Or at least flagged
            actual_outcome=outcome,
            severity=Severity.MEDIUM,
            passed=outcome in [Outcome.REJECTED, Outcome.COERCED],
            explanation="250% may indicate data corruption or unit error",
            danger_in_production="May indicate sensor malfunction, should trigger alert"
        )

    def test_negative_power(self) -> HostileTestResult:
        """Test: Negative power value."""
        test_name = "Invalid Range: Negative Power (kW)"
        input_data = {
            "scenario": "kpi",
            "demoData": {
                "label": "Motor Power",
                "value": -500,  # Negative power
                "unit": "kW"
            }
        }

        outcome = self._attempt_validation("kpi", input_data)

        return HostileTestResult(
            test_name=test_name,
            input_data=input_data,
            expected_outcome=Outcome.REJECTED,
            actual_outcome=outcome,
            severity=Severity.HIGH,
            passed=outcome == Outcome.REJECTED,
            explanation="Motor power cannot be negative (unless regenerative, which needs context)",
            danger_in_production="Misleading data shown to operators"
        )

    def test_unit_mixing_kw_mw(self) -> HostileTestResult:
        """Test: Mixed units without normalization."""
        test_name = "Unit Ambiguity: kW vs MW mixing"
        input_data = {
            "scenario": "comparison",
            "demoData": {
                "label": "Power Comparison",
                "unit": "kW",  # Says kW
                "labelA": "Pump 1",
                "valueA": 500,      # 500 kW
                "labelB": "Pump 2",
                "valueB": 0.5,      # Actually 0.5 MW = 500 kW, but displayed as 0.5
            }
        }

        outcome = self._attempt_validation("comparison", input_data)

        return HostileTestResult(
            test_name=test_name,
            input_data=input_data,
            expected_outcome=Outcome.REJECTED,
            actual_outcome=outcome,
            severity=Severity.HIGH,
            passed=outcome == Outcome.REJECTED,
            explanation="0.5 in kW context is likely MW data leak - 1000x error",
            danger_in_production="Operators see 500 kW vs 0.5 kW, make wrong decisions"
        )

    def test_future_timestamp(self) -> HostileTestResult:
        """Test: Future timestamp in historical data."""
        test_name = "Temporal Violation: Future Timestamp"
        input_data = {
            "scenario": "trend",
            "demoData": {
                "label": "Temperature",
                "unit": "°C",
                "timeSeries": [
                    {"time": "2099-12-31T23:59:59Z", "value": 25}  # Year 2099
                ]
            }
        }

        outcome = self._attempt_validation("trend", input_data)

        return HostileTestResult(
            test_name=test_name,
            input_data=input_data,
            expected_outcome=Outcome.REJECTED,
            actual_outcome=outcome,
            severity=Severity.HIGH,
            passed=outcome == Outcome.REJECTED,
            explanation="Timestamp in year 2099 is impossible for historical data",
            danger_in_production="Chart renders with impossible future data points"
        )

    def test_impossible_date(self) -> HostileTestResult:
        """Test: Invalid date (Feb 30)."""
        test_name = "Temporal Violation: Impossible Date"
        input_data = {
            "scenario": "timeline",
            "demoData": {
                "title": "Events",
                "range": {"start": "2026-02-30", "end": "2026-02-31"},  # Feb 30-31
                "events": []
            }
        }

        outcome = self._attempt_validation("timeline", input_data)

        return HostileTestResult(
            test_name=test_name,
            input_data=input_data,
            expected_outcome=Outcome.REJECTED,
            actual_outcome=outcome,
            severity=Severity.MEDIUM,
            passed=outcome == Outcome.REJECTED,
            explanation="February 30-31 do not exist",
            danger_in_production="Date parsing may silently coerce to invalid date"
        )

    def test_empty_required_array(self) -> HostileTestResult:
        """Test: Empty array where populated array required."""
        test_name = "Empty Required Array"
        input_data = {
            "scenario": "distribution",
            "demoData": {
                "total": 1000,
                "unit": "kW",
                "series": []  # Empty - should have items
            }
        }

        outcome = self._attempt_validation("distribution", input_data)

        return HostileTestResult(
            test_name=test_name,
            input_data=input_data,
            expected_outcome=Outcome.REJECTED,
            actual_outcome=outcome,
            severity=Severity.MEDIUM,
            passed=outcome == Outcome.REJECTED,
            explanation="Distribution with total=1000 but empty series is meaningless",
            danger_in_production="Empty pie chart rendered, confuses operators"
        )

    def test_null_injection(self) -> HostileTestResult:
        """Test: Null values in required fields."""
        test_name = "Null Injection"
        input_data = {
            "scenario": "kpi",
            "demoData": {
                "label": None,  # Null
                "value": None,  # Null
                "unit": None    # Null
            }
        }

        outcome = self._attempt_validation("kpi", input_data)

        return HostileTestResult(
            test_name=test_name,
            input_data=input_data,
            expected_outcome=Outcome.REJECTED,
            actual_outcome=outcome,
            severity=Severity.HIGH,
            passed=outcome == Outcome.REJECTED,
            explanation="All required fields are null",
            danger_in_production="Widget shows 'null' or crashes"
        )

    def test_sql_injection_in_value(self) -> HostileTestResult:
        """Test: SQL injection attempt in string field."""
        test_name = "SQL Injection in Value"
        input_data = {
            "scenario": "kpi",
            "demoData": {
                "label": "'; DROP TABLE equipment;--",
                "value": "42",
                "unit": "kW"
            }
        }

        outcome = self._attempt_validation("kpi", input_data)

        return HostileTestResult(
            test_name=test_name,
            input_data=input_data,
            expected_outcome=Outcome.REJECTED,
            actual_outcome=outcome,
            severity=Severity.CRITICAL if outcome == Outcome.ACCEPTED else Severity.LOW,
            passed=outcome == Outcome.REJECTED,
            explanation="SQL injection attempt in label field",
            danger_in_production="If passed to database, could cause data loss"
        )

    def test_xss_in_value(self) -> HostileTestResult:
        """Test: XSS attempt in string field."""
        test_name = "XSS in Value"
        input_data = {
            "scenario": "kpi",
            "demoData": {
                "label": "<script>alert('xss')</script>",
                "value": "42",
                "unit": "kW"
            }
        }

        outcome = self._attempt_validation("kpi", input_data)

        return HostileTestResult(
            test_name=test_name,
            input_data=input_data,
            expected_outcome=Outcome.REJECTED,
            actual_outcome=outcome,
            severity=Severity.CRITICAL if outcome == Outcome.ACCEPTED else Severity.MEDIUM,
            passed=outcome == Outcome.REJECTED,
            explanation="XSS attempt in label field",
            danger_in_production="Script could execute in operator's browser"
        )

    def test_deeply_nested_object(self) -> HostileTestResult:
        """Test: Deeply nested object (DoS attempt)."""
        # Create deeply nested structure
        nested = {"level": 0}
        current = nested
        for i in range(100):
            current["child"] = {"level": i + 1}
            current = current["child"]

        test_name = "DoS: Deeply Nested Object"
        input_data = {
            "scenario": "kpi",
            "demoData": nested
        }

        outcome = self._attempt_validation("kpi", input_data)

        return HostileTestResult(
            test_name=test_name,
            input_data={"scenario": "kpi", "demoData": "<<100 levels deep>>"},
            expected_outcome=Outcome.REJECTED,
            actual_outcome=outcome,
            severity=Severity.MEDIUM,
            passed=outcome == Outcome.REJECTED,
            explanation="100-level deep nesting could cause stack overflow",
            danger_in_production="Could crash validation or serialization"
        )

    def _attempt_validation(self, scenario: str, data: Dict) -> Outcome:
        """Attempt to validate data through available mechanisms."""
        # First, check if explicit validation function exists
        has_validator, _ = self.find_validation_function()

        if has_validator:
            try:
                from layer2.widget_schemas import validate_widget_data, ValidationError

                # validate_widget_data raises ValidationError on failure
                # It returns None on success
                validate_widget_data(scenario, data.get("demoData", data))
                # If we get here, validation passed
                return Outcome.ACCEPTED

            except ImportError:
                pass
            except ValidationError:
                # ValidationError = proper rejection
                return Outcome.REJECTED
            except Exception as e:
                # Other exceptions = crash
                return Outcome.CRASHED

        # If no validator, check if schema enforcement exists elsewhere
        # Try to use the data in a widget selector or data collector
        try:
            from layer2.widget_selector import WidgetSelector
            # If it accepts without error, it's ACCEPTED
            return Outcome.ACCEPTED
        except ImportError:
            pass
        except Exception as e:
            if "invalid" in str(e).lower():
                return Outcome.REJECTED
            return Outcome.CRASHED

        # No validation found - everything is ACCEPTED by default
        return Outcome.ACCEPTED

    def run_all_tests(self) -> List[HostileTestResult]:
        """Run all hostile tests."""
        tests = [
            self.test_type_mismatch_array_vs_object,
            self.test_type_mismatch_string_vs_number,
            self.test_missing_required_fields,
            self.test_negative_percentage,
            self.test_percentage_over_100,
            self.test_negative_power,
            self.test_unit_mixing_kw_mw,
            self.test_future_timestamp,
            self.test_impossible_date,
            self.test_empty_required_array,
            self.test_null_injection,
            self.test_sql_injection_in_value,
            self.test_xss_in_value,
            self.test_deeply_nested_object,
        ]

        for test in tests:
            result = test()
            self.results.append(result)

        return self.results

    def print_report(self):
        """Print hostile validation report."""
        print("=" * 80)
        print("  HOSTILE DATA VALIDATION REPORT")
        print("=" * 80)
        print()

        # Check for validation function
        has_validator, validator_status = self.find_validation_function()
        print(f"  Validation Function Status: {validator_status}")
        print()

        # Results summary
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        high_failures = [r for r in self.results if not r.passed and r.severity in [Severity.HIGH, Severity.CRITICAL]]
        medium_failures = [r for r in self.results if not r.passed and r.severity == Severity.MEDIUM]

        print("─" * 80)
        print("  RESULTS SUMMARY")
        print("─" * 80)
        print(f"  Total Tests:     {total}")
        print(f"  Passed:          {passed}")
        print(f"  Failed:          {failed}")
        print(f"  HIGH/CRITICAL:   {len(high_failures)}")
        print(f"  MEDIUM:          {len(medium_failures)}")
        print()

        # Detail failures
        if failed > 0:
            print("─" * 80)
            print("  FAILURES DISCOVERED")
            print("─" * 80)

            for r in self.results:
                if not r.passed:
                    print()
                    print(f"  ❌ {r.test_name}")
                    print(f"     Severity: {r.severity.value}")
                    print(f"     Expected: {r.expected_outcome.value}")
                    print(f"     Actual:   {r.actual_outcome.value}")
                    print(f"     Explanation: {r.explanation}")
                    print(f"     DANGER: {r.danger_in_production}")

        # Root cause analysis
        print()
        print("─" * 80)
        print("  ROOT CAUSE ANALYSIS")
        print("─" * 80)
        print()

        if not has_validator:
            print("  ⚠️  PRIMARY ROOT CAUSE: NO VALIDATION FUNCTION EXISTS")
            print()
            print("  The schema definitions exist in widget_schemas.py (WIDGET_SCHEMAS)")
            print("  but there is NO validate_widget_data() function to enforce them.")
            print()
            print("  This is not a bug in validation logic.")
            print("  This is MISSING validation logic entirely.")
            print()
            print("  All 14 failures collapse to a single missing abstraction:")
            print("  → validate_widget_data(scenario, data) -> (bool, errors)")
        else:
            print("  Validation function exists but may have gaps in coverage.")

        # Release decision
        print()
        print("=" * 80)
        print("  RELEASE DECISION")
        print("=" * 80)
        print()

        if len(high_failures) > 0:
            print("  ╔════════════════════════════════════════════════════════════════════╗")
            print("  ║                                                                    ║")
            print("  ║    🛑  RELEASE STATUS: BLOCKED                                     ║")
            print("  ║                                                                    ║")
            print("  ║    HIGH/CRITICAL severity failures: {}                              ║".format(len(high_failures)))
            print("  ║                                                                    ║")
            print("  ╚════════════════════════════════════════════════════════════════════╝")
            print()
            print("  MINIMUM REQUIRED FIXES:")
            print()
            print("  1. Implement validate_widget_data(scenario, data) function")
            print("     Location: backend/layer2/widget_schemas.py")
            print()
            print("  2. Enforce validation at data pipeline entry point")
            print("     Location: backend/layer2/data_collector.py or orchestrator.py")
            print()
            print("  3. Add unit tests proving validation rejects invalid data")
            print()
            print("  4. Re-run this hostile validation - must achieve 0 HIGH failures")
            print()
            print("  ─────────────────────────────────────────────────────────────────────")
            print("  EXPLICIT STATEMENT: This system is NOT SAFE TO SHIP.")
            print("  Invalid data will reach end users without any validation barrier.")
            print("  ─────────────────────────────────────────────────────────────────────")
        else:
            print("  ✅ RELEASE STATUS: PASS")
            print()
            print("  Zero HIGH/CRITICAL severity failures.")
            print("  System is safe to ship with documented MEDIUM risk mitigations.")


if __name__ == "__main__":
    validator = HostileDataValidator()
    validator.run_all_tests()
    validator.print_report()
