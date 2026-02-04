"""
Dimension-Based Normalization Tests — Proves general-purpose design.

This test suite demonstrates that normalization is driven by the DIMENSION REGISTRY,
not by hardcoded examples. Each dimension is tested with its own conversions.

KEY PRINCIPLE: Adding a new dimension to dimensions.py automatically enables
normalization for all units in that dimension. No code changes in widget_normalizer.py.

Dimensions tested:
1. POWER (W, kW, MW, hp)
2. ENERGY (Wh, kWh, MWh, GJ)
3. TEMPERATURE (°C, °F, K) - with offset conversion
4. PRESSURE (bar, psi, Pa, atm)
5. FLOW_RATE (m³/h, L/s, GPM)
6. PERCENTAGE (%, fraction)
7. VOLTAGE (V, kV, mV)
"""
import sys
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "command_center.settings")

import django
django.setup()

from layer2.dimensions import (
    Dimension,
    DIMENSION_REGISTRY,
    convert_value,
    normalize_to_base,
    infer_dimension,
    ConversionError,
)
from layer2.widget_normalizer import normalize_widget_data, NormalizationError


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION 1: POWER
# ═══════════════════════════════════════════════════════════════════════════════

def test_power_w_to_kw():
    """Power: W → kW (base unit)"""
    result = normalize_to_base(5000, "W")
    assert result == (5.0, "kW", Dimension.POWER)
    print("✓ POWER: 5000 W → 5.0 kW")


def test_power_mw_to_kw():
    """Power: MW → kW"""
    result = normalize_to_base(2.5, "MW")
    assert result == (2500.0, "kW", Dimension.POWER)
    print("✓ POWER: 2.5 MW → 2500.0 kW")


def test_power_hp_to_kw():
    """Power: Horsepower → kW"""
    value, unit, dim = normalize_to_base(100, "hp")
    assert dim == Dimension.POWER
    assert abs(value - 74.57) < 0.01  # 100 hp ≈ 74.57 kW
    print(f"✓ POWER: 100 hp → {value:.2f} kW")


def test_power_widget():
    """Power: Full widget normalization"""
    data = {"demoData": {"label": "Motor Power", "value": 7500, "unit": "W"}}
    result = normalize_widget_data("kpi", data)
    assert result.data["demoData"]["value"] == 7.5
    assert result.data["demoData"]["unit"] == "kW"
    print("✓ POWER WIDGET: 7500 W → 7.5 kW")


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION 2: ENERGY
# ═══════════════════════════════════════════════════════════════════════════════

def test_energy_wh_to_kwh():
    """Energy: Wh → kWh (base unit)"""
    result = normalize_to_base(15000, "Wh")
    assert result == (15.0, "kWh", Dimension.ENERGY)
    print("✓ ENERGY: 15000 Wh → 15.0 kWh")


def test_energy_mwh_to_kwh():
    """Energy: MWh → kWh"""
    result = normalize_to_base(0.5, "MWh")
    assert result == (500.0, "kWh", Dimension.ENERGY)
    print("✓ ENERGY: 0.5 MWh → 500.0 kWh")


def test_energy_gj_to_kwh():
    """Energy: GJ → kWh"""
    value, unit, dim = normalize_to_base(1, "GJ")
    assert dim == Dimension.ENERGY
    assert abs(value - 277.8) < 0.1  # 1 GJ ≈ 277.8 kWh
    print(f"✓ ENERGY: 1 GJ → {value:.1f} kWh")


def test_energy_widget():
    """Energy: Full widget normalization"""
    data = {"demoData": {"label": "Daily Consumption", "value": 250000, "unit": "Wh"}}
    result = normalize_widget_data("kpi", data)
    assert result.data["demoData"]["value"] == 250.0
    assert result.data["demoData"]["unit"] == "kWh"
    print("✓ ENERGY WIDGET: 250000 Wh → 250.0 kWh")


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION 3: TEMPERATURE (with offset conversion)
# ═══════════════════════════════════════════════════════════════════════════════

def test_temp_f_to_c():
    """Temperature: °F → °C (offset conversion, not scaling)"""
    value, unit, dim = normalize_to_base(212, "°F")
    assert dim == Dimension.TEMPERATURE
    assert abs(value - 100) < 0.1  # 212°F = 100°C (boiling point)
    print(f"✓ TEMPERATURE: 212 °F → {value:.1f} °C")


def test_temp_k_to_c():
    """Temperature: K → °C"""
    value, unit, dim = normalize_to_base(373.15, "K")
    assert dim == Dimension.TEMPERATURE
    assert abs(value - 100) < 0.1  # 373.15 K = 100°C
    print(f"✓ TEMPERATURE: 373.15 K → {value:.1f} °C")


def test_temp_widget():
    """Temperature: Full widget normalization"""
    data = {"demoData": {"label": "Coolant Temp", "value": 68, "unit": "°F"}}
    result = normalize_widget_data("kpi", data)
    assert abs(result.data["demoData"]["value"] - 20) < 0.1  # 68°F = 20°C
    assert result.data["demoData"]["unit"] == "°C"
    print(f"✓ TEMPERATURE WIDGET: 68 °F → {result.data['demoData']['value']:.1f} °C")


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION 4: PRESSURE
# ═══════════════════════════════════════════════════════════════════════════════

def test_pressure_psi_to_bar():
    """Pressure: psi → bar (base unit)"""
    value, unit, dim = normalize_to_base(14.5, "psi")
    assert dim == Dimension.PRESSURE
    assert abs(value - 1.0) < 0.01  # 14.5 psi ≈ 1 bar
    print(f"✓ PRESSURE: 14.5 psi → {value:.2f} bar")


def test_pressure_pa_to_bar():
    """Pressure: Pa → bar"""
    value, unit, dim = normalize_to_base(100000, "Pa")
    assert dim == Dimension.PRESSURE
    assert abs(value - 1.0) < 0.01  # 100000 Pa = 1 bar
    print(f"✓ PRESSURE: 100000 Pa → {value:.2f} bar")


def test_pressure_atm_to_bar():
    """Pressure: atm → bar"""
    value, unit, dim = normalize_to_base(1, "atm")
    assert dim == Dimension.PRESSURE
    assert abs(value - 1.01325) < 0.001  # 1 atm ≈ 1.01325 bar
    print(f"✓ PRESSURE: 1 atm → {value:.5f} bar")


def test_pressure_widget():
    """Pressure: Full widget normalization"""
    data = {"demoData": {"label": "Tank Pressure", "value": 29, "unit": "psi"}}
    result = normalize_widget_data("kpi", data)
    assert abs(result.data["demoData"]["value"] - 2.0) < 0.1  # 29 psi ≈ 2 bar
    print(f"✓ PRESSURE WIDGET: 29 psi → {result.data['demoData']['value']:.2f} bar")


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION 5: FLOW RATE
# ═══════════════════════════════════════════════════════════════════════════════

def test_flow_lps_to_m3h():
    """Flow Rate: L/s → m³/h (base unit)"""
    value, unit, dim = normalize_to_base(10, "L/s")
    assert dim == Dimension.FLOW_RATE
    assert abs(value - 36.0) < 0.1  # 10 L/s = 36 m³/h
    print(f"✓ FLOW RATE: 10 L/s → {value:.1f} m³/h")


def test_flow_gpm_to_m3h():
    """Flow Rate: GPM → m³/h"""
    value, unit, dim = normalize_to_base(100, "GPM")
    assert dim == Dimension.FLOW_RATE
    assert abs(value - 22.7) < 0.2  # 100 GPM ≈ 22.7 m³/h
    print(f"✓ FLOW RATE: 100 GPM → {value:.1f} m³/h")


def test_flow_widget():
    """Flow Rate: Full widget normalization"""
    data = {"demoData": {"label": "Pump Flow", "value": 500, "unit": "L/s"}}
    result = normalize_widget_data("kpi", data)
    assert abs(result.data["demoData"]["value"] - 1800) < 1  # 500 L/s = 1800 m³/h
    print(f"✓ FLOW RATE WIDGET: 500 L/s → {result.data['demoData']['value']:.0f} m³/h")


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION 6: PERCENTAGE
# ═══════════════════════════════════════════════════════════════════════════════

def test_percentage_fraction_to_percent():
    """Percentage: fraction → %"""
    value, unit, dim = normalize_to_base(0.85, "fraction")
    assert dim == Dimension.PERCENTAGE
    assert abs(value - 85.0) < 0.01  # 0.85 fraction = 85%
    print(f"✓ PERCENTAGE: 0.85 fraction → {value:.1f}%")


def test_percentage_widget():
    """Percentage: Full widget normalization"""
    data = {"demoData": {"label": "Efficiency", "value": 0.92, "unit": "ratio"}}
    result = normalize_widget_data("kpi", data)
    assert abs(result.data["demoData"]["value"] - 92.0) < 0.01
    print(f"✓ PERCENTAGE WIDGET: 0.92 ratio → {result.data['demoData']['value']:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION 7: VOLTAGE
# ═══════════════════════════════════════════════════════════════════════════════

def test_voltage_kv_to_v():
    """Voltage: kV → V (base unit)"""
    value, unit, dim = normalize_to_base(11, "kV")
    assert dim == Dimension.VOLTAGE
    assert value == 11000  # 11 kV = 11000 V
    print(f"✓ VOLTAGE: 11 kV → {value:.0f} V")


def test_voltage_mv_to_v():
    """Voltage: mV → V"""
    value, unit, dim = normalize_to_base(500, "mV")
    assert dim == Dimension.VOLTAGE
    assert abs(value - 0.5) < 0.001  # 500 mV = 0.5 V
    print(f"✓ VOLTAGE: 500 mV → {value:.1f} V")


def test_voltage_widget():
    """Voltage: Full widget normalization"""
    data = {"demoData": {"label": "Grid Voltage", "value": 33, "unit": "kV"}}
    result = normalize_widget_data("kpi", data)
    assert result.data["demoData"]["value"] == 33000
    print(f"✓ VOLTAGE WIDGET: 33 kV → {result.data['demoData']['value']:.0f} V")


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION MISMATCH DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def test_dimension_mismatch_fails():
    """Converting between dimensions MUST fail."""
    try:
        convert_value(100, "kW", "kWh")
        assert False, "Should have raised ConversionError"
    except ConversionError as e:
        assert "Dimension mismatch" in str(e)
        print(f"✓ DIMENSION MISMATCH: kW → kWh correctly fails")


def test_unknown_unit_passthrough():
    """Unknown units pass through for validation to catch."""
    data = {"demoData": {"label": "Custom", "value": 42, "unit": "zorps"}}
    result = normalize_widget_data("kpi", data)
    # Value unchanged, unit unchanged
    assert result.data["demoData"]["value"] == 42
    assert result.data["demoData"]["unit"] == "zorps"
    print("✓ UNKNOWN UNIT: 'zorps' passes through (not guessed)")


# ═══════════════════════════════════════════════════════════════════════════════
# EXTENSIBILITY PROOF
# ═══════════════════════════════════════════════════════════════════════════════

def test_dimension_count():
    """Verify registry has expected dimensions."""
    expected = {
        Dimension.POWER, Dimension.ENERGY, Dimension.TEMPERATURE,
        Dimension.PRESSURE, Dimension.FLOW_RATE, Dimension.PERCENTAGE,
        Dimension.FREQUENCY, Dimension.VOLTAGE, Dimension.CURRENT,
        Dimension.MASS, Dimension.LENGTH, Dimension.TIME, Dimension.DIMENSIONLESS
    }
    actual = set(DIMENSION_REGISTRY.keys())
    assert expected == actual, f"Missing: {expected - actual}"
    print(f"✓ REGISTRY: {len(DIMENSION_REGISTRY)} dimensions registered")


def test_all_dimensions_have_units():
    """Every dimension must have at least one unit."""
    for dim, spec in DIMENSION_REGISTRY.items():
        assert len(spec.units) >= 1, f"{dim.name} has no units"
    print("✓ COMPLETENESS: All dimensions have units defined")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_tests():
    print("=" * 70)
    print("  DIMENSION-BASED NORMALIZATION TESTS")
    print("  Proving general-purpose design (not example-driven)")
    print("=" * 70)
    print()

    tests = [
        # Dimension 1: Power
        ("POWER", test_power_w_to_kw),
        ("POWER", test_power_mw_to_kw),
        ("POWER", test_power_hp_to_kw),
        ("POWER", test_power_widget),

        # Dimension 2: Energy
        ("ENERGY", test_energy_wh_to_kwh),
        ("ENERGY", test_energy_mwh_to_kwh),
        ("ENERGY", test_energy_gj_to_kwh),
        ("ENERGY", test_energy_widget),

        # Dimension 3: Temperature
        ("TEMPERATURE", test_temp_f_to_c),
        ("TEMPERATURE", test_temp_k_to_c),
        ("TEMPERATURE", test_temp_widget),

        # Dimension 4: Pressure
        ("PRESSURE", test_pressure_psi_to_bar),
        ("PRESSURE", test_pressure_pa_to_bar),
        ("PRESSURE", test_pressure_atm_to_bar),
        ("PRESSURE", test_pressure_widget),

        # Dimension 5: Flow Rate
        ("FLOW_RATE", test_flow_lps_to_m3h),
        ("FLOW_RATE", test_flow_gpm_to_m3h),
        ("FLOW_RATE", test_flow_widget),

        # Dimension 6: Percentage
        ("PERCENTAGE", test_percentage_fraction_to_percent),
        ("PERCENTAGE", test_percentage_widget),

        # Dimension 7: Voltage
        ("VOLTAGE", test_voltage_kv_to_v),
        ("VOLTAGE", test_voltage_mv_to_v),
        ("VOLTAGE", test_voltage_widget),

        # Error handling
        ("ERROR_HANDLING", test_dimension_mismatch_fails),
        ("ERROR_HANDLING", test_unknown_unit_passthrough),

        # Extensibility
        ("EXTENSIBILITY", test_dimension_count),
        ("EXTENSIBILITY", test_all_dimensions_have_units),
    ]

    current_category = None
    passed = 0
    failed = 0

    for category, test in tests:
        if category != current_category:
            print(f"\n── {category} ──")
            current_category = category

        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAIL: {test.__name__}")
            print(f"  └─ {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {test.__name__}")
            print(f"  └─ {type(e).__name__}: {e}")
            failed += 1

    print()
    print("=" * 70)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    print()
    print("  KEY INSIGHT: All unit conversions flow from DIMENSION_REGISTRY.")
    print("  To add a new dimension (e.g., LUMINOSITY), simply add it to")
    print("  dimensions.py — no changes needed in widget_normalizer.py.")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
