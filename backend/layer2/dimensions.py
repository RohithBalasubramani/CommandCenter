"""
Dimension Registry — General-purpose unit normalization by physical dimension.

This module provides a CONFIGURATION-DRIVEN approach to unit normalization.
No hardcoded example logic. Extensible by adding to DIMENSION_REGISTRY.

Key concepts:
- DIMENSION: A physical quantity (power, energy, temperature, etc.)
- UNIT: A measurement within a dimension (kW, MW for power)
- SEMANTIC: The meaning within a dimension (capacity vs usage vs peak)
- BASE UNIT: The canonical unit for a dimension (kW for power)

Rules:
1. Normalization ONLY within same dimension
2. Normalization ONLY with explicit dimension metadata OR unambiguous unit inference
3. FAIL when dimensions conflict
4. FAIL when semantics differ (capacity vs usage)
5. FAIL when dimension cannot be determined
"""
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum, auto


# ═══════════════════════════════════════════════════════════════════════════════
# CORE TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class Dimension(Enum):
    """Physical dimensions that can be normalized."""
    POWER = auto()           # Rate of energy transfer (kW, MW, W)
    ENERGY = auto()          # Total energy (kWh, MWh, J)
    TEMPERATURE = auto()     # Thermal (°C, °F, K)
    PRESSURE = auto()        # Force per area (bar, psi, Pa)
    FLOW_RATE = auto()       # Volume per time (m³/h, L/min, GPM)
    PERCENTAGE = auto()      # Ratio (%, fraction)
    FREQUENCY = auto()       # Cycles per time (Hz, RPM)
    VOLTAGE = auto()         # Electrical potential (V, kV)
    CURRENT = auto()         # Electrical flow (A, mA)
    MASS = auto()            # Weight (kg, lb, ton)
    LENGTH = auto()          # Distance (m, ft, in)
    TIME = auto()            # Duration (s, min, h)
    DIMENSIONLESS = auto()   # Pure numbers (count, ratio)


class Semantic(Enum):
    """Semantic meaning within a dimension - MUST match for comparison."""
    # Power semantics
    CAPACITY = auto()        # Maximum rated value
    USAGE = auto()           # Current consumption
    PEAK = auto()            # Peak/max observed
    AVERAGE = auto()         # Average over period
    MINIMUM = auto()         # Minimum observed

    # Generic semantics
    INSTANTANEOUS = auto()   # Point-in-time value
    CUMULATIVE = auto()      # Running total
    DELTA = auto()           # Change from baseline
    TARGET = auto()          # Goal/setpoint
    LIMIT = auto()           # Threshold/limit

    # Default
    UNSPECIFIED = auto()     # No semantic specified


@dataclass
class UnitSpec:
    """Specification for a single unit within a dimension."""
    symbol: str              # Display symbol (e.g., "kW")
    dimension: Dimension     # Physical dimension
    to_base: float           # Multiplier to convert TO base unit
    from_base: float         # Multiplier to convert FROM base unit (usually 1/to_base)
    aliases: list[str] = field(default_factory=list)  # Alternative spellings


@dataclass
class DimensionSpec:
    """Specification for a physical dimension."""
    dimension: Dimension
    base_unit: str           # Canonical unit symbol
    description: str
    units: dict[str, UnitSpec] = field(default_factory=dict)

    def add_unit(self, symbol: str, to_base: float, aliases: list[str] = None) -> "DimensionSpec":
        """Add a unit to this dimension. Returns self for chaining."""
        spec = UnitSpec(
            symbol=symbol,
            dimension=self.dimension,
            to_base=to_base,
            from_base=1.0 / to_base if to_base != 0 else 0,
            aliases=aliases or []
        )
        self.units[symbol] = spec
        # Register aliases
        for alias in spec.aliases:
            self.units[alias] = spec
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION REGISTRY — ADD NEW DIMENSIONS HERE
# ═══════════════════════════════════════════════════════════════════════════════

def _build_dimension_registry() -> dict[Dimension, DimensionSpec]:
    """Build the dimension registry. Extend by adding new DimensionSpec entries."""
    registry = {}

    # ── POWER (base: kW) ──
    power = DimensionSpec(Dimension.POWER, "kW", "Rate of energy transfer")
    power.add_unit("W", 0.001)
    power.add_unit("kW", 1.0)
    power.add_unit("MW", 1000.0)
    power.add_unit("GW", 1000000.0)
    power.add_unit("VA", 0.001, ["va"])      # Apparent power (simplified as equal to W)
    power.add_unit("kVA", 1.0, ["kva"])
    power.add_unit("MVA", 1000.0, ["mva"])
    power.add_unit("hp", 0.7457)             # Horsepower
    power.add_unit("BTU/h", 0.000293)        # BTU per hour
    power.add_unit("TR", 3.517)              # Tons of refrigeration
    registry[Dimension.POWER] = power

    # ── ENERGY (base: kWh) ──
    energy = DimensionSpec(Dimension.ENERGY, "kWh", "Total energy")
    energy.add_unit("Wh", 0.001)
    energy.add_unit("kWh", 1.0)
    energy.add_unit("MWh", 1000.0)
    energy.add_unit("GWh", 1000000.0)
    energy.add_unit("J", 2.778e-7)
    energy.add_unit("kJ", 2.778e-4)
    energy.add_unit("MJ", 0.2778)
    energy.add_unit("GJ", 277.8)
    energy.add_unit("BTU", 0.000293)
    energy.add_unit("therm", 29.3)
    registry[Dimension.ENERGY] = energy

    # ── TEMPERATURE (base: °C) ──
    # Note: Temperature requires offset conversion, not just scaling
    # We handle this specially in the conversion function
    temp = DimensionSpec(Dimension.TEMPERATURE, "°C", "Thermal measurement")
    temp.add_unit("°C", 1.0, ["C", "degC", "celsius"])
    temp.add_unit("°F", 1.0, ["F", "degF", "fahrenheit"])  # Special handling
    temp.add_unit("K", 1.0, ["kelvin"])                     # Special handling
    registry[Dimension.TEMPERATURE] = temp

    # ── PRESSURE (base: bar) ──
    pressure = DimensionSpec(Dimension.PRESSURE, "bar", "Force per area")
    pressure.add_unit("Pa", 0.00001)
    pressure.add_unit("kPa", 0.01)
    pressure.add_unit("MPa", 10.0)
    pressure.add_unit("bar", 1.0)
    pressure.add_unit("mbar", 0.001)
    pressure.add_unit("psi", 0.0689476)
    pressure.add_unit("atm", 1.01325)
    pressure.add_unit("mmHg", 0.00133322)
    pressure.add_unit("inHg", 0.0338639)
    registry[Dimension.PRESSURE] = pressure

    # ── FLOW RATE (base: m³/h) ──
    flow = DimensionSpec(Dimension.FLOW_RATE, "m³/h", "Volume per time")
    flow.add_unit("m³/h", 1.0, ["m3/h", "cmh"])
    flow.add_unit("m³/s", 3600.0, ["m3/s", "cms"])
    flow.add_unit("L/s", 3.6, ["l/s", "lps"])
    flow.add_unit("L/min", 0.06, ["l/min", "lpm"])
    flow.add_unit("GPM", 0.227125, ["gpm", "gal/min"])
    flow.add_unit("CFM", 1.699, ["cfm", "ft³/min"])
    registry[Dimension.FLOW_RATE] = flow

    # ── PERCENTAGE (base: %) ──
    pct = DimensionSpec(Dimension.PERCENTAGE, "%", "Ratio expressed as percentage")
    pct.add_unit("%", 1.0, ["percent", "pct"])
    pct.add_unit("fraction", 100.0)          # 0.5 fraction = 50%
    pct.add_unit("ratio", 100.0)
    pct.add_unit("pp", 1.0, ["percentage point"])  # Percentage points
    registry[Dimension.PERCENTAGE] = pct

    # ── FREQUENCY (base: Hz) ──
    freq = DimensionSpec(Dimension.FREQUENCY, "Hz", "Cycles per time")
    freq.add_unit("Hz", 1.0, ["hz", "hertz"])
    freq.add_unit("kHz", 1000.0, ["khz"])
    freq.add_unit("MHz", 1000000.0, ["mhz"])
    freq.add_unit("RPM", 1/60, ["rpm", "rev/min"])
    freq.add_unit("RPS", 1.0, ["rps", "rev/s"])
    registry[Dimension.FREQUENCY] = freq

    # ── VOLTAGE (base: V) ──
    voltage = DimensionSpec(Dimension.VOLTAGE, "V", "Electrical potential")
    voltage.add_unit("V", 1.0, ["volt", "volts"])
    voltage.add_unit("mV", 0.001, ["millivolt"])
    voltage.add_unit("kV", 1000.0, ["kilovolt"])
    registry[Dimension.VOLTAGE] = voltage

    # ── CURRENT (base: A) ──
    current = DimensionSpec(Dimension.CURRENT, "A", "Electrical current")
    current.add_unit("A", 1.0, ["amp", "amps", "ampere"])
    current.add_unit("mA", 0.001, ["milliamp"])
    current.add_unit("kA", 1000.0, ["kiloamp"])
    registry[Dimension.CURRENT] = current

    # ── MASS (base: kg) ──
    mass = DimensionSpec(Dimension.MASS, "kg", "Mass/weight")
    mass.add_unit("g", 0.001, ["gram"])
    mass.add_unit("kg", 1.0, ["kilogram"])
    mass.add_unit("t", 1000.0, ["tonne", "metric ton"])
    mass.add_unit("lb", 0.453592, ["pound", "lbs"])
    mass.add_unit("oz", 0.0283495, ["ounce"])
    mass.add_unit("ton", 907.185, ["short ton"])  # US ton
    registry[Dimension.MASS] = mass

    # ── LENGTH (base: m) ──
    length = DimensionSpec(Dimension.LENGTH, "m", "Distance")
    length.add_unit("mm", 0.001, ["millimeter"])
    length.add_unit("cm", 0.01, ["centimeter"])
    length.add_unit("m", 1.0, ["meter", "metre"])
    length.add_unit("km", 1000.0, ["kilometer"])
    length.add_unit("in", 0.0254, ["inch", "inches"])
    length.add_unit("ft", 0.3048, ["foot", "feet"])
    length.add_unit("yd", 0.9144, ["yard"])
    length.add_unit("mi", 1609.34, ["mile"])
    registry[Dimension.LENGTH] = length

    # ── TIME (base: s) ──
    time_dim = DimensionSpec(Dimension.TIME, "s", "Duration")
    time_dim.add_unit("ms", 0.001, ["millisecond"])
    time_dim.add_unit("s", 1.0, ["sec", "second"])
    time_dim.add_unit("min", 60.0, ["minute"])
    time_dim.add_unit("h", 3600.0, ["hr", "hour"])
    time_dim.add_unit("d", 86400.0, ["day"])
    registry[Dimension.TIME] = time_dim

    # ── DIMENSIONLESS (base: 1) ──
    dimensionless = DimensionSpec(Dimension.DIMENSIONLESS, "1", "Pure numbers")
    dimensionless.add_unit("1", 1.0, ["", "count", "units"])
    registry[Dimension.DIMENSIONLESS] = dimensionless

    return registry


# Global registry instance
DIMENSION_REGISTRY: dict[Dimension, DimensionSpec] = _build_dimension_registry()

# Build reverse lookup: unit symbol → (Dimension, UnitSpec)
UNIT_LOOKUP: dict[str, tuple[Dimension, UnitSpec]] = {}
for dim, spec in DIMENSION_REGISTRY.items():
    for unit_symbol, unit_spec in spec.units.items():
        UNIT_LOOKUP[unit_symbol] = (dim, unit_spec)


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def infer_dimension(unit: str) -> Optional[tuple[Dimension, UnitSpec]]:
    """
    Infer dimension from unit string.

    Returns (Dimension, UnitSpec) if unambiguous, None if unknown.
    """
    if not unit:
        return None

    unit_clean = unit.strip()

    # Direct lookup
    if unit_clean in UNIT_LOOKUP:
        return UNIT_LOOKUP[unit_clean]

    # Case-insensitive lookup
    unit_lower = unit_clean.lower()
    for symbol, (dim, spec) in UNIT_LOOKUP.items():
        if symbol.lower() == unit_lower:
            return (dim, spec)
        for alias in spec.aliases:
            if alias.lower() == unit_lower:
                return (dim, spec)

    return None


def get_base_unit(dimension: Dimension) -> str:
    """Get the base unit symbol for a dimension."""
    spec = DIMENSION_REGISTRY.get(dimension)
    return spec.base_unit if spec else None


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════

class ConversionError(Exception):
    """Raised when unit conversion fails."""
    pass


def convert_value(
    value: float,
    from_unit: str,
    to_unit: str,
    from_semantic: Semantic = Semantic.UNSPECIFIED,
    to_semantic: Semantic = Semantic.UNSPECIFIED,
) -> float:
    """
    Convert a value between units.

    Args:
        value: The numeric value to convert
        from_unit: Source unit symbol
        to_unit: Target unit symbol
        from_semantic: Semantic meaning of source (e.g., CAPACITY, USAGE)
        to_semantic: Semantic meaning of target

    Returns:
        Converted value in target units

    Raises:
        ConversionError: If conversion is not possible or semantics conflict
    """
    # Same unit, no conversion needed
    if from_unit == to_unit:
        return value

    # Infer dimensions
    from_info = infer_dimension(from_unit)
    to_info = infer_dimension(to_unit)

    if from_info is None:
        raise ConversionError(f"Unknown source unit: '{from_unit}'")

    if to_info is None:
        raise ConversionError(f"Unknown target unit: '{to_unit}'")

    from_dim, from_spec = from_info
    to_dim, to_spec = to_info

    # Dimensions must match
    if from_dim != to_dim:
        raise ConversionError(
            f"Dimension mismatch: '{from_unit}' is {from_dim.name}, "
            f"'{to_unit}' is {to_dim.name}. Cannot convert across dimensions."
        )

    # Semantics must match (unless unspecified)
    if (from_semantic != Semantic.UNSPECIFIED and
        to_semantic != Semantic.UNSPECIFIED and
        from_semantic != to_semantic):
        raise ConversionError(
            f"Semantic mismatch: source is {from_semantic.name}, "
            f"target is {to_semantic.name}. Cannot compare {from_semantic.name} to {to_semantic.name}."
        )

    # Special handling for temperature (offset conversion)
    if from_dim == Dimension.TEMPERATURE:
        return _convert_temperature(value, from_unit, to_unit)

    # Standard linear conversion: value * (from_to_base) * (base_to_target)
    # = value * from_spec.to_base * to_spec.from_base
    return value * from_spec.to_base * to_spec.from_base


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert temperature with proper offset handling.

    Temperature conversion requires offset, not just scaling:
    - °C to °F: (°C × 9/5) + 32
    - °F to °C: (°F - 32) × 5/9
    - °C to K: °C + 273.15
    - K to °C: K - 273.15
    """
    # Normalize unit symbols
    from_norm = from_unit.replace("deg", "°").upper()
    to_norm = to_unit.replace("deg", "°").upper()

    if "C" in from_norm or "CELSIUS" in from_norm:
        from_norm = "C"
    elif "F" in from_norm or "FAHRENHEIT" in from_norm:
        from_norm = "F"
    elif "K" in from_norm or "KELVIN" in from_norm:
        from_norm = "K"

    if "C" in to_norm or "CELSIUS" in to_norm:
        to_norm = "C"
    elif "F" in to_norm or "FAHRENHEIT" in to_norm:
        to_norm = "F"
    elif "K" in to_norm or "KELVIN" in to_norm:
        to_norm = "K"

    if from_norm == to_norm:
        return value

    # Convert to Celsius first
    if from_norm == "C":
        celsius = value
    elif from_norm == "F":
        celsius = (value - 32) * 5 / 9
    elif from_norm == "K":
        celsius = value - 273.15
    else:
        raise ConversionError(f"Unknown temperature unit: {from_unit}")

    # Convert from Celsius to target
    if to_norm == "C":
        return celsius
    elif to_norm == "F":
        return (celsius * 9 / 5) + 32
    elif to_norm == "K":
        return celsius + 273.15
    else:
        raise ConversionError(f"Unknown temperature unit: {to_unit}")


def normalize_to_base(value: float, unit: str) -> tuple[float, str, Dimension]:
    """
    Normalize a value to its dimension's base unit.

    Args:
        value: The numeric value
        unit: The unit symbol

    Returns:
        (normalized_value, base_unit, dimension)

    Raises:
        ConversionError: If unit is unknown
    """
    info = infer_dimension(unit)
    if info is None:
        raise ConversionError(f"Cannot infer dimension for unit: '{unit}'")

    dim, spec = info
    base_unit = get_base_unit(dim)

    if dim == Dimension.TEMPERATURE:
        # For temperature, normalize to °C
        normalized = _convert_temperature(value, unit, "°C")
        return (normalized, "°C", dim)

    # Standard linear conversion
    normalized = value * spec.to_base
    return (normalized, base_unit, dim)
