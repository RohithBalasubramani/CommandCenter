"""
Normalizer — Domain-Aware Normalization.

normalize_domain(data: dict, domain_config) -> NormalizationResult

This is a PLUGIN INTERFACE for domain-specific normalization.
It uses external configuration (not hardcoded rules) to:
- Convert units within the same dimension (kW → MW)
- Normalize to canonical representations
- Apply domain-specific validations

IMPORTANT: This is SEPARATE from syntactic rewriting.
Normalizer uses DOMAIN KNOWLEDGE (unit tables, metric registries).
Rewriter uses only SYNTACTIC patterns (string → number).

The default implementation integrates with the existing dimensions.py.
"""
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from layer2.reconciliation.types import (
    NormalizationResult,
    Provenance,
)
from layer2.reconciliation.errors import NormalizationError


# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DimensionConfig:
    """Configuration for a physical dimension."""
    name: str
    base_unit: str
    units: dict[str, float]  # unit -> multiplier to base

    def convert_to_base(self, value: float, from_unit: str) -> tuple[float, str]:
        """Convert value to base unit."""
        if from_unit not in self.units:
            raise NormalizationError(
                f"Unknown unit '{from_unit}' for dimension {self.name}",
                dimension=self.name,
                from_unit=from_unit,
            )
        multiplier = self.units[from_unit]
        return value * multiplier, self.base_unit

    def convert(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert value between units."""
        if from_unit not in self.units or to_unit not in self.units:
            raise NormalizationError(
                f"Unknown unit for dimension {self.name}",
                dimension=self.name,
                from_unit=from_unit,
                to_unit=to_unit,
            )
        # Convert to base, then to target
        base_value = value * self.units[from_unit]
        return base_value / self.units[to_unit]


@dataclass
class DomainConfig:
    """Complete domain configuration."""
    dimensions: dict[str, DimensionConfig]
    default_units: dict[str, str]  # dimension -> default unit

    def get_dimension_for_unit(self, unit: str) -> Optional[DimensionConfig]:
        """Find which dimension a unit belongs to."""
        for dim in self.dimensions.values():
            if unit in dim.units:
                return dim
        return None


# Default configuration - integrates with existing dimensions.py
def build_default_domain_config() -> DomainConfig:
    """Build default domain configuration from dimensions.py."""
    try:
        from layer2.dimensions import DIMENSION_REGISTRY, Dimension

        dimensions = {}
        default_units = {}

        for dim_enum, spec in DIMENSION_REGISTRY.items():
            units = {}
            for unit_symbol, unit_spec in spec.units.items():
                # Skip aliases (they point to the same UnitSpec)
                if unit_symbol == unit_spec.symbol:
                    units[unit_symbol] = unit_spec.to_base

            dimensions[dim_enum.name] = DimensionConfig(
                name=dim_enum.name,
                base_unit=spec.base_unit,
                units=units,
            )
            default_units[dim_enum.name] = spec.base_unit

        return DomainConfig(
            dimensions=dimensions,
            default_units=default_units,
        )

    except ImportError:
        # Fallback if dimensions.py not available
        return DomainConfig(
            dimensions={
                "POWER": DimensionConfig(
                    name="POWER",
                    base_unit="kW",
                    units={"W": 0.001, "kW": 1.0, "MW": 1000.0, "GW": 1000000.0},
                ),
                "ENERGY": DimensionConfig(
                    name="ENERGY",
                    base_unit="kWh",
                    units={"Wh": 0.001, "kWh": 1.0, "MWh": 1000.0, "GWh": 1000000.0},
                ),
                "PERCENTAGE": DimensionConfig(
                    name="PERCENTAGE",
                    base_unit="%",
                    units={"%": 1.0, "fraction": 100.0, "ratio": 100.0},
                ),
            },
            default_units={"POWER": "kW", "ENERGY": "kWh", "PERCENTAGE": "%"},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NORMALIZER INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

class DomainNormalizer(ABC):
    """
    Abstract interface for domain-aware normalization.

    Implementations must:
    1. Use only configured domain knowledge (no hardcoded rules)
    2. Record all transformations with provenance
    3. Fail explicitly if normalization is ambiguous
    """

    @abstractmethod
    def normalize(self, data: dict, scenario: str) -> NormalizationResult:
        """Normalize data using domain configuration."""
        pass

    @abstractmethod
    def get_config(self) -> DomainConfig:
        """Return the domain configuration being used."""
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

class DefaultDomainNormalizer(DomainNormalizer):
    """
    Default domain normalizer using dimensions.py configuration.

    Normalizes units to base units within each dimension.
    """

    def __init__(self, config: Optional[DomainConfig] = None):
        self.config = config or build_default_domain_config()

    def get_config(self) -> DomainConfig:
        return self.config

    def normalize(self, data: dict, scenario: str) -> NormalizationResult:
        """
        Normalize data using domain configuration.

        Looks for value/unit patterns and normalizes to base units.

        Args:
            data: Data to normalize
            scenario: Widget scenario for context

        Returns:
            NormalizationResult with normalized data and provenance
        """
        result = copy.deepcopy(data)
        provenance: list[Provenance] = []
        warnings: list[str] = []
        transforms_applied = 0

        # Look for demoData wrapper
        if "demoData" in result and isinstance(result["demoData"], dict):
            check_dict = result["demoData"]
        else:
            check_dict = result

        # Normalize value/unit pairs
        if "value" in check_dict and "unit" in check_dict:
            value = check_dict["value"]
            unit = check_dict["unit"]

            if isinstance(value, (int, float)) and isinstance(unit, str) and unit:
                normalized = self._normalize_value_unit(value, unit, "value")
                if normalized:
                    original = f"{value} {unit}"
                    check_dict["value"] = normalized[0]
                    check_dict["unit"] = normalized[1]
                    transforms_applied += 1
                    provenance.append(Provenance.create(
                        rule_id="domain_normalize",
                        description=f"Normalized to {normalized[1]}",
                        original=original,
                        transformed=f"{normalized[0]} {normalized[1]}",
                        reversible=True,
                        inverse_function="domain_denormalize",
                    ))

        # Normalize valueA/valueB for comparisons
        for field in ["valueA", "valueB"]:
            if field in check_dict:
                value = check_dict[field]
                unit = check_dict.get("unit", "")

                if isinstance(value, (int, float)) and isinstance(unit, str) and unit:
                    normalized = self._normalize_value_unit(value, unit, field)
                    if normalized and normalized[1] != unit:
                        original = f"{value} {unit}"
                        check_dict[field] = normalized[0]
                        transforms_applied += 1
                        provenance.append(Provenance.create(
                            rule_id="domain_normalize",
                            description=f"Normalized {field} to base unit",
                            original=original,
                            transformed=f"{normalized[0]} {normalized[1]}",
                            reversible=True,
                        ))

        # Normalize series items
        if "series" in check_dict and isinstance(check_dict["series"], list):
            unit = check_dict.get("unit", "")
            for i, item in enumerate(check_dict["series"]):
                if isinstance(item, dict) and "value" in item:
                    value = item["value"]
                    if isinstance(value, (int, float)) and unit:
                        normalized = self._normalize_value_unit(value, unit, f"series[{i}].value")
                        if normalized and normalized[1] != unit:
                            original = f"{value} {unit}"
                            item["value"] = normalized[0]
                            transforms_applied += 1
                            provenance.append(Provenance.create(
                                rule_id="domain_normalize",
                                description=f"Normalized series[{i}].value",
                                original=original,
                                transformed=f"{normalized[0]} {normalized[1]}",
                                reversible=True,
                            ))

            # Update unit if normalized
            if provenance and "series" in check_dict:
                dim = self.config.get_dimension_for_unit(unit)
                if dim:
                    check_dict["unit"] = dim.base_unit

        return NormalizationResult(
            success=True,
            data=result,
            transforms_applied=transforms_applied,
            provenance=provenance,
            warnings=warnings,
        )

    def _normalize_value_unit(
        self,
        value: float,
        unit: str,
        field_path: str,
    ) -> Optional[tuple[float, str]]:
        """
        Normalize a value/unit pair to base unit.

        Returns (normalized_value, base_unit) or None if no normalization needed.
        """
        dim = self.config.get_dimension_for_unit(unit)
        if dim is None:
            return None  # Unknown unit - pass through

        if unit == dim.base_unit:
            return None  # Already in base unit

        try:
            normalized_value, base_unit = dim.convert_to_base(value, unit)
            return (normalized_value, base_unit)
        except NormalizationError:
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# PLUGIN REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

# Global normalizer instance (can be replaced for testing or custom domains)
_default_normalizer: Optional[DomainNormalizer] = None


def get_normalizer() -> DomainNormalizer:
    """Get the configured domain normalizer."""
    global _default_normalizer
    if _default_normalizer is None:
        _default_normalizer = DefaultDomainNormalizer()
    return _default_normalizer


def set_normalizer(normalizer: DomainNormalizer) -> None:
    """Set a custom domain normalizer."""
    global _default_normalizer
    _default_normalizer = normalizer


def normalize_domain(data: dict, scenario: str) -> NormalizationResult:
    """
    Normalize data using the configured domain normalizer.

    This is the main entry point for domain normalization.

    Args:
        data: Data to normalize
        scenario: Widget scenario for context

    Returns:
        NormalizationResult with normalized data and provenance
    """
    normalizer = get_normalizer()
    return normalizer.normalize(data, scenario)
