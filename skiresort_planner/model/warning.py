"""Warning - Slope segment warnings for construction considerations.

Warnings indicate situations requiring special attention during construction:
- Excavator work for side cuts exceeding threshold
- Slope too steep for safe skiing
- Slope too flat to keep skiers gliding

Reference: DETAILS.md Section 4
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from skiresort_planner.core.terrain_analyzer import SideDirection


@dataclass(frozen=True)
class Warning(ABC):
    """Abstract base class for slope warnings.

    Subclasses store specific parameters and compute their message as a property.
    """

    @property
    @abstractmethod
    def message(self) -> str:
        """Human-readable warning message with emoji prefix."""

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class ExcavatorWarning(Warning):
    """Warning for side cuts requiring excavator work.

    Attributes:
        side_slope_pct: Absolute side slope percentage
        belt_width_m: Width of the piste in meters
        side_slope_dir: Direction terrain leans (SideDirection)
    """

    side_slope_pct: float
    belt_width_m: float
    side_slope_dir: SideDirection

    @property
    def vertical_cut_m(self) -> float:
        """Computed vertical cut depth required."""
        return (abs(self.side_slope_pct) * self.belt_width_m) / 200

    @property
    def message(self) -> str:
        return (
            f"🚜 Excavator Warning: {self.vertical_cut_m:.1f}m vertical side cut required "
            f"across {self.belt_width_m:.0f}m piste width. Cross-slope {self.side_slope_pct:.0f}% "
            f"(terrain leans {self.side_slope_dir.value})"
        )


@dataclass(frozen=True)
class TooSteepWarning(Warning):
    """Warning for slopes exceeding maximum safe gradient.

    Attributes:
        slope_pct: Actual slope percentage
        max_threshold_pct: Maximum safe threshold
    """

    slope_pct: float
    max_threshold_pct: float

    @property
    def message(self) -> str:
        return (
            f"⚠️ Too Steep Warning: Gradient {self.slope_pct:.0f}% exceeds maximum "
            f"safe slope of {self.max_threshold_pct:.0f}% - may require terrain modification"
        )


@dataclass(frozen=True)
class TooFlatWarning(Warning):
    """Warning for slopes below minimum skiable gradient.

    Attributes:
        slope_pct: Actual slope percentage
        min_threshold_pct: Minimum skiable threshold
    """

    slope_pct: float
    min_threshold_pct: float

    @property
    def message(self) -> str:
        return (
            f"📐 Too Flat Warning: Gradient {self.slope_pct:.0f}% is below minimum "
            f"skiable slope of {self.min_threshold_pct:.0f}% - skiers may need to push"
        )
