"""SegmentProfile - classify a committed segment as bridge/tunnel/ground vs the DEM.

A finished slope/road is smoothed (not re-draped), so its deck may float above terrain (bridge) or
cut below it (tunnel). This classifies the WHOLE segment by its worst deviation. Derived on demand
(never stored — points + DEM change on finish/restitch/load).
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from skiresort_planner.constants import StyleConfig

if TYPE_CHECKING:
    from skiresort_planner.core.dem_service import DEMService
    from skiresort_planner.model.path_segment import PathSegment


class SegmentProfile(StrEnum):
    """How a committed segment sits relative to terrain — reload-safe StrEnum compared with ==."""

    GROUND = "ground"
    BRIDGE = "bridge"
    TUNNEL = "tunnel"


# StyleConfig can't import model (keeps its no-model-import rule), so it keys tints by profile value.
# Assert the bijection here at import time so a new profile can't silently miss a tint (mirrors the
# SEGMENT_FLAT_Z ↔ SegmentKind assert in ui/kind_spec.py). GROUND has no tint (never tinted).
assert set(StyleConfig.STRUCTURE_TINT_RGB.keys()) == {SegmentProfile.BRIDGE.value, SegmentProfile.TUNNEL.value}


@dataclass(frozen=True)
class SegmentProfileResult:
    """Classification plus the deviation magnitudes that drove it (both non-negative)."""

    profile: SegmentProfile
    max_above_m: float
    max_below_m: float


def classify_segment_profile(*, segment: "PathSegment", dem: "DEMService", threshold_m: float) -> SegmentProfileResult:
    """Classify a segment as bridge/tunnel/ground by its worst deviation from the terrain surface.

    Deviation = point.elevation - terrain. The whole segment takes one class (its worst point). Ties
    (equal above/below magnitude) resolve to BRIDGE.

    Args:
        segment: Committed segment whose points carry smoothed (non-re-draped) elevations.
        dem: Elevation source, queried once (vectorized) at every point.
        threshold_m: Deviation magnitude below which the segment is GROUND.
    """
    lons = [p.lon for p in segment.points]
    lats = [p.lat for p in segment.points]
    elevs = np.array([p.elevation for p in segment.points], dtype=np.float64)

    # A LOADED backup may have been built against a different DEM (other region / updated file), so some
    # points can sit off the currently-loaded coverage → NaN. That's external file data, not an internal
    # invariant, so classify over the in-coverage points and skip the NaNs rather than crashing.
    ground = dem.get_elevations(lons=lons, lats=lats)
    on_dem = ~np.isnan(ground)
    if not on_dem.any():
        return SegmentProfileResult(
            profile=SegmentProfile.GROUND, max_above_m=0.0, max_below_m=0.0
        )  # nothing to measure

    # Mask the off-DEM points (partial coverage) so their NaN doesn't poison max/min.
    deviation = elevs[on_dem] - ground[on_dem]
    max_above_m = max(0.0, float(deviation.max()))
    max_below_m = max(0.0, float(-deviation.min()))

    worst = max(max_above_m, max_below_m)
    if worst <= threshold_m:
        # Worst devaition inside treshold
        profile = SegmentProfile.GROUND
    # Worst devaition outside treshold, now check if avove or below
    elif max_above_m >= max_below_m:
        # deviation dominated by floating ABOVE terrain
        profile = SegmentProfile.BRIDGE
    else:
        # dominated by cutting BELOW terrain
        profile = SegmentProfile.TUNNEL

    return SegmentProfileResult(profile=profile, max_above_m=max_above_m, max_below_m=max_below_m)
