"""Whole-path smoothing for a finished slope/road across its segment junctions.

Each segment is spline-smoothed independently by the planner, so two segments meet at a
shared node with different tangents — a visible kink. This module fits ONE cubic spline
over the whole joined path and re-slices it back to the original segments, so the ribbon
is C2-continuous across junctions. Pure geometry: no DEM, no UI. Elevation is smoothed
(not DEM-re-queried), so a finished deck may float slightly off ground between nodes —
like a bridge/cut/fill.
"""

import logging
from dataclasses import dataclass
from math import cos, radians

import numpy as np
import numpy.typing as npt
from scipy.interpolate import splev, splprep

from skiresort_planner.constants import MapConfig
from skiresort_planner.model.path_point import PathPoint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SplineFit:
    """A fitted cubic spline plus the local meter frame needed to evaluate it back to lon/lat."""

    tck: tuple[object, ...]
    lon0: float
    lat0: float
    m_per_deg_lon: float
    m_per_deg: float
    cumdist: npt.NDArray[np.float64]  # per-input-point arc length (the spline parameter u)


def _cumulative_distances(points: list[PathPoint]) -> list[float]:
    """Cumulative horizontal distance (m) along a polyline, starting at 0."""
    cum = [0.0]
    for i in range(1, len(points)):
        cum.append(cum[-1] + points[i - 1].distance_to(other=points[i]))
    return cum


def _fit_spline(
    points: list[PathPoint], smoothing_factor: float, step_m: float, weights: npt.NDArray[np.float64] | None
) -> _SplineFit | None:
    """Fit a cubic smoothing spline in a LOCAL METER FRAME (lon/lat projected to meters
    about the first point) so all three dimensions share one scale — otherwise degree-scale
    position is swamped by meter-scale elevation in splprep's residual budget and the curve
    drifts. Returns None when there are too few points or the path is shorter than two steps.

    weights: per-point splprep weights (None = uniform); high weights pin those vertices.
    """
    if len(points) < 4:
        return None

    lon0, lat0 = points[0].lon, points[0].lat
    m_per_deg = MapConfig.METERS_PER_DEGREE_EQUATOR
    m_per_deg_lon = m_per_deg * cos(radians(lat0))
    xs = np.array([(p.lon - lon0) * m_per_deg_lon for p in points])
    ys = np.array([(p.lat - lat0) * m_per_deg for p in points])
    elevs = np.array([p.elevation for p in points])

    cumdist = np.array(_cumulative_distances(points=points))
    if float(cumdist[-1]) < step_m * 2:
        return None

    tck, _ = splprep([xs, ys, elevs], u=cumdist, w=weights, s=smoothing_factor * len(points), k=3)
    return _SplineFit(tck=tck, lon0=lon0, lat0=lat0, m_per_deg_lon=m_per_deg_lon, m_per_deg=m_per_deg, cumdist=cumdist)


def _eval_spline(fit: _SplineFit, dists: npt.NDArray[np.float64]) -> list[PathPoint]:
    """Evaluate a fitted spline at the given arc-length parameters, back in lon/lat/elev."""
    new_x, new_y, new_elev = splev(dists, fit.tck)
    return [
        PathPoint(
            lon=fit.lon0 + float(new_x[i]) / fit.m_per_deg_lon,
            lat=fit.lat0 + float(new_y[i]) / fit.m_per_deg,
            elevation=float(new_elev[i]),
        )
        for i in range(len(new_x))
    ]


def resample_cubic_spline(points: list[PathPoint], smoothing_factor: float, step_m: float) -> list[PathPoint]:
    """Fit a cubic smoothing spline through points and resample it every step_m. Elevation
    is the spline's 3rd dimension (not DEM-sampled).

    Returns points unchanged when there are too few points or the path is shorter than two
    steps. Shared spline core; callers add their own elevation post-processing.
    """
    fit = _fit_spline(points=points, smoothing_factor=smoothing_factor, step_m=step_m, weights=None)
    if fit is None:
        return points
    total_length = float(fit.cumdist[-1])
    new_dists = np.arange(0, total_length + step_m / 2, step_m)
    return _eval_spline(fit=fit, dists=new_dists)


def smooth_joined_path(
    *,
    segment_point_lists: list[list[PathPoint]],
    node_anchors: list[PathPoint],
    step_m: float,
    smoothing_factor: float,
    node_weight: float,
    corridor_weight: float,
) -> list[list[PathPoint]]:
    """Smooth a multi-segment path across its junctions and re-slice it per segment.

    Fits ONE cubic spline over the whole joined path with a WEIGHTED least-squares balance:
    the boundary nodes get a heavy weight (pulled hard onto the curve — they're authoritative
    and any node can be a branch point), while the raw planner corridor points get a light
    weight, acting as a soft "magnetic pull". This lets the spline average the planner's
    staircase / switchback-reversal jitter into a smooth radius instead of threading every
    noisy point and collapsing to a zero-speed CUSP (the sharp-edge bug at switchbacks).
    The path is re-sliced at the junction arc-positions so adjacent segments share the
    junction point by value. A single segment is smoothed too (no junction, just rounded).

    node_anchors: authoritative node coords, one per boundary — [outer start, junction_1,
        ..., junction_{n-1}, outer end]; length == len(segment_point_lists) + 1.
    """
    assert segment_point_lists, "smooth_joined_path needs at least one segment"

    # Join, deduping each segment's first point against the previous segment's last.
    joined: list[PathPoint] = list(segment_point_lists[0])
    junction_after: list[int] = []  # joined-index of each internal junction node
    for seg in segment_point_lists[1:]:
        junction_after.append(len(joined) - 1)
        joined.extend(seg[1:])

    # Set the exact node coords at every boundary, then weight nodes heavily and corridor
    # points lightly so the fit is pulled onto the nodes but only softly toward the corridor.
    node_indices = [0, *junction_after, len(joined) - 1]
    for idx, anchor in zip(node_indices, node_anchors, strict=True):
        joined[idx] = anchor
    weights = np.full(len(joined), corridor_weight)
    weights[node_indices] = node_weight

    fit = _fit_spline(points=joined, smoothing_factor=smoothing_factor, step_m=step_m, weights=weights)
    if fit is None:
        return segment_point_lists  # too short to smooth — leave the raw segments intact

    # Resample on a uniform grid but force an exact sample at every junction arc-position,
    # so each cut lands on its node and adjacent segments share that point by value.
    total_length = float(fit.cumdist[-1])
    base = np.arange(0, total_length + step_m / 2, step_m)
    junction_dists = [float(fit.cumdist[j]) for j in junction_after]
    all_dists = np.array(sorted(set(base.tolist()) | set(junction_dists)))
    smoothed = _eval_spline(fit=fit, dists=all_dists)

    # Pin ONLY the outer endpoints exactly — they are the entity termini.
    smoothed[0] = node_anchors[0]
    smoothed[-1] = node_anchors[-1]
    cut_indices = [int(np.searchsorted(all_dists, d)) for d in junction_dists]

    # Slice inclusive-both-ends so seg[k].points[-1] == seg[k+1].points[0] by value.
    result: list[list[PathPoint]] = []
    start = 0
    for cut in cut_indices:
        result.append(smoothed[start : cut + 1])
        start = cut
    result.append(smoothed[start:])
    return result
