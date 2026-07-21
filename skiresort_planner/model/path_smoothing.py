"""Whole-path smoothing for a finished slope/road across its segment junctions.

Each segment is spline-smoothed independently by the planner, so two segments meet at a
shared node with different tangents — a visible kink. This module fits ONE cubic spline
over the whole joined path and re-slices it back to the original segments, so the ribbon
is C2-continuous across junctions. Pure geometry: no DEM, no UI. Elevation is smoothed
(not DEM-re-queried), so a finished deck may float slightly off ground between nodes —
like a bridge/cut/fill.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.interpolate import PchipInterpolator, splev, splprep
from shapely.geometry import LineString

from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.model.path_point import PathPoint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SplineFit:
    """A fitted HORIZONTAL cubic spline (x/y) plus a shape-preserving elevation profile.

    Horizontal shape and elevation are fitted SEPARATELY, both parameterised by horizontal arc
    length. Horizontal x/y use a cubic *smoothing* spline (rounds the planner's corridor jitter).
    Elevation uses a monotone PCHIP interpolator over arc length: it passes through the input
    elevations and CANNOT overshoot between them, preventing vertical wiggle.
    Bridges/cuts stay intact: elevation is never DEM-requeried.
    """

    tck: tuple[object, ...]  # 2D (x, y) smoothing spline in the local meter frame
    elevation: PchipInterpolator  # arc length (m) → elevation (m), shape-preserving, no overshoot
    lon0: float
    lat0: float
    m_per_deg_lon: float
    m_per_deg_lat: float
    cumdist: npt.NDArray[np.float64]  # per-input-point arc length (the spline parameter u)


def _fit_spline(
    points: list[PathPoint], smoothing_factor: float, step_m: float, weights: npt.NDArray[np.float64] | None
) -> _SplineFit | None:
    """Fit the horizontal shape and the elevation profile SEPARATELY, both over arc length.

    Horizontal x/y (projected to a LOCAL METER FRAME about the first point) get a cubic *smoothing* spline
    that rounds the planner's corridor jitter. Elevation gets a monotone PCHIP interpolator over arc length.
    Returns None when there are too few points or the path is shorter than two steps.

    weights: per-point splprep weights for the horizontal fit (None = uniform); high weights pin
    those vertices. Elevation is interpolated (not weighted): every input elevation is honoured.
    """
    if len(points) < 4:
        return None

    lon0, lat0 = points[0].lon, points[0].lat
    m_per_deg_lon, m_per_deg_lat = GeoCalculator.meters_per_degree(lat=lat0)
    xs = np.array([(p.lon - lon0) * m_per_deg_lon for p in points])
    ys = np.array([(p.lat - lat0) * m_per_deg_lat for p in points])
    elevs = np.array([p.elevation for p in points])

    cumdist = np.array(PathPoint.cumulative_distances(points))
    if float(cumdist[-1]) < step_m * 2:
        return None

    # Both splprep and PCHIP need STRICTLY increasing arc length; drop any point coincident with its
    # predecessor (zero horizontal advance) before fitting. Imported OSM chains can carry such
    # duplicates — feeding them raw made splprep raise "Invalid inputs". `cumdist` (the caller's
    # junction/index reference) stays the FULL array; only the fit inputs are deduped.
    keep = np.concatenate(([True], np.diff(cumdist) > 0))
    if int(keep.sum()) < 4:
        return None  # too few distinct points to fit a cubic spline
    w = weights[keep] if weights is not None else None

    tck, _ = splprep([xs[keep], ys[keep]], u=cumdist[keep], w=w, s=smoothing_factor * int(keep.sum()), k=3)
    elevation = PchipInterpolator(cumdist[keep], elevs[keep])
    return _SplineFit(
        tck=tck,
        elevation=elevation,
        lon0=lon0,
        lat0=lat0,
        m_per_deg_lon=m_per_deg_lon,
        m_per_deg_lat=m_per_deg_lat,
        cumdist=cumdist,
    )


def _eval_spline(fit: _SplineFit, dists: npt.NDArray[np.float64]) -> list[PathPoint]:
    """Evaluate the fit at the given arc-length parameters, back in lon/lat/elev.

    Horizontal x/y come from the 2D smoothing spline; elevation from the shape-preserving PCHIP
    profile (clamped to the fitted arc-length range so end samples never extrapolate).
    """
    new_x, new_y = splev(dists, fit.tck)
    clamped = np.clip(dists, 0.0, float(fit.cumdist[-1]))
    new_elev = fit.elevation(clamped)
    return [
        PathPoint(
            lon=fit.lon0 + float(new_x[i]) / fit.m_per_deg_lon,
            lat=fit.lat0 + float(new_y[i]) / fit.m_per_deg_lat,
            elevation=float(new_elev[i]),
        )
        for i in range(len(new_x))
    ]


def _uniform_resampling_grid(total_length: float, step_m: float) -> npt.NDArray[np.float64]:
    """Uniform distance grid from 0 to total_length inclusive, at step_m intervals."""
    return np.arange(0, total_length + step_m / 2, step_m)


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
    new_dists = _uniform_resampling_grid(total_length, step_m)
    return _eval_spline(fit=fit, dists=new_dists)


def smooth_proposal_points(
    points: list[PathPoint],
    smoothing_factor: float,
    step_m: float,
    elevation_fn: Callable[[float, float], float | None],
) -> list[PathPoint]:
    """Smooth a single-segment proposal (spline) then re-query the DEM at each new position.

    Shared by the fan tracer and the grid planner: both are ground-hugging ribbons, so after
    rounding the horizontal jitter every resampled point takes its DEM elevation (not the
    spline's interpolated one). Returns points unchanged when too short to smooth.
    """
    smoothed = resample_cubic_spline(points=points, smoothing_factor=smoothing_factor, step_m=step_m)
    if smoothed is points:
        return points
    return [PathPoint(lon=p.lon, lat=p.lat, elevation=elevation_fn(p.lon, p.lat) or p.elevation) for p in smoothed]


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
    base = _uniform_resampling_grid(total_length, step_m)
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


def simplify_path_points(points: list[PathPoint], tolerance_m: float) -> list[PathPoint]:
    """Douglas–Peucker (Shapely `LineString.simplify`) dropping interior points within `tolerance_m`
    horizontally of the line between kept neighbours. First/last points are always kept.

    Runs in a local meter frame (lon/lat → metres about the first point) so the tolerance is metres;
    surviving points keep their real elevation, so the ribbon reconstructs within tolerance. Sheds the
    dense ~7 m resampling on straight runs at finish time — cutting render/serialize/transport cost.
    """
    if len(points) <= 2:
        return list(points)
    lon0, lat0 = points[0].lon, points[0].lat
    m_per_deg_lon, m_per_deg_lat = GeoCalculator.meters_per_degree(lat=lat0)
    # LineString in metres about the origin; z carries the elevation so simplify keeps it on survivors.
    line = LineString([((p.lon - lon0) * m_per_deg_lon, (p.lat - lat0) * m_per_deg_lat, p.elevation) for p in points])
    simplified = line.simplify(tolerance_m, preserve_topology=False)
    out = [
        PathPoint(lon=lon0 + x / m_per_deg_lon, lat=lat0 + y / m_per_deg_lat, elevation=z)
        for x, y, z in simplified.coords
    ]
    # DP always keeps the endpoints; restore the originals so the meter-frame round-trip can't drift them.
    out[0], out[-1] = points[0], points[-1]
    return out


def point_at_fraction(points: Sequence[PathPoint], fraction: float) -> PathPoint:
    """The PathPoint at normalized arc-length `fraction` (0..1) along `points`, via Shapely
    `LineString.interpolate` — constant-speed by construction. Used by the flythrough camera + its dot.

    Interpolates in a local meter frame (lon/lat → metres about the first point) so spacing is metric and
    z (elevation) is carried through; fraction 0→first point, 1→last, so endpoints are hit exactly.
    """
    lon0, lat0 = points[0].lon, points[0].lat
    m_per_deg_lon, m_per_deg_lat = GeoCalculator.meters_per_degree(lat=lat0)
    line = LineString([((p.lon - lon0) * m_per_deg_lon, (p.lat - lat0) * m_per_deg_lat, p.elevation) for p in points])
    p = line.interpolate(max(0.0, min(1.0, fraction)), normalized=True)
    return PathPoint(lon=lon0 + p.x / m_per_deg_lon, lat=lat0 + p.y / m_per_deg_lat, elevation=p.z)
