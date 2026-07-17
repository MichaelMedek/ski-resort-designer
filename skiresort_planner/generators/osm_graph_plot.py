"""Plot a built OSM ImportGraph to a PNG map, for reference / manual inspection.

The PNG plots slope runs (blue), lifts (red), and hub nodes (black) in a simple lon/lat projection.
This is the ONLY thing this module does — it is written to the output folder for inspection and is
never read back into the app.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from skiresort_planner.generators.osm_graph_builder import ImportGraph

logger = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]

# Render geometry / colours (kept local — this is presentation, not a domain constant).
_W, _H, _PAD = 1500, 1050, 40
_SLOPE_RGB = (31, 119, 180)  # clean OSM slope geometry
_FABRICATED_RGB = (214, 39, 40)  # geometry our code invented (off-piste pull/connector)
_LIFT_RGB = (148, 0, 211)  # lifts (purple, so red always means "fabricated")
_NODE_RGB = (0, 0, 0)


def _extent(graph: "ImportGraph") -> BBox:
    """(min_lon, min_lat, max_lon, max_lat) enclosing every slope/lift point, padded 2%. Raises if
    the graph is empty (nothing to plot — a loud failure beats a blank canvas).
    """
    pts: list[tuple[float, float]] = [(p.lon, p.lat) for r in graph.slope_runs for p in r.points]
    for lf in graph.lifts:
        pts += [(lf.bottom.lon, lf.bottom.lat), (lf.top.lon, lf.top.lat)]
    pts += [(p.lon, p.lat) for p in graph.node_points.values()]
    if not pts:
        raise ValueError("cannot plot an empty ImportGraph (no slopes, lifts, or nodes)")
    lons = [lon for lon, _ in pts]
    lats = [lat for _, lat in pts]
    min_lon, max_lon, min_lat, max_lat = min(lons), max(lons), min(lats), max(lats)
    pad_lon = (max_lon - min_lon) * 0.02 or 1e-4
    pad_lat = (max_lat - min_lat) * 0.02 or 1e-4
    return (min_lon - pad_lon, min_lat - pad_lat, max_lon + pad_lon, max_lat + pad_lat)


def render_png(graph: "ImportGraph", path: Path) -> None:
    """Render the graph to a PNG at `path`: clean-OSM slope geometry blue, fabricated (off-piste
    pull/connector) geometry red, lifts purple, hub nodes black. The extent is computed from the
    graph's own points, so no bbox is needed.
    """
    min_lon, min_lat, max_lon, max_lat = _extent(graph)

    def px(lon: float, lat: float) -> tuple[float, float]:
        x = _PAD + (lon - min_lon) / (max_lon - min_lon) * (_W - 2 * _PAD)
        y = _H - _PAD - (lat - min_lat) / (max_lat - min_lat) * (_H - 2 * _PAD)
        return (x, y)

    img = Image.new("RGB", (_W, _H), "white")
    draw = ImageDraw.Draw(img)
    for r in graph.slope_runs:
        # Colour each leg by whether its higher-index endpoint is fabricated (off-piste): a contiguous
        # pulled connector renders red, the on-OSM body blue. Empty mask (no source) → all blue.
        pts = [px(p.lon, p.lat) for p in r.points]
        fab = r.fabricated or [False] * len(pts)
        for i in range(len(pts) - 1):
            leg_fab = fab[i] or fab[i + 1]
            draw.line([pts[i], pts[i + 1]], fill=_FABRICATED_RGB if leg_fab else _SLOPE_RGB, width=2)
    for lf in graph.lifts:
        draw.line([px(lf.bottom.lon, lf.bottom.lat), px(lf.top.lon, lf.top.lat)], fill=_LIFT_RGB, width=4)
    for p in graph.node_points.values():
        x, y = px(p.lon, p.lat)
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=_NODE_RGB)
    fab_runs = sum(1 for r in graph.slope_runs if any(r.fabricated))
    title = (
        f"OSM import — {len(graph.node_points)} nodes, {len(graph.slope_runs)} slopes "
        f"(blue=OSM, red=fabricated in {fab_runs}), {len(graph.lifts)} lifts (purple)"
    )
    draw.text((_PAD, 12), title, fill=_NODE_RGB, font=ImageFont.load_default())
    img.save(path)
