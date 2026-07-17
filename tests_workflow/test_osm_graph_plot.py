"""Tests for the OSM ImportGraph reference plot (generators/osm_graph_plot.py).

render_png is a reference-only artifact (never read back), so these are smoke + contract tests:
it produces a non-empty PNG, refuses an empty graph loudly, and the fabricated (off-piste) overlay
is driven by SlopeRun.fabricated so a reader can tell clean OSM from geometry our code invented.
"""

import pytest

from skiresort_planner.generators.osm_graph_builder import ImportGraph, LiftLine, SlopeRun
from skiresort_planner.generators.osm_graph_plot import _FABRICATED_RGB, _SLOPE_RGB, render_png
from skiresort_planner.model.path_point import PathPoint


def _graph_with_fabricated() -> ImportGraph:
    g = ImportGraph()
    g.node_points = {
        1: PathPoint(lon=10.0, lat=47.0, elevation=2000.0),
        2: PathPoint(lon=10.01, lat=46.99, elevation=1800.0),
    }
    mid = PathPoint(lon=10.005, lat=46.995, elevation=1900.0)
    g.slope_runs = [
        SlopeRun(points=[g.node_points[1], mid, g.node_points[2]], node_a=1, node_b=2, fabricated=[False, True, False])
    ]
    g.lifts = [LiftLine(bottom=g.node_points[2], top=g.node_points[1], lift_type="chairlift", node_a=2, node_b=1)]
    return g


class TestRenderPng:
    def test_writes_a_png(self, tmp_path) -> None:
        out = tmp_path / "osm_import.png"
        render_png(_graph_with_fabricated(), out)
        assert out.exists() and out.stat().st_size > 0

    def test_empty_graph_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="empty ImportGraph"):
            render_png(ImportGraph(), tmp_path / "x.png")

    def test_fabricated_and_clean_use_distinct_colours(self) -> None:
        # The overlay must actually distinguish the two — a guard against them collapsing to one colour.
        assert _FABRICATED_RGB != _SLOPE_RGB

    def test_missing_mask_defaults_to_all_clean(self, tmp_path) -> None:
        # A run with no fabricated mask (e.g. source-less build) still renders (treated as clean OSM).
        g = _graph_with_fabricated()
        g.slope_runs[0].fabricated = []
        render_png(g, tmp_path / "y.png")
        assert (tmp_path / "y.png").exists()


class TestMarkFabricated:
    """The builder flags each run point off every source piste (> PISTE_TOL_M) as fabricated — the
    same off-piste test the pull-shape gate uses, so the plot's red overlay matches what was invented.
    """

    def test_off_piste_points_flagged_on_source_none_all_false(self) -> None:
        from shapely.geometry import LineString, MultiLineString

        from skiresort_planner.constants import OSMConfig
        from skiresort_planner.core.dem_service import DEMService
        from skiresort_planner.generators.osm_graph_builder import OSMGraphBuilder

        class _DEM(DEMService):
            def __new__(cls):
                return object.__new__(cls)

            def __init__(self):
                pass

            def get_elevation(self, lon: float, lat: float) -> float:
                return 2000.0

        bbox = (10.0, 47.0, 10.05, 47.05)
        builder = OSMGraphBuilder(dem=_DEM(), bbox=bbox)

        # A source piste running along y=0 (metres); a run whose middle point sits far off it.
        source = MultiLineString([LineString([(0.0, 0.0), (500.0, 0.0)])])
        on1, on2 = builder._to_deg(50.0, 0.0), builder._to_deg(450.0, 0.0)
        off = builder._to_deg(250.0, OSMConfig.PISTE_TOL_M + 40.0)  # well beyond the off-piste band
        run = SlopeRun(
            points=[
                PathPoint(lon=on1[0], lat=on1[1], elevation=2000.0),
                PathPoint(lon=off[0], lat=off[1], elevation=1950.0),
                PathPoint(lon=on2[0], lat=on2[1], elevation=1900.0),
            ],
            node_a=1,
            node_b=2,
        )
        graph = ImportGraph(slope_runs=[run])
        builder._mark_fabricated(graph, source)
        assert run.fabricated == [False, True, False], "only the off-piste middle point is fabricated"

        # No source → nothing is fabricated (all clean).
        builder._mark_fabricated(graph, None)
        assert run.fabricated == [False, False, False]
