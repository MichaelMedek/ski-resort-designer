"""Completeness guards — tripwires that make it impossible to add a feature *half*-wired.

These tests don't check behavior; they check that parallel code sites stay in sync. Each guards a
hazard where adding an enum member / dataclass field / new module would compile and pass every
other test, yet ship a silent bug or a rarely-hit crash:

- Serialization: a new dataclass field that `from_dict` forgets → silently lost on save/load.
- Enum dispatch: a new enum member a total (`else: raise`) dispatcher forgets → runtime crash.
- Layering: a model/core/generator module importing `ui` → an architecture violation.

The ActionType undo-dispatcher guard lives in test_resort_graph.py (it's undo-specific).
"""

import ast
import dataclasses
import inspect

from skiresort_planner.constants import PACKAGE_DIR

# =============================================================================
# 1. Serialization round-trip: every dataclass field survives to_dict → from_dict
# =============================================================================


class TestSerializationCompleteness:
    """A dataclass field that `from_dict` doesn't restore is silently lost on reload.

    Two complementary checks: a source scan (new field not named in from_dict) and a real
    round-trip (the field's value actually survives). Either catches the osm_key/rename-class bug.
    """

    def _manual_from_dict_classes(self):
        from skiresort_planner.model.lift import Lift
        from skiresort_planner.model.path_segment import PathSegment
        from skiresort_planner.model.segment_path import SegmentPath

        # Classes with a HAND-WRITTEN from_dict that lists fields (the drop hazard). Node/Pylon/
        # PathPoint deserialize via ** unpack, so they can't drop a field and need no guard.
        return [SegmentPath, PathSegment, Lift]

    def test_every_field_named_in_from_dict(self) -> None:
        offenders = {}
        for cls in self._manual_from_dict_classes():
            src = inspect.getsource(cls.__dict__["from_dict"])
            missing = [f.name for f in dataclasses.fields(cls) if f.name not in src]
            if missing:
                offenders[cls.__name__] = missing
        assert not offenders, f"from_dict must restore every field; missing: {offenders}"

    def test_graph_roundtrip_preserves_all_entity_fields(self, empty_graph, path_points_blue, mock_dem_blue_slope):
        """Build one of each entity, round-trip the whole graph, assert every entity is identical.

        Fields deserialized with `.get(key, default)` (side_slope_pct/side_slope_dir) can be silently
        dropped yet still equal the default — so we force NON-DEFAULT values on a segment first, or a
        dropped field would slip through this equality check.
        """
        import json

        from skiresort_planner.model.path_point import PathPoint
        from skiresort_planner.model.path_segment import SegmentKind, SideDirection
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.model.resort_graph import ResortGraph

        graph, dem = empty_graph, mock_dem_blue_slope
        m = 111320.0
        # A slope.
        graph.commit_paths(paths=[ProposedPathSegment(points=path_points_blue, target_difficulty="blue")])
        graph.finish_slope(segment_ids=list(graph.segments.keys()))
        # A road.
        road_pts = [
            PathPoint(lon=0.3, lat=0.3, elevation=2000.0),
            PathPoint(lon=0.3 + 300 / m, lat=0.3, elevation=1990.0),
        ]
        graph.commit_paths(paths=[ProposedPathSegment(points=road_pts, is_connector=True, kind=SegmentKind.ROAD)])
        graph.finish_road(segment_ids=[list(graph.segments.keys())[-1]])
        # A lift.
        bottom, _ = graph.get_or_create_node(
            lon=0.0, lat=-1000 / m, elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / m)
        )
        top, _ = graph.get_or_create_node(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
        graph.add_lift(start_node_id=bottom.id, end_node_id=top.id, lift_type="chairlift", dem=dem)

        # Force non-default values on every optional/`.get`-defaulted field so dropping it in
        # from_dict produces a DIFFERENT object (otherwise default==default masks the loss).
        for seg in graph.segments.values():
            seg.side_slope_pct = 12.3
            seg.side_slope_dir = SideDirection.LEFT

        restored = ResortGraph.from_dict(data=json.loads(json.dumps(graph.to_dict())))

        # dataclass __eq__ compares every field, so a dropped field fails here.
        assert restored.slopes == graph.slopes
        assert restored.roads == graph.roads
        assert restored.lifts == graph.lifts
        assert restored.segments == graph.segments
        assert restored.nodes == graph.nodes


# =============================================================================
# 2. Enum dispatch: total (`else: raise`) dispatchers must handle every member
# =============================================================================


class TestEnumDispatchCompleteness:
    """A total dispatcher (one with a final `raise` on unknown) must name every member of its enum.

    Only TOTAL dispatchers are listed — functions that legitimately handle a subset (e.g. a
    click handler that only cares about some marker types) are excluded on purpose.
    """

    def _cases(self):
        from skiresort_planner.model.click_info import ClickInfo, MarkerType
        from skiresort_planner.model.path_segment import SegmentKind
        from skiresort_planner.ui import bottom_chart, context
        from skiresort_planner.ui.context import BuildMode, EntityKind

        marker_names = [m.name for m in MarkerType]
        segkind_names = [m.name for m in SegmentKind]
        entity_names = [m.name for m in EntityKind]
        # BuildMode is a plain-attr class (not an Enum); BuildMode.ALL is its source-of-truth list.
        buildmode_names = [m.upper() for m in BuildMode.ALL]

        # (label, function-or-source, qualifier, member-names)
        return [
            ("ClickInfo._validate_marker_ids", ClickInfo.__dict__["_validate_marker_ids"], "MarkerType", marker_names),
            ("ClickInfo.display_name", ClickInfo.__dict__["display_name"].fget, "MarkerType", marker_names),
            (
                "bottom_chart.render_building_profile",
                bottom_chart.render_building_profile,
                "SegmentKind",
                segkind_names,
            ),
            ("bottom_chart.render_viewing_profile", bottom_chart.render_viewing_profile, "EntityKind", entity_names),
            ("context.BuildMode.display_name", context.BuildMode.display_name, "BuildMode", buildmode_names),
            ("context.BuildMode.icon", context.BuildMode.icon, "BuildMode", buildmode_names),
        ]

    def test_total_dispatchers_handle_every_member(self) -> None:
        offenders = {}
        for label, func, qual, names in self._cases():
            src = inspect.getsource(func)
            missing = [n for n in names if f"{qual}.{n}" not in src]
            if missing:
                offenders[label] = missing
        assert not offenders, f"total dispatchers missing branches: {offenders}"


# =============================================================================
# 3. Layering: model / core / generators must never import ui
# =============================================================================


class TestLayering:
    """The model/core/generator layers must not depend on the ui layer.

    ui builds on model+core+generators, never the reverse. A model module importing ui (e.g. an
    EntityKind leak into the graph) would tangle the layers and risk import cycles.
    """

    def test_lower_layers_do_not_import_ui(self) -> None:
        lower = [PACKAGE_DIR / "model", PACKAGE_DIR / "core", PACKAGE_DIR / "generators"]
        offenders = []
        for layer in lower:
            for py in layer.rglob("*.py"):
                tree = ast.parse(py.read_text(), filename=str(py))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("skiresort_planner.ui"):
                        offenders.append(f"{py.relative_to(PACKAGE_DIR)}:{node.lineno} imports {node.module}")
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name.startswith("skiresort_planner.ui"):
                                offenders.append(f"{py.relative_to(PACKAGE_DIR)}:{node.lineno} imports {alias.name}")
        assert not offenders, "lower layers must not import ui:\n" + "\n".join(offenders)


# =============================================================================
# 4. NodeConnected contract: every subclass exposes the endpoint interface
# =============================================================================


class TestNodeConnectedContract:
    """Every concrete NodeConnected subclass (Slope, Road, Lift) must expose `id`, `start_node_id`,
    and `end_node_id` — as a dataclass field (Lift: stored) or a property (Slope/Road: derived).
    """

    def test_every_subclass_provides_the_endpoint_members(self) -> None:
        import dataclasses as dc

        from skiresort_planner.model.lift import Lift  # noqa: F401 — import registers the subclass
        from skiresort_planner.model.node_connected import NodeConnected
        from skiresort_planner.model.road import Road  # noqa: F401
        from skiresort_planner.model.slope import Slope  # noqa: F401

        required = ("id", "start_node_id", "end_node_id")

        def all_descendants(cls: type) -> set[type]:
            subs = set(cls.__subclasses__())
            return subs.union(*(all_descendants(s) for s in subs))

        # Every descendant must satisfy the contract (SegmentPath supplies it via properties, Lift via
        # fields, Slope/Road inherit SegmentPath's) — no leaf/abstract filtering needed: the check is
        # uniform, a class either exposes each member as a property or a dataclass field, or it fails.
        descendants = all_descendants(NodeConnected)
        assert {c.__name__ for c in descendants} >= {"Slope", "Road", "Lift"}, "expected the 3 known entities"

        offenders = {}
        for cls in descendants:
            field_names = {f.name for f in dc.fields(cls)} if dc.is_dataclass(cls) else set()
            missing = [m for m in required if not (isinstance(getattr(cls, m, None), property) or m in field_names)]
            if missing:
                offenders[cls.__name__] = missing
        assert not offenders, f"NodeConnected subclasses missing endpoint members: {offenders}"
