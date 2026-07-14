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

    Two complementary checks: a source scan (every field named somewhere in the deserialization
    source) and a real round-trip (the field's value actually survives). Either catches the
    osm_key/rename-class bug.
    """

    def _deserialization_targets(self):
        """Discover (dataclass, deserialization-source) pairs whose from_dict hand-lists fields.

        Returns a list of (cls, source_str). A class maps to the source of the from_dict that
        actually constructs it: its own, or an inherited one (for subclasses that don't override).
        Nested reconstruction (Pylon built inside Lift.from_dict) is added explicitly below, derived
        from "which model dataclasses are serialized but have NO from_dict of their own and are not
        `**`-safe" — i.e. rebuilt by a keyword constructor in some other class's from_dict.
        """
        import dataclasses as dc

        from skiresort_planner.model.lift import Lift
        from skiresort_planner.model.node import Node
        from skiresort_planner.model.path_segment import PathSegment
        from skiresort_planner.model.pylon import Pylon
        from skiresort_planner.model.road import Road
        from skiresort_planner.model.segment_path import SegmentPath
        from skiresort_planner.model.slope import Slope

        targets: list[tuple[type, str]] = []

        def fd_source(owner: type) -> str:
            # Source of the from_dict that constructs `owner` — its own or the inherited one.
            for klass in owner.__mro__:
                if "from_dict" in klass.__dict__:
                    return inspect.getsource(klass.__dict__["from_dict"])
            raise AssertionError(f"{owner.__name__} has no from_dict in its MRO")

        # Classes rebuilt by a hand-listed from_dict (own or inherited). SegmentPath itself plus its
        # concrete subclasses (whose extra fields must appear in the inherited source), the segment,
        # the lift, and the node (lists `id` explicitly, only `location` is ** unpacked).
        for cls in (SegmentPath, Slope, Road, PathSegment, Lift, Node):
            targets.append((cls, fd_source(cls)))

        # Pylon has no from_dict and is NOT ** unpacked — Lift.from_dict rebuilds it field-by-field,
        # so a new Pylon field would be dropped there. Check its fields against Lift's from_dict.
        assert "from_dict" not in Pylon.__dict__ and dc.is_dataclass(Pylon)
        targets.append((Pylon, inspect.getsource(Lift.__dict__["from_dict"])))

        return targets

    def test_every_field_named_in_from_dict(self) -> None:
        offenders = {}
        for cls, src in self._deserialization_targets():
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

        from skiresort_planner.core.terrain_analyzer import SideDirection
        from skiresort_planner.model.path_point import PathPoint
        from skiresort_planner.model.path_segment import SegmentKind
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

        # The id counters must round-trip too — a dropped/mismatched counter would reissue an
        # existing id (collision) or, with a `[]` read on an old save, KeyError on load.
        assert restored._node_counter == graph._node_counter
        assert restored._segment_counter == graph._segment_counter
        assert restored._slope_counter == graph._slope_counter
        assert restored._lift_counter == graph._lift_counter
        assert restored._road_counter == graph._road_counter


# =============================================================================
# 2. Enum dispatch: total (`else: raise`) dispatchers must handle every member
# =============================================================================


class TestEnumDispatchCompleteness:
    """Every TOTAL dispatcher (a match/if chain ending in `raise` on an unhandled member) must name
    every member of the enum it switches on — else a new member ships a rarely-hit runtime crash.

    The enum MEMBERS are derived from production (that's the part that must not drift — a new member
    is exactly what we're guarding against). The list of dispatchers, and which members a
    deliberately PARTIAL one may omit, is stated explicitly here with a reason: "is this function
    meant to be exhaustive?" is human intent that has to live somewhere, and an explicit reasoned
    table in the test is honest and robust — better than magic opt-out comments in production.
    """

    def _cases(self):
        """Yield (label, source, member_names, qualifier, allowed_omissions) per total dispatcher.

        - source: the function source, scanned for `qualifier.MEMBER` mentions.
        - member_names/qualifier: derived from the production enum (BuildMode is a plain-attr class,
          so its members come from BuildMode.ALL and are qualified as `BuildMode.MEMBER`).
        - allowed_omissions: member names a legitimately-partial dispatcher may skip (reason inline).
        """
        import inspect

        from skiresort_planner.model.click_info import ClickInfo, MapClickType, MarkerType
        from skiresort_planner.model.message import Message, MessageLevel
        from skiresort_planner.model.path_segment import SegmentKind
        from skiresort_planner.ui import bottom_chart, click_handlers, context
        from skiresort_planner.ui.context import BuildMode, EntityKind

        src = inspect.getsource
        marker = [m.name for m in MarkerType]
        clicktype = [m.name for m in MapClickType]
        segkind = [m.name for m in SegmentKind]
        entity = [m.name for m in EntityKind]
        msglevel = [m.name for m in MessageLevel]
        buildmode = [m.upper() for m in BuildMode.ALL]  # attr names, e.g. SLOPE, CHAIRLIFT

        # IMPORT_CENTER only appears in import_placing; the idle & slope-building handlers legitimately
        # don't route it (it hits their safety-net raise), so they may omit exactly that one member.
        idle_omit = {"IMPORT_CENTER"}

        # (label, source, member_names, qualifier, allowed_omissions)
        return [
            (
                "ClickInfo._validate_marker_ids",
                src(ClickInfo.__dict__["_validate_marker_ids"]),
                marker,
                "MarkerType",
                set(),
            ),
            (
                "ClickInfo.display_name[Marker]",
                src(ClickInfo.__dict__["display_name"].fget),
                marker,
                "MarkerType",
                set(),
            ),
            (
                "ClickInfo.display_name[Click]",
                src(ClickInfo.__dict__["display_name"].fget),
                clicktype,
                "MapClickType",
                set(),
            ),
            ("ClickInfo.__post_init__", src(ClickInfo.__dict__["__post_init__"]), clicktype, "MapClickType", set()),
            (
                "bottom_chart.render_building_profile",
                src(bottom_chart.render_building_profile),
                segkind,
                "SegmentKind",
                set(),
            ),
            (
                "bottom_chart.render_viewing_profile",
                src(bottom_chart.render_viewing_profile),
                entity,
                "EntityKind",
                set(),
            ),
            ("context.BuildMode.display_name", src(context.BuildMode.display_name), buildmode, "BuildMode", set()),
            ("context.BuildMode.icon", src(context.BuildMode.icon), buildmode, "BuildMode", set()),
            ("Message.display", src(Message.__dict__["display"]), msglevel, "MessageLevel", set()),
            ("handle_idle_click", src(click_handlers.handle_idle_click), marker, "MarkerType", idle_omit),
            (
                "handle_slope_building_click",
                src(click_handlers.handle_slope_building_click),
                marker,
                "MarkerType",
                idle_omit,
            ),
        ]

    def test_total_dispatchers_handle_every_member(self) -> None:
        offenders: dict[str, list[str]] = {}
        for label, source, members, qual, allowed in self._cases():
            missing = [m for m in members if m not in allowed and f"{qual}.{m}" not in source]
            if missing:
                offenders[f"{label} [{qual}]"] = missing
        assert not offenders, f"total dispatchers missing enum branches: {offenders}"


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
