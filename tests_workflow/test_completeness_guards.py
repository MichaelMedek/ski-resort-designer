"""Completeness guards — tripwires that make it impossible to add a feature *half*-wired.

These tests don't check behavior; they check that parallel code sites stay in sync. Each guards a
hazard where adding an enum member / dataclass field / new module would compile and pass every
other test, yet ship a silent bug or a rarely-hit crash.

The ActionType undo-dispatcher guard lives in test_resort_graph.py (it's undo-specific).
"""

import ast
import dataclasses
import inspect
import textwrap

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
                "handle_path_building_click",
                src(click_handlers.handle_path_building_click),
                marker,
                "MarkerType",
                idle_omit,
            ),
        ]

    @staticmethod
    def _strip_docstring(source: str) -> str:
        """Return the function/property source with its leading docstring blanked out.

        The scan below asks "is ``Qualifier.MEMBER`` mentioned in the branch logic?" — a docstring
        that happens to name a member (e.g. ``MarkerType.NODE`` in a handler's prose) would satisfy
        that substring test even if no branch actually routes it, silently defeating the guard. We
        strip the docstring so only real code is scanned.
        """
        dedented = textwrap.dedent(source)
        node = ast.parse(dedented).body[0]
        assert isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)  # every scanned source is a def
        if ast.get_docstring(node, clean=False) is None:
            return dedented
        doc_stmt = node.body[0]
        assert doc_stmt.end_lineno is not None  # docstring statement always has an end line
        lines = dedented.splitlines()
        for i in range(doc_stmt.lineno - 1, doc_stmt.end_lineno):
            lines[i] = ""
        return "\n".join(lines)

    def test_total_dispatchers_handle_every_member(self) -> None:
        offenders: dict[str, list[str]] = {}
        for label, source, members, qual, allowed in self._cases():
            scanned = self._strip_docstring(source)  # docstring mentions must not satisfy the guard
            missing = [m for m in members if m not in allowed and f"{qual}.{m}" not in scanned]
            if missing:
                offenders[f"{label} [{qual}]"] = missing
        assert not offenders, f"total dispatchers missing enum branches: {offenders}"

    def test_docstring_mention_does_not_satisfy_the_guard(self) -> None:
        """A member named only in a docstring must NOT count as handled.

        Guards the guard: before docstring-stripping, a handler that documents ``MarkerType.SLOPE``
        in its prose but never routes it would slip through the substring scan. This pins the fix.
        """
        handled_only = (
            "def _handler():\n"
            '    """Routes MarkerType.NODE and MarkerType.SLOPE clicks."""\n'
            "    if x == MarkerType.NODE:\n"
            "        return 1\n"
            "    raise RuntimeError\n"
        )
        scanned = self._strip_docstring(handled_only)
        assert "MarkerType.NODE" in scanned  # real branch survives stripping
        assert "MarkerType.SLOPE" not in scanned  # docstring-only mention is gone


# =============================================================================
# 3. Reload-safe enum comparisons: enum members compared with enum_eq, never ==/!=
# =============================================================================


# =============================================================================
# 3. Reload-safe enums: every domain enum is a StrEnum; no isinstance on domain classes
# =============================================================================


class TestReloadSafeEnums:
    """Reload safety is enforced structurally, not by a comparison helper.

    Streamlit re-imports modules on rerun, creating a FRESH class object per reload. Two patterns
    break across that boundary and are BANNED in production source:
      1. A plain ``Enum`` domain type — ``==`` is identity-based, so an old-class member fails against
         a new-class member. A ``StrEnum`` compares by string value and is reload-safe, so raw ``==``
         is correct and no ``enum_eq`` helper is needed.
      2. ``isinstance(x, <domain class>)`` — an object built before the reload (it lives in the
         preserved graph/session_state) is an instance of the OLD class and fails ``isinstance``
         against the freshly-imported class. Branch on a reload-safe ``.kind`` StrEnum instead.

    These guards make either pattern impossible to (re)introduce. Genuine builtin/library isinstance
    (dict/list/float/requests.RequestException on external input) is fine and not matched here.
    """

    # Domain classes whose identity is destroyed by a Streamlit reload. isinstance against these in
    # production is the reload-fragility bug (the `unhandled parent entity Slope` crash).
    _DOMAIN_CLASSES = {"Slope", "Road", "SegmentPath", "Lift"}

    def test_every_domain_enum_is_a_str_enum(self) -> None:
        """No production enum may subclass a bare ``Enum`` — StrEnum (or IntEnum) only.

        A bare ``class X(Enum)`` is identity-compared and breaks after a reload. This scans every
        production module's class defs and flags any that inherit ``Enum`` directly rather than the
        value-based ``StrEnum``/``IntEnum``.
        """
        offenders: list[str] = []
        for py in PACKAGE_DIR.rglob("*.py"):
            tree = ast.parse(py.read_text(), filename=str(py))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                base_names = {b.id for b in node.bases if isinstance(b, ast.Name)}
                # `Enum` directly (not StrEnum/IntEnum) is the reload-fragile case.
                if "Enum" in base_names:
                    offenders.append(f"{py.relative_to(PACKAGE_DIR)}:{node.lineno} class {node.name}(Enum)")
        assert not offenders, (
            "Domain enums must subclass StrEnum (reload-safe value comparison), never bare Enum. "
            f"Offenders: {offenders}"
        )

    def test_no_isinstance_on_domain_classes_in_production(self) -> None:
        """``isinstance(x, Slope|Road|SegmentPath|Lift)`` is reload-unsafe and banned in source.

        Branch on the entity's ``.kind`` StrEnum instead. Genuine builtin/library isinstance checks
        (dict, list|tuple, int|float, requests.RequestException) are not matched — only our own
        domain classes.
        """
        offenders: list[str] = []
        for py in PACKAGE_DIR.rglob("*.py"):
            tree = ast.parse(py.read_text(), filename=str(py))
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "isinstance"
                ):
                    continue
                if len(node.args) < 2:
                    continue
                # Second arg is the type(s): a Name, or a tuple/BinOp of Names.
                type_arg = node.args[1]
                named = {n.id for n in ast.walk(type_arg) if isinstance(n, ast.Name)}
                hit = named & self._DOMAIN_CLASSES
                if hit:
                    offenders.append(f"{py.relative_to(PACKAGE_DIR)}:{node.lineno} isinstance(..., {sorted(hit)})")
        assert not offenders, (
            "isinstance on domain classes is reload-unsafe (fails after a Streamlit reimport). "
            f"Branch on the .kind StrEnum instead. Offenders: {offenders}"
        )

    def test_enum_eq_helper_is_gone(self) -> None:
        """The enum_eq helper must not exist — StrEnum + raw ``==`` replaces it entirely.

        Guards against re-adding the crutch (and against a stray import of a deleted module).
        """
        assert not (PACKAGE_DIR / "enum_utils.py").exists(), "enum_utils.py must be deleted; use StrEnum + raw =="
        offenders: list[str] = []
        for py in PACKAGE_DIR.rglob("*.py"):
            if "enum_eq" in py.read_text():
                offenders.append(str(py.relative_to(PACKAGE_DIR)))
        assert not offenders, f"enum_eq is removed; no production file may reference it. Offenders: {offenders}"


# =============================================================================
# 4. Layering: model / core / generators must never import ui
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
# 5. NodeConnected contract: every subclass exposes the endpoint interface
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


# =============================================================================
# 6. segment_path_entities covers every buildable SegmentKind (extensibility guard)
# =============================================================================


class TestSegmentPathEntitiesCoversEveryKind:
    """ResortGraph.segment_path_entities is the single source for "all segment-group entities"
    (merge repoint, boundary snapshot, segment→entity lookup). It is hand-written
    (`[*self.slopes.values(), *self.roads.values()]`), so a new SegmentKind whose finished entity
    lands in a NEW collection the property doesn't include would silently vanish from every
    consumer. This ties the property to the SegmentKind ground truth: finish one entity of each
    kind and assert it shows up in segment_path_entities.
    """

    def test_every_buildable_kind_appears_in_segment_path_entities(self) -> None:
        from skiresort_planner.model.path_point import PathPoint
        from skiresort_planner.model.path_segment import SegmentKind
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.model.resort_graph import ResortGraph
        from skiresort_planner.ui.kind_spec import KIND_SPECS

        m = 111320.0
        for kind in SegmentKind:
            graph = ResortGraph()
            pts = [PathPoint(lon=0.0, lat=0.0, elevation=2000.0), PathPoint(lon=0.0, lat=-300 / m, elevation=1970.0)]
            graph.commit_paths(paths=[ProposedPathSegment(points=pts, kind=kind)], record_undo=False)
            seg_id = list(graph.segments.keys())[-1]
            entity = KIND_SPECS[kind].finish(graph, [seg_id])
            assert entity is not None, f"finishing a {kind.value} must produce an entity"
            assert entity in graph.segment_path_entities, (
                f"finished {kind.value} entity is missing from segment_path_entities — "
                "a new SegmentKind must be added to that property or it drops out of merge/lookup/snapshot"
            )

    def test_every_buildable_kind_survives_serialization_roundtrip(self) -> None:
        """GAP-C guard: ResortGraph's per-kind dicts/counters/serialization are hand-written
        (slopes/roads, _slope_counter/_road_counter, the to_dict/from_dict blocks). A new SegmentKind
        whose entity dict is forgotten in to_dict/from_dict would silently fail to persist. This ties
        persistence to the SegmentKind ground truth: finish one entity of each kind, round-trip the
        whole graph, and assert the entity (by id) survives.
        """
        import json

        from skiresort_planner.model.path_point import PathPoint
        from skiresort_planner.model.path_segment import SegmentKind
        from skiresort_planner.model.proposed_path import ProposedPathSegment
        from skiresort_planner.model.resort_graph import ResortGraph
        from skiresort_planner.ui.kind_spec import KIND_SPECS

        m = 111320.0
        for kind in SegmentKind:
            graph = ResortGraph()
            pts = [PathPoint(lon=0.0, lat=0.0, elevation=2000.0), PathPoint(lon=0.0, lat=-300 / m, elevation=1970.0)]
            graph.commit_paths(paths=[ProposedPathSegment(points=pts, kind=kind)], record_undo=False)
            entity = KIND_SPECS[kind].finish(graph, [list(graph.segments.keys())[-1]])
            assert entity is not None
            before_ids = {e.id for e in graph.segment_path_entities}

            restored = ResortGraph.from_dict(data=json.loads(json.dumps(graph.to_dict())))
            after_ids = {e.id for e in restored.segment_path_entities}
            assert before_ids <= after_ids, (
                f"a finished {kind.value} entity did not survive to_dict→from_dict — "
                f"its per-kind dict is likely missing from the serialization block (GAP-C)"
            )
