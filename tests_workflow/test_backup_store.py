"""Tests for resort auto-backup: store primitives, graph helpers,
startup URL routing, and the central dirty-checked autosave hook."""

from __future__ import annotations

from pathlib import Path

import pytest

from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.persistence import backup_store


class FakeSessionState(dict):
    """Mimics st.session_state: both attribute and item access over one store."""

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value) -> None:
        self[key] = value


@pytest.fixture(autouse=True)
def _isolate_backup_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect backup_store to a temp dir so tests never touch the real backups/."""
    monkeypatch.setattr(backup_store, "BACKUP_DIR", tmp_path / "backups")
    return tmp_path / "backups"


def _populate(graph: ResortGraph, path_points_blue: list) -> str:
    """Commit one slope so the graph has content worth saving; return its name."""
    proposal = ProposedPathSegment(
        points=path_points_blue,
        target_slope_pct=20.0,
        target_difficulty="blue",
        sector_name="Test",
    )
    graph.commit_paths(paths=[proposal])
    slope = graph.finish_slope(segment_ids=list(graph.segments.keys()), name="Testrun")
    return slope.name


# =============================================================================
# Store primitives
# =============================================================================


class TestBackupStore:
    def test_new_resort_id_is_unique_and_short(self) -> None:
        ids = {backup_store.new_resort_id() for _ in range(20)}
        assert len(ids) == 20
        assert all(len(i) == 8 for i in ids)

    def test_save_load_roundtrip(self, empty_graph: ResortGraph, path_points_blue) -> None:
        slope_name = _populate(empty_graph, path_points_blue)
        resort_id = backup_store.new_resort_id()

        backup_store.save(graph=empty_graph, resort_id=resort_id)
        loaded = backup_store.load(resort_id=resort_id)

        assert loaded is not None
        assert len(loaded.slopes) == 1
        assert list(loaded.slopes.values())[0].name == slope_name
        assert len(loaded.nodes) == len(empty_graph.nodes)
        assert len(loaded.segments) == len(empty_graph.segments)

    def test_load_missing_returns_none(self) -> None:
        assert backup_store.load(resort_id="nonexistent") is None

    def test_save_skips_empty_graph(self, empty_graph: ResortGraph) -> None:
        resort_id = backup_store.new_resort_id()
        backup_store.save(graph=empty_graph, resort_id=resort_id)
        assert backup_store.load(resort_id=resort_id) is None

    def test_delete_removes_file(self, empty_graph: ResortGraph, path_points_blue) -> None:
        _populate(empty_graph, path_points_blue)
        resort_id = backup_store.new_resort_id()
        backup_store.save(graph=empty_graph, resort_id=resort_id)
        assert backup_store.load(resort_id=resort_id) is not None

        backup_store.delete(resort_id=resort_id)
        assert backup_store.load(resort_id=resort_id) is None

    def test_delete_missing_is_silent(self) -> None:
        backup_store.delete(resort_id="nonexistent")  # must not raise

    def test_largest_resort_id_none_when_empty(self) -> None:
        assert backup_store.largest_resort_id() is None

    def test_largest_resort_id_picks_most_nodes(self, empty_graph: ResortGraph, path_points_blue) -> None:
        small = empty_graph
        _populate(small, path_points_blue)
        small_id = backup_store.new_resort_id()
        backup_store.save(graph=small, resort_id=small_id)

        big = ResortGraph()
        _populate(big, path_points_blue)
        big.get_or_create_node(lon=11.0, lat=47.0, elevation=2000.0)
        big_id = backup_store.new_resort_id()
        backup_store.save(graph=big, resort_id=big_id)

        assert len(big.nodes) > len(small.nodes)
        assert backup_store.largest_resort_id() == big_id

    def test_largest_resort_id_skips_corrupt_file(
        self, _isolate_backup_dir: Path, empty_graph, path_points_blue
    ) -> None:
        _populate(empty_graph, path_points_blue)
        good_id = backup_store.new_resort_id()
        backup_store.save(graph=empty_graph, resort_id=good_id)
        # Drop a garbage .json into the backup dir
        (_isolate_backup_dir / "broken.json").write_text("{ not valid json")

        assert backup_store.largest_resort_id() == good_id  # skips broken, no raise

    def test_save_is_atomic_no_tmp_left(self, _isolate_backup_dir: Path, empty_graph, path_points_blue) -> None:
        _populate(empty_graph, path_points_blue)
        resort_id = backup_store.new_resort_id()
        backup_store.save(graph=empty_graph, resort_id=resort_id)
        assert (_isolate_backup_dir / f"{resort_id}.json").exists()
        assert list(_isolate_backup_dir.glob("*.tmp")) == []


# =============================================================================
# Graph helpers
# =============================================================================


class TestGetElevationRange:
    def test_empty_graph_returns_none(self, empty_graph: ResortGraph) -> None:
        assert empty_graph.get_elevation_range() is None

    def test_populated_graph_returns_min_max(self, empty_graph: ResortGraph, path_points_blue) -> None:
        _populate(empty_graph, path_points_blue)
        result = empty_graph.get_elevation_range()
        assert result is not None
        min_elev, max_elev = result
        assert min_elev == min(n.elevation for n in empty_graph.nodes.values())
        assert max_elev == max(n.elevation for n in empty_graph.nodes.values())


class TestGetCenter:
    def test_empty_graph_returns_none(self, empty_graph: ResortGraph) -> None:
        assert empty_graph.get_center() is None

    def test_populated_graph_returns_mean_lon_lat(self, empty_graph: ResortGraph, path_points_blue) -> None:
        _populate(empty_graph, path_points_blue)
        result = empty_graph.get_center()
        assert result is not None
        lon, lat = result
        nodes = empty_graph.nodes.values()
        assert lon == sum(n.lon for n in nodes) / len(nodes)
        assert lat == sum(n.lat for n in nodes) / len(nodes)


class TestChangeToken:
    def test_token_changes_on_mutation(self, empty_graph: ResortGraph, path_points_blue) -> None:
        before = empty_graph.change_token()
        _populate(empty_graph, path_points_blue)
        assert empty_graph.change_token() != before

    def test_token_stable_without_mutation(self, empty_graph: ResortGraph, path_points_blue) -> None:
        _populate(empty_graph, path_points_blue)
        assert empty_graph.change_token() == empty_graph.change_token()


# =============================================================================
# Central autosave hook (infra._autosave_if_dirty)
# =============================================================================


class TestAutosaveHook:
    """The hook that trigger_rerun() calls before every rerun."""

    def _fake_session(self, monkeypatch: pytest.MonkeyPatch, state: dict) -> FakeSessionState:
        import skiresort_planner.ui.infra as infra

        session = FakeSessionState(state)
        monkeypatch.setattr(infra.st, "session_state", session, raising=False)
        return session

    def test_saves_when_dirty(self, monkeypatch, empty_graph, path_points_blue) -> None:
        import skiresort_planner.ui.infra as infra

        _populate(empty_graph, path_points_blue)
        resort_id = backup_store.new_resort_id()
        session = self._fake_session(monkeypatch, {"resort_id": resort_id, "graph": empty_graph})

        infra._autosave_if_dirty()

        assert backup_store.load(resort_id=resort_id) is not None
        assert session["_saved_token"] == empty_graph.change_token()

    def test_skips_when_unchanged(self, monkeypatch, empty_graph, path_points_blue) -> None:
        import skiresort_planner.ui.infra as infra

        _populate(empty_graph, path_points_blue)
        resort_id = backup_store.new_resort_id()
        self._fake_session(
            monkeypatch,
            {"resort_id": resort_id, "graph": empty_graph, "_saved_token": empty_graph.change_token()},
        )

        infra._autosave_if_dirty()

        # token matched → no file written
        assert backup_store.load(resort_id=resort_id) is None

    def test_noop_without_resort_id(self, monkeypatch, empty_graph, path_points_blue) -> None:
        import skiresort_planner.ui.infra as infra

        _populate(empty_graph, path_points_blue)
        self._fake_session(monkeypatch, {"graph": empty_graph})
        infra._autosave_if_dirty()  # must not raise


# =============================================================================
# Startup routing (app._init_resort_from_url_or_new)
# =============================================================================


class TestStartupRouting:
    """URL-param → backup resolution on session init."""

    def _patch_app(self, monkeypatch: pytest.MonkeyPatch, query: dict, state: dict) -> FakeSessionState:
        import skiresort_planner.app as app

        session = FakeSessionState(state)
        monkeypatch.setattr(app.st, "query_params", query, raising=False)
        monkeypatch.setattr(app.st, "session_state", session, raising=False)
        return session

    def test_no_param_no_backups_starts_fresh(self, monkeypatch) -> None:
        import skiresort_planner.app as app

        query: dict = {}
        session = self._patch_app(monkeypatch, query, {})

        app._init_resort_from_url_or_new()

        assert "graph" not in session  # fresh: no graph loaded
        assert len(session["resort_id"]) == 8
        assert query["resort"] == session["resort_id"]

    def test_param_with_backup_loads_it(self, monkeypatch, empty_graph, path_points_blue) -> None:
        import skiresort_planner.app as app

        _populate(empty_graph, path_points_blue)
        resort_id = backup_store.new_resort_id()
        backup_store.save(graph=empty_graph, resort_id=resort_id)

        session = self._patch_app(monkeypatch, {"resort": resort_id}, {})

        app._init_resort_from_url_or_new()

        assert session["resort_id"] == resort_id
        assert len(session["graph"].slopes) == 1
        assert session["_saved_token"] == session["graph"].change_token()

    def test_param_missing_file_falls_through_to_fresh(self, monkeypatch) -> None:
        import skiresort_planner.app as app

        query: dict = {"resort": "ghost123"}
        session = self._patch_app(monkeypatch, query, {})

        app._init_resort_from_url_or_new()

        assert "graph" not in session
        assert session["resort_id"] != "ghost123"
        assert query["resort"] == session["resort_id"]

    def test_bare_link_loads_biggest_backup(self, monkeypatch, empty_graph, path_points_blue) -> None:
        import skiresort_planner.app as app

        _populate(empty_graph, path_points_blue)
        biggest_id = backup_store.new_resort_id()
        backup_store.save(graph=empty_graph, resort_id=biggest_id)

        query: dict = {}
        session = self._patch_app(monkeypatch, query, {})

        app._init_resort_from_url_or_new()

        assert session["resort_id"] == biggest_id
        assert len(session["graph"].slopes) == 1
        assert query["resort"] == biggest_id
