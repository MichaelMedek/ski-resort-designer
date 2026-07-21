"""Tests for resort auto-backup: store primitives, graph helpers,
startup URL routing, and the central dirty-checked autosave hook.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.persistence import backup_store


class FakeSessionState(dict[str, object]):
    """Mimics st.session_state: both attribute and item access over one store."""

    def __getattr__(self, key: str) -> object:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: object) -> None:
        self[key] = value


@pytest.fixture(autouse=True)
def _isolate_backup_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect backup_store to a temp dir so tests never touch the real backups/."""
    monkeypatch.setattr(backup_store, "BACKUP_DIR", tmp_path / "backups")
    return tmp_path / "backups"


def _populate(graph: ResortGraph, path_points_blue: list[PathPoint]) -> str:
    """Commit one slope so the graph has content worth saving; return its name."""
    proposal = ProposedPathSegment(
        points=path_points_blue,
        target_slope_pct=20.0,
        target_difficulty="blue",
        sector_name="Test",
    )
    graph.commit_paths(paths=[proposal])
    slope = graph.finish_slope(segment_ids=list(graph.segments.keys()), name="Testrun")
    assert slope is not None
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

    def test_delete_soft_deletes_and_hides_from_load(
        self, _isolate_backup_dir: Path, empty_graph: ResortGraph, path_points_blue
    ) -> None:
        _populate(empty_graph, path_points_blue)
        resort_id = backup_store.new_resort_id()
        backup_store.save(graph=empty_graph, resort_id=resort_id)
        assert backup_store.load(resort_id=resort_id) is not None

        backup_store.delete(resort_id=resort_id)
        # Gone from load() and from largest (it's a reset), but RENAMED (not removed) for recovery.
        assert backup_store.load(resort_id=resort_id) is None
        assert not (_isolate_backup_dir / f"{resort_id}.json").exists()
        recovery = list(_isolate_backup_dir.glob(f"{resort_id}_DELETED_*.json"))
        assert len(recovery) == 1, "the live backup is renamed to a _DELETED recovery copy, not removed"

    def test_delete_twice_keeps_both_recovery_copies(
        self, _isolate_backup_dir: Path, empty_graph: ResortGraph, path_points_blue
    ) -> None:
        # Resetting the SAME resort id twice must never clobber the first recovery copy (unique uuid suffix).
        _populate(empty_graph, path_points_blue)
        resort_id = backup_store.new_resort_id()
        backup_store.save(graph=empty_graph, resort_id=resort_id)
        backup_store.delete(resort_id=resort_id)
        backup_store.save(graph=empty_graph, resort_id=resort_id)
        backup_store.delete(resort_id=resort_id)
        assert len(list(_isolate_backup_dir.glob(f"{resort_id}_DELETED_*.json"))) == 2

    def test_largest_resort_id_excludes_soft_deleted(
        self, _isolate_backup_dir: Path, empty_graph: ResortGraph, path_points_blue
    ) -> None:
        _populate(empty_graph, path_points_blue)
        resort_id = backup_store.new_resort_id()
        backup_store.save(graph=empty_graph, resort_id=resort_id)
        backup_store.delete(resort_id=resort_id)
        # The only backup is soft-deleted → bare-link load finds nothing to restore.
        assert backup_store.largest_resort_id() is None

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

    def test_save_persists_segments_without_finished_slope(self, empty_graph: ResortGraph, path_points_blue) -> None:
        # The save skip-guard keys off `not graph.segments` (not slopes/lifts): a graph with
        # committed-but-unfinished segments must still be SAVED (skip only truly-empty graphs).
        proposal = ProposedPathSegment(
            points=path_points_blue,
            target_slope_pct=20.0,
            target_difficulty="blue",
            sector_name="Test",
        )
        empty_graph.commit_paths(paths=[proposal])
        assert empty_graph.segments and not empty_graph.slopes and not empty_graph.lifts

        resort_id = backup_store.new_resort_id()
        backup_store.save(graph=empty_graph, resort_id=resort_id)

        # The empty-guard keys off segments, so the file IS written (load returns a graph, not None).
        # Note: on load the unowned segments + their orphaned nodes are discarded, so the reloaded
        # graph is empty — this test's point is only that `save` did not skip a segment-bearing graph.
        loaded = backup_store.load(resort_id=resort_id)
        assert loaded is not None, "a graph with segments must not be skipped by the empty-guard"


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

    def test_populated_graph_returns_median_lon_lat(self, empty_graph: ResortGraph, path_points_blue) -> None:
        import statistics

        _populate(empty_graph, path_points_blue)
        result = empty_graph.get_center()
        assert result is not None
        lon, lat = result
        nodes = empty_graph.nodes.values()
        assert lon == statistics.median(n.lon for n in nodes)
        assert lat == statistics.median(n.lat for n in nodes)

    def test_center_uses_median_not_mean_for_two_clusters(self, empty_graph: ResortGraph) -> None:
        # Two clusters: 3 nodes near lon 0 (spaced >30m snap threshold so they stay distinct), one
        # far outlier at lon 100. Mean would drift toward the outlier (~25); median stays near 0.
        for i in range(3):
            empty_graph.get_or_create_node(lon=i * 5e-4, lat=0.0, elevation=2000.0)
        empty_graph.get_or_create_node(lon=100.0, lat=0.0, elevation=2000.0)
        assert len(empty_graph.nodes) == 4, "cluster nodes must not have snap-merged"
        result = empty_graph.get_center()
        assert result is not None
        lon, _lat = result
        assert lon < 1.0, f"median must stay in the dense cluster, not drift to the outlier: {lon}"


class TestChangeToken:
    def test_token_changes_on_mutation(self, empty_graph: ResortGraph, path_points_blue) -> None:
        before = empty_graph.change_token()
        _populate(empty_graph, path_points_blue)
        assert empty_graph.change_token() != before

    def test_token_stable_without_mutation(self, empty_graph: ResortGraph, path_points_blue) -> None:
        _populate(empty_graph, path_points_blue)
        token = empty_graph.change_token()
        # Read-only accesses (what a rerun performs) must not bump the token.
        empty_graph.get_center()
        empty_graph.get_elevation_range()
        assert empty_graph.change_token() == token


# =============================================================================
# Central autosave hook (infra._autosave_if_dirty)
# =============================================================================


class TestAutosaveHook:
    """The hook that trigger_rerun() calls before every rerun."""

    def _fake_session(self, monkeypatch: pytest.MonkeyPatch, state: dict[str, object]) -> FakeSessionState:
        session = FakeSessionState(state)
        monkeypatch.setattr("skiresort_planner.ui.infra.st.session_state", session, raising=False)
        return session

    def test_saves_when_dirty(self, monkeypatch, empty_graph, path_points_blue) -> None:
        from skiresort_planner.ui import infra

        _populate(empty_graph, path_points_blue)
        resort_id = backup_store.new_resort_id()
        session = self._fake_session(monkeypatch, {"resort_id": resort_id, "graph": empty_graph})

        infra._autosave_if_dirty()

        assert backup_store.load(resort_id=resort_id) is not None
        assert session["_saved_token"] == empty_graph.change_token()

    def test_skips_when_unchanged(self, monkeypatch, empty_graph, path_points_blue) -> None:
        from skiresort_planner.ui import infra

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
        from skiresort_planner.ui import infra

        _populate(empty_graph, path_points_blue)
        self._fake_session(monkeypatch, {"graph": empty_graph})
        infra._autosave_if_dirty()  # must not raise


# =============================================================================
# Startup routing (app._init_resort_from_url_or_new)
# =============================================================================


class TestStartupRouting:
    """URL-param → backup resolution on session init."""

    def _patch_app(
        self, monkeypatch: pytest.MonkeyPatch, query: dict[str, object], state: dict[str, object]
    ) -> FakeSessionState:
        session = FakeSessionState(state)
        monkeypatch.setattr("skiresort_planner.app.st.query_params", query, raising=False)
        monkeypatch.setattr("skiresort_planner.app.st.session_state", session, raising=False)
        return session

    def test_no_param_no_backups_starts_fresh(self, monkeypatch) -> None:
        from skiresort_planner import app

        query: dict[str, object] = {}
        session = self._patch_app(monkeypatch, query, {})

        app._init_resort_from_url_or_new()

        assert "graph" not in session  # fresh: no graph loaded
        assert len(cast(str, session["resort_id"])) == 8
        assert query["resort"] == session["resort_id"]

    def test_param_with_backup_loads_it(self, monkeypatch, empty_graph, path_points_blue) -> None:
        from skiresort_planner import app

        _populate(empty_graph, path_points_blue)
        resort_id = backup_store.new_resort_id()
        backup_store.save(graph=empty_graph, resort_id=resort_id)

        session = self._patch_app(monkeypatch, {"resort": resort_id}, {})

        app._init_resort_from_url_or_new()

        assert session["resort_id"] == resort_id
        loaded_graph = cast(ResortGraph, session["graph"])
        assert len(loaded_graph.slopes) == 1
        assert session["_saved_token"] == loaded_graph.change_token()

    def test_param_missing_file_falls_through_to_fresh(self, monkeypatch) -> None:
        from skiresort_planner import app

        query: dict[str, object] = {"resort": "ghost123"}
        session = self._patch_app(monkeypatch, query, {})

        app._init_resort_from_url_or_new()

        assert "graph" not in session
        assert session["resort_id"] != "ghost123"
        assert query["resort"] == session["resort_id"]

    def test_bare_link_loads_biggest_backup(self, monkeypatch, empty_graph, path_points_blue) -> None:
        from skiresort_planner import app

        _populate(empty_graph, path_points_blue)
        biggest_id = backup_store.new_resort_id()
        backup_store.save(graph=empty_graph, resort_id=biggest_id)

        query: dict[str, object] = {}
        session = self._patch_app(monkeypatch, query, {})

        app._init_resort_from_url_or_new()

        assert session["resort_id"] == biggest_id
        assert len(cast(ResortGraph, session["graph"]).slopes) == 1
        assert query["resort"] == biggest_id
