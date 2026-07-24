"""Tests for the shared Dialog hierarchy (ui/dialogs.py) and its concrete subclasses.

The base classes are exercised through a tiny concrete subclass, and the real panel dialogs are
driven via `fake_st.clicked_keys` (the standard button keys `dialog_confirm`/`dialog_cancel`/
`dialog_save`) to assert the confirm/cancel branches fire the right callbacks.
"""

import pytest

from skiresort_planner.ui.dialogs import ConfirmDialog, InputDialog


class _RecordingConfirm(ConfirmDialog):
    """Minimal ConfirmDialog that records whether its confirm/cancel branches ran."""

    def __init__(self) -> None:
        self.confirmed = False
        self.cancelled = False

    @property
    def title(self) -> str:
        return "T"

    def _body(self) -> None:
        pass

    def _on_confirm(self) -> None:
        self.confirmed = True

    def _on_cancel(self) -> None:
        self.cancelled = True


class _RecordingInput(InputDialog):
    """Minimal InputDialog that records the saved value."""

    def __init__(self) -> None:
        self.saved: str | None = None

    @property
    def title(self) -> str:
        return "T"

    def _input(self) -> str:
        return "typed-name"

    def _on_save(self, value: str) -> None:
        self.saved = value


@pytest.fixture(autouse=True)
def _no_rerun(monkeypatch):
    """Neutralise trigger_rerun so dialog button branches don't call st.rerun in tests."""
    monkeypatch.setattr("skiresort_planner.ui.dialogs.trigger_rerun", lambda *a, **k: None)


class TestConfirmDialog:
    def test_confirm_click_runs_on_confirm(self, fake_st) -> None:
        fake_st.clicked_keys = {"dialog_confirm"}
        dlg = _RecordingConfirm()
        dlg.show()
        assert dlg.confirmed is True

    def test_cancel_click_does_not_confirm(self, fake_st) -> None:
        fake_st.clicked_keys = {"dialog_cancel"}
        dlg = _RecordingConfirm()
        dlg.show()
        assert dlg.confirmed is False

    def test_no_click_does_not_confirm(self, fake_st) -> None:
        dlg = _RecordingConfirm()
        dlg.show()
        assert dlg.confirmed is False

    def test_cancel_click_runs_on_cancel_hook(self, fake_st) -> None:
        fake_st.clicked_keys = {"dialog_cancel"}
        dlg = _RecordingConfirm()
        dlg.show()
        assert dlg.cancelled is True and dlg.confirmed is False

    def test_confirm_click_does_not_run_on_cancel(self, fake_st) -> None:
        fake_st.clicked_keys = {"dialog_confirm"}
        dlg = _RecordingConfirm()
        dlg.show()
        assert dlg.confirmed is True and dlg.cancelled is False


class TestInputDialog:
    def test_save_click_passes_input_value(self, fake_st) -> None:
        fake_st.clicked_keys = {"dialog_save"}
        dlg = _RecordingInput()
        dlg.show()
        assert dlg.saved == "typed-name"

    def test_cancel_click_does_not_save(self, fake_st) -> None:
        fake_st.clicked_keys = {"dialog_cancel"}
        dlg = _RecordingInput()
        dlg.show()
        assert dlg.saved is None


class TestConcreteDialogs:
    def test_reset_dialog_confirm_performs_reset(self, fake_st, monkeypatch) -> None:
        from skiresort_planner.ui import left_panel

        called: list[bool] = []
        monkeypatch.setattr(left_panel, "perform_reset_resort", lambda: called.append(True))

        fake_st.clicked_keys = {"dialog_confirm"}
        left_panel._ResetResortDialog().show()
        assert called == [True], "confirm invokes perform_reset_resort"

    def test_reset_dialog_cancel_leaves_resort(self, fake_st, monkeypatch) -> None:
        from skiresort_planner.ui import left_panel

        called: list[bool] = []
        monkeypatch.setattr(left_panel, "perform_reset_resort", lambda: called.append(True))

        fake_st.clicked_keys = {"dialog_cancel"}
        left_panel._ResetResortDialog().show()
        assert called == [], "cancel must not reset"


class TestChangeLiftTypeDialog:
    """The confirm-gated lift retype: confirm re-types + stays viewing; cancel keeps the lift and
    closes the view so the already-armed new type drives the next build.
    """

    def _viewed_lift(self, fake_st, dem):
        from skiresort_planner.constants import MapConfig
        from skiresort_planner.model.resort_graph import ResortGraph
        from skiresort_planner.ui.state_machine import PlannerStateMachine

        graph = ResortGraph()
        bottom, _ = graph.get_or_create_node(
            lon=0.0,
            lat=-1000 / MapConfig.METERS_PER_DEGREE_EQUATOR,
            elevation=dem.get_elevation_or_raise(lon=0.0, lat=-1000 / MapConfig.METERS_PER_DEGREE_EQUATOR),
        )
        top, _ = graph.get_or_create_node(lon=0.0, lat=0.0, elevation=dem.get_elevation_or_raise(lon=0.0, lat=0.0))
        lift = graph.add_lift(start_node_id=bottom.id, end_node_id=top.id, lift_type="chairlift", dem=dem)
        sm, ctx = PlannerStateMachine.create(graph=graph, add_ui_listener=False)
        fake_st.session_state["graph"] = graph
        fake_st.session_state["state_machine"] = sm
        fake_st.session_state["context"] = ctx
        sm.view_lift(lift_id=lift.id)
        return graph, sm, lift

    def test_confirm_retypes_and_keeps_view(self, fake_st, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.right_panel import _ChangeLiftTypeDialog

        graph, sm, lift = self._viewed_lift(fake_st, mock_dem_blue_slope)
        fake_st.clicked_keys = {"dialog_confirm"}
        _ChangeLiftTypeDialog(lift_id=lift.id, old_type="chairlift", new_type="gondola", sm=sm).show()

        assert graph.lifts[lift.id].lift_type == "gondola", "confirm re-types the lift"
        assert sm.is_idle_viewing_lift, "confirm keeps the lift view open"

    def test_cancel_keeps_lift_and_closes_view(self, fake_st, mock_dem_blue_slope) -> None:
        from skiresort_planner.ui.right_panel import _ChangeLiftTypeDialog

        graph, sm, lift = self._viewed_lift(fake_st, mock_dem_blue_slope)
        fake_st.clicked_keys = {"dialog_cancel"}
        _ChangeLiftTypeDialog(lift_id=lift.id, old_type="chairlift", new_type="gondola", sm=sm).show()

        assert graph.lifts[lift.id].lift_type == "chairlift", "cancel leaves the lift unchanged"
        assert sm.is_idle_ready, "cancel closes the lift view so the armed new type drives the next build"
