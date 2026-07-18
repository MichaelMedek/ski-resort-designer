"""Tests for the shared Dialog hierarchy (ui/dialogs.py) and its concrete subclasses.

The base classes are exercised through a tiny concrete subclass, and the real panel dialogs are
driven via `fake_st.clicked_keys` (the standard button keys `dialog_confirm`/`dialog_cancel`/
`dialog_save`) to assert the confirm/cancel branches fire the right callbacks.
"""

import pytest

from skiresort_planner.ui.dialogs import ConfirmDialog, InputDialog


class _RecordingConfirm(ConfirmDialog):
    """Minimal ConfirmDialog that records whether its confirm branch ran."""

    def __init__(self) -> None:
        self.confirmed = False

    @property
    def title(self) -> str:
        return "T"

    def _body(self) -> None:
        pass

    def _on_confirm(self) -> None:
        self.confirmed = True


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
        monkeypatch.setattr(left_panel, "_perform_reset_resort", lambda: called.append(True))

        fake_st.clicked_keys = {"dialog_confirm"}
        left_panel._ResetResortDialog().show()
        assert called == [True], "confirm invokes _perform_reset_resort"

    def test_reset_dialog_cancel_leaves_resort(self, fake_st, monkeypatch) -> None:
        from skiresort_planner.ui import left_panel

        called: list[bool] = []
        monkeypatch.setattr(left_panel, "_perform_reset_resort", lambda: called.append(True))

        fake_st.clicked_keys = {"dialog_cancel"}
        left_panel._ResetResortDialog().show()
        assert called == [], "cancel must not reset"
