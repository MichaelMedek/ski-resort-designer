"""Dialog - user-facing modal dialogs. Base + confirm/input children; concrete dialogs live
next to their panel logic. Mirrors model/message.py's Message hierarchy.
"""

from abc import ABC, abstractmethod

import streamlit as st

from skiresort_planner.ui.infra import trigger_rerun


class Dialog(ABC):
    """Abstract modal dialog. `.show()` opens it under @st.dialog; subclasses render a body + a
    standard button row. Two children fix the row shape (confirm vs. input).

    CANCEL_LABEL is fixed here as the single source — every dialog cancels identically, never
    overridden. Primary-button labels vary per dialog and are abstract (each subclass supplies one).
    """

    CANCEL_LABEL = "✖️ Cancel"

    @property
    @abstractmethod
    def title(self) -> str:
        """Dialog title shown in the modal chrome."""
        raise NotImplementedError

    @abstractmethod
    def _body(self) -> None:
        """Render the message/inputs above the button row."""
        raise NotImplementedError

    @abstractmethod
    def _buttons(self) -> None:
        """Render the action button row (child-specific)."""
        raise NotImplementedError

    def show(self) -> None:
        """Open the dialog: body then buttons, wrapped in @st.dialog."""

        @st.dialog(self.title)
        def _run() -> None:
            self._body()
            self._buttons()

        _run()


class ConfirmDialog(Dialog):
    """Destructive/confirmation dialog: primary + cancel row. Subclasses supply title, body, and
    _on_confirm().

    CONFIRM_LABEL is fixed here — the title already names the action, so a per-dialog primary label
    is redundant; every dialog confirms with the same green checkmark. Never overridden.
    """

    CONFIRM_LABEL = "✅ Confirm"

    @abstractmethod
    def _on_confirm(self) -> None:
        """Run when the primary button is clicked."""
        raise NotImplementedError

    def _buttons(self) -> None:
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button(self.CONFIRM_LABEL, type="primary", use_container_width=True, key="dialog_confirm"):
                self._on_confirm()
                trigger_rerun()
        with col_cancel:
            if st.button(self.CANCEL_LABEL, use_container_width=True, key="dialog_cancel"):
                trigger_rerun()


class InputDialog(Dialog):
    """Text-input dialog: an input + save/cancel row. Subclasses supply title, _input() (renders
    the widget, returns its value) and _on_save(value).

    SAVE_LABEL is fixed here for the same reason as ConfirmDialog.CONFIRM_LABEL — the title names
    the dialog, so a per-dialog save label is redundant. Never overridden.
    """

    SAVE_LABEL = "💾 Save"

    @abstractmethod
    def _input(self) -> str:
        """Render the input widget and return its current value."""
        raise NotImplementedError

    @abstractmethod
    def _on_save(self, value: str) -> None:
        """Run with the input value when Save is clicked."""
        raise NotImplementedError

    def _body(self) -> None:
        # Value read in _body, consumed in _buttons via a per-instance stash.
        self._value = self._input()

    def _buttons(self) -> None:
        col_save, col_cancel = st.columns(2)
        with col_save:
            if st.button(self.SAVE_LABEL, type="primary", use_container_width=True, key="dialog_save"):
                self._on_save(self._value)
                trigger_rerun()
        with col_cancel:
            if st.button(self.CANCEL_LABEL, use_container_width=True, key="dialog_cancel"):
                trigger_rerun()
