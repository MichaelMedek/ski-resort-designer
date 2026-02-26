"""Message - User-facing messages for the ski resort planner UI.

Architecture:
- LEFT (sidebar): ONE blue info message showing current mode, progress, and general capabilities
- CENTER (under map): Red error messages when user clicks invalid locations, blue for loading
- RIGHT (control panel): ONE yellow instruction message for what to do NOW

Design Principles:
- Maximum ONE message per panel location at any time
- LEFT = CONTEXT (blue) - Mode, stats, general info
- CENTER = ERRORS (red) - Invalid clicks only / LOADING (blue)
- RIGHT = ACTION (yellow) - Specific next step

All data (elevations, node names, stats) must be preserved in the consolidated messages.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MessageLevel(Enum):
    """Display level for UI messages."""

    INFO = "info"  # Blue - context/status/loading
    WARNING = "warning"  # Yellow - action instructions
    ERROR = "error"  # Red - user mistakes


@dataclass(frozen=True)
class Message(ABC):
    """Abstract base class for user-facing messages displayed inline (sidebars/panels).

    These messages are rendered as st.info/st.warning/st.error blocks that persist
    in the UI until replaced. Used for context, instructions, and status.
    """

    @property
    @abstractmethod
    def message(self) -> str:
        """Formatted message for display in Streamlit."""
        raise NotImplementedError

    @property
    @abstractmethod
    def level(self) -> MessageLevel:
        """Display level."""
        raise NotImplementedError

    def display(self) -> None:
        """Render this message using the appropriate Streamlit function."""
        import streamlit as st

        render_fn = {
            MessageLevel.INFO: st.info,
            MessageLevel.WARNING: st.warning,
            MessageLevel.ERROR: st.error,
        }[self.level]
        render_fn(self.message)


@dataclass(frozen=True)
class ToastMessage(ABC):
    """Abstract base class for transient popup notifications.

    "Toast" is a UI term for brief notifications that appear temporarily and
    disappear (like bread popping up from a toaster). Use these for transient
    feedback about user actions, not persistent context.

    Good for: click errors, validation failures, quick confirmations
    Bad for: context messages, status displays, instruction panels
    """

    @property
    @abstractmethod
    def message(self) -> str:
        """Formatted message for the toast notification."""
        raise NotImplementedError

    @property
    @abstractmethod
    def icon(self) -> str:
        """Icon to show in toast. Override in subclasses."""
        raise NotImplementedError

    def display(self) -> None:
        """Show this message as a toast notification and log it."""
        import streamlit as st

        logger = logging.getLogger(__name__)
        logger.info(f"[TOAST] {self.icon} {self.message}")
        st.toast(f"{self.icon} {self.message}")


# =============================================================================
# TOAST MESSAGES - Transient popup notifications for errors/feedback
# =============================================================================


@dataclass(frozen=True)
class InvalidClickMessage(ToastMessage):
    """User clicked something not allowed in current state."""

    action: str  # e.g., "view slope", "click terrain"
    reason: str  # e.g., "while building slope", "without Custom Connect enabled"

    @property
    def icon(self) -> str:
        return "⚠️"

    @property
    def message(self) -> str:
        return f"Cannot {self.action} — {self.reason}"


@dataclass(frozen=True)
class OutsideTerrainMessage(ToastMessage):
    """User clicked outside DEM/terrain coverage."""

    lat: float
    lon: float

    @property
    def icon(self) -> str:
        return "📍"

    @property
    def message(self) -> str:
        return f"Outside Terrain — Point ({self.lat:.4f}, {self.lon:.4f}) has no elevation data."


@dataclass(frozen=True)
class LiftMustGoUphillMessage(ToastMessage):
    """User clicked downhill for lift top station."""

    start_elevation_m: float
    end_elevation_m: float

    @property
    def icon(self) -> str:
        return "🚡"

    @property
    def message(self) -> str:
        diff = self.end_elevation_m - self.start_elevation_m
        return f"Lift Must Go Uphill — {self.start_elevation_m:.0f}m → {self.end_elevation_m:.0f}m ({diff:+.0f}m)"


@dataclass(frozen=True)
class SameNodeLiftMessage(ToastMessage):
    """User clicked same location for lift start and end."""

    @property
    def icon(self) -> str:
        return "🚡"

    @property
    def message(self) -> str:
        return "Same Location — Top station cannot be at the same point as bottom station."


@dataclass(frozen=True)
class TargetTooFarMessage(ToastMessage):
    """User clicked too far away in custom connect mode."""

    distance_m: float
    max_distance_m: float

    @property
    def icon(self) -> str:
        return "📏"

    @property
    def message(self) -> str:
        return f"Target Too Far — {self.distance_m:.0f}m (max: {self.max_distance_m:.0f}m)"


@dataclass(frozen=True)
class TargetNotDownhillMessage(ToastMessage):
    """User clicked uphill or flat in custom connect mode."""

    start_elevation_m: float
    target_elevation_m: float
    min_drop_m: float

    @property
    def icon(self) -> str:
        return "⛰️"

    @property
    def message(self) -> str:
        drop = self.start_elevation_m - self.target_elevation_m
        return f"Not Downhill Enough — drop: {drop:.0f}m, need: {self.min_drop_m:.0f}m"


@dataclass(frozen=True)
class FileLoadErrorMessage(ToastMessage):
    """User uploaded invalid resort file."""

    error: str

    @property
    def icon(self) -> str:
        return "📁"

    @property
    def message(self) -> str:
        return f"Load Failed — {self.error}"


# =============================================================================
# TOAST MESSAGES - Undo feedback
# =============================================================================


@dataclass(frozen=True)
class UndoSegmentMessage(ToastMessage):
    """Feedback for undoing a segment addition."""

    segment_id: str
    was_last_segment: bool = False

    @property
    def icon(self) -> str:
        return "↩️"

    @property
    def message(self) -> str:
        if self.was_last_segment:
            return f"Removed {self.segment_id} — slope canceled (no segments left)"
        return f"Removed segment {self.segment_id}"


@dataclass(frozen=True)
class UndoFinishSlopeMessage(ToastMessage):
    """Feedback for undoing a slope finish."""

    slope_name: str
    num_segments: int

    @property
    def icon(self) -> str:
        return "↩️"

    @property
    def message(self) -> str:
        return f"Reopened {self.slope_name} — {self.num_segments} segments to continue editing"


@dataclass(frozen=True)
class UndoLiftMessage(ToastMessage):
    """Feedback for undoing a lift addition."""

    lift_name: str

    @property
    def icon(self) -> str:
        return "↩️"

    @property
    def message(self) -> str:
        return f"Removed lift {self.lift_name}"


@dataclass(frozen=True)
class UndoDeleteSlopeMessage(ToastMessage):
    """Feedback for undoing a slope deletion."""

    slope_name: str

    @property
    def icon(self) -> str:
        return "↩️"

    @property
    def message(self) -> str:
        return f"Restored slope {self.slope_name}"


@dataclass(frozen=True)
class UndoDeleteLiftMessage(ToastMessage):
    """Feedback for undoing a lift deletion."""

    lift_name: str

    @property
    def icon(self) -> str:
        return "↩️"

    @property
    def message(self) -> str:
        return f"Restored lift {self.lift_name}"


@dataclass(frozen=True)
class UndoCancelSlopeMessage(ToastMessage):
    """Feedback when user presses undo but there's nothing to undo (exits slope mode)."""

    @property
    def icon(self) -> str:
        return "↩️"

    @property
    def message(self) -> str:
        return "Exited slope mode — nothing to undo"


# =============================================================================
# CENTER (UNDER MAP) - Loading states (BLUE)
# =============================================================================


@dataclass(frozen=True)
class DEMLoadingMessage(Message):
    """Shown while DEM terrain data is loading."""

    @property
    def level(self) -> MessageLevel:
        return MessageLevel.INFO

    @property
    def message(self) -> str:
        return "🗺️ **Loading Terrain Data** — This takes a few seconds on first load..."


# =============================================================================
# RIGHT PANEL - Building Context Messages
# =============================================================================


@dataclass(frozen=True)
class SlopeStartingContextMessage(Message):
    """RIGHT panel: Starting a new slope (no segments yet).

    Shows the start location when user just started building.
    """

    slope_name: str
    start_node_id: Optional[str] = None
    start_lat: Optional[float] = None
    start_lon: Optional[float] = None

    @property
    def level(self) -> MessageLevel:
        return MessageLevel.INFO

    @property
    def message(self) -> str:
        if self.start_node_id:
            start_loc = f"Node **{self.start_node_id}**"
        elif self.start_lat is not None and self.start_lon is not None:
            start_loc = f"({self.start_lat:.4f}, {self.start_lon:.4f})"
        else:
            raise ValueError("SlopeStartingContextMessage requires start_node_id or start_lat/lon")
        return f"🎿 **{self.slope_name}** — New Slope\n\n- 📍 Start: {start_loc}\n- ↔️ No segments committed yet"


@dataclass(frozen=True)
class SlopeBuildingContextMessage(Message):
    """RIGHT panel: Slope building progress message.

    Shows committed progress while actively building a slope.
    """

    slope_name: str
    num_segments: int
    difficulty_emoji: str
    total_drop_m: float
    total_length_m: float
    avg_gradient_pct: float
    max_gradient_pct: float
    start_elevation_m: float
    current_elevation_m: float

    @property
    def level(self) -> MessageLevel:
        return MessageLevel.INFO

    @property
    def message(self) -> str:
        return (
            f"🎿 **{self.slope_name}** — Committed Progress — {self.num_segments} ↔️\n\n"
            f"- {self.difficulty_emoji} • ↓{self.total_drop_m:.0f}m drop • {self.total_length_m:.0f}m\n"
            f"- 📐 {self.avg_gradient_pct:.0f}% overall / {self.max_gradient_pct:.0f}% steepest\n"
            f"- 📍 {self.start_elevation_m:.0f}m → {self.current_elevation_m:.0f}m"
        )


@dataclass(frozen=True)
class LiftPlacingContextMessage(Message):
    """RIGHT panel: Lift placing progress message.

    Shows bottom station info while awaiting top station selection.
    """

    lift_type: str = "chairlift"
    lift_icon: str = "🚡"
    bottom_node_id: Optional[str] = None
    bottom_lat: Optional[float] = None
    bottom_lon: Optional[float] = None
    bottom_elevation_m: float = 0.0

    @property
    def level(self) -> MessageLevel:
        return MessageLevel.INFO

    @property
    def lift_name(self) -> str:
        return self.lift_type.replace("_", " ").title()

    @property
    def message(self) -> str:
        if self.bottom_node_id:
            location = f"Node **{self.bottom_node_id}**"
        elif self.bottom_lat is not None and self.bottom_lon is not None:
            location = f"({self.bottom_lat:.4f}, {self.bottom_lon:.4f})"
        else:
            raise ValueError("LiftPlacingContextMessage requires bottom_node_id or bottom_lat/lon")
        return (
            f"{self.lift_icon} **{self.lift_name}** — Placing\n\n"
            f"- 🚉 Bottom station: {location}\n"
            f"- 📍 Elevation: {self.bottom_elevation_m:.0f}m"
        )


# =============================================================================
# RIGHT PANEL (CONTROL) - Action Instructions (YELLOW)
# One message telling user exactly what to do NOW
# =============================================================================


@dataclass(frozen=True)
class SlopeActionMessage(Message):
    """RIGHT panel: Specific action instruction for slope building.

    Covers: path selection, custom direction mode
    """

    # Action state flags
    is_selecting_path: bool = False
    is_custom_direction: bool = False
    is_custom_path: bool = False  # True if proposals came from custom connection

    # Path selection info (when is_selecting_path=True)
    num_paths: int = 0
    selected_path_idx: int = 0
    path_difficulty: str = ""
    path_difficulty_emoji: str = ""
    actual_gradient_pct: float = 0.0
    target_gradient_pct: float = 0.0
    path_length_m: float = 0.0
    path_drop_m: float = 0.0
    start_elevation_m: float = 0.0
    end_elevation_m: float = 0.0
    is_connector: bool = False
    target_node_id: Optional[str] = None

    @property
    def level(self) -> MessageLevel:
        return MessageLevel.WARNING

    @property
    def message(self) -> str:
        if self.is_custom_direction:
            return (
                "🎯 **Select Target**\n\n"
                "- 👆 Click **downhill** to set target direction\n"
                "- ⚪ Click near **node** to connect & finish slope"
            )

        if self.is_selecting_path:
            is_conn = self.is_connector and self.target_node_id
            path_label = "Custom Proposal" if self.is_custom_path else "Proposed Segment"
            if is_conn:
                header = f"🏁 **{path_label} {self.selected_path_idx + 1}/{self.num_paths}** → {self.target_node_id}"
                action = "- ✅ **Commit to finish slope** or use ◀▶ to browse"
            else:
                header = f"🎯 **{path_label} {self.selected_path_idx + 1}/{self.num_paths}**"
                action = "- ✅ **Commit** to add segment or use ◀▶ to browse"
            return (
                f"{header}\n\n"
                f"- {self.path_difficulty_emoji} {self.path_difficulty.capitalize()} • "
                f"↓{self.path_drop_m:.0f}m drop • {self.path_length_m:.0f}m\n"
                f"- 📐 {self.actual_gradient_pct:.0f}% overall ({self.target_gradient_pct:.0f}% target)\n"
                f"- 📍 {self.start_elevation_m:.0f}m → {self.end_elevation_m:.0f}m\n"
                f"{action}"
            )

        # No flags set - show fallback message for empty paths (terrain too steep, etc.)
        return "⚠️ **No Paths Available**\n\n- Try 🎯 **Custom Direction** to pick a target\n- Or ↩️ **Undo** to go back"


@dataclass(frozen=True)
class LiftActionMessage(Message):
    """RIGHT panel: Specific action instruction for lift placement.

    Covers: selecting top station (bottom station selection has no right panel message)
    """

    is_awaiting_top: bool = False
    bottom_elevation_m: float = 0.0

    @property
    def level(self) -> MessageLevel:
        return MessageLevel.WARNING

    @property
    def message(self) -> str:
        if self.is_awaiting_top:
            return (
                "⬆️ **Select Top Station**\n\n"
                f"- 👆 Click terrain **above {self.bottom_elevation_m:.0f}m**\n"
                "- ⚪ Or click a higher **node**"
            )
        # No right panel message needed for lift idle
        raise ValueError("No action message to display - all flags are False")


# =============================================================================
# STATS PANELS - Segment warnings
# =============================================================================


@dataclass(frozen=True)
class SegmentWarningMessage(Message):
    """Warning in slope stats panel about segment issues."""

    warning_text: str

    @property
    def level(self) -> MessageLevel:
        return MessageLevel.WARNING

    @property
    def message(self) -> str:
        return f"⚠️ {self.warning_text}"
