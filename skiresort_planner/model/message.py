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
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from skiresort_planner.model.path_segment import SegmentKind


class MessageLevel(StrEnum):
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
        logger.debug(f"[TOAST] {self.icon} {self.message}")
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
        # Fires only when end <= start. Show 1-decimal elevations so a sub-metre downhill is
        # visible (integer metres would render an identical-looking "2500m → 2500m") and the
        # diff stays arithmetically consistent with the two shown numbers.
        start, end = round(self.start_elevation_m, 1), round(self.end_elevation_m, 1)
        return f"Lift Must Go Uphill — {start:.1f}m → {end:.1f}m ({end - start:+.1f}m)"


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
        # Round the distance UP to 0.1 m so it never renders equal to the max: this fires
        # only when distance strictly exceeds the max, and ".0f" could show "1000m (max: 1000m)".
        distance_shown = math.ceil(self.distance_m * 10) / 10
        return f"Target Too Far — {distance_shown:.1f}m (max: {self.max_distance_m:.0f}m)"


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
        drop_explainer = f" (Target is {abs(drop):.0f}m above your current point)" if drop < 0 else ""
        # Round the drop DOWN to 0.1 m so it never renders equal to the minimum: this fires
        # only when drop is strictly under min_drop_m, and ".0f" could show "drop: 5m, need at least 5m".
        drop_shown = math.floor(drop * 10) / 10
        return f"Not Downhill Enough — drop: {drop_shown:.1f}m, need at least {self.min_drop_m:.0f}m" + drop_explainer


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


@dataclass(frozen=True)
class PlaceNotFoundMessage(ToastMessage):
    """The sidebar place search returned no match (or the lookup failed)."""

    query: str

    @property
    def icon(self) -> str:
        return "🔍"

    @property
    def message(self) -> str:
        return f"No place found for “{self.query}”."


@dataclass(frozen=True)
class OSMImportErrorMessage(ToastMessage):
    """An OpenStreetMap import could not run (off-coverage viewport or network/parse error)."""

    error: str

    @property
    def icon(self) -> str:
        return "🗺️"

    @property
    def message(self) -> str:
        return f"OSM import failed — {self.error}"


@dataclass(frozen=True)
class MergeTooFarMessage(ToastMessage):
    """Selected nodes span too far to merge (any pair exceeds MergeConfig.MAX_SPAN_M)."""

    span_m: float
    max_span_m: float

    @property
    def icon(self) -> str:
        return "📏"

    @property
    def message(self) -> str:
        # Round the span UP to 0.1 m so it never renders equal to the max (this fires only when
        # the span strictly exceeds the max, and ".0f" could show "500m (max: 500m)").
        span_shown = math.ceil(self.span_m * 10) / 10
        return f"Nodes Too Far Apart — {span_shown:.1f}m (max: {self.max_span_m:.0f}m)"


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
class PathBuildingContextMessage(Message):
    """RIGHT panel: build-progress context for ANY path kind (slope OR road).

    One class covers both the STARTING case (no segments yet → shows the origin) and the BUILDING
    case (≥1 segment → shows committed progress). The per-kind bits are derived from ``kind``: the
    UI noun is ``kind.capitalize()`` (no separate stored arg), and ``difficulty_emoji`` is empty for
    roads (no ski difficulty), which drops the difficulty from the stats line. No Slope*/Road* subclasses.
    """

    icon: str
    kind: SegmentKind
    name: str
    num_segments: int = 0
    # Origin (STARTING case, num_segments == 0): a node id OR a lat/lon.
    start_node_id: str | None = None
    start_lat: float | None = None
    start_lon: float | None = None
    # Committed stats (BUILDING case, num_segments > 0).
    difficulty_emoji: str = ""  # "" for roads → difficulty omitted from the line
    total_drop_m: float = 0.0
    total_length_m: float = 0.0
    avg_gradient_pct: float = 0.0
    max_gradient_pct: float = 0.0
    start_elevation_m: float = 0.0
    current_elevation_m: float = 0.0

    @property
    def level(self) -> MessageLevel:
        return MessageLevel.INFO

    @property
    def message(self) -> str:
        if self.num_segments == 0:
            if self.start_node_id:
                start_loc = f"Node **{self.start_node_id}**"
            elif self.start_lat is not None and self.start_lon is not None:
                start_loc = f"({self.start_lat:.4f}, {self.start_lon:.4f})"
            else:
                raise ValueError("PathBuildingContextMessage (starting) requires start_node_id or start_lat/lon")
            # self.kind is a StrEnum → capitalize() gives the title-case noun ("Slope"/"Road").
            return f"{self.icon} **{self.name}** — New {self.kind.capitalize()}\n\n- 📍 Start: {start_loc}\n- ↔️ No segments committed yet"

        # Building: roads have no ski difficulty, so lead the stats line with the drop instead.
        # Show drop/gradient as MAGNITUDES (the backend has "downhill is positive").
        lead = f"{self.difficulty_emoji} • " if self.difficulty_emoji else ""
        return (
            f"{self.icon} **{self.name}** — Committed Progress — {self.num_segments} ↔️\n\n"
            f"- {lead}↕{abs(self.total_drop_m):.0f}m • {self.total_length_m:.0f}m\n"
            f"- 📐 {abs(self.avg_gradient_pct):.0f}% overall / {abs(self.max_gradient_pct):.0f}% steepest\n"
            f"- 📍 {self.start_elevation_m:.0f}m → {self.current_elevation_m:.0f}m"
        )


@dataclass(frozen=True)
class LiftPlacingContextMessage(Message):
    """RIGHT panel: Lift placing progress message.

    Shows bottom station info while awaiting top station selection.
    """

    lift_type: str = "chairlift"
    lift_icon: str = "🚡"
    bottom_node_id: str | None = None
    bottom_lat: float | None = None
    bottom_lon: float | None = None
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


@dataclass(frozen=True)
class ImportPlacingContextMessage(Message):
    """RIGHT panel: OSM import placement progress — shows the placed box center + area size."""

    center_lat: float = 0.0
    center_lon: float = 0.0
    half_width_km: float = 0.0

    @property
    def level(self) -> MessageLevel:
        return MessageLevel.INFO

    @property
    def message(self) -> str:
        side_km = self.half_width_km * 2
        return (
            "🗺️ **Import from OpenStreetMap** — Placing\n\n"
            f"- 📍 Center: ({self.center_lat:.4f}, {self.center_lon:.4f})\n"
            f"- ⬜ Area: {side_km:.1f} × {side_km:.1f} km"
        )


@dataclass(frozen=True)
class MergePlacingContextMessage(Message):
    """RIGHT panel: node-merge selection progress — shows how many nodes are selected + their span."""

    selected_count: int = 0
    span_m: float = 0.0

    @property
    def level(self) -> MessageLevel:
        return MessageLevel.INFO

    @property
    def message(self) -> str:
        if self.selected_count == 0:
            return "🔗 **Merge Nodes** — Selecting\n\n- 👆 Click node markers to select them"
        return (
            "🔗 **Merge Nodes** — Selecting\n\n"
            f"- ⚪ Selected: {self.selected_count} node(s)\n"
            f"- 📏 Span: {self.span_m:.0f}m"
        )


# =============================================================================
# RIGHT PANEL (CONTROL) - Action Instructions (YELLOW)
# One message telling user exactly what to do NOW
# =============================================================================


def too_steep_detail(gentlest_pct: float | None, max_grade_pct: float, subject: str, *, two_sided: bool) -> str:
    """The "why" line for a too-steep path result (slope OR road), shared by the panels.

    ``subject`` ("to ski" / "for a car road") and ``two_sided`` (roads cap a ±band, slopes a
    single-sided ceiling) are the two per-kind differences, passed as data. ``gentlest_pct`` is the
    magnitude of the gentlest route found, or None when the planner found no route at all.
    """
    band = f"±{max_grade_pct:.0f}%" if two_sided else f"{max_grade_pct:.0f}%"
    if gentlest_pct is None:
        reach = f"within {band}" if two_sided else f"under {band}"
        return f"Too steep {subject} — no route to that point {reach}."
    # Round the gentlest UP to 0.1% so it never renders equal to the cap
    gentlest_shown = math.ceil(gentlest_pct * 10) / 10
    return f"Too steep {subject} — gentlest possible is {gentlest_shown:.1f}%, over the {band} limit."


@dataclass(frozen=True)
class PathActionMessage(Message):
    """RIGHT panel: action instruction while selecting a path proposal (slope OR road).

    Covers fan-out and custom-connect proposals for any path kind. The commit/finish wording noun
    is derived from ``kind`` (``kind`` value, e.g. "slope"/"road" — no separate stored arg), and
    ``path_difficulty`` (empty for roads) keeps the stats line kind-correct: the difficulty line is
    omitted when there is no difficulty (roads).
    """

    kind: SegmentKind

    # Action state flags
    is_selecting_path: bool = False
    is_custom_path: bool = False  # True if proposals came from custom connection

    # Path selection info (when is_selecting_path=True)
    num_paths: int = 0
    selected_path_idx: int = 0
    path_difficulty: str = ""  # "" for roads → the difficulty line is omitted
    path_difficulty_emoji: str = ""
    actual_gradient_pct: float = 0.0
    target_gradient_pct: float = 0.0
    path_length_m: float = 0.0
    path_drop_m: float = 0.0
    start_elevation_m: float = 0.0
    end_elevation_m: float = 0.0
    is_connector: bool = False
    target_node_id: str | None = None

    # Too-steep detail for the empty-paths branch. `too_steep_gentlest_pct`:
    # None means NOT too steep (plain guidance); a value means too steep and IS the gentlest grade to report.
    too_steep_gentlest_pct: float | None = None
    too_steep_cap_pct: float = 0.0
    too_steep_subject: str = ""
    too_steep_two_sided: bool = False

    @property
    def level(self) -> MessageLevel:
        return MessageLevel.WARNING

    @property
    def message(self) -> str:
        # Two states: either a proposal is selected (show its stats) or the list is empty (guidance).
        if self.is_selecting_path:
            return self._selecting_message()
        return self._empty_message()

    def _selecting_message(self) -> str:
        """The stats block for the currently-selected proposal (is_selecting_path)."""
        is_conn = self.is_connector and self.target_node_id
        path_label = "Custom Proposal" if self.is_custom_path else "Proposed Segment"
        if is_conn:
            header = f"🏁 **{path_label} {self.selected_path_idx + 1}/{self.num_paths}** → {self.target_node_id}"
            # self.kind is a StrEnum → renders as "slope"/"road" directly.
            action = f"- ✅ **Commit to finish {self.kind}** or use ◀▶ to browse"
        else:
            header = f"🎯 **{path_label} {self.selected_path_idx + 1}/{self.num_paths}**"
            action = "- ✅ **Commit** to add segment or use ◀▶ to browse"
        # Roads carry no ski difficulty, so drop that line for them.
        difficulty_line = (
            f"- {self.path_difficulty_emoji} {self.path_difficulty.capitalize()} • " if self.path_difficulty else "- "
        )
        # Show drop and gradient as MAGNITUDES: the backend has "downhill is positive".
        return (
            f"{header}\n\n"
            f"{difficulty_line}↕{abs(self.path_drop_m):.0f}m • {self.path_length_m:.0f}m\n"
            f"- 📐 {abs(self.actual_gradient_pct):.0f}% overall ({abs(self.target_gradient_pct):.0f}% target)\n"
            f"- 📍 {self.start_elevation_m:.0f}m → {self.end_elevation_m:.0f}m\n"
            f"{action}"
        )

    def _empty_message(self) -> str:
        """The "No Paths Available" block, led by the too-steep reason when that's the cause."""
        # too_steep_gentlest_pct is the single signal: set → too steep (lead with the why line).
        if self.too_steep_gentlest_pct is not None:
            reason = (
                too_steep_detail(
                    gentlest_pct=self.too_steep_gentlest_pct,
                    max_grade_pct=self.too_steep_cap_pct,
                    subject=self.too_steep_subject,
                    two_sided=self.too_steep_two_sided,
                )
                + "\n\n"
            )
        else:
            reason = ""

        # The escape differs by origin: a custom target returns to the fan via Cancel Custom Path,
        # a fan extension steps back via Undo.
        if self.is_custom_path:
            guidance = (
                "- 👆 Click a **gentler** point or **node** to route there\n"
                "- Or ✖️ **Cancel Custom Path** to go back to the fan-out"
            )
        else:
            guidance = "- 👆 Click a **downhill** point or **node** to route a path to it\n- Or ↩️ **Undo** to go back"

        return f"⚠️ **No Paths Available**\n\n{reason}{guidance}"


@dataclass(frozen=True)
class LiftActionMessage(Message):
    """RIGHT panel: instruction to select the top station during lift placement."""

    bottom_elevation_m: float = 0.0

    @property
    def level(self) -> MessageLevel:
        return MessageLevel.WARNING

    @property
    def message(self) -> str:
        return (
            "⬆️ **Select Top Station**\n\n"
            f"- 👆 Click terrain **above {self.bottom_elevation_m:.0f}m**\n"
            "- ⚪ Or click a higher **node**"
        )


@dataclass(frozen=True)
class ImportActionMessage(Message):
    """RIGHT panel: action instruction while placing an OSM import box."""

    @property
    def level(self) -> MessageLevel:
        return MessageLevel.WARNING

    @property
    def message(self) -> str:
        return (
            "🗺️ **Confirm the Import Area**\n\n"
            "- ↔️ Resize with the **half-width slider** (left)\n"
            "- 👆 Click terrain to **re-place** the center\n"
            "- ✅ Click the **center dot** or **Confirm Import** to fetch"
        )


@dataclass(frozen=True)
class MergeActionMessage(Message):
    """RIGHT panel: action instruction while selecting nodes to merge."""

    selected_count: int = 0

    @property
    def level(self) -> MessageLevel:
        return MessageLevel.WARNING

    @property
    def message(self) -> str:
        if self.selected_count < 2:
            return (
                "🔗 **Select Nodes to Merge**\n\n"
                "- 👆 Click **node markers** to select (click again to deselect)\n"
                "- Select at least **2 nodes** to merge them into one"
            )
        return (
            "🔗 **Merge the Selected Nodes**\n\n"
            "- 👆 Click more **node markers** to add/remove\n"
            "- ✅ Click **Confirm Merge** to collapse them to their median position"
        )


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
