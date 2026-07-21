"""Message - User-facing messages for the ski resort planner UI.

Architecture:
- LEFT (sidebar): ONE blue info message showing current mode, progress, and general capabilities
- CENTER (under map): blue loading / yellow invalid-click messages
- RIGHT (control panel): ONE yellow instruction message for what to do NOW

Design Principles:
- Maximum ONE message per panel location at any time
- Two levels only: INFO (blue) = context/status/loading, WARNING (yellow) = invalid input / next step.
- Subclass InfoMessage/WarningMessage for inline messages (level/icon fixed by the base). Transient
  toasts are always WarningToast (yellow); a subclass supplies only its text.

All data (elevations, node names, stats) must be preserved in the consolidated messages.
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from skiresort_planner.constants import LiftType, OSMImportMode, StyleConfig
from skiresort_planner.model.path_segment import SegmentKind


class MessageLevel(StrEnum):
    """Display level for UI messages. Only two levels: blue INFO and yellow WARNING."""

    INFO = "info"  # Blue - context/status/loading
    WARNING = "warning"  # Yellow - action instructions / invalid input


@dataclass(frozen=True)
class Message(ABC):
    """Abstract base class for user-facing messages displayed inline (sidebars/panels).

    These messages are rendered as st.info/st.warning blocks that persist in the UI until replaced.
    Used for context, instructions, and status. Subclass via InfoMessage/WarningMessage — those fix
    the level so a subclass only supplies its text.
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
        }[self.level]
        render_fn(self.message)


@dataclass(frozen=True)
class InfoMessage(Message):
    """Inline blue (INFO) message — loading/status/context. Subclasses supply only `message`."""

    @property
    def level(self) -> MessageLevel:
        return MessageLevel.INFO


@dataclass(frozen=True)
class WarningMessage(Message):
    """Inline yellow (WARNING) message — next-step instructions / invalid input. Text-only subclasses."""

    @property
    def level(self) -> MessageLevel:
        return MessageLevel.WARNING


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


@dataclass(frozen=True)
class WarningToast(ToastMessage):
    """Transient yellow (WARNING) toast — invalid input / rejected action. The only toast level:
    toasts are always transient warnings (a subclass may override `icon` for a topical glyph).
    """

    @property
    def icon(self) -> str:
        return "⚠️"


# =============================================================================
# TOAST MESSAGES - Transient popup notifications for errors/feedback
# =============================================================================


@dataclass(frozen=True)
class InvalidClickMessage(WarningToast):
    """User clicked something not allowed in current state."""

    action: str  # e.g., "view slope", "click terrain"
    reason: str  # e.g., "while building slope", "without Custom Connect enabled"

    @property
    def message(self) -> str:
        return f"Cannot {self.action} — {self.reason}"


@dataclass(frozen=True)
class OutsideTerrainMessage(WarningToast):
    """User clicked outside DEM/terrain coverage."""

    lat: float
    lon: float

    @property
    def message(self) -> str:
        return f"Outside Terrain — Point ({self.lat:.4f}, {self.lon:.4f}) has no elevation data."


@dataclass(frozen=True)
class SameNodeLiftMessage(WarningToast):
    """User clicked the same location for both lift stations."""

    @property
    def message(self) -> str:
        return "Same Location — a lift needs two different stations."


@dataclass(frozen=True)
class TargetTooFarMessage(WarningToast):
    """User clicked too far away in custom connect mode."""

    distance_m: float
    max_distance_m: float

    @property
    def message(self) -> str:
        # Round the distance UP to the next whole metre so it never renders equal to the max: this
        # fires only when distance strictly exceeds the max, and ".0f" could show "1000m (max: 1000m)".
        distance_shown = math.ceil(self.distance_m)
        return f"Target Too Far — {distance_shown:.0f}m (max: {self.max_distance_m:.0f}m)"


@dataclass(frozen=True)
class TargetNotDownhillMessage(WarningToast):
    """User clicked uphill or flat in custom connect mode."""

    start_elevation_m: float
    target_elevation_m: float
    min_drop_m: float

    @property
    def message(self) -> str:
        drop = self.start_elevation_m - self.target_elevation_m
        drop_explainer = f" (Target is {abs(drop):.0f}m above your current point)" if drop < 0 else ""
        # Round the drop DOWN to the next whole metre so it never renders equal to the minimum: this
        # fires only when drop is strictly under min_drop_m, and ".0f" could show "drop: 5m, need at least 5m".
        drop_shown = math.floor(drop)
        return f"Not Downhill Enough — drop: {drop_shown:.0f}m, need at least {self.min_drop_m:.0f}m" + drop_explainer


@dataclass(frozen=True)
class FileLoadErrorMessage(WarningToast):
    """User uploaded invalid resort file."""

    error: str

    @property
    def message(self) -> str:
        return f"Load Failed — {self.error}"


@dataclass(frozen=True)
class UploadBlockedMessage(WarningToast):
    """Upload attempted while the resort still has content."""

    @property
    def message(self) -> str:
        return "Clear the resort first — use “🗑️ Reset to Empty” before loading a file."


@dataclass(frozen=True)
class PlaceNotFoundMessage(WarningToast):
    """The sidebar place search returned no match (or the lookup failed)."""

    query: str

    @property
    def message(self) -> str:
        return f"No place found for “{self.query}”."


@dataclass(frozen=True)
class OSMImportErrorMessage(WarningToast):
    """An OpenStreetMap import could not run (off-coverage viewport or network/parse error)."""

    error: str

    @property
    def message(self) -> str:
        return f"OSM import failed — {self.error}"


@dataclass(frozen=True)
class MergeTooFarMessage(WarningToast):
    """Selected nodes span too far to merge (any pair exceeds MergeConfig.MAX_SPAN_M)."""

    span_m: float
    max_span_m: float

    @property
    def message(self) -> str:
        # Round the span UP to the next whole metre so it never renders equal to the max (this fires
        # only when the span strictly exceeds the max, and ".0f" could show "500m (max: 500m)").
        span_shown = math.ceil(self.span_m)
        return f"Nodes Too Far Apart — {span_shown:.0f}m (max: {self.max_span_m:.0f}m)"


@dataclass(frozen=True)
class UnableToDeleteMessage(WarningToast):
    """A selected node can't be deleted (lift station, shared/branch junction, or sole segment)."""

    reason: str  # human sentence, e.g. "N5 is a lift station — delete the lift first"

    @property
    def message(self) -> str:
        return f"Cannot delete — {self.reason}"


@dataclass(frozen=True)
class ClickingDisabledIn3DToast(WarningToast):
    """User clicked the map in 3D view, where deck.gl picking is unreliable (default ⚠️ icon)."""

    @property
    def message(self) -> str:
        return "Clicking disabled in 3D view. Return to 2D to interact with the map."


# =============================================================================
# CENTER (UNDER MAP) - Loading states (BLUE)
# =============================================================================


@dataclass(frozen=True)
class DEMLoadingMessage(InfoMessage):
    """Shown while DEM terrain data is loading."""

    @property
    def message(self) -> str:
        return "🗺️ **Loading Terrain Data** — This takes a few seconds on first load…"


@dataclass(frozen=True)
class SizingMapMessage(InfoMessage):
    """Shown once on first load while the browser viewport height resolves (no map this pass)."""

    @property
    def message(self) -> str:
        return "📐 Sizing map to your window…"


@dataclass(frozen=True)
class OSMImportLoadingMessage(InfoMessage):
    """Shown while an OSM import fetches + builds (blocks the whole render; no map this pass).

    Icons come from StyleConfig (the single source the import buttons use) so the two never drift.
    """

    mode: OSMImportMode

    @property
    def message(self) -> str:
        if self.mode == OSMImportMode.LIFTS_ONLY:
            return f"{StyleConfig.LIFT_ICONS[LiftType.GONDOLA]} Importing lifts from OpenStreetMap…"
        return f"{StyleConfig.LIFT_ICONS[LiftType.GONDOLA]}{StyleConfig.SLOPE_ICON} Importing lifts + slopes from OpenStreetMap…"


# =============================================================================
# RIGHT PANEL - Building Context Messages
# =============================================================================


@dataclass(frozen=True)
class PathBuildingContextMessage(InfoMessage):
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
class LiftPlacingContextMessage(InfoMessage):
    """RIGHT panel: Lift placing progress message.

    Shows the first-station info while awaiting the second station selection.
    """

    lift_type: str = "chairlift"
    first_node_id: str | None = None
    first_lat: float | None = None
    first_lon: float | None = None
    first_elevation_m: float = 0.0

    @property
    def lift_icon(self) -> str:
        # Derive from StyleConfig (the single source the type buttons use) so the two never drift.
        return StyleConfig.LIFT_ICONS[self.lift_type]

    @property
    def lift_name(self) -> str:
        return self.lift_type.replace("_", " ").title()

    @property
    def message(self) -> str:
        if self.first_node_id:
            location = f"Node **{self.first_node_id}**"
        elif self.first_lat is not None and self.first_lon is not None:
            location = f"({self.first_lat:.4f}, {self.first_lon:.4f})"
        else:
            raise ValueError("LiftPlacingContextMessage requires first_node_id or first_lat/lon")
        return (
            f"{self.lift_icon} **{self.lift_name}** — Placing\n\n"
            f"- 🚉 First station: {location}\n"
            f"- 📍 Elevation: {self.first_elevation_m:.0f}m"
        )


@dataclass(frozen=True)
class ImportSelectingContextMessage(InfoMessage):
    """RIGHT panel: OSM import placement progress — shows the placed box center + area size."""

    center_lat: float = 0.0
    center_lon: float = 0.0
    half_width_km: float = 0.0

    @property
    def message(self) -> str:
        side_km = self.half_width_km * 2
        return (
            "🗺️ **Import from OpenStreetMap** — Placing\n\n"
            f"- 📍 Center: ({self.center_lat:.4f}, {self.center_lon:.4f})\n"
            f"- ⬜ Area: {side_km:.1f} × {side_km:.1f} km"
        )


@dataclass(frozen=True)
class NodeEditContextMessage(InfoMessage):
    """RIGHT panel: node-editor selection progress — shows how many nodes are selected + their span."""

    selected_count: int = 0
    span_m: float = 0.0

    @property
    def message(self) -> str:
        # One header, body varies by selection count (mirrors LiftPlacingContextMessage's single return).
        if self.selected_count == 0:
            body = "- 👆 Click node markers, or a path to add a node"
        else:
            body = f"- ⚪ Selected: {self.selected_count} node(s)\n- 📏 Span: {self.span_m:.0f}m"
        return f"🔗 **Edit Nodes** — Selecting\n\n{body}"


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
class PathActionMessage(WarningMessage):
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
class LiftActionMessage(WarningMessage):
    """RIGHT panel: instruction to select the second station during lift placement."""

    @property
    def message(self) -> str:
        return (
            "⬆️ **Select Second Station**\n\n"
            "- 👆 Click terrain or a **node** for the other station\n"
            "- ↕️ The lift auto-orients low → high (always goes up)"
        )


@dataclass(frozen=True)
class ImportActionMessage(WarningMessage):
    """RIGHT panel: action instruction while placing an OSM import box."""

    @property
    def message(self) -> str:
        return (
            "🗺️ **Confirm the Import Area**\n\n"
            "- ↔️ Resize with the **half-width slider** (left)\n"
            "- 👆 Click terrain to **re-place** the center\n"
            "- ✅ Click **Import lifts + slopes** or **Import lifts only** to fetch"
        )


@dataclass(frozen=True)
class NodeEditActionMessage(WarningMessage):
    """RIGHT panel: action instruction while editing nodes (select → add/delete/merge)."""

    selected_count: int = 0

    @property
    def message(self) -> str:
        if self.selected_count < 2:
            return (
                "🔗 **Select Nodes** — merge, delete, or click a path\n\n"
                "- 👆 Click **node markers** to select (again to deselect)\n"
                "- 🗑️ **Delete** trims 1 node • 🔗 **Merge** needs 2 • or click a **path** to add a node"
            )
        return (
            "🔗 **Merge or Delete the Selected Nodes**\n\n"
            "- 👆 Click more **node markers** to add/remove\n"
            "- ✅ **Confirm Merge** to collapse them • 🗑️ **Delete** to remove them"
        )


# =============================================================================
# STATS PANELS - Segment warnings
# =============================================================================


@dataclass(frozen=True)
class SegmentWarningMessage(WarningMessage):
    """Warning in slope stats panel about segment issues."""

    warning_text: str

    @property
    def message(self) -> str:
        return f"⚠️ {self.warning_text}"


@dataclass(frozen=True)
class DisconnectedEntityMessage(WarningMessage):
    """Warning that the viewed slope/lift can't be reached from the core resort."""

    entity_noun: str  # "slope" / "lift"
    core_lift_name: str  # longest lift in the core area — orients the user

    @property
    def message(self) -> str:
        return (
            f"⚠️ This {self.entity_noun} is disconnected from the core area (with {self.core_lift_name}) — "
            "it can't be reached by skiing slopes or taking lifts."
        )


@dataclass(frozen=True)
class NoReturnEntityMessage(WarningMessage):
    """Warning that after taking the viewed slope/lift you can't get back to ride it again."""

    entity_noun: str  # "slope" / "lift"

    @property
    def message(self) -> str:
        return (
            f"⚠️ This {self.entity_noun} is a one-way trip — once you take it, no sequence of slopes "
            "and lifts brings you back to ride it again."
        )


# =============================================================================
# ROUTE PLANNER — pick start/end, then browse the best routes by criterion
# =============================================================================


@dataclass(frozen=True)
class RoutePlacingContextMessage(InfoMessage):
    """RIGHT panel (blue): where the route START was placed (node + elevation).

    Mirrors LiftPlacingContextMessage's first-station block. In route_placing the start is always
    set (the first node click sets it before the transition), so this always renders — no branch.
    """

    start_node_id: str
    start_elevation_m: float

    @property
    def message(self) -> str:
        return (
            f"🧭 **Route Planner** — Placing\n\n"
            f"- 🚩 Start: node **{self.start_node_id}**\n"
            f"- 📍 Elevation: {self.start_elevation_m:.0f}m"
        )


@dataclass(frozen=True)
class RoutePlacingActionMessage(WarningMessage):
    """RIGHT panel (yellow): instruction to click the end node — a DIFFERENT node for shortest routes,
    or the SAME start node again for a scenic tour of every reachable lift.
    """

    @property
    def message(self) -> str:
        return (
            "🏁 **Select End Node**\n\n"
            "- 👆 Click another **node** for the fastest routes there.\n"
            "- 🔁 Click the **same start node** again for a scenic tour of every lift, back to start."
        )


@dataclass(frozen=True)
class RouteResultsContextMessage(InfoMessage):
    """RIGHT panel (blue): how many routes were found under the shown difficulty cap, and which one."""

    total: int = 0
    selected_index: int = 0  # 0-based
    difficulty_cap: str = "black"  # the premise: hardest band allowed for the shown routes

    @property
    def message(self) -> str:
        return f"🛣️ **Routes** — showing {self.selected_index + 1} of {self.total} (max **{self.difficulty_cap}**)"


@dataclass(frozen=True)
class RouteNoResultsMessage(WarningMessage):
    """RIGHT panel (yellow): no routes under the shown cap — distinguish "cap too strict" from "none exist"."""

    cap_restrictive: bool = False  # True when an easier cap is hiding an otherwise-reachable route

    @property
    def message(self) -> str:
        if self.cap_restrictive:
            return (
                "⚠️ **No route within this difficulty**\n\n- 🎚️ Raise the max-difficulty selector to allow harder slopes."
            )
        return (
            "⚠️ **No route exists**\n\n"
            "- 🎯 No way to ski/lift from the start to the end.\n- Build connecting slopes or lifts."
        )
