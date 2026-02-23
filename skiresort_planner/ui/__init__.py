"""User interface components for ski resort planner.

File Structure (layout-based naming):
- left_panel.py: Sidebar with mode selection, building controls, stats
- center_map.py: Folium map with slopes, lifts, proposals
- right_panel.py: Control panels, path selection, stats panels
- bottom_chart.py: Plotly elevation profile charts

Core Components:
- state_machine.py: PlannerStateMachine (4 states) + PlannerContext
- actions.py: All action functions (commit, finish, undo, etc.)
- click_handlers.py: State-specific map click processing
- validators.py: Input validation with Optional[Message] returns

UI workflow documented in DETAILS_UI.md.
"""

from skiresort_planner.ui.actions import (
    bump_map_version,
    cancel_connection_mode,
    cancel_current_slope,
    cancel_custom_direction_mode,
    center_on_lift,
    center_on_slope,
    commit_selected_path,
    enter_custom_direction_mode,
    finish_current_slope,
    handle_deferred_actions,
    recompute_paths,
    undo_last_action,
)
from skiresort_planner.ui.bottom_chart import ProfileChart
from skiresort_planner.ui.center_map import MapRenderer
from skiresort_planner.ui.click_detector import ClickDetector
from skiresort_planner.ui.click_handlers import dispatch_click
from skiresort_planner.ui.left_panel import SidebarRenderer
from skiresort_planner.ui.right_panel import (
    LiftStatsPanel,
    PathSelectionPanel,
    SlopeStatsPanel,
    render_control_panel,
)
from skiresort_planner.ui.state_machine import (
    PlannerContext,
    PlannerStateMachine,
    StreamlitUIListener,
)

__all__ = [
    "PlannerStateMachine",
    "PlannerContext",
    "StreamlitUIListener",
    "MapRenderer",
    "ProfileChart",
    "SidebarRenderer",
    "PathSelectionPanel",
    "SlopeStatsPanel",
    "LiftStatsPanel",
    "ClickDetector",
    "dispatch_click",
    "render_control_panel",
    "bump_map_version",
    "cancel_connection_mode",
    "cancel_current_slope",
    "cancel_custom_direction_mode",
    "center_on_lift",
    "center_on_slope",
    "commit_selected_path",
    "enter_custom_direction_mode",
    "finish_current_slope",
    "handle_deferred_actions",
    "recompute_paths",
    "undo_last_action",
]
