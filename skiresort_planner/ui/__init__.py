"""User interface components for ski resort planner.

File Structure (layout-based naming):
- left_panel.py: Sidebar with mode selection, building controls, stats
- center_map.py: Pydeck map with slopes, lifts, proposals
- right_panel.py: Control panels, path selection, stats panels
- bottom_chart.py: Plotly elevation profile charts
- pydeck_click_handler.py: Custom click handling for Pydeck maps

Core Components:
- state_machine.py: PlannerStateMachine (all states) + PlannerContext
- actions.py: All action functions (commit, finish, undo, etc.)
- click_handlers.py: State-specific map click processing
- validators.py: Input validation with Optional[Message] returns

UI workflow documented in DETAILS_UI.md.
"""

from skiresort_planner.ui.actions import (
    bump_map_version,
    cancel_current_road,
    cancel_current_slope,
    cancel_custom_path,
    center_on_lift,
    center_on_road,
    center_on_slope,
    commit_selected_path,
    confirm_import_action,
    finish_current_road,
    finish_current_slope,
    handle_fast_deferred_actions,
    process_custom_connect_deferred,
    process_osm_import_deferred,
    process_path_generation_deferred,
    recompute_paths,
    reload_map,
    trigger_rerun,
    undo_last_action,
)
from skiresort_planner.ui.bottom_chart import (
    ProfileChart,
    render_building_profile,
    render_viewing_profile,
)
from skiresort_planner.ui.center_map import MapRenderer
from skiresort_planner.ui.click_detector import ClickDetector
from skiresort_planner.ui.click_handlers import dispatch_click
from skiresort_planner.ui.infra import viewport_map_height
from skiresort_planner.ui.left_panel import SidebarRenderer
from skiresort_planner.ui.pydeck_click_handler import PydeckClickResult, render_pydeck_map
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
    "render_building_profile",
    "render_viewing_profile",
    "SidebarRenderer",
    "PathSelectionPanel",
    "SlopeStatsPanel",
    "LiftStatsPanel",
    "ClickDetector",
    "dispatch_click",
    "render_control_panel",
    "bump_map_version",
    "cancel_custom_path",
    "cancel_current_road",
    "cancel_current_slope",
    "center_on_lift",
    "center_on_road",
    "center_on_slope",
    "commit_selected_path",
    "finish_current_road",
    "finish_current_slope",
    "handle_fast_deferred_actions",
    "confirm_import_action",
    "process_custom_connect_deferred",
    "process_osm_import_deferred",
    "process_path_generation_deferred",
    "recompute_paths",
    "reload_map",
    "trigger_rerun",
    "undo_last_action",
    "viewport_map_height",
    "PydeckClickResult",
    "render_pydeck_map",
]
