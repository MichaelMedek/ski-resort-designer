"""Infrastructure utilities for Streamlit UI operations.

This module abstracts Streamlit-specific infrastructure (st.rerun, st.session_state)
to enable mockability in tests while keeping actions.py as the orchestrator.

Pattern: Actions import from this module. Tests mock these functions instead of
mocking 10+ places where st.rerun might be called directly.

IMPORTANT: Only infrastructure belongs here (rerun, map version).
- Session state object access (sm, ctx, graph) stays in actions.py
- UI presentation (st.spinner) stays in app.py caller around process_*_deferred calls
"""

import logging
from collections.abc import Callable
from typing import Literal

import streamlit as st
from streamlit_js_eval import streamlit_js_eval  # type: ignore[import-untyped]

from skiresort_planner.constants import ChartConfig
from skiresort_planner.persistence import backup_store

logger = logging.getLogger(__name__)


def _autosave_if_dirty() -> None:
    """Persist the resort to disk if it changed since the last save.

    Called from trigger_rerun() — the single choke point that runs after
    every graph mutation and before every rerun (st.rerun raises
    StopExecution, so any code placed *after* a mutation-triggered rerun
    never executes; this is the only reliable central location).

    A cheap change token (entity counters + undo-stack length) gates the
    write so no-op reruns don't re-serialize the graph.
    """
    resort_id = st.session_state.get("resort_id")
    graph = st.session_state.get("graph")
    if resort_id is None or graph is None:
        return

    token = graph.change_token()
    if token == st.session_state.get("_saved_token"):
        return

    backup_store.save(graph=graph, resort_id=resort_id)
    st.session_state._saved_token = token


def trigger_rerun(scope: Literal["app", "fragment"] = "app") -> None:
    """Trigger Streamlit rerun with optional scope.

    This is a mockable wrapper around st.rerun() for testability.
    In tests, patch 'skiresort_planner.ui.infra.trigger_rerun' to prevent
    actual reruns (which raise StopExecution).

    Autosaves the resort first (dirty-checked) so every mutation is
    persisted before the browser re-requests.

    Args:
        scope: Rerun scope - "app" for full rerun, "fragment" for partial.
    """
    _autosave_if_dirty()
    st.rerun(scope=scope)


def bump_map_version() -> None:
    """Increment map_version to create fresh Pydeck component.

    This eliminates ghost clicks by creating a new component instance
    with no memory of previous click events. Call this when completing
    actions that should clear stale click state.
    """
    old_version = st.session_state.get("map_version", 0)
    new_version = old_version + 1
    st.session_state.map_version = new_version
    logger.info(f"[MAP] Bumped map_version: {old_version} -> {new_version}")


def reload_map(before: Callable[[], None] | None = None) -> None:
    """Reload map with optional pre-reload callback.

    This is the canonical way to reload the map. It provides a single point
    for all map reloads, making the pattern explicit and consistent.

    The flow is:
    1. Execute before callback (if provided) - runs BEFORE st.rerun()
    2. Bump map version to clear stale click state
    3. Call trigger_rerun() which raises StopExecution

    For actions that need to run AFTER the reload, use the deferred action
    pattern (set ctx.deferred.* flags before calling this).

    Args:
        before: Optional callback to execute before rerun.
                Use for state updates that must happen before reload.

    Example:
        # Simple reload
        reload_map()

        # Reload with pre-action
        def setup_for_reload():
            ctx.set_selection(lon=x, lat=y, elevation=e)
            ctx.deferred.fan_generation.add(SegmentKind.SLOPE)
        reload_map(before=setup_for_reload)
    """
    if before is not None:
        before()
    bump_map_version()
    trigger_rerun()


def viewport_map_height(reserved_below_px: int = 0) -> int | None:
    """Map height in px that fills the browser window, or None only on first load.

    The  JS component reports parent.innerHeight (the real browser window), and only on
    the render its round-trip resolved — every other rerun returns None. We cache the
    last real value so the map never blanks on the constant reruns a stateful app makes.

    Args:
        reserved_below_px: Height to leave free below the map (e.g. an elevation
            profile chart) so it stays visible without scrolling. 0 = map fills all.

    Returns None only before the very first successful read (caller shows a placeholder);
    thereafter the cached viewport height minus reserved space, floored at a minimum.
    """
    value = streamlit_js_eval(js_expressions="parent.innerHeight", key="window_inner_height")
    if isinstance(value, int | float):
        st.session_state.window_height_px = int(value)
    window_height = st.session_state.get("window_height_px")
    if window_height is None:
        return None
    available: int = int(window_height) - ChartConfig.MAP_TOP_OFFSET_PX - reserved_below_px
    return max(available, ChartConfig.MAP_MIN_HEIGHT_PX)
