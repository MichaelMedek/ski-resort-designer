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

import streamlit as st

logger = logging.getLogger(__name__)


def trigger_rerun(scope: str = "app") -> None:
    """Trigger Streamlit rerun with optional scope.

    This is a mockable wrapper around st.rerun() for testability.
    In tests, patch 'skiresort_planner.ui.infra.trigger_rerun' to prevent
    actual reruns (which raise StopExecution).

    Args:
        scope: Rerun scope - "app" for full rerun, "fragment" for partial.
    """
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
            ctx.deferred.path_generation = True
        reload_map(before=setup_for_reload)
    """
    if before is not None:
        before()
    bump_map_version()
    trigger_rerun()
