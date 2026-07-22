"""Infrastructure utilities for Streamlit UI operations.

This module abstracts Streamlit-specific infrastructure (st.rerun, st.session_state)
to enable mockability in tests while keeping actions.py as the orchestrator.

Pattern: Actions import from this module. Tests mock these functions instead of
mocking 10+ places where st.rerun might be called directly.

IMPORTANT: Only infrastructure belongs here (rerun, map version).
- Session state object access (sm, ctx, graph) stays in actions.py
- UI presentation (loading messages, progress bars, toasts) stays in app.py's pending-action helpers
"""

import logging
from typing import Literal

import streamlit as st
from streamlit_js_eval import streamlit_js_eval  # type: ignore[import-untyped]

from skiresort_planner.constants import ChartConfig, MapConfig
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
        logger.warning(
            f"Autosave skipped: uninitialized state (resort_id={resort_id}, graph={'set' if graph else None})"
        )
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


def _bump_epoch(key: str) -> None:
    """Increment a session-state remount epoch. Tolerant get is load-bearing (pre-bootstrap in
    tests), NOT defensive — see the autosave-choke-point tolerant-gets note.
    """
    old = st.session_state.get(key, 0)
    st.session_state[key] = old + 1
    logger.debug(f"[MAP] Bumped {key}: {old} -> {old + 1}")


def bump_camera_epoch() -> None:
    """Increment camera_epoch → remount the Pydeck component so it re-reads initial_view_state.

    This is the ONLY way the camera intentionally re-frames (finish/show-view/reset/3D/search). It
    also gives the fresh component clean click state. Do NOT call it for in-place interactions
    (commit/cancel/undo/start/toggle) — those must keep the user's live pan.
    """
    _bump_epoch("camera_epoch")


def bump_dedup_epoch() -> None:
    """Increment dedup_epoch → make regenerated proposals/markers clickable again.

    Embedded in click ids only (NOT the component key), so bumping it does NOT remount or move the
    camera. Bump it whenever the proposal/marker set is regenerated so that re-clicking the same
    proposal index (or re-toggling the same node) after regeneration counts as a fresh click.
    """
    _bump_epoch("dedup_epoch")


def bump_window_size_epoch() -> None:
    """Increment window_size_epoch → remount the js-eval probe so it re-reads parent.innerHeight.

    The only way to re-measure the browser window (e.g. after a resize); the static key otherwise
    reads it once per session and caches it forever.
    """
    _bump_epoch("window_size_epoch")


def reload_map(*, center: tuple[float, float], zoom: float, pitch: float = MapConfig.VIEWING_PITCH) -> None:
    """Recenter on an explicit (lon,lat) frame IN PLACE, then rerun — no camera_epoch bump.

    The new view flows via ctx.map → the deck's initialViewState, which deck.gl applies to the mounted
    component (bumping the epoch would remount the iframe = the ~0.5s gray-out). 2D↔3D still remounts via
    the pitch-delta path in app.py. Mandatory args so no caller reframes on a stale view.
    """
    ctx = st.session_state.context
    ctx.map.set_view(lon=center[0], lat=center[1], zoom=zoom, pitch=pitch)
    trigger_rerun()


def viewport_map_height() -> int | None:
    """Map height in px that fills the browser window, or None only on first load.

    The JS component reports parent.innerHeight (the real browser window), and only on the render its
    round-trip resolved — every other rerun returns None. We cache the last real value so the map
    never blanks on the constant reruns a stateful app makes. Height is state-independent (the profile
    lives in the right column, not below the map) so it stays constant and never remounts the deck.

    Returns None only before the very first successful read (caller shows a placeholder);
    thereafter the cached viewport height, floored at a minimum.
    """
    epoch = st.session_state.get("window_size_epoch", 0)
    value = streamlit_js_eval(js_expressions="parent.innerHeight", key=f"window_inner_height_{epoch}")
    if isinstance(value, int | float):
        st.session_state.window_height_px = int(value)
    window_height = st.session_state.get("window_height_px")
    if window_height is None:
        return None
    available: int = int(window_height) - ChartConfig.MAP_TOP_OFFSET_PX
    result = max(available, ChartConfig.MAP_MIN_HEIGHT_PX)
    logger.debug(f"[MAP] viewport_map_height: js_eval={value!r} cached_window={window_height} -> height={result}")
    return result
