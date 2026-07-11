# Test suite conventions

Tests **mirror the production module structure**: each production module with
unique logic has a matching `test_<module>.py` holding its atomic unit tests.
Cross-cutting concerns get their own dedicated files.

## Layout

| Kind | Files |
|------|-------|
| **Atomic (one per prod module)** | `test_geo_calculator`, `test_terrain_analyzer`, `test_path_tracer`, `test_path_point`, `test_node`, `test_segment_path`, `test_road`, `test_proposed_path`, `test_resort_graph`, `test_serialization`, `test_click_info`, `test_path_factory`, `test_connection_planners`, `test_backup_store`, `test_state_machine`, `test_context`, `test_actions`, `test_click_handlers`, `test_click_detector`, `test_validators`, `test_center_map`, `test_bottom_chart`, `test_left_panel`, `test_right_panel`, `test_pydeck_click_handler`, `test_app` |
| **Cross-cutting workflows** (state-machine end-to-end) | `test_workflow_slope`, `test_workflow_lift`, `test_workflow_road`, `test_workflow_custom_path` |
| **Cross-cutting other** | `test_smoke` (imports + `constants.py` config), `test_apptest_e2e` (real Streamlit AppTest), `test_real_dem` (integration on real EuroDEM) |

## Rules

- **One home per behavior.** Each entity's graph-level add/delete/**undo** is asserted only in `test_resort_graph.py`; the `ui/actions.undo_last_action` **dispatch/routing** is asserted only in `test_actions.py`.
- **Workflows stay thin.** `test_workflow_*.py` drive the state machine end-to-end; atomic model/graph/render assertions belong in the mirror module, not the workflow file.
- **No `# pragma: no cover`.** UI glue is tested through the shared `fake_st` harness (see below), not excluded.
- **No duplicate class names within a file.**

## The `fake_st` harness (conftest.py)

`fake_st` installs a no-op Streamlit into every `skiresort_planner.ui` module plus `app`, so render code runs without a browser. `fake_st.button(key=...)` returns `True` only for keys in `fake_st.clicked_keys`, so a test can fire one specific button and assert the real state change. `@st.dialog` bodies are decorated with real Streamlit at import, so their action logic is extracted into plain helpers (e.g. `_perform_reset_resort`, `_request_pending_undo`) that are unit-tested directly.
