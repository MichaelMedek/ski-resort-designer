"""Smoke tests for module imports and configuration validation.

Quick tests that verify the system is correctly installed and configured.
"""

import pytest

from tests_workflow.conftest import SMAndCtx


# =============================================================================
# MODULE IMPORT TESTS
# =============================================================================


class TestModuleImports:
    """Parametrized smoke tests for module imports."""

    @pytest.mark.parametrize(
        "module_path,class_name",
        [
            # Core modules
            pytest.param("skiresort_planner.core.dem_service", "DEMService", id="core_dem"),
            pytest.param("skiresort_planner.core.geo_calculator", "GeoCalculator", id="core_geo"),
            pytest.param("skiresort_planner.core.path_tracer", "PathTracer", id="core_tracer"),
            pytest.param("skiresort_planner.core.terrain_analyzer", "TerrainAnalyzer", id="core_terrain"),
            # Model modules
            pytest.param("skiresort_planner.model.lift", "Lift", id="model_lift"),
            pytest.param("skiresort_planner.model.node", "Node", id="model_node"),
            pytest.param("skiresort_planner.model.path_point", "PathPoint", id="model_pathpoint"),
            pytest.param("skiresort_planner.model.resort_graph", "ResortGraph", id="model_graph"),
            pytest.param("skiresort_planner.model.slope", "Slope", id="model_slope"),
            # UI modules
            pytest.param("skiresort_planner.ui.click_detector", "ClickDetector", id="ui_detector"),
            pytest.param("skiresort_planner.ui.context", "PlannerContext", id="ui_context"),
            pytest.param("skiresort_planner.ui.state_machine", "PlannerStateMachine", id="ui_statemachine"),
            # Generator modules
            pytest.param("skiresort_planner.generators.connection_planners", "LeastCostPathPlanner", id="gen_planner"),
            pytest.param("skiresort_planner.generators.path_factory", "PathFactory", id="gen_factory"),
        ],
    )
    def test_module_import(self, module_path: str, class_name: str) -> None:
        """Module can be imported without errors."""
        import importlib

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        assert cls is not None


# =============================================================================
# CONFIGURATION VALIDATION TESTS
# =============================================================================


class TestConfigurationValidation:
    """Tests that configuration constants are valid and consistent."""

    def test_difficulty_thresholds_are_contiguous(self) -> None:
        """Difficulty thresholds cover full range without gaps."""
        from skiresort_planner.constants import SlopeConfig

        thresholds = SlopeConfig.DIFFICULTY_THRESHOLDS

        assert "green" in thresholds
        assert "blue" in thresholds
        assert "red" in thresholds
        assert "black" in thresholds

        assert thresholds["green"][1] == thresholds["blue"][0]
        assert thresholds["blue"][1] == thresholds["red"][0]
        assert thresholds["red"][1] == thresholds["black"][0]

    @pytest.mark.parametrize("difficulty", ["green", "blue", "red", "black"])
    def test_difficulty_target_within_threshold(self, difficulty: str) -> None:
        """Difficulty target falls within its threshold range."""
        from skiresort_planner.constants import SlopeConfig

        low, high = SlopeConfig.DIFFICULTY_THRESHOLDS[difficulty]
        gentle = SlopeConfig.DIFFICULTY_TARGETS[difficulty]["gentle"]
        steep = SlopeConfig.DIFFICULTY_TARGETS[difficulty]["steep"]

        assert low <= gentle < high
        assert low <= steep < high
        assert gentle < steep

    @pytest.mark.parametrize("lift_type", ["chairlift", "gondola", "surface_lift", "aerial_tram"])
    def test_lift_type_exists(self, lift_type: str) -> None:
        """Lift configuration has required type."""
        from skiresort_planner.constants import LiftConfig

        assert lift_type in LiftConfig.TYPES

    def test_entity_prefixes_are_unique(self) -> None:
        """Entity ID prefixes are unique to prevent ID collisions."""
        from skiresort_planner.constants import EntityPrefixes

        prefixes = [EntityPrefixes.NODE, EntityPrefixes.SEGMENT, EntityPrefixes.SLOPE, EntityPrefixes.LIFT]
        assert len(prefixes) == len(set(prefixes))


# =============================================================================
# STATE MACHINE TESTS
# =============================================================================


class TestStateMachineConfiguration:
    """Tests for state machine setup."""

    @pytest.mark.parametrize(
        "state_name",
        [
            "idle_ready",
            "idle_viewing_slope",
            "idle_viewing_lift",
            "slope_starting",
            "slope_building",
            "slope_custom_picking",
            "slope_custom_path",
            "lift_placing",
        ],
    )
    def test_state_machine_has_state(self, empty_graph, state_name: str) -> None:
        """PlannerStateMachine defines expected state."""
        from skiresort_planner.ui.state_machine import PlannerStateMachine

        sm, ctx = PlannerStateMachine.create(graph=empty_graph)
        state_values = [s.value for s in sm.states]
        assert state_name in state_values

    def test_state_machine_starts_in_idle_ready(self, sm_and_ctx: SMAndCtx) -> None:
        """State machine starts in IdleReady state."""
        sm, ctx = sm_and_ctx
        assert sm.current_state_value == "idle_ready"
