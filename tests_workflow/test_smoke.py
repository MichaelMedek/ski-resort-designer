"""Smoke tests for module imports and configuration validation.

Quick tests that verify the system is correctly installed and configured.
"""

from tests_workflow.conftest import SMAndCtx


class TestModuleImports:
    """Smoke tests for module imports."""

    def test_core_imports(self) -> None:
        """Core modules can be imported without errors."""
        from skiresort_planner.core.dem_service import DEMService
        from skiresort_planner.core.geo_calculator import GeoCalculator
        from skiresort_planner.core.path_tracer import PathTracer
        from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer

        assert GeoCalculator is not None
        assert TerrainAnalyzer is not None
        assert PathTracer is not None
        assert DEMService is not None

    def test_model_imports(self) -> None:
        """Model modules can be imported without errors."""
        from skiresort_planner.model.lift import Lift
        from skiresort_planner.model.node import Node
        from skiresort_planner.model.path_point import PathPoint
        from skiresort_planner.model.resort_graph import ResortGraph
        from skiresort_planner.model.slope import Slope

        assert PathPoint is not None
        assert Node is not None
        assert Slope is not None
        assert Lift is not None
        assert ResortGraph is not None

    def test_ui_imports(self) -> None:
        """UI modules can be imported without errors."""
        from skiresort_planner.ui.click_detector import ClickDetector
        from skiresort_planner.ui.context import PlannerContext
        from skiresort_planner.ui.state_machine import PlannerStateMachine
        from skiresort_planner.ui.validators import (
            validate_custom_target_downhill,
            validate_lift_goes_uphill,
        )

        assert PlannerStateMachine is not None
        assert PlannerContext is not None
        assert ClickDetector is not None
        assert validate_lift_goes_uphill is not None
        assert validate_custom_target_downhill is not None

    def test_generator_imports(self) -> None:
        """Generator modules can be imported without errors."""
        from skiresort_planner.generators.connection_planners import LeastCostPathPlanner
        from skiresort_planner.generators.path_factory import PathFactory

        assert PathFactory is not None
        assert LeastCostPathPlanner is not None


class TestConfigurationValidation:
    """Tests that configuration constants are valid and consistent."""

    def test_difficulty_thresholds_are_contiguous(self) -> None:
        """Difficulty thresholds cover full range without gaps."""
        from skiresort_planner.constants import SlopeConfig

        thresholds = SlopeConfig.DIFFICULTY_THRESHOLDS

        # Check all difficulties are present
        assert "green" in thresholds
        assert "blue" in thresholds
        assert "red" in thresholds
        assert "black" in thresholds

        # Check thresholds are contiguous
        assert thresholds["green"][1] == thresholds["blue"][0], "Green-Blue boundary"
        assert thresholds["blue"][1] == thresholds["red"][0], "Blue-Red boundary"
        assert thresholds["red"][1] == thresholds["black"][0], "Red-Black boundary"

    def test_difficulty_targets_within_thresholds(self) -> None:
        """Difficulty targets fall within their respective threshold ranges."""
        from skiresort_planner.constants import SlopeConfig

        thresholds = SlopeConfig.DIFFICULTY_THRESHOLDS
        targets = SlopeConfig.DIFFICULTY_TARGETS

        for difficulty in ["green", "blue", "red", "black"]:
            low, high = thresholds[difficulty]
            gentle = targets[difficulty]["gentle"]
            steep = targets[difficulty]["steep"]

            assert low <= gentle < high, f"{difficulty} gentle target should be in range"
            assert low <= steep < high, f"{difficulty} steep target should be in range"
            assert gentle < steep, f"{difficulty} gentle should be less than steep"

    def test_lift_types_are_valid(self) -> None:
        """Lift configuration has all required types."""
        from skiresort_planner.constants import LiftConfig

        expected_types = ["chairlift", "gondola", "surface_lift", "aerial_tram"]

        for lift_type in expected_types:
            assert lift_type in LiftConfig.TYPES, f"{lift_type} should be in TYPES"

    def test_entity_prefixes_are_unique(self) -> None:
        """Entity ID prefixes are unique to prevent ID collisions."""
        from skiresort_planner.constants import EntityPrefixes

        prefixes = [
            EntityPrefixes.NODE,
            EntityPrefixes.SEGMENT,
            EntityPrefixes.SLOPE,
            EntityPrefixes.LIFT,
        ]

        assert len(prefixes) == len(set(prefixes)), "Prefixes should be unique"


class TestStateMachineConfiguration:
    """Tests for state machine setup."""

    def test_state_machine_has_all_states(self, empty_graph) -> None:
        """PlannerStateMachine defines all expected states."""
        from skiresort_planner.ui.state_machine import PlannerStateMachine

        expected_states = [
            "idle_ready",
            "idle_viewing_slope",
            "idle_viewing_lift",
            "slope_starting",
            "slope_building",
            "slope_custom_picking",
            "slope_custom_path",
            "lift_placing",
        ]

        sm, ctx = PlannerStateMachine.create(graph=empty_graph)

        state_values = [s.value for s in sm.states]

        for state_name in expected_states:
            assert state_name in state_values, f"State {state_name} should exist"

    def test_state_machine_starts_in_idle_ready(self, sm_and_ctx: SMAndCtx) -> None:
        """State machine starts in IdleReady state."""
        sm, ctx = sm_and_ctx

        assert sm.current_state_value == "idle_ready", "Should start in idle_ready"
