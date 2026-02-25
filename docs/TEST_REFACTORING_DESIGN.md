# Test Suite Refactoring Design Document

## Executive Summary

This document analyzes the current 204 tests across 8 test files (1,800+ lines of test code) and proposes a streamlined, maintainable architecture that:
- Reduces test count by ~50% while maintaining coverage
- Eliminates redundant initialization tests
- Consolidates related tests into workflow-based integration tests
- Preserves granular unit tests only for core mathematical/logic functions
- **Enforces strict state machine workflow testing based on python-statemachine best practices**

---

## 0. State Machine Testing Principles (MANDATORY)

All workflow tests MUST validate the python-statemachine architecture. This section defines the **strict testing contract** that ensures the state machine implementation follows best practices.

### 0.1 The Four Pillars of State Machine Testing

Every workflow test must verify these four principles:

#### Pillar 1: Single Point of Truth (enter_* hooks)

**Rule**: `on_enter_*` functions are the ONLY place where state-specific guarantees are established.

```python
# CORRECT: Test that enter_idle_viewing_slope ALWAYS shows panel
def test_enter_viewing_slope_guarantees_panel_visible(sm, ctx):
    """Panel visibility is guaranteed by enter hook, not by transition."""
    # Multiple paths to same state should all result in panel visible

    # Path 1: view_slope from idle
    sm.view_slope(slope_id="SL1")
    assert ctx.viewing.panel_visible is True, "enter_idle_viewing_slope must show panel"

    # Reset
    sm.close_slope_panel()

    # Path 2: finish_slope from building
    # ... setup slope building ...
    sm.finish_slope(slope_id="SL2")
    assert ctx.viewing.panel_visible is True, "Same state, same guarantee"

    # Path 3: switch_slope (self-loop)
    sm.switch_slope(slope_id="SL3")
    assert ctx.viewing.panel_visible is True, "Self-loop re-enters state"
```

#### Pillar 2: before_* Hooks Store Data Only

**Rule**: `before_*` hooks MUST only store data (via setters), NEVER control UI visibility.

```python
# CORRECT: Test that before_view_slope sets ID but doesn't show panel
def test_before_hook_stores_data_not_visibility(sm, ctx):
    """Before hooks store data, enter hooks control visibility."""
    # Access before hook directly (white-box test)
    sm.before_view_slope(slope_id="SL1")

    # ID is set
    assert ctx.viewing.slope_id == "SL1", "before_* sets the data"

    # But panel visibility is NOT touched (still False from initial state)
    # Panel visibility is ONLY set by enter_* function
```

#### Pillar 3: Self-Loops Execute on_exit + on_enter

**Rule**: Self-transitions (e.g., `switch_slope`) MUST trigger both exit and enter hooks.

```python
# CORRECT: Test self-loop behavior
def test_self_loop_executes_exit_and_enter(sm, ctx):
    """Self-loops are external transitions (not internal=True)."""
    sm.view_slope(slope_id="SL1")

    # Self-loop to different slope
    sm.switch_slope(slope_id="SL2")

    # Exit cleared old slope_id
    # Enter re-established panel visibility with new ID
    assert ctx.viewing.slope_id == "SL2", "ID updated by before_switch_slope"
    assert ctx.viewing.panel_visible is True, "enter re-showed panel"
```

#### Pillar 4: Guards (cond/unless) Over Manual If-Statements

**Rule**: Conditional transitions use guards, not if-statements in action code.

```python
# CORRECT: Test guard-based transitions
def test_undo_uses_guards_not_if_statements(sm, ctx, graph):
    """Undo destination determined by guards, not manual checks."""
    # Setup: 2 segments committed
    # ...

    # undo_continue guard: unless="undo_leaves_no_segments" (2 > 1, so False)
    sm.send("undo", removed_segment_id="S2", new_endpoint_node_id="N1")
    assert sm.current_state_value == "SlopeBuilding", "Guard kept us in building"

    # Now 1 segment left
    # undo_to_idle guard: cond="undo_leaves_no_segments" (1 <= 1, so True)
    sm.send("undo", removed_segment_id="S1")
    assert sm.current_state_value == "IdleReady", "Guard sent to idle"
```

### 0.2 State Machine Test Checklist (Verify in Each Workflow Test)

| Checklist Item | How to Verify |
|----------------|---------------|
| ✅ enter_* is Single Point of Truth | Test multiple paths to same state → same outcome |
| ✅ before_* only stores data | Verify ID set but panel_visible unchanged before enter |
| ✅ exit_* only cleans up | Verify stale references cleared, NOT visibility |
| ✅ Self-loops run exit+enter | Self-transition refreshes state (e.g., new slope ID) |
| ✅ Guards control flow | Use `cond=`/`unless=` transitions, not if-statements |
| ✅ StreamlitUIListener runs st.rerun() | After transition, rerun is triggered |
| ✅ Context Object Pattern | State machine has NO business logic, only flow control |
| ✅ Orthogonal flags (view_3d) | Flags are NOT separate states |

### 0.3 Anti-Patterns to Test Against

```python
# ANTI-PATTERN 1: Testing transition SETS visibility (wrong!)
def test_wrong_view_slope_sets_panel_visible():  # BAD
    sm.view_slope(slope_id="SL1")
    # WRONG: This tests implementation detail, not contract

# CORRECT: Test that ENTERING the state guarantees visibility
def test_correct_viewing_slope_state_guarantees_panel():  # GOOD
    sm.view_slope(slope_id="SL1")
    assert sm.is_idle_viewing_slope, "In viewing state"
    assert ctx.viewing.panel_visible, "Viewing state guarantees panel"

# ANTI-PATTERN 2: Testing before_* sets visibility (wrong!)
def test_wrong_before_view_slope():  # BAD
    sm.before_view_slope(slope_id="SL1")
    assert ctx.viewing.panel_visible  # WRONG! before_* doesn't do this

# ANTI-PATTERN 3: Not testing self-loops refresh state (missing!)
# MUST test that switch_slope actually updates the viewed slope
```

---

## 1. Current Test Inventory

### 1.1 test_core.py (13 tests, 250 lines)

| Test Class | Test Function | What It Tests | Keep/Delete/Merge |
|------------|---------------|---------------|-------------------|
| `TestGeoCalculator` | `test_haversine_distance_one_degree_latitude` | Haversine formula for latitude | **KEEP** - core math |
| `TestGeoCalculator` | `test_haversine_distance_one_degree_longitude` | Haversine formula for longitude | **MERGE** with above |
| `TestGeoCalculator` | `test_bearing_north` | Initial bearing 0° | **MERGE** into single bearing test |
| `TestGeoCalculator` | `test_bearing_south` | Initial bearing 180° | **MERGE** into single bearing test |
| `TestGeoCalculator` | `test_bearing_east` | Initial bearing 90° | **MERGE** into single bearing test |
| `TestGeoCalculator` | `test_destination_roundtrip` | destination() + haversine consistency | **KEEP** - validates inverse operation |
| `TestTerrainAnalyzer` | `test_classify_difficulty_green` | Green threshold <15% | **MERGE** all difficulty tests |
| `TestTerrainAnalyzer` | `test_classify_difficulty_blue` | Blue threshold 15-25% | **MERGE** all difficulty tests |
| `TestTerrainAnalyzer` | `test_classify_difficulty_red` | Red threshold 25-40% | **MERGE** all difficulty tests |
| `TestTerrainAnalyzer` | `test_classify_difficulty_black` | Black threshold >=40% | **MERGE** all difficulty tests |
| `TestTerrainAnalyzer` | `test_compute_gradient_on_mock_terrain` | Gradient computation | **MERGE** into integration |
| `TestTerrainAnalyzer` | `test_get_orientation_on_diagonal_slope` | Orientation on diagonal | **MERGE** into integration |
| `TestTerrainAnalyzer` | `test_terrain_on_real_dem` | Real DEM integration | **KEEP** - validates real data |
| `TestPathTracerWithMockDEM` | `test_trace_downhill_basic` | Path tracing returns valid path | **KEEP** as integration |
| `TestPathTracerWithMockDEM` | `test_trace_downhill_respects_target_length` | Length constraint | **MERGE** into above |
| `TestPathTracerWithMockDEM` | `test_trace_downhill_side_affects_direction` | Left/right diverge | **MERGE** into above |
| `TestDEMServiceReal` | `test_dem_file_exists` | DEM file present | **DELETE** - infrastructure |
| `TestDEMServiceReal` | `test_load_real_dem_and_sample` | Sample from real DEM | **KEEP** - validates real data |
| `TestPathTracerWithRealDEM` | `test_tracer_terrain_analysis` | PathTracer on real DEM | **MERGE** into real DEM test |

**Summary**: 13 tests → 5 tests (keep 3 unit + 2 integration)

---

### 1.2 test_model.py (36 tests, 761 lines)

| Test Class | Test Function | What It Tests | Keep/Delete/Merge |
|------------|---------------|---------------|-------------------|
| `TestPathPoint` | `test_path_point_creation_and_distance` | PathPoint properties + distance | **KEEP** - core data structure |
| `TestPathPoint` | `test_lat_lon_property` | (lat, lon) tuple order | **DELETE** - trivial getter |
| `TestPathPoint` | `test_lon_lat_property` | (lon, lat) tuple order | **DELETE** - trivial getter |
| `TestNode` | `test_node_distance_to_point` | Node.distance_to() | **MERGE** into PathPoint test |
| `TestNode` | `test_node_lat_lon_property` | Delegation to location | **DELETE** - trivial getter |
| `TestNode` | `test_node_lon_lat_property` | Delegation to location | **DELETE** - trivial getter |
| `TestProposedSlopeSegment` | `test_computed_properties` | drop, length, slope, difficulty | **KEEP** - core computed props |
| `TestResortGraph` | `test_empty_graph` | Empty state | **DELETE** - initialization test |
| `TestResortGraph` | `test_node_creation` | get_or_create_node | **MERGE** into workflow |
| `TestResortGraph` | `test_commit_finish_undo_workflow` | Full commit-finish-undo | **KEEP** - key workflow |
| `TestResortGraph` | `test_add_lift` | Add lift between nodes | **MERGE** into lift workflow |
| `TestSlope` | `test_number_from_id_single_digit` | ID parsing | **MERGE** into single test |
| `TestSlope` | `test_number_from_id_multi_digit` | ID parsing | **MERGE** into single test |
| `TestSlope` | `test_number_property_matches_id` | Number derivation | **MERGE** into single test |
| `TestSlope` | `test_generate_name_format` | Name generation | **MERGE** into single test |
| `TestLift` | `test_number_from_id_single_digit` | ID parsing | **MERGE** into single test |
| `TestLift` | `test_number_from_id_multi_digit` | ID parsing | **MERGE** into single test |
| `TestLift` | `test_number_property_matches_id` | Number derivation | **MERGE** into single test |
| `TestLift` | `test_generate_name_different_types` | Name for all lift types | **MERGE** into single test |
| `TestPylon` | `test_pylon_lat_lon_property` | Coordinate getters | **DELETE** - trivial getter |
| `TestPylon` | `test_pylon_lon_lat_property` | Coordinate getters | **DELETE** - trivial getter |
| `TestResortGraphSerialization` | `test_to_dict_and_from_dict` | Save/load roundtrip | **KEEP** - critical functionality |
| `TestSlopeSegment` | `test_blue_slope_difficulty` | Difficulty classification | **MERGE** into computed props |
| `TestCleanupIsolatedNodes` | `test_removes_isolated_nodes` | Node cleanup | **MERGE** into undo workflow |
| `TestCleanupIsolatedNodes` | `test_no_nodes_removed_when_all_connected` | No false cleanup | **MERGE** into above |
| `TestUndoStackBasics` | `test_empty_undo_stack_raises` | Empty stack error | **KEEP** - error handling |
| `TestUndoStackBasics` | `test_undo_stack_size_limit` | Max stack enforcement | **MERGE** into undo workflow |
| `TestUndoAddSegments` | `test_undo_removes_segment_from_graph` | Segment removal | **MERGE** into undo workflow |
| `TestUndoAddSegments` | `test_undo_multiple_segments_at_once` | Multi-segment undo | **MERGE** into undo workflow |
| `TestUndoAddSegments` | `test_undo_cleans_up_isolated_nodes` | Node cleanup on undo | **MERGE** into undo workflow |
| `TestUndoFinishSlope` | `test_undo_finish_slope_removes_slope_keeps_segments` | Finish undo | **MERGE** into undo workflow |
| `TestUndoFinishSlope` | `test_undo_finish_slope_returns_correct_segment_ids` | Segment IDs preserved | **MERGE** into above |
| `TestUndoDeleteSlope` | `test_undo_delete_slope_restores_slope_and_segments` | Delete undo | **MERGE** into undo workflow |
| `TestUndoLift` | `test_undo_add_lift_removes_lift` | Lift undo | **MERGE** into lift workflow |
| `TestUndoLift` | `test_undo_delete_lift_restores_lift` | Lift restore | **MERGE** into lift workflow |
| `TestUndoStateMachineIntegration` | All 3 tests | SM + undo integration | **MERGE** into workflow tests |
| `TestUndoEdgeCases` | `test_undo_after_undo_works_correctly` | Multiple undos | **MERGE** into undo workflow |
| `TestUndoEdgeCases` | `test_undo_stack_empty_after_all_undos` | Stack exhaustion | **MERGE** into undo workflow |
| `TestUndoEdgeCases` | `test_undo_preserves_other_data` | Isolation | **MERGE** into undo workflow |
| `TestUndoEdgeCases` | `test_node_sharing_preserved_after_partial_undo` | Node sharing | **MERGE** into undo workflow |

**Summary**: 36 tests → 8 tests (5 unit + 3 workflow integration)

---

### 1.3 test_ui.py (53 tests, 1121 lines)

| Test Class | Tests | What It Tests | Keep/Delete/Merge |
|------------|-------|---------------|-------------------|
| `TestPlannerContext` | 4 tests | Context defaults, clears, checks | **DELETE** - initialization tests |
| `TestMapContext` | 4 tests | Coordinate properties | **DELETE** - trivial getters |
| `TestStateMachineInitialization` | 2 tests | Initial state | **DELETE** - initialization tests |
| `TestSlopeModeTransitions` | 6 tests | Slope state transitions | **MERGE** into slope workflow |
| `TestLiftModeTransitions` | 5 tests | Lift state transitions | **MERGE** into lift workflow |
| `TestUndoTransitions` | 2 tests | Undo state changes | **MERGE** into undo workflow |
| `TestInvalidTransitions` | 4 tests | Forbidden transitions | **KEEP** as single guard test |
| `TestTryTransition` | 2 tests | Safe transition method | **DELETE** - API test |
| `TestStateMachineHelpers` | 1 test | get_state_name() | **DELETE** - trivial getter |
| `TestFullWorkflow` | 10 tests | End-to-end workflows | **CONSOLIDATE** into 2 workflows |
| `TestDeferredActionFlags` | 4 tests | Deferred flags | **DELETE** - initialization tests |
| `TestLayerCollection` | 1 test | Layer ordering | **MERGE** into rendering test |
| `TestMapRendererSegmentLayers` | 2 tests | Segment layer structure | **MERGE** into rendering test |
| `TestMapRendererLiftLayers` | 2 tests | Lift layer structure | **MERGE** into rendering test |
| `TestMapRendererNodeLayer` | 2 tests | Node layer structure | **MERGE** into rendering test |
| `TestMapRendererEmptyGraph` | 3 tests | Empty rendering | **MERGE** into rendering test |

**Summary**: 53 tests → 10 tests (2 workflow + 1 guards + 1 rendering + 6 specific)

---

### 1.4 test_generators.py (34 tests, 900 lines)

| Test Class | Tests | What It Tests | Keep/Delete/Merge |
|------------|-------|---------------|-------------------|
| `TestPathFactory` | 3 tests | Fan generation | **MERGE** into path generation test |
| `TestPathFactoryDifficulties` | 1 test | Configuration validation | **MERGE** into config test |
| `TestLeastCostPathPlanner` | 3 tests | Dijkstra path planning | **KEEP** - core algorithm |
| `TestLeastCostPathPlannerSmoothness` | 1 test | Endpoint matching | **MERGE** into above |
| `TestGeneratorsWithRealDEM` | 2 tests | Real DEM integration | **KEEP** - real data validation |
| `TestArePathsSimilar` | 6 tests | Path similarity | **MERGE** into 2 tests |
| `TestDeduplicatePaths` | 4 tests | Deduplication | **MERGE** into 1 test |
| `TestGenerateManualPaths` | 3 tests | Manual path gen | **MERGE** into 1 test |
| `TestCreateStraightLinePath` | 4 tests | Fallback path | **MERGE** into 1 test |
| `TestPathFactoryHypothesis` | 2 tests | Property-based tests | **KEEP** - edge cases |
| `TestGradeConfig` | 2 tests | GradeConfig properties | **DELETE** - trivial getters |

**Summary**: 34 tests → 8 tests

---

### 1.5 test_integration.py (30 tests, 848 lines)

| Test Class | Tests | What It Tests | Keep/Delete/Merge |
|------------|-------|---------------|-------------------|
| `TestImportSmoke` | 6 tests | Module imports | **KEEP** as single smoke test |
| `TestConfigurationValidation` | 6 tests | Config constants | **KEEP** as single validation test |
| `TestClickHandlerPipeline` | 3 tests | Click dispatch | **MERGE** into click handling test |
| `TestStateMachineIntegration` | 3 tests | SM creation | **DELETE** - duplicates ui tests |
| `TestMapRendererLayerData` | 6 tests | Layer structure | **DELETE** - duplicates ui tests |
| `TestProfileChartIntegration` | 1 test | Chart creation | **DELETE** - duplicates chart tests |
| `TestContextInitialization` | 3 tests | Context defaults | **DELETE** - initialization tests |
| `TestResortGraphIntegration` | 3 tests | Graph operations | **DELETE** - duplicates model tests |
| `TestUndoSegmentIntegration` | 5 tests | Undo state | **DELETE** - duplicates model tests |
| `TestViewToggleIntegration` | 5 tests | 3D toggle | **MERGE** into view workflow |
| `TestDeferredToastIntegration` | 2 tests | Toast mechanism | **DELETE** - UI detail |

**Summary**: 30 tests → 4 tests

---

### 1.6 test_click_detector.py (18 tests, 260 lines)

| Test Class | Tests | What It Tests | Keep/Delete/Merge |
|------------|-------|---------------|-------------------|
| `TestTerrainClicks` | 3 tests | Terrain click parsing | **MERGE** into single click test |
| `TestInvisibleTerrainLayerClicks` | 3 tests | Terrain layer clicks | **MERGE** into single click test |
| `TestNodeClicks` | 2 tests | Node click parsing | **MERGE** into single click test |
| `TestSlopeClicks` | 2 tests | Slope/segment parsing | **MERGE** into single click test |
| `TestLiftClicks` | 4 tests | Lift/pylon parsing | **MERGE** into single click test |
| `TestProposalClicks` | 3 tests | Proposal click parsing | **MERGE** into single click test |
| `TestUnknownTypes` | 2 tests | Error handling | **MERGE** into single click test |
| `TestDeduplicationWithObjects` | 2 tests | Deduplication | **MERGE** into single click test |

**Summary**: 18 tests → 2 tests (click parsing + deduplication)

---

### 1.7 test_profile_chart.py (13 tests, 260 lines)

| Test Class | Tests | What It Tests | Keep/Delete/Merge |
|------------|-------|---------------|-------------------|
| `TestProposalRendering` | 5 tests | Proposal charts | **MERGE** into 1 chart test |
| `TestSegmentRendering` | 3 tests | Segment charts | **MERGE** into 1 chart test |
| `TestSlopeRendering` | 2 tests | Slope charts | **MERGE** into 1 chart test |
| `TestChartConfiguration` | 2 tests | Dimensions | **DELETE** - trivial config |
| `TestDifficultyColors` | 1 test | Color assignment | **DELETE** - config test |
| `TestInputValidation` | 2 tests | Error handling | **KEEP** - error cases |

**Summary**: 13 tests → 3 tests

---

### 1.8 test_validators.py (10 tests, 130 lines)

| Test Class | Tests | What It Tests | Keep/Delete/Merge |
|------------|-------|---------------|-------------------|
| `TestValidateLiftGoesUphill` | 3 tests | Lift validation | **MERGE** into 1 validator test |
| `TestValidateLiftDifferentNodes` | 2 tests | Same-node check | **MERGE** into 1 validator test |
| `TestValidateCustomTargetDownhill` | 4 tests | Downhill validation | **MERGE** into 1 validator test |
| `TestValidateCustomTargetDistance` | 3 tests | Distance validation | **MERGE** into 1 validator test |

**Summary**: 10 tests → 2 tests

---

## 2. Redundancy Analysis

### 2.1 Heavily Overlapping Test Areas

1. **State Machine Transitions** (duplicated in test_ui.py and test_integration.py)
   - Both test the same state machine transitions
   - test_integration.py `TestStateMachineIntegration` duplicates test_ui.py tests

2. **Undo Operations** (duplicated in test_model.py and test_integration.py)
   - `TestUndoSegmentIntegration` duplicates `TestUndoAddSegments`
   - Similar assertions about segment removal and node cleanup

3. **Layer Structure** (duplicated in test_ui.py and test_integration.py)
   - `TestMapRendererLayerData` duplicates `TestMapRendererSegmentLayers`
   - Same assertions about layer data structure

4. **Context Initialization** (in test_ui.py and test_integration.py)
   - Both test that PlannerContext initializes with defaults
   - No value in testing constructor defaults

### 2.2 Low-Value Tests (Delete Candidates)

1. **Trivial Getter Tests**: `test_lat_lon_property`, `test_lon_lat_property` for every model
2. **Empty/Default State Tests**: `test_empty_graph`, `test_context_defaults`
3. **API Tests**: `test_try_transition_success`, `test_get_state_name`
4. **File Existence**: `test_dem_file_exists`

---

## 3. Proposed New Test Architecture

### 3.1 Directory Structure

```
tests_workflow/
├── conftest.py                    # Shared MockDEMService + fixtures
├── test_core_math.py              # Unit tests: GeoCalculator, difficulty thresholds
├── test_model_computations.py     # Unit tests: PathPoint distance, segment properties
├── test_graph_operations.py       # Unit tests: commit, finish, id parsing
├── test_click_parsing.py          # Unit tests: ClickDetector parsing logic
├── test_validators.py             # Unit tests: all validation functions
├── test_workflow_slope.py         # Integration: complete slope building workflow
├── test_workflow_lift.py          # Integration: complete lift placement workflow
├── test_workflow_undo.py          # Integration: comprehensive undo scenarios
├── test_workflow_custom_path.py   # Integration: custom path connection workflow
├── test_rendering.py              # Integration: map + chart rendering
├── test_serialization.py          # Integration: save/load roundtrip
├── test_real_dem.py               # Integration: tests with real DEM data
└── test_smoke.py                  # Smoke: imports + config validation
```

### 3.2 Test Count Comparison

| Category | Current | Proposed | Reduction |
|----------|---------|----------|-----------|
| Core Math | 13 | 3 | -77% |
| Model | 36 | 6 | -83% |
| UI/State Machine | 53 | 10 | -81% |
| Generators | 34 | 8 | -76% |
| Integration | 30 | 4 | -87% |
| Click Detector | 18 | 2 | -89% |
| Profile Chart | 13 | 3 | -77% |
| Validators | 10 | 2 | -80% |
| **Total** | **204** | **38** | **-81%** |

---

## 4. Detailed Test Specifications

### 4.1 test_core_math.py (3 tests)

```python
"""Unit tests for core mathematical functions.

Tests GeoCalculator geodesic functions and TerrainAnalyzer difficulty classification.
These are pure functions with deterministic outputs - perfect for unit testing.
"""

class TestGeoCalculator:
    def test_haversine_and_bearing_cardinal_directions(self):
        """Haversine distance and bearing for cardinal directions.

        Tests:
        - 1° latitude = ~111km
        - 1° longitude at 46°N = ~77km
        - Bearing north = 0°, south = 180°, east = 90°
        - destination() roundtrip consistent with haversine()
        """
        # All assertions in one test with intermediate asserts
        dist_lat = GeoCalculator.haversine_distance_m(lat1=46.0, lon1=10.0, lat2=47.0, lon2=10.0)
        assert 110_000 < dist_lat < 112_000, "1° lat should be ~111km"

        bearing_north = GeoCalculator.initial_bearing_deg(lon1=10.0, lat1=46.0, lon2=10.0, lat2=47.0)
        assert bearing_north < 1 or bearing_north > 359, "North should be ~0°"

        bearing_south = GeoCalculator.initial_bearing_deg(lon1=10.0, lat1=47.0, lon2=10.0, lat2=46.0)
        assert 179 < bearing_south < 181, "South should be ~180°"

        # Roundtrip validation
        lon_end, lat_end = GeoCalculator.destination(lon=10.0, lat=46.0, bearing_deg=45.0, distance_m=1000.0)
        dist_check = GeoCalculator.haversine_distance_m(lat1=46.0, lon1=10.0, lat2=lat_end, lon2=lon_end)
        assert abs(dist_check - 1000) < 10, "Roundtrip should be consistent"


class TestDifficultyClassification:
    def test_classify_difficulty_thresholds(self):
        """Difficulty classification at all threshold boundaries.

        Tests boundary values for green/blue/red/black classification.
        """
        # Test each threshold boundary in one test
        assert TerrainAnalyzer.classify_difficulty(5.0) == "green"
        assert TerrainAnalyzer.classify_difficulty(14.9) == "green"
        assert TerrainAnalyzer.classify_difficulty(15.0) == "blue"
        assert TerrainAnalyzer.classify_difficulty(24.9) == "blue"
        assert TerrainAnalyzer.classify_difficulty(25.0) == "red"
        assert TerrainAnalyzer.classify_difficulty(39.9) == "red"
        assert TerrainAnalyzer.classify_difficulty(40.0) == "black"
        assert TerrainAnalyzer.classify_difficulty(60.0) == "black"


class TestPathTracerOnMockTerrain:
    def test_trace_downhill_produces_valid_diverging_paths(self, mock_dem_blue_slope):
        """PathTracer generates valid downhill paths with left/right divergence.

        Tests:
        - trace_downhill returns non-None on valid terrain
        - Path goes downhill (end elevation < start)
        - Left/right paths diverge significantly
        - Path length approximates target
        """
        tracer = PathTracer(dem=mock_dem_blue_slope)

        left = tracer.trace_downhill(start_lon=0.0, start_lat=0.0, target_slope_pct=15.0, side="left", target_length_m=300)
        right = tracer.trace_downhill(start_lon=0.0, start_lat=0.0, target_slope_pct=15.0, side="right", target_length_m=300)

        # Both paths exist
        assert left is not None and right is not None

        # Both go downhill
        assert left.points[-1].elevation < left.points[0].elevation
        assert right.points[-1].elevation < right.points[0].elevation

        # They diverge
        end_dist = GeoCalculator.haversine_distance_m(
            lat1=left.points[-1].lat, lon1=left.points[-1].lon,
            lat2=right.points[-1].lat, lon2=right.points[-1].lon
        )
        assert end_dist > 50, "Left/right should diverge"

        # Approximate target length
        assert 0.8 * 300 < left.length_m < 1.2 * 300
```

### 4.2 test_workflow_slope.py (State Machine Workflow Test)

```python
"""Integration test for complete slope building workflow.

Tests the full slope lifecycle with STRICT state machine contract validation.
Verifies the Four Pillars (see Section 0) at each workflow step.
"""

class TestSlopeBuildingWorkflow:
    def test_complete_slope_workflow_validates_state_machine_contract(self, workflow_complete_setup):
        """Build slope while validating state machine architecture.

        This test verifies the STATE MACHINE CONTRACT at each step:
        - Pillar 1: enter_* is Single Point of Truth for state guarantees
        - Pillar 2: before_* hooks only store data
        - Pillar 3: Self-loops (switch_slope) run exit+enter
        - Pillar 4: Guards control conditional transitions

        Workflow: start → generate → commit → add more → finish → switch view
        """
        sm, ctx, graph, factory, dem = workflow_complete_setup

        # === Phase 1: Start Slope (IdleReady → SlopeStarting) ===
        start_elev = dem.get_elevation_or_raise(lon=0.0, lat=0.0)

        # BEFORE: Panel should be hidden (IdleReady)
        assert ctx.viewing.panel_visible is False, "IdleReady: no panel"

        sm.start_slope(lon=0.0, lat=0.0, elevation=start_elev, node_id=None)

        # VERIFY Pillar 1: enter_slope_starting guarantees panel hidden
        assert sm.current_state_value == "SlopeStarting"
        assert ctx.viewing.panel_visible is False, "enter_slope_starting hides panel"
        assert ctx.building.name == "Slope 1", "before_start_slope set name"

        # === Phase 2: Commit First Path (SlopeStarting → SlopeBuilding) ===
        proposals = list(factory.generate_fan(...))
        end_nodes = graph.commit_paths(paths=[proposals[0]])
        seg1_id = list(graph.segments.keys())[0]

        sm.commit_path(segment_id=seg1_id, endpoint_node_id=end_nodes[0])

        # VERIFY: commit_first_path leads to SlopeBuilding
        assert sm.current_state_value == "SlopeBuilding"
        assert ctx.viewing.panel_visible is False, "Building states: panel hidden"

        # === Phase 3: Finish Slope (SlopeBuilding → IdleViewingSlope) ===
        slope = graph.finish_slope(segment_ids=ctx.building.segments)

        # VERIFY Pillar 2: before_finish_slope stores ID only
        sm.before_finish_slope(slope_id=slope.id)
        assert ctx.viewing.slope_id == slope.id, "before_* stored slope_id"
        # Panel NOT visible yet (before_* doesn't touch visibility!)

        # Trigger actual transition
        sm.finish_slope(slope_id=slope.id)

        # VERIFY Pillar 1: enter_idle_viewing_slope guarantees panel visible
        assert sm.current_state_value == "IdleViewingSlope"
        assert ctx.viewing.panel_visible is True, "enter_* made panel visible"
        assert ctx.viewing.slope_id == slope.id

        # === Phase 4: Switch to Different Slope (Self-Loop Test) ===
        slope2 = create_another_slope(graph)  # Helper to create 2nd slope

        # VERIFY Pillar 3: Self-loop runs exit+enter
        sm.switch_slope(slope_id=slope2.id)

        assert sm.current_state_value == "IdleViewingSlope", "Same state"
        assert ctx.viewing.slope_id == slope2.id, "exit cleared old, before set new"
        assert ctx.viewing.panel_visible is True, "enter re-showed panel"

        # === Phase 5: Close Panel (IdleViewingSlope → IdleReady) ===
        sm.close_slope_panel()

        # VERIFY: enter_idle_ready clears everything
        assert sm.current_state_value == "IdleReady"
        assert ctx.viewing.panel_visible is False, "enter_idle_ready hides panel"
        assert ctx.viewing.slope_id is None, "All viewing state cleared"


class TestSelfLoopBehavior:
    """Dedicated tests for self-loop transitions (Pillar 3)."""

    def test_switch_slope_refreshes_panel_with_new_slope(self, sm, ctx, graph):
        """Self-loop switch_slope must trigger exit+enter to refresh state."""
        # Setup: viewing slope SL1
        sm.view_slope(slope_id="SL1")
        assert ctx.viewing.slope_id == "SL1"

        # Self-loop to SL2
        sm.switch_slope(slope_id="SL2")

        # exit_idle_viewing_slope cleared slope_id
        # before_switch_slope set new slope_id
        # enter_idle_viewing_slope called show_panel()
        assert ctx.viewing.slope_id == "SL2", "New slope active"
        assert ctx.viewing.panel_visible is True, "Panel still visible"

    def test_switch_lift_refreshes_panel_with_new_lift(self, sm, ctx, graph):
        """Self-loop switch_lift must trigger exit+enter to refresh state."""
        sm.view_lift(lift_id="L1")
        sm.switch_lift(lift_id="L2")

        assert ctx.viewing.lift_id == "L2"
        assert ctx.viewing.panel_visible is True


class TestGuardedTransitions:
    """Dedicated tests for guard-based flow control (Pillar 4)."""

    def test_undo_to_idle_when_last_segment(self, sm, ctx, graph):
        """Guard undo_leaves_no_segments sends to IdleReady."""
        # Setup: 1 segment committed in SlopeBuilding
        setup_building_with_one_segment(sm, ctx, graph)
        assert len(ctx.building.segments) == 1

        # undo_to_idle has cond="undo_leaves_no_segments" (1 <= 1 = True)
        sm.send("undo_segment", removed_segment_id=ctx.building.segments[0])

        assert sm.current_state_value == "IdleReady", "Guard sent to idle"

    def test_undo_continue_when_multiple_segments(self, sm, ctx, graph):
        """Guard unless='undo_leaves_no_segments' stays in SlopeBuilding."""
        # Setup: 2 segments committed in SlopeBuilding
        setup_building_with_two_segments(sm, ctx, graph)
        assert len(ctx.building.segments) == 2

        # undo_continue has unless="undo_leaves_no_segments" (2 > 1, so True)
        sm.undo_continue(removed_segment_id=ctx.building.segments[-1], new_endpoint_node_id="N1")

        assert sm.current_state_value == "SlopeBuilding", "Guard kept in building"
        assert len(ctx.building.segments) == 1
```

### 4.3 test_workflow_undo.py (1 comprehensive test)

```python
"""Integration test for comprehensive undo operations.

Tests undo across all action types with intermediate assertions.
"""

class TestUndoWorkflow:
    def test_comprehensive_undo_scenarios(self, mock_dem_blue_slope):
        """Test undo for all action types in realistic sequence.

        Covers:
        - Undo add segments (single and multiple)
        - Undo finish slope (keeps segments)
        - Undo delete slope (restores)
        - Undo add lift
        - Undo delete lift (restores)
        - Multiple consecutive undos
        - Node cleanup on undo
        - Stack size limit enforcement
        """
        graph = ResortGraph()
        sm, ctx = PlannerStateMachine.create(graph=graph)

        # === Scenario 1: Undo Single Segment ===
        proposal1 = make_proposal(index=0)
        graph.commit_paths(paths=[proposal1])
        seg1_id = list(graph.segments.keys())[0]

        assert len(graph.segments) == 1
        assert len(graph.nodes) == 2

        graph.undo_last()

        assert len(graph.segments) == 0
        assert len(graph.nodes) == 0  # Cleanup isolated

        # === Scenario 2: Undo Keeps Other Data ===
        graph.commit_paths(paths=[make_proposal(index=0)])
        graph.commit_paths(paths=[make_proposal(index=1)])

        assert len(graph.segments) == 2

        graph.undo_last()  # Remove second

        assert len(graph.segments) == 1
        first_seg = list(graph.segments.keys())[0]
        assert "S1" in first_seg or first_seg.startswith("S")

        # === Scenario 3: Undo Finish Slope ===
        seg_id = list(graph.segments.keys())[0]
        slope = graph.finish_slope(segment_ids=[seg_id])

        assert len(graph.slopes) == 1

        undone = graph.undo_last()

        assert isinstance(undone, FinishSlopeAction)
        assert len(graph.slopes) == 0
        assert len(graph.segments) == 1  # Segment preserved

        # === Scenario 4: Undo Delete Restores ===
        slope = graph.finish_slope(segment_ids=[seg_id])
        slope_name = slope.name
        graph.delete_slope(slope_id=slope.id)

        assert len(graph.slopes) == 0

        graph.undo_last()

        assert len(graph.slopes) == 1
        assert graph.slopes[slope.id].name == slope_name

        # === Scenario 5: Empty Stack Raises ===
        graph.undo_stack.clear()

        with pytest.raises(RuntimeError, match="empty"):
            graph.undo_last()
```

### 4.4 test_click_parsing.py (2 tests)

```python
"""Unit tests for ClickDetector parsing logic.

Tests all click type parsing in consolidated tests.
"""

class TestClickDetectorParsing:
    def test_parse_all_click_types(self, detector):
        """Parse all supported click types and validate ClickInfo fields.

        Tests:
        - Terrain click from coordinate
        - Terrain click from invisible layer
        - Node click with ID extraction
        - Slope/segment click
        - Lift/pylon click with metadata
        - Proposal endpoint/body
        - Unknown type returns None
        """
        # Terrain from coordinate
        result = detector.detect(clicked_object=None, clicked_coordinate=[10.27, 46.97])
        assert result.click_type == MapClickType.TERRAIN
        assert result.lon == 10.27

        detector.dedup.clear()

        # Node
        result = detector.detect(clicked_object={"type": "node", "id": "N1"}, clicked_coordinate=None)
        assert result.marker_type == MarkerType.NODE
        assert result.node_id == "N1"

        detector.dedup.clear()

        # Pylon with metadata
        result = detector.detect(clicked_object={"type": "pylon", "lift_id": "L1", "pylon_index": 2}, clicked_coordinate=None)
        assert result.marker_type == MarkerType.PYLON
        assert result.lift_id == "L1"
        assert result.pylon_index == 2

        # Unknown type
        result = detector.detect(clicked_object={"type": "unknown"}, clicked_coordinate=None)
        assert result is None

        # Missing required field
        result = detector.detect(clicked_object={"type": "node"}, clicked_coordinate=None)  # no id
        assert result is None

    def test_click_deduplication_rejects_duplicates(self, detector):
        """Same click is rejected, different clicks accepted."""
        obj1 = {"type": "node", "id": "N1"}

        result1 = detector.detect(clicked_object=obj1, clicked_coordinate=None)
        assert result1 is not None

        result2 = detector.detect(clicked_object=obj1, clicked_coordinate=None)
        assert result2 is None  # Duplicate rejected

        result3 = detector.detect(clicked_object={"type": "node", "id": "N2"}, clicked_coordinate=None)
        assert result3 is not None  # Different object accepted
```

---

## 5. Implementation Guidelines

### 5.1 Test Naming Convention

```python
def test_<what>_<when/how>():
    """<One-liner description>.

    Tests:
    - <Specific assertion 1>
    - <Specific assertion 2>
    ...
    """
```

### 5.2 Assertion Strategy

- Use `assert ... , "explanation"` for every assertion
- Test intermediate state, not just final outcome
- Each test should teach the reader about the system

### 5.3 Fixture Philosophy

- Keep `conftest.py` minimal: only MockDEMService and path fixtures
- Each test module can have its own `@pytest.fixture` for module-specific needs
- Avoid fixture chains deeper than 2 levels

### 5.4 What NOT to Test

1. **Trivial getters**: `.lat_lon`, `.lon_lat`, `.name`
2. **Default values**: Context/model initialization
3. **External library behavior**: Pydeck layer rendering, Plotly figure creation
4. **Error messages**: Exact error text (test exception type only)

### 5.5 State Machine Testing Guidelines (MANDATORY)

Every workflow test that involves the state machine MUST follow these rules:

#### 5.5.1 Test State Outcomes, Not Transitions

```python
# WRONG: Testing the transition itself
def test_view_slope_transition():
    sm.view_slope(slope_id="SL1")
    # What are we even testing here?

# CORRECT: Testing the state guarantee
def test_viewing_slope_state_shows_panel():
    sm.view_slope(slope_id="SL1")
    assert sm.current_state_value == "IdleViewingSlope", "Transition completed"
    assert ctx.viewing.panel_visible is True, "enter_* guarantees visibility"
```

#### 5.5.2 Verify Hook Responsibilities

```python
# Test that before_* only stores data
def test_before_hooks_dont_control_visibility():
    # Call before_* hook directly
    sm.before_view_slope(slope_id="SL1")

    # Data is set
    assert ctx.viewing.slope_id == "SL1"

    # But panel is NOT visible (that's enter_*'s job)
    # Note: This is a white-box test to verify architecture
```

#### 5.5.3 Test Self-Loop Refresh Behavior

```python
# Self-loops MUST run exit + enter (not internal=True)
def test_switch_slope_is_external_self_loop():
    sm.view_slope(slope_id="SL1")
    old_id = ctx.viewing.slope_id

    sm.switch_slope(slope_id="SL2")

    # exit_* cleared old reference
    # before_* set new ID
    # enter_* re-established panel visibility
    assert ctx.viewing.slope_id == "SL2" != old_id
```

#### 5.5.4 Test Guards, Not If-Statements

```python
# Guards determine transition destination
def test_undo_guard_controls_destination():
    # With 2 segments: undo_continue (stays in SlopeBuilding)
    setup_two_segments(...)
    sm.undo_continue(...)
    assert sm.current_state_value == "SlopeBuilding"

    # With 1 segment: undo_to_idle (goes to IdleReady)
    sm.undo_to_idle(...)
    assert sm.current_state_value == "IdleReady"
```

#### 5.5.5 State Machine Test Smell Checklist

| Smell | Fix |
|-------|-----|
| Testing `panel_visible` changes in before_* | Move to enter_* test |
| Not testing multiple paths to same state | Add coverage for all transitions |
| Missing self-loop test | Add switch_slope/switch_lift tests |
| Manual if-statement for conditional transition | Use guards (cond=/unless=) |
| Testing internal implementation details | Test externally observable state |
3. **External library behavior**: Pydeck layer rendering, Plotly figure creation
4. **Error messages**: Exact error text (test exception type only)

---

## 6. Migration Plan

### Phase 1: Create tests_v2/ directory
1. Create new structure
2. Copy conftest.py with minimal fixtures

### Phase 2: Write core math tests
1. `test_core_math.py` - GeoCalculator, thresholds
2. Run both old and new in parallel

### Phase 3: Write workflow integration tests
1. `test_workflow_slope.py`
2. `test_workflow_lift.py`
3. `test_workflow_undo.py`

### Phase 4: Write remaining unit tests
1. `test_model_computations.py`
2. `test_click_parsing.py`
3. `test_validators.py`

### Phase 5: Verify coverage parity
1. Run coverage on both test suites
2. Identify any gaps
3. Add targeted tests for gaps

### Phase 6: Switch over
1. Remove old tests/ directory
2. Rename tests_v2/ to tests/

---

## 7. Expected Benefits

1. **Maintainability**: 81% fewer tests to maintain
2. **Readability**: Each test teaches about the system
3. **Speed**: Fewer tests = faster CI
4. **Confidence**: Workflow tests catch integration issues
5. **Onboarding**: New developers understand system from tests

---

## 8. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Coverage drop | Run coverage comparison before deletion |
| Missing edge case | Keep Hypothesis tests for property-based edge case coverage |
| Real DEM regression | Keep dedicated real_dem test file |
| Breaking change unnoticed | Workflow tests cover realistic scenarios |
