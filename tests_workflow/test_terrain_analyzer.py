"""Unit tests for TerrainAnalyzer difficulty classification (core/terrain_analyzer.py)."""

from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer


class TestDifficultyClassification:
    """Tests for slope difficulty classification."""

    def test_classify_difficulty_all_thresholds(self) -> None:
        """Difficulty classification at all threshold boundaries.

        Tests boundary values for green/blue/red/black classification.
        """
        # Green: 0-15%
        assert TerrainAnalyzer.classify_difficulty(slope_pct=0.0) == "green"
        assert TerrainAnalyzer.classify_difficulty(slope_pct=5.0) == "green"
        assert TerrainAnalyzer.classify_difficulty(slope_pct=14.9) == "green"

        # Blue: 15-25%
        assert TerrainAnalyzer.classify_difficulty(slope_pct=15.0) == "blue"
        assert TerrainAnalyzer.classify_difficulty(slope_pct=20.0) == "blue"
        assert TerrainAnalyzer.classify_difficulty(slope_pct=24.9) == "blue"

        # Red: 25-40%
        assert TerrainAnalyzer.classify_difficulty(slope_pct=25.0) == "red"
        assert TerrainAnalyzer.classify_difficulty(slope_pct=30.0) == "red"
        assert TerrainAnalyzer.classify_difficulty(slope_pct=39.9) == "red"

        # Black: 40%+
        assert TerrainAnalyzer.classify_difficulty(slope_pct=40.0) == "black"
        assert TerrainAnalyzer.classify_difficulty(slope_pct=60.0) == "black"
        assert TerrainAnalyzer.classify_difficulty(slope_pct=100.0) == "black"
