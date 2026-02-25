"""Tests for ProfileChart - elevation profile rendering without Streamlit.

These tests verify that ProfileChart correctly creates Plotly figures
without needing Streamlit. Tests cover:
- Proposal rendering
- Segment rendering
- Slope rendering (combined segments)
- Error handling for invalid inputs
- Chart configuration
"""

import pytest
import plotly.graph_objects as go

from skiresort_planner.constants import ChartConfig, MapConfig
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.proposed_path import ProposedSlopeSegment
from skiresort_planner.model.resort_graph import ResortGraph
from skiresort_planner.model.slope_segment import SlopeSegment
from skiresort_planner.ui.bottom_chart import ProfileChart


@pytest.fixture
def chart() -> ProfileChart:
    """Standard chart for testing."""
    return ProfileChart(width=800, height=400)


def make_proposal(index: int = 0, base_elev: float = 2500, difficulty: str = "blue") -> ProposedSlopeSegment:
    """Helper to create ProposedSlopeSegment with correct parameters."""
    M = MapConfig.METERS_PER_DEGREE_EQUATOR
    return ProposedSlopeSegment(
        points=[
            PathPoint(lon=0.0 + index * 0.01, lat=0.0, elevation=base_elev),
            PathPoint(lon=0.0 + index * 0.01 + 0.005, lat=-250 / M, elevation=base_elev - 50),
            PathPoint(lon=0.0 + index * 0.01 + 0.01, lat=-500 / M, elevation=base_elev - 100),
        ],
        target_slope_pct=20.0,
        target_difficulty=difficulty,
    )


def make_segment(seg_id: str = "S1", name: str = "Segment 1", base_elev: float = 2500) -> SlopeSegment:
    """Helper to create SlopeSegment with correct parameters."""
    M = MapConfig.METERS_PER_DEGREE_EQUATOR
    return SlopeSegment(
        id=seg_id,
        name=name,
        points=[
            PathPoint(lon=0.0, lat=0.0, elevation=base_elev),
            PathPoint(lon=0.005, lat=-250 / M, elevation=base_elev - 50),
            PathPoint(lon=0.01, lat=-500 / M, elevation=base_elev - 100),
        ],
        start_node_id="N1",
        end_node_id="N2",
    )


@pytest.fixture
def sample_proposal() -> ProposedSlopeSegment:
    """Sample proposal with blue difficulty."""
    return make_proposal(difficulty="blue")


@pytest.fixture
def sample_segment() -> SlopeSegment:
    """Sample committed segment."""
    return make_segment()


class TestProposalRendering:
    """Tests for render_proposal method."""

    def test_render_proposal_returns_figure(self, chart: ProfileChart, sample_proposal: ProposedSlopeSegment) -> None:
        """render_proposal returns a Plotly Figure."""
        fig = chart.render_proposal(
            proposal=sample_proposal,
            proposed_segment_title="Test Proposal",
        )
        assert isinstance(fig, go.Figure)

    def test_render_proposal_with_empty_points_raises(self, chart: ProfileChart) -> None:
        """render_proposal raises for proposal without points."""
        empty_proposal = ProposedSlopeSegment(
            points=[],
            target_slope_pct=10.0,
            target_difficulty="blue",
        )
        with pytest.raises(ValueError, match="points"):
            chart.render_proposal(
                proposal=empty_proposal,
                proposed_segment_title="Empty",
            )

    def test_render_proposal_has_traces(self, chart: ProfileChart, sample_proposal: ProposedSlopeSegment) -> None:
        """render_proposal creates traces for elevation profile."""
        fig = chart.render_proposal(
            proposal=sample_proposal,
            proposed_segment_title="Test",
        )
        # Should have at least one trace (elevation line)
        assert len(fig.data) >= 1

    def test_render_proposal_uses_correct_dimensions(self, sample_proposal: ProposedSlopeSegment) -> None:
        """render_proposal respects width/height settings."""
        custom_chart = ProfileChart(width=1000, height=500)
        fig = custom_chart.render_proposal(
            proposal=sample_proposal,
            proposed_segment_title="Test",
        )
        assert fig.layout.width == 1000
        assert fig.layout.height == 500

    def test_render_proposal_title(self, chart: ProfileChart, sample_proposal: ProposedSlopeSegment) -> None:
        """render_proposal sets the title correctly."""
        title = "My Custom Title"
        fig = chart.render_proposal(
            proposal=sample_proposal,
            proposed_segment_title=title,
        )
        assert fig.layout.title.text == title


class TestSegmentRendering:
    """Tests for render_segment method."""

    def test_render_segment_returns_figure(self, chart: ProfileChart, sample_segment: SlopeSegment) -> None:
        """render_segment returns a Plotly Figure."""
        fig = chart.render_segment(
            segment=sample_segment,
            difficulty="blue",
            title="Test Segment",
        )
        assert isinstance(fig, go.Figure)

    def test_render_segment_with_empty_points_raises(self, chart: ProfileChart) -> None:
        """render_segment raises for segment without points."""
        empty_segment = SlopeSegment(
            id="S1",
            name="Empty",
            points=[],
            start_node_id="N1",
            end_node_id="N2",
        )
        with pytest.raises(ValueError, match="points"):
            chart.render_segment(
                segment=empty_segment,
                difficulty="blue",
            )

    def test_render_segment_different_difficulties(self, chart: ProfileChart, sample_segment: SlopeSegment) -> None:
        """render_segment works for all difficulty levels."""
        for difficulty in ["green", "blue", "red", "black"]:
            fig = chart.render_segment(
                segment=sample_segment,
                difficulty=difficulty,
            )
            assert isinstance(fig, go.Figure)


class TestSlopeRendering:
    """Tests for render_slope method (combined segments)."""

    @pytest.fixture
    def graph_with_slope(self) -> ResortGraph:
        """Graph with a finished slope."""
        graph = ResortGraph()

        # Create two segments
        for i in range(2):
            proposal = make_proposal(index=i, base_elev=2500 - i * 100)
            graph.commit_paths(paths=[proposal])

        seg_ids = list(graph.segments.keys())
        graph.finish_slope(segment_ids=seg_ids)
        return graph

    def test_render_slope_returns_figure(self, chart: ProfileChart, graph_with_slope: ResortGraph) -> None:
        """render_slope returns a Plotly Figure."""
        slope = list(graph_with_slope.slopes.values())[0]
        fig = chart.render_slope(slope=slope, graph=graph_with_slope)
        assert isinstance(fig, go.Figure)

    def test_render_slope_with_missing_segment_raises(self, chart: ProfileChart, graph_with_slope: ResortGraph) -> None:
        """render_slope raises if segment is missing."""
        slope = list(graph_with_slope.slopes.values())[0]
        # Remove a segment
        seg_id = slope.segment_ids[0]
        del graph_with_slope.segments[seg_id]

        with pytest.raises(ValueError, match="segment"):
            chart.render_slope(slope=slope, graph=graph_with_slope)


class TestChartConfiguration:
    """Tests for chart dimensions and configuration."""

    def test_default_dimensions(self) -> None:
        """Chart uses provided dimensions."""
        chart = ProfileChart(width=600, height=300)
        assert chart.width == 600
        assert chart.height == 300

    def test_y_axis_range_includes_padding(self, chart: ProfileChart, sample_proposal: ProposedSlopeSegment) -> None:
        """Y-axis range includes padding around data."""
        fig = chart.render_proposal(
            proposal=sample_proposal,
            proposed_segment_title="Test",
        )

        y_range = fig.layout.yaxis.range
        min_elev = min(p.elevation for p in sample_proposal.points)
        max_elev = max(p.elevation for p in sample_proposal.points)

        # Range should extend beyond data
        assert y_range[0] < min_elev
        assert y_range[1] > max_elev


class TestDifficultyColors:
    """Tests for difficulty-based coloring."""

    def test_all_difficulties_have_colors(self, chart: ProfileChart) -> None:
        """All difficulty levels can be rendered with colors."""
        for difficulty in ["green", "blue", "red", "black"]:
            proposal = make_proposal(difficulty=difficulty)
            fig = chart.render_proposal(
                proposal=proposal,
                proposed_segment_title=f"{difficulty.title()} Proposal",
            )
            assert isinstance(fig, go.Figure)


class TestInputValidation:
    """Strictness tests - internal errors should raise."""

    def test_invalid_difficulty_in_segment_raises(self, chart: ProfileChart) -> None:
        """Invalid difficulty key raises KeyError."""
        segment = make_segment()
        with pytest.raises(KeyError):
            chart.render_segment(segment=segment, difficulty="invalid")

    def test_single_point_proposal_still_works(self, chart: ProfileChart) -> None:
        """Proposal with single point can be rendered (edge case)."""
        single_point = ProposedSlopeSegment(
            points=[PathPoint(lon=10.0, lat=47.0, elevation=2500)],
            target_slope_pct=0.0,
            target_difficulty="blue",
        )
        # Should not raise - single point is technically valid
        fig = chart.render_proposal(
            proposal=single_point,
            proposed_segment_title="Single Point",
        )
        assert isinstance(fig, go.Figure)
