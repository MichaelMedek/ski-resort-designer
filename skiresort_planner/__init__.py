"""Ski Resort Planner - Design ski resorts on real terrain.

A modular, professional-grade ski resort planning application featuring:
- High-precision terrain analysis using DEM data
- Smart path generation with difficulty-aware routing
- State machine-based UI for robust user interactions
- Comprehensive resort management (slopes, lifts, nodes)

Modules:
    core: Foundation classes (geo calculations, DEM service, terrain analysis)
    model: Data structures (PathPoint, Node, SlopeSegment, Slope, Lift)
    generators: Path generation algorithms (fan patterns, custom direction paths)
    ui: Streamlit interface components (state machine, renderers, sidebar)

Example:
    from skiresort_planner.core import DEMService, TerrainAnalyzer
    from skiresort_planner.model import ResortGraph
    from skiresort_planner.generators import PathFactory
"""
