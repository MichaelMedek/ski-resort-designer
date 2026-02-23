"""Path generation algorithms for ski slope planning.

Provides the PathFactory for generating proposed paths using nested loops:
- Fan pattern generation: difficulty → grade → side (up to 16 paths)
- Manual path generation: tries all 8 difficulty-grade combinations
- Force-connect: simple interpolation fallback

Path planning algorithm documented in DETAILS.md.
"""

from skiresort_planner.generators.path_factory import (
    GradeConfig,
    PathFactory,
    Side,
)

__all__ = [
    "PathFactory",
    "GradeConfig",
    "Side",
]
