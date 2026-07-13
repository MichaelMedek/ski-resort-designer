"""Reload-safe comparison for Enum members.

Streamlit re-imports modules on rerun, creating a FRESH class object for each Enum.
Values that persist across reruns (in session state, the preserved graph, cached
proposals) are instances of the OLD class, so `is` identity — and even `==` in some
mixed cases — can surprise. `enum_eq` sidesteps all of it by comparing the stable
`repr()` form (`'<ClassName.MEMBER: value>'`), which is identical across reloads and
class-prefixed so members of different enums never compare equal.

Use `enum_eq(a, b)` for EVERY comparison of these enums instead of `is`/`==`.
"""

from enum import Enum


def enum_eq(a: Enum, b: Enum) -> bool:
    """True if two Enum members are the same member, regardless of Streamlit reloads."""
    return repr(a) == repr(b)
