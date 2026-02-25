"""Validators - Input validation for ski resort planner.

Centralizes all validation logic. Validators return Optional[Message]:
- None if valid
- A Message object if invalid (caller displays it)

Design Principles:
- No exceptions for expected validation failures
- Messages know their own display level (error/warning/info)
- Caller controls when/how to display the message
"""

from skiresort_planner.constants import ConnectionConfig
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.model.message import (
    LiftMustGoUphillMessage,
    SameNodeLiftMessage,
    TargetNotDownhillMessage,
    TargetTooFarMessage,
    ToastMessage,
)
from skiresort_planner.model.node import Node


def validate_lift_goes_uphill(
    start_node: Node,
    end_node: Node,
) -> ToastMessage | None:
    """Validate that lift end station is higher than start station.

    Returns:
        None if valid, LiftMustGoUphillMessage if end is not higher.
    """
    if end_node.elevation <= start_node.elevation:
        return LiftMustGoUphillMessage(
            start_elevation_m=start_node.elevation,
            end_elevation_m=end_node.elevation,
        )
    return None


def validate_lift_different_nodes(
    start_node_id: str,
    end_node_id: str,
) -> ToastMessage | None:
    """Validate that lift start and end are different nodes.

    Returns:
        None if valid, SameNodeLiftMessage if same node.
    """
    if start_node_id == end_node_id:
        return SameNodeLiftMessage()
    return None


def validate_custom_target_downhill(
    start_elevation: float,
    target_elevation: float,
) -> ToastMessage | None:
    """Validate that custom target is sufficiently downhill.

    Returns:
        None if valid, TargetNotDownhillMessage if not enough drop.
    """
    elevation_drop = start_elevation - target_elevation
    if elevation_drop < ConnectionConfig.MIN_DROP_M:
        return TargetNotDownhillMessage(
            start_elevation_m=start_elevation,
            target_elevation_m=target_elevation,
            min_drop_m=ConnectionConfig.MIN_DROP_M,
        )
    return None


def validate_custom_target_distance(
    start_lat: float,
    start_lon: float,
    target_lat: float,
    target_lon: float,
    max_distance_m: float,
) -> ToastMessage | None:
    """Validate that custom target is within allowed distance.

    Returns:
        None if valid, TargetTooFarMessage if target is too far.
    """
    distance_m = GeoCalculator.haversine_distance_m(
        lat1=start_lat,
        lon1=start_lon,
        lat2=target_lat,
        lon2=target_lon,
    )
    if distance_m > max_distance_m:
        return TargetTooFarMessage(
            distance_m=distance_m,
            max_distance_m=max_distance_m,
        )
    return None
