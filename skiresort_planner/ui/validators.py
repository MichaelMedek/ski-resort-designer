"""Validators - Input validation for ski resort planner.

Centralizes all validation logic. Validators return Optional[Message]:
- None if valid
- A Message object if invalid (caller displays it)

Design Principles:
- No exceptions for expected validation failures
- Messages know their own display level (error/warning/info)
- Caller controls when/how to display the message
"""

from skiresort_planner.constants import ConnectionConfig, LiftConfig, LiftType, PathConfig
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.model.message import (
    LiftTooShortMessage,
    SameNodeLiftMessage,
    TargetNotDownhillMessage,
    TargetTooFarMessage,
    ToastMessage,
)


def validate_lift_stations(
    first_lon: float,
    first_lat: float,
    second_lon: float,
    second_lat: float,
    lift_type: str,
) -> ToastMessage | None:
    """Validate the two lift stations: distinct points, and long enough to host a pylon.

    A pylon needs min_spacing_m clearance to EACH station, so the lift must be >= 2*min_spacing_m long to
    fit even one; shorter lifts are refused. Returns None if valid, else the toast to show.
    """
    if first_lon == second_lon and first_lat == second_lat:
        return SameNodeLiftMessage()
    length_m = GeoCalculator.haversine_distance_m(lat1=first_lat, lon1=first_lon, lat2=second_lat, lon2=second_lon)
    min_length_m = 2 * float(LiftConfig.PYLON_CONFIG[LiftType(lift_type)]["min_spacing_m"])
    if length_m < min_length_m:
        return LiftTooShortMessage(length_m=length_m, min_length_m=min_length_m)
    return None


def validate_custom_target_downhill(
    start_elevation: float,
    target_elevation: float,
    *,
    may_climb: bool,
) -> ToastMessage | None:
    """Validate that a custom target is sufficiently downhill — unless the kind may climb.

    Kinds that may climb (roads) route uphill freely, so the check is skipped for them.
    Returns None if valid (or climbing allowed), TargetNotDownhillMessage otherwise.
    """
    if may_climb:
        return None
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
    if distance_m > PathConfig.SEGMENT_LENGTH_MAX_M:
        return TargetTooFarMessage(
            distance_m=distance_m,
            max_distance_m=PathConfig.SEGMENT_LENGTH_MAX_M,
        )
    return None
