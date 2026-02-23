"""ResortGraph - Central manager for ski resort entities.

Owns and manages all nodes, segments, slopes, and lifts.
Provides operations for:
- Committing proposed paths to segments
- Finishing slopes (grouping segments)
- Adding/removing lifts
- Undo/redo operations
- Serialization/deserialization
- Graph cleanup (isolated nodes, auto-backup)

Reference: DETAILS.md
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from skiresort_planner.constants import OUTPUT_DIR, EntityPrefixes, PathConfig, UndoConfig
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
from skiresort_planner.model.lift import Lift
from skiresort_planner.model.node import Node
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.proposed_path import ProposedSlopeSegment
from skiresort_planner.model.slope import Slope
from skiresort_planner.model.slope_segment import SlopeSegment

if TYPE_CHECKING:
    from skiresort_planner.core.dem_service import DEMService

logger = logging.getLogger(__name__)


# =============================================================================
# Undo Action Types
# =============================================================================


@dataclass(frozen=True)
class AddSegmentsAction:
    """Undo action for committed path segments."""

    segment_ids: tuple[str, ...]
    node_ids: tuple[str, ...]


@dataclass(frozen=True)
class FinishSlopeAction:
    """Undo action for finishing a slope."""

    slope_id: str
    segment_ids: tuple[str, ...]
    slope_name: str
    start_node_id: str | None


@dataclass(frozen=True)
class AddLiftAction:
    """Undo action for creating a lift."""

    lift_id: str


@dataclass(frozen=True)
class DeleteSlopeAction:
    """Undo action for deleting a slope (stores data for restore)."""

    slope_id: str
    deleted_slope: "Slope"
    deleted_segments: tuple["SlopeSegment", ...]


@dataclass(frozen=True)
class DeleteLiftAction:
    """Undo action for deleting a lift (stores data for restore)."""

    lift_id: str
    deleted_lift: "Lift"


UndoAction = AddSegmentsAction | FinishSlopeAction | AddLiftAction | DeleteSlopeAction | DeleteLiftAction


class ResortGraph:
    """Graph representing the ski resort.

    Central manager owning all nodes, segments, slopes, and lifts.
    Provides operations for building and modifying the resort.

    Example:
        graph = ResortGraph()
        graph.commit_paths(paths=[proposed_path])
        graph.finish_slope(segment_ids=["S1", "S2"])
    """

    def __init__(self) -> None:
        """Initialize empty resort graph."""
        self.nodes: dict[str, Node] = {}
        self.segments: dict[str, SlopeSegment] = {}
        self.slopes: dict[str, Slope] = {}
        self.lifts: dict[str, Lift] = {}
        self.undo_stack: list[UndoAction] = []

        self._node_counter = 0
        self._segment_counter = 0
        self._slope_counter = 0
        self._lift_counter = 0

    def _next_node_id(self) -> str:
        self._node_counter += 1
        return f"{EntityPrefixes.NODE}{self._node_counter}"

    def _next_segment_id(self) -> str:
        self._segment_counter += 1
        return f"{EntityPrefixes.SEGMENT}{self._segment_counter}"

    def _next_slope_id(self) -> str:
        self._slope_counter += 1
        return f"{EntityPrefixes.SLOPE}{self._slope_counter}"

    def _next_lift_id(self) -> str:
        self._lift_counter += 1
        return f"{EntityPrefixes.LIFT}{self._lift_counter}"

    def _push_undo(self, action: UndoAction) -> None:
        """Push action to undo stack with size limiting.

        Discards oldest actions when stack exceeds MAX_UNDO_STACK_SIZE.
        """
        self.undo_stack.append(action)
        # Trim oldest actions if stack is too large
        while len(self.undo_stack) > UndoConfig.MAX_UNDO_STACK_SIZE:
            self.undo_stack.pop(0)

    # =========================================================================
    # Node Operations
    # =========================================================================

    def find_nearest_node(
        self,
        lon: float,
        lat: float,
        threshold_m: float = PathConfig.STEP_SIZE_M,
    ) -> Optional[Node]:
        """Find nearest node within threshold distance.

        Args:
            lon, lat: Target coordinates
            threshold_m: Maximum distance in meters

        Returns:
            Nearest Node or None if none within threshold.
        """
        best_dist = threshold_m
        best_node = None

        for node in self.nodes.values():
            dist = node.distance_to(lon=lon, lat=lat)
            if dist < best_dist:
                best_dist = dist
                best_node = node

        return best_node

    def get_or_create_node(
        self,
        lon: float,
        lat: float,
        elevation: float,
    ) -> tuple[Node, bool]:
        """Get existing node or create new one.

        Args:
            lon, lat: Node coordinates
            elevation: Node elevation

        Returns:
            Tuple of (node, was_created)
        """
        existing = self.find_nearest_node(lon=lon, lat=lat)
        if existing:
            return existing, False

        node = Node(
            id=self._next_node_id(),
            location=PathPoint(lon=lon, lat=lat, elevation=elevation),
        )
        self.nodes[node.id] = node
        return node, True

    def get_connection_count(self, node_id: str) -> int:
        """Count connections (segments + lifts) for a node.

        Args:
            node_id: ID of the node to check

        Returns:
            Total number of segments and lifts connected to this node.
        """
        segment_count = sum(1 for s in self.segments.values() if s.start_node_id == node_id or s.end_node_id == node_id)
        lift_count = sum(
            1 for lift in self.lifts.values() if lift.start_node_id == node_id or lift.end_node_id == node_id
        )
        return segment_count + lift_count

    # =========================================================================
    # Commit Operations
    # =========================================================================

    def commit_paths(
        self,
        paths: list[ProposedSlopeSegment],
    ) -> list[str]:
        """Commit proposed paths to the graph.

        Simple workflow:
        1. Get or create node at path start (snaps to existing node if nearby)
        2. Get or create node at path end (snaps to existing node if nearby)
        3. Create segment connecting them

        Note: No auto-snap to segment lines. Endpoints only snap to existing
        nodes (via get_or_create_node's find_nearest_node check).

        Args:
            paths: List of ProposedSlopeSegment to commit

        Returns:
            List of end node IDs for continuation.
        """
        new_segment_ids = []
        new_node_ids = []
        end_node_ids = []

        for path in paths:
            if not path.points:
                continue

            # Get or create start node
            start_pt = path.start
            start_node, start_created = self.get_or_create_node(
                lon=start_pt.lon,
                lat=start_pt.lat,
                elevation=start_pt.elevation,
            )
            if start_created:
                new_node_ids.append(start_node.id)

            # Get or create end node
            end_pt = path.end
            end_node, end_created = self.get_or_create_node(
                lon=end_pt.lon,
                lat=end_pt.lat,
                elevation=end_pt.elevation,
            )
            if end_created:
                new_node_ids.append(end_node.id)

            # Calculate side slope (requires terrain analysis, stored in segment)
            if len(path.points) < 2:
                raise ValueError(
                    f"Path must have at least 2 points to compute side slope, got {len(path.points)}: {path}"
                )
            side_info = TerrainAnalyzer.compute_side_slope(
                start_lon=path.points[0].lon,
                start_lat=path.points[0].lat,
                end_lon=path.points[1].lon,
                end_lat=path.points[1].lat,
            )
            side_slope_pct = side_info.slope_pct
            side_slope_dir = side_info.direction

            # Create segment (metrics computed as properties from points)
            segment_id = self._next_segment_id()
            segment = SlopeSegment(
                id=segment_id,
                name=f"Segment {self._segment_counter}",
                points=path.points,
                start_node_id=start_node.id,
                end_node_id=end_node.id,
                side_slope_pct=side_slope_pct,
                side_slope_dir=side_slope_dir,
            )
            self.segments[segment_id] = segment
            new_segment_ids.append(segment_id)
            end_node_ids.append(end_node.id)

        # Record for undo
        if new_segment_ids:
            self._push_undo(
                AddSegmentsAction(
                    segment_ids=tuple(new_segment_ids),
                    node_ids=tuple(new_node_ids),
                )
            )

        return end_node_ids

    def finish_slope(
        self,
        segment_ids: list[str],
        name: Optional[str] = None,
    ) -> Optional[Slope]:
        """Finish a slope by grouping segments.

        Args:
            segment_ids: List of segment IDs to group
            name: Optional custom name (generates creative name if None)

        Returns:
            Created Slope or None if invalid.
        """
        if not segment_ids:
            return None

        # Get first and last segment
        first_seg = self.segments.get(segment_ids[0])
        last_seg = self.segments.get(segment_ids[-1])

        if not first_seg or not last_seg:
            return None

        # Calculate metrics for naming
        start_node = self.nodes.get(first_seg.start_node_id)
        end_node = self.nodes.get(last_seg.end_node_id)

        if not start_node or not end_node:
            return None

        # Calculate average bearing
        avg_bearing = GeoCalculator.initial_bearing_deg(
            lon1=start_node.lon,
            lat1=start_node.lat,
            lon2=end_node.lon,
            lat2=end_node.lat,
        )

        slope_id = self._next_slope_id()

        # Determine difficulty
        max_slope = max(self.segments[sid].avg_slope_pct for sid in segment_ids if sid in self.segments)
        difficulty = TerrainAnalyzer.classify_difficulty(slope_pct=max_slope)

        # Generate name
        if name is None:
            name = Slope.generate_name(
                difficulty=difficulty,
                slope_id=slope_id,
                start_elevation=start_node.elevation,
                end_elevation=end_node.elevation,
                avg_bearing=avg_bearing,
            )

        logger.info(f"Slope finished: {name}, {len(segment_ids)} segments, difficulty={difficulty}")

        slope = Slope(
            id=slope_id,
            name=name,
            segment_ids=segment_ids,
            start_node_id=first_seg.start_node_id,
            end_node_id=last_seg.end_node_id,
        )
        self.slopes[slope_id] = slope

        # Update segment names
        for seg_id in segment_ids:
            seg = self.segments.get(seg_id)
            if seg:
                seg.name = name

        # Record for undo (store slope name and start node for context restoration)
        self._push_undo(
            FinishSlopeAction(
                slope_id=slope_id,
                segment_ids=tuple(segment_ids),
                slope_name=name,
                start_node_id=first_seg.start_node_id,
            )
        )

        return slope

    # =========================================================================
    # Lift Operations
    # =========================================================================

    def add_lift(
        self,
        start_node_id: str,
        end_node_id: str,
        lift_type: str,
        dem: "DEMService",
    ) -> Lift:
        """Add a lift between two nodes.

        Args:
            start_node_id: ID of bottom station
            end_node_id: ID of top station
            lift_type: Type of lift
            dem: DEM service for terrain sampling

        Returns:
            Created Lift.

        Raises:
            KeyError: If start or end node not found.
        """
        start = self.nodes[start_node_id]
        end = self.nodes[end_node_id]

        lift_id = self._next_lift_id()

        lift = Lift.create(
            start_node=start,
            end_node=end,
            dem=dem,
            lift_type=lift_type,
            lift_id=lift_id,
        )

        self.lifts[lift_id] = lift
        self._push_undo(AddLiftAction(lift_id=lift_id))

        return lift

    # =========================================================================
    # Undo Operations
    # =========================================================================

    def undo_last(self) -> UndoAction:
        """Undo the last action.

        Returns:
            The undone action.

        Raises:
            RuntimeError: If undo stack is empty (caller should check first).
        """
        if not self.undo_stack:
            raise RuntimeError("undo_last called with empty undo_stack")

        action = self.undo_stack.pop()

        if isinstance(action, AddSegmentsAction):
            for seg_id in action.segment_ids:
                self.segments.pop(seg_id, None)
            self.cleanup_isolated_nodes()  # Remove orphaned nodes

        elif isinstance(action, AddLiftAction):
            self.lifts.pop(action.lift_id, None)
            self.cleanup_isolated_nodes()  # Remove orphaned nodes created for lift

        elif isinstance(action, FinishSlopeAction):
            self.slopes.pop(action.slope_id, None)
            for seg_id in action.segment_ids:
                seg = self.segments.get(seg_id)
                if seg:
                    seg.name = f"Segment {seg_id[1:]}"
            # No nodes are added in finish_slope only in last segment, so no cleanup needed

        elif isinstance(action, DeleteSlopeAction):
            # Restore deleted slope and its segments
            self.slopes[action.slope_id] = action.deleted_slope
            for seg in action.deleted_segments:
                self.segments[seg.id] = seg
            logger.info(f"Restored slope {action.slope_id} with {len(action.deleted_segments)} segments")

        elif isinstance(action, DeleteLiftAction):
            # Restore deleted lift
            self.lifts[action.lift_id] = action.deleted_lift
            logger.info(f"Restored lift {action.lift_id}")

        return action

    def delete_slope(self, slope_id: str) -> bool:
        """Delete a slope and its segments.

        Args:
            slope_id: ID of slope to delete

        Returns:
            True if deleted, False if not found.
        """
        slope = self.slopes.get(slope_id)
        if not slope:
            return False

        # Collect segments to store for undo
        deleted_segments = [self.segments[seg_id] for seg_id in slope.segment_ids if seg_id in self.segments]

        # Remove all segments belonging to this slope
        for seg_id in slope.segment_ids:
            self.segments.pop(seg_id, None)

        # Remove the slope
        del self.slopes[slope_id]

        # Push to undo stack with full data for restore
        self._push_undo(
            DeleteSlopeAction(
                slope_id=slope_id,
                deleted_slope=slope,
                deleted_segments=tuple(deleted_segments),
            )
        )

        # Cleanup isolated nodes
        self.cleanup_isolated_nodes()

        logger.info(f"Deleted slope {slope.name} with {len(slope.segment_ids)} segments")
        return True

    def delete_lift(self, lift_id: str) -> bool:
        """Delete a lift.

        Args:
            lift_id: ID of lift to delete

        Returns:
            True if deleted, False if not found.
        """
        lift = self.lifts.get(lift_id)
        if not lift:
            return False

        # Remove the lift
        del self.lifts[lift_id]

        # Push to undo stack with full data for restore
        self._push_undo(
            DeleteLiftAction(
                lift_id=lift_id,
                deleted_lift=lift,
            )
        )

        # Cleanup isolated nodes (lift stations may become isolated)
        self.cleanup_isolated_nodes()

        logger.info(f"Deleted lift {lift.name}")
        return True

    # =========================================================================
    # Query Operations
    # =========================================================================

    def get_slope_by_segment_id(self, segment_id: str) -> Optional[Slope]:
        """Find the slope containing a given segment.

        Args:
            segment_id: ID of segment to find

        Returns:
            Slope containing the segment, or None if segment is not in any slope.
        """
        for slope in self.slopes.values():
            if segment_id in slope.segment_ids:
                return slope
        return None

    def get_segment_stats(self, segment_ids: list[str]) -> dict:
        """Get statistics for specific segments (used for running stats during building).

        Args:
            segment_ids: List of segment IDs to calculate stats for

        Returns:
            Dict with: total_drop, total_length, avg_gradient, max_gradient, difficulty, start_elev, current_elev
        """
        if not segment_ids:
            return {
                "total_drop": 0.0,
                "total_length": 0.0,
                "avg_gradient": 0.0,
                "max_gradient": 0.0,
                "difficulty": "green",
                "start_elev": None,
                "current_elev": None,
            }

        first_seg = self.segments.get(segment_ids[0])
        last_seg = self.segments.get(segment_ids[-1])

        if not first_seg or not last_seg:
            return {
                "total_drop": 0.0,
                "total_length": 0.0,
                "avg_gradient": 0.0,
                "max_gradient": 0.0,
                "difficulty": "green",
                "start_elev": None,
                "current_elev": None,
            }

        start_elev = first_seg.start.elevation
        current_elev = last_seg.end.elevation

        total_length = sum(seg.length_m for seg_id in segment_ids if (seg := self.segments.get(seg_id)))

        total_drop = start_elev - current_elev
        avg_gradient = (total_drop / total_length * 100) if total_length > 0 else 0.0

        # Difficulty is based on steepest segment (not average)
        max_slope = max(self.segments[sid].avg_slope_pct for sid in segment_ids if sid in self.segments)
        difficulty = TerrainAnalyzer.classify_difficulty(slope_pct=max_slope)

        return {
            "total_drop": total_drop,
            "total_length": total_length,
            "avg_gradient": avg_gradient,
            "max_gradient": max_slope,
            "difficulty": difficulty,
            "start_elev": start_elev,
            "current_elev": current_elev,
        }

    def get_stats(self) -> dict:
        """Get resort statistics."""
        if not self.segments:
            return {
                "total_slopes": 0,
                "total_segments": 0,
                "total_vertical_m": 0,
                "total_length_m": 0,
                "longest_run_m": 0,
                "total_lifts": len(self.lifts),
            }

        total_vertical = sum(s.total_drop_m for s in self.segments.values())
        total_length = sum(s.length_m for s in self.segments.values())

        longest = 0.0
        for slope in self.slopes.values():
            slope_length = slope.get_total_length(segments=self.segments)
            if slope_length > longest:
                longest = slope_length
        for seg in self.segments.values():
            if seg.length_m > longest:
                longest = seg.length_m

        return {
            "total_slopes": len(self.slopes),
            "total_segments": len(self.segments),
            "total_vertical_m": total_vertical,
            "total_length_m": total_length,
            "longest_run_m": longest,
            "total_lifts": len(self.lifts),
        }

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> dict:
        """Serialize entire graph to JSON-compatible dict."""
        return {
            "version": "2.0",
            "nodes": {nid: asdict(node) for nid, node in self.nodes.items()},
            "segments": {sid: asdict(seg) for sid, seg in self.segments.items()},
            "slopes": {slid: asdict(slope) for slid, slope in self.slopes.items()},
            "lifts": {lid: asdict(lift) for lid, lift in self.lifts.items()},
            "counters": {
                "node": self._node_counter,
                "segment": self._segment_counter,
                "slope": self._slope_counter,
                "lift": self._lift_counter,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ResortGraph":
        """Deserialize graph from dict."""
        graph = cls()

        for nid, node_data in data["nodes"].items():
            graph.nodes[nid] = Node.from_dict(data=node_data)

        for sid, seg_data in data["segments"].items():
            graph.segments[sid] = SlopeSegment.from_dict(data=seg_data)

        for slid, slope_data in data["slopes"].items():
            graph.slopes[slid] = Slope.from_dict(data=slope_data)

        for lid, lift_data in data["lifts"].items():
            graph.lifts[lid] = Lift.from_dict(data=lift_data)

        counters = data["counters"]
        graph._node_counter = counters["node"]
        graph._segment_counter = counters["segment"]
        graph._slope_counter = counters["slope"]
        graph._lift_counter = counters["lift"]

        return graph

    def to_gpx(self) -> str:
        """Export resort to GPX format."""
        import xml.etree.ElementTree as ET

        gpx_ns = "http://www.topografix.com/GPX/1/1"
        ET.register_namespace("", gpx_ns)

        gpx = ET.Element("gpx", xmlns=gpx_ns, version="1.1", creator="Ski Resort Planner")

        metadata = ET.SubElement(gpx, "metadata")
        ET.SubElement(metadata, "name").text = "Ski Resort Planner Export"
        ET.SubElement(metadata, "time").text = datetime.now().isoformat()

        # Track finished slope segments
        finished_segment_ids = set()
        for slope in self.slopes.values():
            finished_segment_ids.update(slope.segment_ids)

        # Export finished slopes
        for slope in self.slopes.values():
            all_points = slope.get_all_points(segments=self.segments)
            difficulty = slope.get_difficulty(segments=self.segments)
            total_length = slope.get_total_length(segments=self.segments)
            total_drop = slope.get_total_drop(segments=self.segments)

            trk = ET.SubElement(gpx, "trk")
            ET.SubElement(trk, "name").text = slope.name
            ET.SubElement(
                trk, "desc"
            ).text = f"{difficulty.capitalize()} - Drop {total_drop:.0f}m - Length {total_length:.0f}m"
            ET.SubElement(trk, "type").text = f"slope_{difficulty}"

            trkseg = ET.SubElement(trk, "trkseg")
            for pt in all_points:
                trkpt = ET.SubElement(trkseg, "trkpt", lat=str(pt.lat), lon=str(pt.lon))
                ET.SubElement(trkpt, "ele").text = f"{pt.elevation:.1f}"

        # Export lifts (using terrain_points for accurate terrain following)
        for lift in self.lifts.values():
            start_node = self.nodes.get(lift.start_node_id)
            end_node = self.nodes.get(lift.end_node_id)
            if not start_node or not end_node:
                raise ValueError(f"Lift {lift.name} has invalid start or end node")

            trk = ET.SubElement(gpx, "trk")
            ET.SubElement(trk, "name").text = f"{lift.name} - {lift.lift_type}"
            vertical = lift.get_vertical_rise(nodes=self.nodes)
            length = lift.get_length_m(nodes=self.nodes)
            ET.SubElement(trk, "desc").text = f"Rise {vertical:.0f}m - Length {length:.0f}m"
            ET.SubElement(trk, "type").text = f"lift_{lift.lift_type}"

            trkseg = ET.SubElement(trk, "trkseg")

            # Use cable_points for 3D visualization (follows cable line with sag)
            if not lift.cable_points:
                raise ValueError(f"Lift {lift.name} must have cable_points for GPX export")
            for pt in lift.cable_points:
                trkpt = ET.SubElement(trkseg, "trkpt", lat=str(pt.lat), lon=str(pt.lon))
                ET.SubElement(trkpt, "ele").text = f"{pt.elevation:.1f}"

        return ET.tostring(gpx, encoding="unicode", method="xml")

    # =========================================================================
    # Cleanup and Maintenance
    # =========================================================================

    def cleanup_isolated_nodes(self) -> int:
        """Remove nodes not connected to any segment or lift.

        Returns:
            Number of nodes removed.
        """
        isolated_node_ids = [node_id for node_id in self.nodes if self.get_connection_count(node_id=node_id) == 0]

        for node_id in isolated_node_ids:
            del self.nodes[node_id]

        return len(isolated_node_ids)

    def create_auto_backup(self) -> None:
        """Create automatic backup of the resort graph.

        Saves a JSON file without timestamp to overwrite an existing backup if it exists.

        Args:
            backup_dir: Directory for backups. Defaults to
                output/skiresort_planner/backups/
        """
        BACKUP_DIR = Path(OUTPUT_DIR) / "skiresort_planner" / "backups"

        BACKUP_DIR.mkdir(parents=True, exist_ok=True)

        # Create timestamped backup
        backup_path = BACKUP_DIR / "resort_backup.json"

        try:
            with open(backup_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
            logger.info(f"Auto-backup created: {backup_path.name}")
        except Exception as e:
            logger.error(f"Failed to create auto-backup: {e}")
            return

    def perform_cleanup(self) -> None:
        """Perform maintenance tasks on the graph.

        Called by StreamlitUIListener.after_transition() before st.rerun().
        Ensures the graph is always in a clean state after any state change.

        Current cleanup tasks:
        - Remove isolated nodes (nodes not connected to any segment or lift)
        - Create automatic backup (JSON file)
        """
        # Remove isolated nodes and log how many were removed
        removed_count = self.cleanup_isolated_nodes()
        if removed_count > 0:
            logger.info(f"Cleanup: removed {removed_count} isolated node(s)")

        # Create auto-backup if there's any content
        if self.nodes or self.segments or self.slopes or self.lifts:
            self.create_auto_backup()
