"""On-disk backup store for ResortGraph state.

Silently checkpoints the resort after each graph change so users survive
browser reloads and dropped connections. Long-term storage is still the
user's job via the JSON export — this is only a crash/outage safety net.

Layout: `backups/<resort_id>.json` — one file per resort. Filenames
are short UUIDs; each session gets one via `st.query_params`.

Writes are atomic (tempfile + os.replace) so a crash mid-write cannot
leave a truncated JSON. No locking: filenames are distinct per resort,
and two tabs sharing the same URL accept last-write-wins.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path

from skiresort_planner.constants import BACKUP_DIR
from skiresort_planner.model.resort_graph import ResortGraph

logger = logging.getLogger(__name__)


def new_resort_id() -> str:
    """Return a short filename-safe id for a fresh resort."""
    return uuid.uuid4().hex[:8]


def _path_for(resort_id: str) -> Path:
    return BACKUP_DIR / f"{resort_id}.json"


def save(graph: ResortGraph, resort_id: str) -> None:
    """Atomically write graph.to_dict() to backups/<resort_id>.json.

    Skips empty graphs — no point creating a file for an unused session.
    """
    if not graph.slopes and not graph.lifts and not graph.segments:
        return

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    target = _path_for(resort_id)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(graph.to_dict(), indent=2))
    os.replace(tmp, target)
    logger.info(f"Auto-saved resort {resort_id} to {target}")


def load(resort_id: str) -> ResortGraph | None:
    """Read backups/<resort_id>.json and return the graph, or None if missing."""
    path = _path_for(resort_id)
    if not path.exists():
        return None
    return ResortGraph.from_dict(data=json.loads(path.read_text()))


def delete(resort_id: str) -> None:
    """Remove backups/<resort_id>.json (silent if missing)."""
    _path_for(resort_id).unlink(missing_ok=True)
    logger.info(f"Deleted backup {resort_id}")


def largest_resort_id() -> str | None:
    """Return the id of the backup with the most nodes, or None if none exist.

    The "in doubt, restore my work" fallback when a user opens the bare link
    with no ?resort= param: the biggest resort is almost always theirs, since
    users typically work on one resort at a time.
    """
    if not BACKUP_DIR.exists():
        return None

    best_id: str | None = None
    best_nodes = -1
    for path in BACKUP_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Skipping unreadable backup {path.name}: {e}")
            continue
        node_count = len(data.get("nodes", {}))
        if node_count > best_nodes:
            best_nodes = node_count
            best_id = path.stem

    return best_id
