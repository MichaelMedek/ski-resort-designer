"""Digital Elevation Model (DEM) service for terrain elevation queries.

Provides singleton access to EuroDEM GeoTIFF data:
- Fast O(1) elevation lookup using pre-loaded NumPy array
- Automatic coordinate transformation from WGS84 to DEM's native CRS
- Auto-download from Hugging Face if local file missing
- Thread-safe singleton pattern

Data Source:
    EuroDEM - 60m resolution covering Europe
    Download: https://www.mapsforeurope.org/datasets/euro-dem
    Hosted: https://huggingface.co/datasets/MichaelMedek/alps_eurodem

Reference: DETAILS.md Section 1.1
"""

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import rasterio
import requests
from rasterio.warp import transform

from skiresort_planner.constants import DEMConfig

logger = logging.getLogger(__name__)


def download_dem_from_huggingface(
    target_path: Path = DEMConfig.EURODEM_PATH,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Path:
    """Download Alps DEM from Hugging Face if not already present.

    Args:
        target_path: Local path to save the DEM file.
        progress_callback: Optional callback receiving progress 0.0-1.0.

    Returns:
        Path to the downloaded (or existing) DEM file.

    Raises:
        requests.RequestException: If download fails.
    """
    if target_path.exists():
        logger.info(f"DEM already exists at {target_path}")
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    url = DEMConfig.HF_DOWNLOAD_URL
    logger.info(f"Downloading Alps DEM (~285MB) from {url}...")

    response = requests.get(url, stream=True, timeout=180)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if progress_callback and total_size > 0:
                progress_callback(downloaded / total_size)

    logger.info(f"DEM downloaded to {target_path}")
    return target_path


class DEMService:
    """Singleton service for elevation sampling from EuroDEM GeoTIFF.

    Uses the singleton pattern to ensure only one DEM file is loaded into memory.
    The DEM array is loaded on first access and cached for fast subsequent queries.

    Example:
        dem = DEMService()
        elevation = dem.get_elevation(lon=10.295, lat=46.985)
    """

    _instance: Optional["DEMService"] = None
    _load_lock = threading.Lock()
    _dem = None
    _dem_crs: Optional[str] = None
    _dem_array: Optional[np.ndarray] = None
    _dem_transform = None
    _dem_nodata = None

    def __new__(cls, dem_path: Optional[Path] = None) -> "DEMService":
        """Create or return the singleton instance.

        Args:
            dem_path: Optional path to DEM file (uses EURODEM_PATH by default)

        Returns:
            The singleton DEMService instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._dem_path = dem_path or DEMConfig.EURODEM_PATH
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        """Check if DEM data has been fully loaded into memory."""
        return self._dem_transform is not None

    def _ensure_loaded(self) -> None:
        """Load DEM into memory on first access (thread-safe)."""
        # Fast path: already loaded
        if self.is_loaded:
            return

        # Slow path: acquire lock and load (or wait for another thread to finish)
        with self._load_lock:
            # Double-check after acquiring lock
            if self.is_loaded:
                return

            dem_path = self._dem_path

            if not dem_path.exists():
                raise FileNotFoundError(
                    f"DEM file not found at {dem_path}. Run the Streamlit app first to download it."
                )

            logger.info(f"Loading EuroDEM from {dem_path}...")
            start_time = time.time()

            self._dem = rasterio.open(dem_path)
            self._dem_crs = self._dem.crs.to_string() if self._dem.crs else "EPSG:4326"
            self._dem_array = self._dem.read(1)
            self._dem_nodata = self._dem.nodata
            # Set _dem_transform LAST - this is what is_loaded checks
            self._dem_transform = self._dem.transform

            elapsed = time.time() - start_time
            logger.info(f"EuroDEM loaded in {elapsed:.2f}s (shape: {self._dem_array.shape}, CRS: {self._dem_crs})")

    def get_elevation(self, lon: float, lat: float) -> float | None:
        """Get elevation at a single point using direct NumPy array lookup.

        Args:
            lon: Longitude in decimal degrees (WGS84)
            lat: Latitude in decimal degrees (WGS84)

        Returns:
            Elevation in meters, or None if outside coverage or invalid.
        """
        self._ensure_loaded()

        # Transform WGS84 to DEM CRS if needed
        if self._dem_crs != "EPSG:4326":
            proj_coords = transform("EPSG:4326", self._dem_crs, [lon], [lat])
            x, y = proj_coords[0][0], proj_coords[1][0]
        else:
            x, y = lon, lat

        # Convert coordinates to array indices using inverse transform
        col, row = ~self._dem_transform * (x, y)
        col, row = int(col), int(row)

        # Check bounds
        if row < 0 or row >= self._dem_array.shape[0] or col < 0 or col >= self._dem_array.shape[1]:
            logger.warning(
                f"Coordinates outside DEM bounds: lon={lon}, lat={lat} (row={row}, col={col}, shape={self._dem_array.shape})"
            )
            return None

        elev = self._dem_array[row, col]

        # Check for no-data values
        if self._dem_nodata is not None and elev == self._dem_nodata:
            logger.warning(
                f"No-data value at coordinates: lon={lon}, lat={lat} (raw_value={elev}, nodata={self._dem_nodata})"
            )
            return None
        if np.isnan(elev):
            logger.warning(f"NaN elevation at coordinates: lon={lon}, lat={lat}")
            return None

        return float(elev)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return (west, south, east, north) bounds in WGS84.

        Returns:
            Tuple of (min_lon, min_lat, max_lon, max_lat) in decimal degrees.
        """
        self._ensure_loaded()
        b = self._dem.bounds

        if self._dem_crs != "EPSG:4326":
            # Transform corners to WGS84
            corners_x = [b.left, b.right, b.left, b.right]
            corners_y = [b.bottom, b.bottom, b.top, b.top]
            lons, lats = transform(self._dem_crs, "EPSG:4326", corners_x, corners_y)
            return min(lons), min(lats), max(lons), max(lats)

        return b.left, b.bottom, b.right, b.top
