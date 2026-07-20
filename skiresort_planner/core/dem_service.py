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
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import rasterio
import requests
from pyproj import Transformer
from rasterio.io import DatasetReader
from rasterio.warp import transform

from skiresort_planner.constants import DEMConfig

logger = logging.getLogger(__name__)


def download_dem_from_huggingface(
    target_path: Path = DEMConfig.EURODEM_PATH,
    progress_callback: Callable[[float], None] | None = None,
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
        logger.debug(f"DEM already exists at {target_path}")
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
    _dem_path: Path
    _dem: DatasetReader | None = None
    _dem_crs: str | None = None
    _dem_array: npt.NDArray[np.float64] | None = None
    _dem_transform: object = None
    _dem_nodata: object = None
    # Cached WGS84→DEM-CRS transformer, built once at load. None when the DEM is already EPSG:4326
    # (identity, no reprojection). Replaces per-call rasterio.warp.transform (which rebuilds the GDAL
    # env every call) — same result, ~25× faster.
    _to_dem: Transformer | None = None

    def __new__(cls, dem_path: Path | None = None) -> "DEMService":
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
                return  # type: ignore[unreachable]  # Thread-safe double-check pattern

            dem_path = self._dem_path

            if not dem_path.exists():
                raise FileNotFoundError(
                    f"DEM file not found at {dem_path}. Run the Streamlit app first to download it."
                )

            logger.info(f"Loading EuroDEM from {dem_path}...")
            start_time = time.time()

            dataset = rasterio.open(dem_path)
            self._dem = dataset
            self._dem_crs = dataset.crs.to_string() if dataset.crs else "EPSG:4326"
            self._dem_array = dataset.read(1)
            self._dem_nodata = dataset.nodata
            # Cache the WGS84→DEM-CRS transformer once (only when reprojection is needed).
            if self._dem_crs != "EPSG:4326":
                self._to_dem = Transformer.from_crs("EPSG:4326", self._dem_crs, always_xy=True)
            # Set _dem_transform LAST - this is what is_loaded checks
            self._dem_transform = dataset.transform

            elapsed = time.time() - start_time
            logger.info(f"EuroDEM loaded in {elapsed:.2f}s (shape: {self._dem_array.shape}, CRS: {self._dem_crs})")

    def get_elevation(self, lon: float, lat: float) -> float | None:
        """Get elevation at a single point.

        A thin wrapper over the vectorized `get_elevations` (single sampling implementation); returns
        None for an out-of-coverage / nodata / invalid point.

        Args:
            lon: Longitude in decimal degrees (WGS84)
            lat: Latitude in decimal degrees (WGS84)

        Returns:
            Elevation in meters, or None if outside coverage or invalid.
        """
        elev = float(self.get_elevations([lon], [lat])[0])
        return None if np.isnan(elev) else elev

    def get_elevations(
        self, lons: "npt.NDArray[np.float64] | list[float]", lats: "npt.NDArray[np.float64] | list[float]"
    ) -> npt.NDArray[np.float64]:
        """Batch elevation lookup — vectorized WGS84→CRS transform, inverse-affine, and array gather.

        One `transform` call + numpy indexing over all points at once (the fast path for grid planners).
        Out-of-coverage / nodata / NaN cells come back as `np.nan` (the scalar `get_elevation` wrapper
        maps that to None; grid callers raise on any NaN, preserving the no-missing-data invariant).

        Args:
            lons: Longitudes in decimal degrees (WGS84).
            lats: Latitudes in decimal degrees (WGS84).

        Returns:
            1-D float64 array of elevations, `np.nan` where a point is out of coverage or nodata.
        """
        self._ensure_loaded()
        assert self._dem_transform is not None
        assert self._dem_array is not None
        lons = np.asarray(lons, dtype=np.float64)
        lats = np.asarray(lats, dtype=np.float64)

        # WGS84 → DEM CRS if needed (one batched transform via the cached pyproj transformer; takes/
        # returns numpy arrays directly — byte-identical to the old per-point rasterio.warp.transform).
        if self._to_dem is not None:
            xs, ys = self._to_dem.transform(lons, lats)
        else:
            xs, ys = lons, lats

        # Inverse affine → array indices (vectorized `~transform * (x, y)`; int() truncates toward zero).
        inv = ~self._dem_transform  # type: ignore[operator]
        cols = (inv.a * xs + inv.b * ys + inv.c).astype(np.int64)
        rows = (inv.d * xs + inv.e * ys + inv.f).astype(np.int64)

        h, w = self._dem_array.shape
        in_bounds = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        out = np.full(lons.shape, np.nan, dtype=np.float64)
        out[in_bounds] = self._dem_array[rows[in_bounds], cols[in_bounds]]
        if self._dem_nodata is not None:
            out[out == self._dem_nodata] = np.nan
        return out

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return (west, south, east, north) bounds in WGS84.

        Returns:
            Tuple of (min_lon, min_lat, max_lon, max_lat) in decimal degrees.
        """
        self._ensure_loaded()
        assert self._dem is not None
        assert self._dem_crs is not None, "DEM CRS must be set after _ensure_loaded()"
        b = self._dem.bounds

        if self._dem_crs != "EPSG:4326":
            # Transform corners to WGS84
            corners_x = [b.left, b.right, b.left, b.right]
            corners_y = [b.bottom, b.bottom, b.top, b.top]
            lons, lats = transform(self._dem_crs, "EPSG:4326", corners_x, corners_y)
            return min(lons), min(lats), max(lons), max(lats)

        return b.left, b.bottom, b.right, b.top
