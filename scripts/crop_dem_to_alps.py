"""Crop the full Euro DEM to the Alps region for faster loading and smaller file size.

Developer utility: This script was used to create the cropped alps_dem.tif that ships
with the package. Normal users don't need to run this - the cropped data is included.

To create a new cropped DEM:
1. Download full EuroDEM from https://www.mapsforeurope.org/datasets/euro-dem
2. Update INPUT_FILE path below to point to eurodem.tif
3. Run: python -m skiresort_planner.crop_dem_to_alps
"""

from pathlib import Path

import rasterio
from rasterio.mask import mask
from shapely.geometry import box

from skiresort_planner.constants import DEMConfig

# Paths - update INPUT_FILE to point to your downloaded full EuroDEM
INPUT_FILE = Path.home() / "Downloads" / "euro-dem-tif" / "data" / "eurodem" / "eurodem.tif"
OUTPUT_FILE = DEMConfig.EURODEM_PATH

# Alps bounding box in degrees (WGS84)
ALPS_WEST_DEG = 4.5
ALPS_EAST_DEG = 17.0
ALPS_SOUTH_DEG = 42.75
ALPS_NORTH_DEG = 48.5

# The EuroDEM uses arcseconds as units (1 degree = 3600 arcseconds)
ARCSECONDS_PER_DEGREE = 3600


def crop_dem_to_alps() -> None:
    """Load full Euro DEM, crop to Alps region, and save as compressed GeoTIFF."""
    # Convert degrees to arcseconds for the DEM's native CRS
    west = ALPS_WEST_DEG * ARCSECONDS_PER_DEGREE
    east = ALPS_EAST_DEG * ARCSECONDS_PER_DEGREE
    south = ALPS_SOUTH_DEG * ARCSECONDS_PER_DEGREE
    north = ALPS_NORTH_DEG * ARCSECONDS_PER_DEGREE

    print(f"Alps bbox in arcseconds: W={west}, E={east}, S={south}, N={north}")

    # Create bounding box in the DEM's native CRS (arcseconds)
    alps_bbox = box(west, south, east, north)
    geo = [alps_bbox.__geo_interface__]

    with rasterio.open(INPUT_FILE) as src:
        print(f"Input CRS: {src.crs}")
        print(f"Input bounds: {src.bounds}")
        print(f"Input shape: {src.width} x {src.height}")
        print(f"Input size: {src.width * src.height * 4 / 1024 / 1024:.1f} MB (uncompressed)")

        # Crop to Alps region
        out_image, out_transform = mask(dataset=src, shapes=geo, crop=True)
        out_meta = src.meta.copy()

        # Update metadata with new dimensions and compression
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw",  # Lossless compression
            }
        )

        print(f"Output shape: {out_meta['width']} x {out_meta['height']}")
        print(f"Output size: {out_meta['width'] * out_meta['height'] * 4 / 1024 / 1024:.1f} MB (uncompressed)")

        with rasterio.open(OUTPUT_FILE, "w", **out_meta) as dest:
            dest.write(out_image)

    print(f"Saved cropped DEM to {OUTPUT_FILE}")


if __name__ == "__main__":
    crop_dem_to_alps()
