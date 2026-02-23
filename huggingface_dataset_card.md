---
license: mit
task_categories:
  - other
tags:
  - geospatial
  - elevation
  - dem
  - terrain
  - alps
  - gis
pretty_name: Alps Digital Elevation Model (EuroDEM)
size_categories:
  - n<1K
---

# Alps Digital Elevation Model (EuroDEM)

A cropped Digital Elevation Model (DEM) covering the European Alps at 60m resolution.

## Dataset Description

This dataset contains elevation data for the Alps mountain range, extracted from the [EuroDEM](https://www.mapsforeurope.org/datasets/euro-dem) dataset. It is used by the [Ski Resort Planner](https://github.com/MichaelMedek/Ski-Resort-Planner) application for terrain analysis and ski slope design.

### Coverage

| Parameter | Value |
|-----------|-------|
| **West** | 4.5° E |
| **East** | 17.0° E |
| **South** | 42.75° N |
| **North** | 48.5° N |
| **Resolution** | ~60m |
| **CRS** | EPSG:4326 (arcseconds) |

### Countries Covered

Austria, Switzerland, France, Italy, Germany, Slovenia, Liechtenstein

## File Information

| File | Size | Format |
|------|------|--------|
| `alps_dem.tif` | ~285 MB | GeoTIFF (LZW compressed) |

## Usage

### With Hugging Face Hub

```python
from huggingface_hub import hf_hub_download

dem_path = hf_hub_download(
    repo_id="MichaelMedek/alps_eurodem",
    filename="alps_dem.tif",
    repo_type="dataset"
)
```

### Reading with Rasterio

```python
import rasterio

with rasterio.open(dem_path) as src:
    elevation = src.read(1)
    print(f"Shape: {elevation.shape}")
    print(f"Bounds: {src.bounds}")
```

## Source

Original data from [EuroGeographics](https://www.mapsforeurope.org/datasets/euro-dem) - EuroDEM dataset.
