"""Terrain visualization for ski resort maps using free OpenTopoMap tiles.

Provides two basemap modes:

3D Mode (TerrainLayer):
    Uses AWS Terrarium elevation tiles for 3D mesh with OpenTopoMap texture.
    The TerrainLayer handles both elevation data and texture rendering internally.

2D Mode (OPENTOPOMAP_STYLE dict):
    Uses Mapbox GL style specification to define a custom raster basemap.
    This is the official deck.gl approach for XYZ raster tiles because pydeck's
    TileLayer alone only fetches tiles but doesn't render them - it requires a
    renderSubLayers callback which pydeck doesn't expose to Python.

    The style dict defines:
    - sources: Where to fetch tiles (OpenTopoMap a/b/c subdomains)
    - layers: How to render them (as raster with zoom limits)

No API key required - uses free OpenTopoMap (CC-BY-SA) and AWS tiles.
"""

import logging

import pydeck as pdk

logger = logging.getLogger(__name__)

# AWS Terrain Tiles (free, open, no API key)
AWS_TERRAIN_TILES = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

# Terrarium decoder for AWS tiles
AWS_ELEVATION_DECODER = {
    "rScaler": 256,
    "gScaler": 1,
    "bScaler": 1 / 256,
    "offset": -32768,
}

# OpenTopoMap - best free topographic map with contour lines (no API key needed)
# Attribution: © OpenStreetMap contributors, OpenTopoMap
# Use explicit subdomain 'a' - pydeck TileLayer doesn't support {s} placeholder
OPENTOPOMAP_TILES = "https://a.tile.opentopomap.org/{z}/{x}/{y}.png"

# Alternative: Multiple subdomains for load balancing
OPENTOPOMAP_TILES_ABC = [
    "https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
    "https://b.tile.opentopomap.org/{z}/{x}/{y}.png",
    "https://c.tile.opentopomap.org/{z}/{x}/{y}.png",
]

# Standard OpenStreetMap tiles (simpler, more reliable)
OSM_TILES = "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"


def create_aws_terrain_layer(mesh_max_error: float = 1.0) -> pdk.Layer:
    """Create a 3D TerrainLayer using free AWS terrain tiles with OpenTopoMap texture.

    Uses AWS S3 hosted Terrarium elevation tiles with OpenTopoMap texture.
    OpenTopoMap shows contour lines and terrain features - ideal for ski planning.
    No API key required.

    Note: TerrainLayer picking has limitations - clicks may hit z=0 plane instead
    of actual terrain surface. Use mesh_max_error to improve precision.

    Args:
        mesh_max_error: Mesh approximation error in meters. Lower = more precise
            picking but slower rendering. Default 1.0 for high precision.

    Returns:
        pdk.Layer with global 3D terrain mesh
    """
    return pdk.Layer(
        "TerrainLayer",
        elevation_data=AWS_TERRAIN_TILES,
        elevation_decoder=AWS_ELEVATION_DECODER,
        texture=OPENTOPOMAP_TILES,
        mesh_max_error=mesh_max_error,
        id="terrain_3d_aws",
        pickable=False,
    )


# Mapbox GL style specification for 2D raster basemap.
#
# Why map_style dict instead of TileLayer?
# - pydeck's TileLayer is a "tile manager" that fetches tiles but doesn't render them
# - It requires a renderSubLayers callback (JavaScript) to create BitmapLayers per tile
# - pydeck doesn't expose renderSubLayers to Python, so TileLayer shows blank/white
# - The same OpenTopoMap URL works in TerrainLayer because it handles texturing internally
#
# The map_style dict approach:
# - Follows Mapbox GL style spec (same format as pdk.map_styles.CARTO_ROAD etc.)
# - deck.gl natively understands this format and renders raster tiles correctly
# - Requires map_provider="mapbox" in pdk.Deck() (works without API key for raster)
# - Multiple tile subdomains (a/b/c) enable parallel loading for better performance
OPENTOPOMAP_STYLE: dict[str, object] = {
    "version": 8,
    "sources": {
        "opentopomap": {
            "type": "raster",
            "tiles": OPENTOPOMAP_TILES_ABC,
            "tileSize": 256,
            "attribution": (
                '© <a href="https://www.opentopomap.org/">OpenTopoMap</a> '
                '(<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)'
            ),
        }
    },
    "layers": [
        {
            "id": "opentopomap",
            "type": "raster",
            "source": "opentopomap",
            "minzoom": 0,
            "maxzoom": 17,
        }
    ],
}
