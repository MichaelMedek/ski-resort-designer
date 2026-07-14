"""Unit tests for the terrain basemap layer (ui/terrain_layer.py).

The 3D basemap is a single pydeck TerrainLayer built from free AWS Terrarium elevation tiles with an
OpenTopoMap texture; the 2D basemap is a Mapbox-GL style dict. These tests pin the layer's wiring
(tile URLs, elevation decoder, non-pickable) and the style spec, since a silent change to a tile URL
or the decoder scalers renders a blank or mis-elevated map with no error.
"""

from skiresort_planner.ui import terrain_layer as tl


class TestAwsTerrainLayer:
    def test_builds_a_terrain_layer_with_aws_tiles_and_texture(self) -> None:
        layer = tl.create_aws_terrain_layer()
        assert layer.type == "TerrainLayer"
        assert layer.id == "terrain_3d_aws"
        # Elevation mesh comes from AWS Terrarium tiles; texture from OpenTopoMap.
        assert layer.elevation_data == tl.AWS_TERRAIN_TILES
        assert layer.texture == tl.OPENTOPOMAP_TILES
        assert layer.elevation_decoder == tl.AWS_ELEVATION_DECODER

    def test_layer_is_not_pickable(self) -> None:
        # The basemap must not steal clicks — terrain/marker picking is handled by other layers.
        layer = tl.create_aws_terrain_layer()
        assert layer.pickable is False

    def test_mesh_max_error_is_forwarded(self) -> None:
        # Lower error = more precise picking; the caller's value must reach the layer verbatim.
        default = tl.create_aws_terrain_layer()
        assert default.mesh_max_error == 1.0
        precise = tl.create_aws_terrain_layer(mesh_max_error=0.25)
        assert precise.mesh_max_error == 0.25


class TestTerrariumDecoder:
    def test_decoder_matches_the_terrarium_formula(self) -> None:
        # Terrarium encodes elevation as (R*256 + G + B/256) - 32768. The decoder scalers must match
        # exactly or the whole mesh is offset/scaled wrong.
        d = tl.AWS_ELEVATION_DECODER
        assert d["rScaler"] == 256
        assert d["gScaler"] == 1
        assert d["bScaler"] == 1 / 256
        assert d["offset"] == -32768


class TestOpenTopoMapStyle:
    def test_style_is_a_valid_v8_raster_source(self) -> None:
        style = tl.OPENTOPOMAP_STYLE
        assert style["version"] == 8
        source = style["sources"]["opentopomap"]  # type: ignore[index]
        assert source["type"] == "raster"
        assert source["tiles"] == tl.OPENTOPOMAP_TILES_ABC
        assert source["tileSize"] == 256

    def test_style_layer_references_its_source_within_zoom_limits(self) -> None:
        layer = tl.OPENTOPOMAP_STYLE["layers"][0]  # type: ignore[index]
        assert layer["type"] == "raster"
        assert layer["source"] == "opentopomap", "the render layer must point at the declared source"
        assert layer["minzoom"] == 0 and layer["maxzoom"] == 17

    def test_abc_subdomains_are_distinct_opentopomap_hosts(self) -> None:
        # Three distinct subdomains enable parallel tile loading; a copy-paste bug collapsing them
        # would serialize all fetches through one host.
        hosts = tl.OPENTOPOMAP_TILES_ABC
        assert len(hosts) == 3
        assert len({h.split("//")[1].split(".")[0] for h in hosts}) == 3, "a/b/c subdomains must differ"
        assert all(h.endswith("/{z}/{x}/{y}.png") for h in hosts)
