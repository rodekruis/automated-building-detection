from rasterio.crs import CRS
from rasterio.warp import transform_geom
from rasterio.features import rasterize
from rasterio.transform import from_bounds

import mercantile
from supermercado import burntiles
from shapely.geometry import shape, mapping

from neat_eo.tiles import tile_bbox


def geojson_parse_feature(zoom, srid, feature_map, feature, buffer=0):
    def geojson_parse_polygon(zoom, srid, feature_map, polygon):

        if isinstance(polygon["coordinates"], list):  # https://github.com/Toblerity/Shapely/issues/245
            for i, ring in enumerate(polygon["coordinates"]):  # GeoJSON coordinates could be N dimensionals
                polygon["coordinates"][i] = [[x, y] for point in ring for x, y in zip([point[0]], [point[1]])]

        if srid != 4326:
            try:
                polygon = transform_geom(CRS.from_epsg(srid), CRS.from_epsg(4326), polygon)
            except:  # negative buffer could lead to empty/invalid geom
                return feature_map

        try:
            for tile in burntiles.burn([{"type": "feature", "geometry": polygon}], zoom=zoom):
                feature_map[mercantile.Tile(*tile)].append({"type": "feature", "geometry": polygon})
        except:
            pass

        return feature_map

    def geojson_parse_geometry(zoom, srid, feature_map, geometry, buffer):
        if buffer:
            geometry = transform_geom(CRS.from_epsg(srid), CRS.from_epsg(3857), geometry)  # be sure to be planar
            geometry = mapping(shape(geometry).buffer(buffer))
            srid = 3857

        if geometry["type"] == "Polygon":
            feature_map = geojson_parse_polygon(zoom, srid, feature_map, geometry)

        elif geometry["type"] == "MultiPolygon":
            for polygon in geometry["coordinates"]:
                feature_map = geojson_parse_polygon(zoom, srid, feature_map, {"type": "Polygon", "coordinates": polygon})

        return feature_map

    if not feature or not feature["geometry"]:
        return feature_map

    if feature["geometry"]["type"] == "GeometryCollection":
        for geometry in feature["geometry"]["geometries"]:
            feature_map = geojson_parse_geometry(zoom, srid, feature_map, geometry, buffer)
    else:
        feature_map = geojson_parse_geometry(zoom, srid, feature_map, feature["geometry"], buffer)

    return feature_map


def geojson_srid(feature_collection):

    try:
        crs_mapping = {"CRS84": "4326", "900913": "3857"}
        srid = feature_collection["crs"]["properties"]["name"].split(":")[-1]
        srid = int(srid) if srid not in crs_mapping else int(crs_mapping[srid])
    except:
        srid = int(4326)

    return srid


def geojson_tile_burn(tile, features, srid, ts, burn_value=1):
    """Burn tile with GeoJSON features."""

    crs = (CRS.from_epsg(srid), CRS.from_epsg(3857))
    shapes = ((transform_geom(*crs, feature["geometry"]), burn_value) for feature in features)

    try:
        return rasterize(shapes, out_shape=ts, transform=from_bounds(*tile_bbox(tile, mercator=True), *ts))
    except:
        return None
