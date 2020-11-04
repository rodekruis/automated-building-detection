"""Slippy Map Tiles.
   See: https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
"""

import io
import os
import re
import glob
import warnings

import numpy as np
from PIL import Image
from rasterio import open as rasterio_open
import cv2

import json
import psycopg2
import rasterio
import mercantile
import supermercado

warnings.simplefilter("ignore", UserWarning)  # To prevent rasterio NotGeoreferencedWarning


def tile_pixel_to_location(tile, dx, dy):
    """Converts a pixel in a tile to lon/lat coordinates."""

    assert 0 <= dx <= 1 and 0 <= dy <= 1, "x and y offsets must be in [0, 1]"

    w, s, e, n = mercantile.bounds(tile)

    def lerp(a, b, c):
        return a + c * (b - a)

    return lerp(w, e, dx), lerp(s, n, dy)  # lon, lat


def tiles_from_csv(path, xyz=True, extra_columns=False):
    """Retrieve tiles from a line-delimited csv file."""

    assert os.path.isfile(os.path.expanduser(path)), "'{}' seems not a valid CSV file".format(path)
    with open(os.path.expanduser(path)) as fp:

        for row in fp:
            row = row.replace("\n", "")
            if not row:
                continue

            row = re.split(",|\t", row)  # use either comma or tab as separator
            if xyz:
                assert len(row) >= 3, "Invalid Cover"
                if not extra_columns or len(row) == 3:
                    yield mercantile.Tile(int(row[0]), int(row[1]), int(row[2]))
                else:
                    yield [mercantile.Tile(int(row[0]), int(row[1]), int(row[2])), *map(float, row[3:])]

            if not xyz:
                assert len(row) >= 1, "Invalid Cover"
                if not extra_columns:
                    yield row[0]
                else:
                    yield [row[0], *map(float, row[1:])]


def tiles_from_dir(root, cover=None, xyz=True, xyz_path=False):
    """Loads files from an on-disk dir."""
    root = os.path.expanduser(root)

    if xyz is True:
        paths = glob.glob(os.path.join(root, "[0-9]*/[0-9]*/[0-9]*.*"))

        for path in paths:
            tile_xyz = re.match(os.path.join(root, "(?P<z>[0-9]+)/(?P<x>[0-9]+)/(?P<y>[0-9]+).+"), path)
            if not tile_xyz:
                continue
            tile = mercantile.Tile(int(tile_xyz["x"]), int(tile_xyz["y"]), int(tile_xyz["z"]))

            if cover is not None and tile not in cover:
                continue

            if xyz_path is True:
                yield tile, path
            else:
                yield tile

    else:
        paths = glob.glob(root, "**/*.*", recursive=True)

        for path in paths:
            return path


def tile_from_xyz(root, x, y, z):
    """Retrieve a single tile from a slippy map dir."""

    path = glob.glob(os.path.join(os.path.expanduser(root), str(z), str(x), str(y) + ".*"))
    if not path:
        return None

    assert len(path) == 1, "ambiguous tile path"

    return mercantile.Tile(x, y, z), path[0]


def tile_bbox(tile, mercator=False):

    if isinstance(tile, mercantile.Tile):
        if mercator:
            return mercantile.xy_bounds(tile)  # EPSG:3857
        else:
            return mercantile.bounds(tile)  # EPSG:4326

    else:
        with open(rasterio_open(tile)) as r:

            if mercator:
                w, s, e, n = r.bounds
                w, s = mercantile.xy(w, s)
                e, n = mercantile.xy(e, n)
                return w, s, e, n  # EPSG:3857
            else:
                return r.bounds  # EPSG:4326

        assert False, "Unable to open tile"


def tiles_to_geojson(tiles, union=True):
    """Convert tiles to their footprint GeoJSON."""

    first = True
    geojson = '{"type":"FeatureCollection","features":['

    if union:  # smaller tiles union geometries (but losing properties)
        tiles = [str(tile.z) + "-" + str(tile.x) + "-" + str(tile.y) + "\n" for tile in tiles]
        for feature in supermercado.uniontiles.union(tiles, True):
            geojson += json.dumps(feature) if first else "," + json.dumps(feature)
            first = False
    else:  # keep each tile geometry and properties (but fat)
        for tile in tiles:
            prop = '"properties":{{"x":{},"y":{},"z":{}}}'.format(tile.x, tile.y, tile.z)
            geom = '"geometry":{}'.format(json.dumps(mercantile.feature(tile, precision=6)["geometry"]))
            geojson += '{}{{"type":"Feature",{},{}}}'.format("," if not first else "", geom, prop)
            first = False

    geojson += "]}"
    return geojson


def tiles_to_granules(tiles, pg):
    """Retrieve Intersecting Sentinel Granules from tiles."""

    conn = psycopg2.connect(pg)
    db = conn.cursor()
    assert db

    granules = set()
    tiles = [str(tile.z) + "-" + str(tile.x) + "-" + str(tile.y) + "\n" for tile in tiles]
    for feature in supermercado.uniontiles.union(tiles, True):
        geom = json.dumps(feature["geometry"])
        query = """SELECT id FROM neo.s2_granules
                   WHERE ST_Intersects(geom, ST_SetSRID(ST_GeomFromGeoJSON('{}'), 4326))""".format(
            geom
        )
        db.execute(query)
        granules.update(db.fetchone()[:])

    return granules


def tile_image_from_file(path, bands=None, force_rgb=False):
    """Return a multiband image numpy array, from an image file path, or None."""

    try:
        if path[-3:] == "png" and force_rgb:  # PIL PNG Color Palette handling
            return np.array(Image.open(os.path.expanduser(path)).convert("RGB"))
        elif path[-3:] == "png":
            return np.array(Image.open(os.path.expanduser(path)))
        else:
            raster = rasterio_open(os.path.expanduser(path))
    except:
        return None

    image = None
    for i in raster.indexes if bands is None else bands:
        data_band = raster.read(i)
        data_band = data_band.reshape(data_band.shape[0], data_band.shape[1], 1)  # H,W -> H,W,C
        image = np.concatenate((image, data_band), axis=2) if image is not None else data_band

    assert image is not None, "Unable to open {}".format(path)
    return image


def tile_image_to_file(root, tile, image, ext=None):
    """ Write an image tile on disk. """

    H, W, C = image.shape

    root = os.path.expanduser(root)
    path = os.path.join(root, str(tile.z), str(tile.x)) if isinstance(tile, mercantile.Tile) else root
    os.makedirs(path, exist_ok=True)

    if C == 1:
        ext = "png"
    elif C == 3:
        ext = ext if ext is not None else "webp"  # allow to switch to jpeg (for old browser)
    else:
        ext = "tiff"

    if isinstance(tile, mercantile.Tile):
        path = os.path.join(path, "{}.{}".format(str(tile.y), ext))
    else:
        path = os.path.join(path, "{}.{}".format(tile, ext))

    try:
        if C == 1:
            Image.fromarray(image.reshape(H, W), mode="L").save(path)
        elif C == 3:
            cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            rasterio.open(path, "w", driver="GTiff", compress="lzw", height=H, width=W, count=C, dtype=image.dtype).write(
                np.moveaxis(image, 2, 0)  # H,W,C -> C,H,W
            )
    except:
        assert False, "Unable to write {}".format(path)


def tile_label_from_file(path, silent=True):
    """Return a numpy array, from a label file path, or None."""

    try:
        return np.array(Image.open(os.path.expanduser(path))).astype(int)
    except:
        assert silent, "Unable to open existing label: {}".format(path)


def tile_label_to_file(root, tile, palette, transparency, label, append=False, margin=0):
    """ Write a label (or a mask) tile on disk. """

    root = os.path.expanduser(root)
    dir_path = os.path.join(root, str(tile.z), str(tile.x)) if isinstance(tile, mercantile.Tile) else root
    path = os.path.join(dir_path, "{}.png".format(str(tile.y)))

    if len(label.shape) == 3:  # H,W,C -> H,W
        assert label.shape[2] == 1
        label = label.reshape((label.shape[0], label.shape[1]))

    if append and os.path.isfile(path):
        previous = tile_label_from_file(path, silent=False)
        label = np.uint8(np.maximum(previous, label))
    else:
        os.makedirs(dir_path, exist_ok=True)

    try:
        out = Image.fromarray(label, mode="P")
        out.putpalette(palette)
        if transparency is not None:
            out.save(path, optimize=True, transparency=transparency)
        else:
            out.save(path, optimize=True)
    except:
        assert False, "Unable to write {}".format(path)


def tile_image_from_url(requests_session, url, timeout=10):
    """Fetch a tile image using HTTP, and return it or None """

    try:
        resp = requests_session.get(url, timeout=timeout)
        resp.raise_for_status()
        image = np.fromstring(io.BytesIO(resp.content).read(), np.uint8)
        return cv2.cvtColor(cv2.imdecode(image, cv2.IMREAD_ANYCOLOR), cv2.COLOR_BGR2RGB)

    except Exception:
        return None


def tile_is_neighboured(tile, tiles):
    """Check if a tile is surrounded by others tiles"""

    tiles = dict(tiles)
    try:
        # 3x3 matrix (upper, center, bottom) x (left, center, right)
        tiles[mercantile.Tile(x=int(tile.x) - 1, y=int(tile.y) - 1, z=int(tile.z))]  # ul
        tiles[mercantile.Tile(x=int(tile.x) + 0, y=int(tile.y) - 1, z=int(tile.z))]  # uc
        tiles[mercantile.Tile(x=int(tile.x) + 1, y=int(tile.y) - 1, z=int(tile.z))]  # ur
        tiles[mercantile.Tile(x=int(tile.x) - 1, y=int(tile.y) + 0, z=int(tile.z))]  # cl
        tiles[mercantile.Tile(x=int(tile.x) + 1, y=int(tile.y) + 0, z=int(tile.z))]  # cr
        tiles[mercantile.Tile(x=int(tile.x) - 1, y=int(tile.y) + 1, z=int(tile.z))]  # bl
        tiles[mercantile.Tile(x=int(tile.x) + 0, y=int(tile.y) + 1, z=int(tile.z))]  # bc
        tiles[mercantile.Tile(x=int(tile.x) + 1, y=int(tile.y) + 1, z=int(tile.z))]  # br
    except KeyError:
        return False

    return True


def tile_image_buffer(tile, tiles, bands):
    """Buffers a tile image adding borders on all sides based on adjacent tile, or zeros padded if not possible."""

    def tile_image_neighbour(tile, dx, dy, tiles, bands):
        """Retrieves neighbour tile image if exists."""
        try:
            path = tiles[mercantile.Tile(x=int(tile.x) + dx, y=int(tile.y) + dy, z=int(tile.z))]
        except KeyError:
            return None

        return tile_image_from_file(path, bands)

    tiles = dict(tiles)
    # 3x3 matrix (upper, center, bottom) x (left, center, right)
    ul = tile_image_neighbour(tile, -1, -1, tiles, bands)
    uc = tile_image_neighbour(tile, +0, -1, tiles, bands)
    ur = tile_image_neighbour(tile, +1, -1, tiles, bands)
    cl = tile_image_neighbour(tile, -1, +0, tiles, bands)
    cc = tile_image_neighbour(tile, +0, +0, tiles, bands)
    cr = tile_image_neighbour(tile, +1, +0, tiles, bands)
    bl = tile_image_neighbour(tile, -1, +1, tiles, bands)
    bc = tile_image_neighbour(tile, +0, +1, tiles, bands)
    br = tile_image_neighbour(tile, +1, +1, tiles, bands)

    b = len(bands)
    ts = cc.shape[1]
    o = int(ts / 4)
    oo = o * 2

    img = np.zeros((ts + oo, ts + oo, len(bands))).astype(np.uint8)

    # fmt:off
    img[0:o,        0:o,        :] = ul[-o:ts, -o:ts, :] if ul is not None else np.zeros((o,   o, b)).astype(np.uint8)
    img[0:o,        o:ts+o,     :] = uc[-o:ts,  0:ts, :] if uc is not None else np.zeros((o,  ts, b)).astype(np.uint8)
    img[0:o,        ts+o:ts+oo, :] = ur[-o:ts,   0:o, :] if ur is not None else np.zeros((o,   o, b)).astype(np.uint8)
    img[o:ts+o,     0:o,        :] = cl[0:ts,  -o:ts, :] if cl is not None else np.zeros((ts,  o, b)).astype(np.uint8)
    img[o:ts+o,     o:ts+o,     :] = cc                  if cc is not None else np.zeros((ts, ts, b)).astype(np.uint8)
    img[o:ts+o,     ts+o:ts+oo, :] = cr[0:ts,    0:o, :] if cr is not None else np.zeros((ts,  o, b)).astype(np.uint8)
    img[ts+o:ts+oo, 0:o,        :] = bl[0:o,   -o:ts, :] if bl is not None else np.zeros((o,   o, b)).astype(np.uint8)
    img[ts+o:ts+oo, o:ts+o,     :] = bc[0:o,    0:ts, :] if bc is not None else np.zeros((o,  ts, b)).astype(np.uint8)
    img[ts+o:ts+oo, ts+o:ts+oo, :] = br[0:o,     0:o, :] if br is not None else np.zeros((o,   o, b)).astype(np.uint8)
    # fmt:on

    return img
