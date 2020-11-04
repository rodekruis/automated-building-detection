import math
from shapely.geometry import box, mapping
import fiona


class Tile():

    def __init__(self, xmin=None, ymin=None, xmax=None, ymax=None, x=None, y=None, z=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        if self.is_set():
            bbox = '%s, %s, %s, %s' % (self.xmin, self.ymin, self.xmax, self.ymax)
        else:
            bbox = 'not set'
        return 'Tile[bbox: %s, x: %s, y: %s, z: %s]' % (bbox, self.x, self.y, self.z)

    def is_set(self):
        return self.xmin is not None and self.ymin is not None and self.xmax is not None and self.ymax is not None

    def get_geometry(self):
        return box(self.xmin, self.ymin, self.xmax, self.ymax)

    def get_feature(self):
        pass


class TileCollection(list):

    def __init__(self):
        self.geom = None
        self.extent = None

    def __str__(self):
        return 'TileCollection[tiles: %s]' % len(self)

    def generate_tiles(self, geom, z):
        self.geom = geom
        self.extent = geom.bounds

        from_tile = self.deg2tile(self.extent[0], self.extent[1], z)
        to_tile = self.deg2tile(self.extent[2], self.extent[3], z)
        x_start = min(from_tile[0], to_tile[0])
        x_end = max(from_tile[0], to_tile[0])
        y_start = min(from_tile[1], to_tile[1])
        y_end = max(from_tile[1], to_tile[1])

        for x in range(x_start, x_end + 1):
            for y in range(y_start, y_end + 1):
                t = self.tileGeometry(x, y, z)
                if t.get_geometry().intersects(geom):
                    self.append(t)

    def export_shapefile(self, filename):
        if len(self) < 1:
            print('no tiles to save')
            return

        schema = {
            'geometry': 'Polygon',
            'properties': {
                'id': 'int',
                'x': 'int',
                'y': 'int',
                'z': 'int'
            },
        }

        with fiona.open(filename, 'w', 'ESRI Shapefile', schema) as c:
            tile_count = 1
            for t in self:
                geom = t.get_geometry()
                c.write({
                    'geometry': mapping(geom),
                    'properties': {
                        'id': tile_count,
                        'x': t.x,
                        'y': t.y,
                        'z': t.z
                    },
                })
                tile_count += 1

    def export_geometry_shapefile(self, filename):
        if self.geom is None:
            print('no tiles to save')
            return

        schema = {
            'geometry': 'Polygon',
            'properties': {'id': 'int'},
        }

        with fiona.open(filename, 'w', 'ESRI Shapefile', schema) as c:
            c.write({
                'geometry': mapping(self.geom),
                'properties': {'id': 0},
            })

    def deg2tile(self, lat_deg, lon_deg, zoom):
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n)

        return (xtile, ytile)

    def tileGeometry(self, x, y, z):
        n = 2.0 ** z
        ymin = x / n * 360.0 - 180.0
        ymax = (x + 1) / n * 360.0 - 180
        xmin = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
        xmax = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
        return Tile(xmin, ymin, xmax, ymax, x, y, z)