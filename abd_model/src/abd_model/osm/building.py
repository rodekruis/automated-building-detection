import sys
import osmium
import geojson
import shapely.geometry


class BuildingHandler(osmium.SimpleHandler):
    """Extracts building polygon features"""

    def __init__(self):
        super().__init__()
        self.features = []

    def way(self, w):
        if not w.is_closed() or len(w.nodes) < 4:
            return

        if not list(set(["building", "construction"]) & set([k for k in dict(w.tags).keys()])):
            return

        if "building" in w.tags and w.tags["building"] in set(
            ["houseboat", "static_caravan", "stadium", "digester", "ruins"]
        ):
            return

        if "location" in w.tags and w.tags["location"] in set(["underground", "underwater"]):
            return

        geometry = geojson.Polygon([[(n.lon, n.lat) for n in w.nodes]])
        shape = shapely.geometry.shape(geometry)

        if shape.is_valid:
            feature = geojson.Feature(geometry=geometry)
            self.features.append(feature)
        else:
            print("Warning: invalid feature: https://www.openstreetmap.org/way/{}".format(w.id), file=sys.stderr)

    def save(self, out):
        collection = geojson.FeatureCollection(self.features)

        with open(out, "w") as fp:
            geojson.dump(collection, fp)
