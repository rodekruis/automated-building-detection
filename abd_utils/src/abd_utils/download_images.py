# -*- coding: utf-8 -*-
from urllib.request import urlretrieve
from pygeotile.tile import Tile as pyTile
from tqdm import tqdm
import os
import csv
import urllib
import click
import pandas as pd
import geopandas as gpd
from .tiles import Tile, TileCollection
from shapely.geometry import box
import subprocess
from dotenv import load_dotenv
load_dotenv()


def quadkey_to_url(quadKey, api_key):
    # Read textfile with Bing Maps API key
    # See: https://msdn.microsoft.com/en-us/library/ff428642.aspx
    tile_url = ("http://a0.ortho.tiles.virtualearth.net/tiles/a{}.jpeg?"
                "g=854&mkt=en-US&token={}".format(quadKey, api_key))

    return tile_url


def read_tiles(csv_filename):
    with open(csv_filename) as csvfile:
        reader = csv.reader(csvfile)
        list_x, list_y, list_boundaries = [], [], []
        zoom = next(reader, None)[6]
        print("Reading rows:")
        for row in tqdm(reader):
            list_x.append(row[0])
            list_y.append(row[1])
            list_boundaries.append(row[2:6])

        tilelist = zip(list_x, list_y, list_boundaries)
    tilelist = list(tilelist)
    return tilelist, zoom


def retrieve_bing_image_old_api(tile_url, image_name):
    tile_url = urllib.parse.quote(tile_url, ':/&=-?').replace("%EF%BB%BF",
                                                              "")  # Convert encoding from ascii to utf8 so that urllib accepts it
    with urllib.request.urlopen(tile_url) as response:
        image = open(image_name, 'wb')
        image.write(response.read())
        image.close()
    if os.path.getsize(image_name) <= 1033:
        os.remove(image_name)
        return False
    else:
        return True


def process_print(command_args):
    process = subprocess.Popen(command_args, stdout=subprocess.PIPE)
    stdout = process.communicate()[0]
    print('{}'.format(stdout))


@click.command()
@click.option('--aoi', help='area of interest (vector format)')
@click.option('--output', help='output directory')
@click.option('--zoom', default=17, help='zoom level [default: 17]')
def main(aoi, output, zoom):
    """ download tiled images from Bing Maps API in a given AOI """

    # read AOI file
    gdf_aoi = gpd.read_file(aoi).iloc[0]
    aoi_bounds = gdf_aoi['geometry'].bounds

    # create output directories
    os.makedirs(output, exist_ok=True)
    outdir_img = os.path.join(output, 'images')
    os.makedirs(outdir_img, exist_ok=True)

    # create tile collection
    print("Creating tiles")
    tc = TileCollection()
    geom_bounds = aoi_bounds
    geom_box = box(geom_bounds[1], geom_bounds[0], geom_bounds[3], geom_bounds[2])
    tc.generate_tiles(geom_box, zoom)
    coords = []
    for tile in tc:
        coords.append([tile.x, tile.y, tile.xmin, tile.ymin, tile.xmax, tile.ymax])

    print("Number of tiles created:")
    print(len(tc))

    # save it in csv
    df_tiles = pd.DataFrame.from_records(coords)
    df_tiles.columns = ['x', 'y', 'x_min', 'y_min', 'x_max', 'y_max']
    df_tiles['zoom'] = zoom
    df_tiles['xc'] = (df_tiles['x_min'] + df_tiles['x_max']) / 2.
    df_tiles['yc'] = (df_tiles['y_min'] + df_tiles['y_max']) / 2.
    df_tiles.to_csv(os.path.join(output, 'coords.csv'))

    # loop over tile collection and download images
    print("Downloading tile images:")
    previous_quadKey = ""  # fixes the problem of overlapping tiles later on
    for ix, tile in tqdm(df_tiles.iterrows()):
        tile_center_x = int(tile['x'])
        tile_center_y = int(tile['y'])
        boundaries = [tile['x_min'], tile['y_min'], tile['x_max'], tile['y_max']]
        zoom_level = tile['zoom']

        # Obtain quadkey
        coord_center_lat = (float(boundaries[0]) + float(boundaries[2])) / 2.0
        coord_center_lon = (float(boundaries[1]) + float(boundaries[3])) / 2.0
        tile_obj = pyTile.for_latitude_longitude(coord_center_lat, coord_center_lon, int(zoom_level))
        quadKey = tile_obj.quad_tree
        if quadKey == previous_quadKey:
            # Tile is already downloaded because it has the same quadKey
            continue
        else:
            previous_quadKey = quadKey

        # Get Tile URL
        tile_url = quadkey_to_url(quadKey, os.environ.get("BING_API_KEY"))

        # Retrieve image from old API
        image_name = os.path.join(outdir_img, f"{int(zoom_level)}.{tile_center_x}.{tile_center_y}.png")
        result = retrieve_bing_image_old_api(tile_url, image_name)
        if not result:
            print(f'WARNING: failed to download image {image_name}, check AOI')


if __name__ == "__main__":
    main()
