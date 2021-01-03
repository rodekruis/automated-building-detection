import json
import overpy
api = overpy.Overpass()
import geopandas as gpd
from shapely.geometry import Polygon
import click
import re
from tqdm import tqdm


def select_non_overlapping_ml(gdf_ml, gdf_osm):
    """select osm buildings non-overlapping with ml predictions"""
    gdf_osm_buffered = gdf_osm.copy()
    gdf_osm_buffered['geometry'] = gdf_osm_buffered.buffer(1)
    df_sj = gpd.sjoin(gdf_ml, gdf_osm_buffered, how='left', op='intersects')
    if '@osmId' in df_sj.columns:
        df_sj = df_sj[df_sj['@osmId'].isna()]
    elif 'OBJECTID' in df_sj.columns:
        df_sj = df_sj[df_sj['OBJECTID'].isna()]
    else:
        print('WARNING: MISSING OSM INDEX, RETURNING ALL')
    return df_sj


@click.command()
@click.option('--aoi', help='input Area of Interest (geojson)')
@click.option('--ml', help='input ML predictions (geojson)')
@click.option('--dest', help='output ML predictions + OSM (geojson)')
def main(aoi, ml, dest):
    """download OSM data in the AOI and add to ML predictions"""

    with open(aoi) as f:
        data = json.load(f)

    coords = data['features'][0]['geometry']['coordinates'][0]
    geom = Polygon([(x[1], x[0]) for x in coords])
    print(geom)

    # convert the bounding box into string for query
    bbounds = geom.bounds
    bbox_query = "s=\"" + str(bbounds[0]) + "\" w=\"" + str(bbounds[1]) + "\" n=\"" + str(
        bbounds[2]) + "\" e=\"" + str(bbounds[3]) + "\""
    # call API
    print('call API:')
    r = api.query("""
    <osm-script output="json">
        <query type="way">
          <has-kv k="building"/>
          <bbox-query """ + bbox_query + """/>
        </query>
      <print mode="body"/>
      <recurse type="down"/>
      <print mode="skeleton"/>
    </osm-script>
    """)

    build_osm = gpd.GeoDataFrame()
    build_osm['geometry'] = None
    index_start = len(build_osm)
    index_g = 0
    id = re.compile(r'id=([0-9]+)')

    print('transform to GeoDataframe')
    for way in tqdm(r.ways):
        nodes = way.get_nodes(resolve_missing=True)
        coordinates = []
        for node in nodes:
            coordinates.append([float(node.lon), float(node.lat)])

        if coordinates[0] == coordinates[-1]:
            geom = Polygon(coordinates)
            build_osm.at[index_start + index_g, 'geometry'] = geom
            build_osm.at[index_start + index_g, 'OBJECTID'] = id.findall(str(way).strip())[0]
            index_g += 1
    build_osm = build_osm[~(build_osm.geometry.is_empty | build_osm.geometry.isna())]
    build_osm = build_osm.set_crs("EPSG:4326")

    # add to ml predictions
    build_ml = gpd.read_file(ml)
    build_ml = build_ml.to_crs("EPSG:8857")
    build_osm = build_osm.to_crs("EPSG:8857")
    build_ml_non_overlap = select_non_overlapping_ml(build_ml, build_osm)
    gdf = build_osm.copy()
    gdf = gdf.append(build_ml_non_overlap, ignore_index=True)
    print(f'ml: {len(build_ml)}, osm: {len(build_osm)}, combo: {len(gdf)}')

    # reproject to WGS84 and save
    gdf = gdf.to_crs("EPSG:4326")
    gdf.to_file(dest, driver='GeoJSON')



if __name__ == "__main__":
    main()