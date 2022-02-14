import geopandas as gpd
import click
import math
from tqdm import tqdm
import os
import numpy as np

import shapely
from datetime import datetime

SPLIT_SIZE = 1000

@click.command()
@click.option('--data', help='input (vector format)')
@click.option('--dest', help='output (vector format)')
@click.option('--crsmeters', default='EPSG:4087', help='CRS in unit meters, to filter small buildings [default: EPSG:4087]')
@click.option('--waterbodies', default='', help='vector file of water bodies, to filter artifacts')
@click.option('--area', default=10, help='minimum building area, in m2 [default: 10]')
@click.option('--building_markers', default=False, type=bool, help='replace buildings with a circle marker with same area')
@click.option('--marker', default='circle', help='geometry of the building marker polygon')

def main(data, dest, crsmeters, waterbodies, area, building_markers, marker):
    """ merge touching buildings, filter small ones, simplify geometry """
    # shapely.speedups.disable()
    # start = datetime.now()
    gdf = gpd.read_file(data)
    # print(datetime.now() - start)

    # merge touching buildings
    if len(gdf)>SPLIT_SIZE and not building_markers:
        print(f'pre-processing ({len(gdf)} entries)')
        list_gdfs = [gdf[x*SPLIT_SIZE:(x+1)*SPLIT_SIZE] for x in range(math.ceil(len(gdf)/SPLIT_SIZE))]
        gdf = gpd.GeoDataFrame()
        for gdf_ in tqdm(list_gdfs):
            df_sj = gpd.sjoin(gdf_, gdf_, how='left', op='intersects')
            df_sj = df_sj.reset_index().rename(columns={'index': 'index_left'})
            num_disj = len(df_sj[df_sj['index_left'] != df_sj['index_right']])
            while num_disj > 0:
                df_sj = df_sj.dissolve(by='index_right').rename_axis(index={'index_right': 'index'})
                df_sj = df_sj.drop_duplicates(subset=['geometry'])
                df_sj = df_sj[['geometry']]
                df_sj = gpd.sjoin(df_sj, df_sj, how='left', op='intersects')
                df_sj = df_sj.reset_index().rename(columns={'index': 'index_left'})
                num_disj = len(df_sj[df_sj['index_left'] != df_sj['index_right']])
            gdf = gdf.append(df_sj.copy(), ignore_index=True)
        gdf = gdf.drop(columns=['index_left', 'index_right'])
        print(f'done ({len(gdf)} entries)')
        print(f'saving intermediate')
        gdf.to_file(dest, driver='GeoJSON')

    if not building_markers:
        print('merging all')
        df_sj = gpd.sjoin(gdf, gdf, how='left', op='intersects')
        df_sj = df_sj.reset_index().rename(columns={'index': 'index_left'})
        num_disj = len(df_sj[df_sj['index_left'] != df_sj['index_right']])
        while num_disj > 0:
            df_sj = df_sj.dissolve(by='index_right').rename_axis(index={'index_right': 'index'})
            df_sj = df_sj.drop_duplicates(subset=['geometry'])
            df_sj = df_sj[['geometry']]
            df_sj = gpd.sjoin(df_sj, df_sj, how='left', op='intersects')
            df_sj = df_sj.reset_index().rename(columns={'index': 'index_left'})
            num_disj = len(df_sj[df_sj['index_left'] != df_sj['index_right']])
        gdf = df_sj.copy()

    # filter small stuff
    crs_original = gdf.crs
    gdf = gdf.to_crs(crsmeters)
    gdf['area'] = gdf['geometry'].area
    gdf = gdf[gdf.area > area]
    gdf = gdf[['geometry']]
    # simplify geometry
    gdf = gdf.simplify(tolerance=1., preserve_topology=True)
    gdf = gdf.to_crs(crs_original)
    gdf = gpd.GeoDataFrame(geometry=gdf)
    gdf = gdf[~(gdf.geometry.is_empty | gdf.geometry.isna())]
    # filter by water bodies
    if os.path.exists(waterbodies):
        print('filtering by water bodies')
        gdf_water = gpd.read_file(waterbodies)
        if gdf.crs != gdf_water.crs:
            gdf = gdf.to_crs(gdf_water.crs)
        gdf = gpd.sjoin(gdf, gdf_water, how='left', op='intersects')
        gdf = gdf[gdf['TYPE'].isna()]
        gdf = gdf[['geometry']]

    if building_markers:
        crs_original = gdf.crs
        gdf = gdf.to_crs(crsmeters)
        if marker == 'circle':
            gdf = gdf.centroid.buffer(np.sqrt(gdf.area / np.pi))
        elif marker == 'square':
            gdf = gdf.centroid.buffer(np.sqrt(gdf.area)/2, cap_style=3)
        else:
            print("no supported markers")
        gdf = gdf.to_crs(crs_original)

    # project to WGS84 and save
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    gdf.to_file(dest, driver='GeoJSON')

if __name__ == "__main__":
    main()
