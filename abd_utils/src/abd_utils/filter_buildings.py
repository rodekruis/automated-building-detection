import geopandas as gpd
import click


@click.command()
@click.option('--data', help='input (vector format)')
@click.option('--dest', help='output (vector format)')
@click.option('--crsmeters', default='EPSG:4087', help='CRS in unit meters, to filter small buildings [default: EPSG:4087]')
@click.option('--area', default=10, help='minimum building area, in m2 [default: 10]')
def main(data, dest, crsmeters, area):

    gdf = gpd.read_file(data)
    crs_original = gdf.crs

    # merge touching buildings
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

    # convert to CRS with unit meters
    gdf = gdf.to_crs(crsmeters)
    gdf = gdf[['geometry']]

    # filter small stuff
    gdf['area'] = gdf['geometry'].area
    gdf = gdf[gdf.area > area]
    gdf = gdf[['geometry']]

    # simplify geometry
    gdf = gdf.simplify(tolerance=1., preserve_topology=True)

    # reproject to original CRS and save
    gdf = gdf.to_crs(crs_original)
    gdf.to_file(dest, driver='GeoJSON')


if __name__ == "__main__":
    main()
