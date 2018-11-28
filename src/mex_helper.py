import geopandas as gp
import pandas as pd
from shapely.geometry import Point

from src.creds import mex_root, mex_tower_fn


def tower():
    tower_info_path = mex_root + mex_tower_fn
    towers = pd.read_csv(tower_info_path, header=None, sep='|')
    towers['lonlat'] = towers.apply(lambda x: '%.6f' % (x[2]) + ',' + '%.6f' % (x[3]), axis=1)

    # group towers with same location
    towers_gp = towers.groupby('lonlat')[0].apply(list).to_frame()
    towers_gp['gtid'] = towers_gp[0].apply(lambda x: '-'.join(x))
    # get group id
    # gt2loc = {row['gtid']: loc.split(',') for loc, row in towers_gp.iterrows()}
    # t2gt = {}
    # for _, row in towers_gp.iterrows():
    #     for tid in row[0]:
    #         t2gt[tid] = row['gtid']

    towers_shp = towers_gp.reset_index()
    towers_shp['lonlat'] = towers_shp.lonlat.apply(lambda x: eval(x))
    towers_shp['geometry'] = towers_shp.lonlat.apply(lambda x: Point(x))
    towers_shp = gp.GeoDataFrame(towers_shp)
    towers_shp.crs = {'init': 'epsg:4326'}
    return towers_shp


def cities():
    c = gp.read_file('data/cities_mexico.geojson')
    c.set_index('cname', inplace=True)
    return c
