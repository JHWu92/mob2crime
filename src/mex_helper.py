import os

import geopandas as gp
import pandas as pd
from shapely.geometry import Point
import glob
from collections import defaultdict
import datetime
from src.creds import mex_root, mex_tower_fn
from src.utils.gis import lonlats2vor_gp, polys2polys, gp_polys_to_grids, assign_crs, clip_if_not_within
import pickle

CLAT, CLON = 19.381495, -99.139095
# source: https://epsg.io/102010
EQDC_CRS = '+proj=eqdc +lat_0=40 +lon_0=-96 +lat_1=20 +lat_2=60 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs'
AREA_CRS = 6362
REGION_KINDS = ('cities','urban_areas_16')


def stat_tw_dow_aver_hr_uniq_user(call_direction='out'):
    """return average hourly nunique users for each tower on each day of week (dow, weekday or weekend)"""
    path = f'stats/stat_tw_dow_aver_hr_uniq_user-{call_direction}.pickle'

    if os.path.exists(path):
        print('loading cached tw average', path)
        average = pickle.load(open(path, 'rb'))
        return average

    print('stats dir:', f'stats/MexTwHrUniqCnt-{call_direction}/')
    fns = sorted(glob.glob(f'stats/MexTwHrUniqCnt-{call_direction}/*-located.csv'))
    if len(fns) == 0:
        print('no file is found')
    print('loading stats by weekday or weekend')
    store = {'wd': defaultdict(list), 'wk': defaultdict(list)}
    for i, fn in enumerate(fns):
        if i % 50 == 0:
            print('loading %dth file %s' % (i, fn))
        date = os.path.basename(fn)[:10]
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        dow = 'wd' if date.weekday() < 5 else 'wk'
        tmp_df = pd.read_csv(fn, index_col=0)
        for gtid, row in tmp_df.iterrows():
            store[dow][gtid].append(row)

    print('computing average of hourly vector by weekday or weekend')
    average = {'wd': dict(), 'wk': dict()}
    for dow in ['wd', 'wk']:
        for gtid, rows in store[dow].items():
            avg_row = pd.DataFrame(rows).fillna(0).mean(axis=0)
            average[dow][gtid] = avg_row
    pickle.dump(average, open(path, 'wb'))

    return average


def tower2grid(rkind, side, redo=False, t2r_intxn_only=False):
    """

    :param rkind: region kind: cities only
    :param side: side of grids in meters
    :param redo: ignore existing t2g mapping
    :param t2r_intxn_only: keep only the intersection of the regions to compute the distribution weight.
    :return:
    """
    t2g_path = f'data/mex_tower/mex_t2g_{rkind}_{side}m.csv'

    if not redo and os.path.exists(t2g_path):
        print('reading existing t2g file:', t2g_path)
        t2g = pd.read_csv(t2g_path, index_col=0)
        return t2g

    tvor = tower_vor()
    tname = tvor.index.name

    rs = regions(rkind)
    rname = rs.index.name

    print('keep tower voronoi within', rkind, 'intersection only:', t2r_intxn_only)
    t2r = polys2polys(tvor, rs, tname, rname, cur_crs=4326, area_crs=AREA_CRS, intersection_only=t2r_intxn_only)

    gs = grids(rkind, side)

    print('building tower to grid mapping')
    t2g = []
    for n in rs.index:
        tr = t2r[t2r[rname] == n]
        gr = gs[gs[rname] == n]
        tr2gr = polys2polys(tr, gr, pname1='towerInRegion', pname2='grid', cur_crs=4326, area_crs=AREA_CRS,
                            intersection_only=True)
        tr2gr = tr2gr.merge(tr[[tname, rname, f'{tname}_area', 'weight']], left_on='towerInRegion', right_index=True)
        tr2gr.rename(columns={'weight_x': 'w_Grid2towerInRegion', 'weight_y': 'w_towerInRegion',
                              'iarea': 'gridInTowerInRegion_area'}, inplace=True)
        tr2gr['weight'] = tr2gr.w_Grid2towerInRegion * tr2gr.w_towerInRegion
        tr2gr = tr2gr[[rname, tname, 'towerInRegion', 'grid', 'weight', 'w_towerInRegion', 'w_Grid2towerInRegion',
                       'gridInTowerInRegion_area', 'towerInRegion_area', 'gtid_area', 'geometry', ]]
        t2g.append(tr2gr[[rname, tname, 'grid', 'weight']])

    t2g = pd.concat(t2g, ignore_index=True).drop(rname, axis=1)
    print('saving tower to grid mapping:', t2g_path)
    t2g.to_csv(t2g_path)
    return t2g


def tower_vor(rkind=None, intersection_only=False, in_country=True):
    in_or_not = 'in_country' if in_country else 'raw_vor'
    path = f'data/mex_tower/mex_tvor_{in_or_not}.geojson'

    if os.path.exists(path):
        tvor = gp.read_file(path)
        tvor.set_index('gtid', inplace=True)
        assign_crs(tvor, 4326, ignore_gpdf_crs=True)
        print(f'loading existing tvor file: {path}')

    else:
        t = tower()
        # voronoi polygons across mexico
        tvor = lonlats2vor_gp(t.lonlat.tolist(), dataframe=True)
        tvor['gtid'] = t.gtid

        if in_country:
            print('clipping tvor outside mexico country boarder')
            country_poly = country().geometry.values[0]
            tvor['geometry'] = tvor.geometry.apply(lambda x: clip_if_not_within(x, country_poly))

        assign_crs(tvor, 4326)
        print(f'saving tvor file: {path}')
        tvor.to_file(path, driver='GeoJSON')
        tvor.set_index('gtid', inplace=True)

    if rkind is None:
        return tvor
    else:
        rgns = regions(rkind)
        return polys2polys(tvor, rgns, tvor.index.name, rgns.index.name, cur_crs=4326, area_crs=AREA_CRS,
                           intersection_only=intersection_only)


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
    towers_shp.crs = assign_crs(towers_shp, 4326)
    return towers_shp


def grids(rkind, side, redo=False):
    """
    build grids and save it using gzip, gp.to_json;
    read grids if it exists unless redo=True

    :param rkind: which kind of regions is used, now only cities
    :param side: the side of grid in meter
    :param redo: if True, build grids regardless the existing file
    :return:
    """
    import gzip
    rgns = regions(rkind)
    rname = rgns.index.name
    grid_path = f'data/mex_grid_{rkind}_{side}m.geojson.gz'

    if not redo and os.path.exists(grid_path):
        print('reading existing grids')
        return gp.read_file(f'gzip://{grid_path}')

    print('building grids')
    g = gp_polys_to_grids(rgns, side, cur_crs=4326, eqdc_crs=EQDC_CRS, pname=rname)
    g['grid'] = g.index
    print('writing grids as gzip')
    with gzip.open(grid_path, 'wt') as fout:
        fout.write(g.to_json())
    return g


def regions(rkind='cities'):
    if rkind not in REGION_KINDS:
        raise ValueError(f'Regions kind={rkind} is not implemented, it should be one of {REGION_KINDS}')
    rgns = globals()[rkind]()
    return rgns


def cities():
    """
    It is actually metropolitan areas.
    """
    c = gp.read_file('data/cities_mexico.geojson')
    c.set_index('cname', inplace=True)
    c.index.name = 'city'
    c.crs = None
    assign_crs(c, 4326)
    return c


def urban_areas_16():
    u = gp.read_file('data/mex_16_munic_urban_merge.geojson')
    u.set_index('name', inplace=True)
    u.index.name = 'urban'
    u.crs = None
    assign_crs(u, 4326)
    return u


def states():
    return gp.read_file('data/mexico/mge2014v6_2/mge2015v6_2.shp')


def country():
    c = gp.read_file('data/mexico_country.geojson')
    c.crs = None
    assign_crs(c, 4326)
    return c
