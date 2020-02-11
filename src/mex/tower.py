import sys

sys.path.insert(0, '../../')

import os
# if not os.getcwd().endswith('mob2crime'):
#     os.chdir('..')

from src.creds import mex_root, mex_tower_fn
import src.mex as mex
import src.mex.regions2010 as region
import pandas as pd
import src.utils.gis as gis
import geopandas as gp
from shapely.geometry import Point

DIR_INTPL = 'data/mex_tw_intpl'


def pts(to_4326=False):
    tower_info_path = mex_root + mex_tower_fn
    towers = pd.read_csv(tower_info_path, header=None, sep='|')
    towers['lonlat'] = towers.apply(lambda x: '%.6f' % (x[2]) + ',' + '%.6f' % (x[3]), axis=1)

    # group towers with same location
    towers_gp = towers.groupby('lonlat')[0].apply(list).to_frame()
    towers_gp['gtid'] = towers_gp[0].apply(lambda x: '-'.join(x))

    towers_shp = towers_gp.reset_index()
    towers_shp['lonlat'] = towers_shp.lonlat.apply(lambda x: eval(x))
    towers_shp['geometry'] = towers_shp.lonlat.apply(lambda x: Point(x))
    towers_shp = gp.GeoDataFrame(towers_shp)
    gis.assign_crs(towers_shp, 4326)
    if not to_4326:
        towers_shp = towers_shp.to_crs(mex.crs)
    return towers_shp


def voronoi(to_4326=False, load_pop=False):
    fn = f'{DIR_INTPL}/voronoi.geojson'

    if os.path.exists(fn):
        tvor = gp.read_file(fn)
        print(f'loading existing mexico tower voronoi file: {fn}')

    else:
        t = pts(to_4326=False)
        tvor = gis.lonlats2vor_gp(t.geometry.apply(lambda x: x.coords[0]).tolist(), dataframe=True,
                                  lonlat_bounded=False)

        print('clipping tvor outside mexico country boarder')
        country_poly = region.country(to_4326=False).geometry.values[0]
        tvor['geometry'] = tvor.geometry.apply(lambda x: gis.clip_if_not_within(x, country_poly))

        print(f'saving tvor file: {fn}')
        tvor.to_file(fn, driver='GeoJSON')

    tvor.crs = mex.crs
    tvor.set_index('gtid', inplace=True)
    if load_pop:
        tvor_pop = voronoi_pop_by_ageb()
        tvor = tvor.join(tvor_pop)
    if to_4326:
        tvor = tvor.to_crs(epsg=4326)
    return tvor


def voronoi_pop_by_ageb():
    fn = f'{DIR_INTPL}/voronoi_pop_by_ageb.csv'

    if os.path.exists(fn):
        tvor_pop = pd.read_csv(fn).set_index('gtid')
        print(f'loading existing mexico tower voronoi population file: {fn}')

    else:
        towers_vor = voronoi(load_pop=False)
        agebs = region.agebs()
        t2a = gis.polys2polys(towers_vor, agebs, 'gtid', 'ageb', area_crs=mex.crs, intersection_only=False)
        t2a = t2a.merge(agebs[['pobtot']], left_on='ageb', right_index=True)

        # ageb area is the sum area covered by towers
        # in case ageb' polgyons are not exactly the same as the official map (happens for localidads)
        # also, the points are bufferred, which adds fake areas.
        ageb_area = t2a.groupby('ageb').iarea.sum()
        ageb_area.name = 'ageb_area'
        t2a = t2a.drop(['ageb_area', 'weight'], axis=1).merge(ageb_area.to_frame(), left_on='ageb', right_index=True)

        # iPop is the population of the intersected area between a tower and a ageb
        # within a ageb, the population is assumed to be distributed evenly over space
        # therefore the population is divided proportionally to the intersection area
        # area of intersection / area of ageb * pop of ageb
        t2a['Pop'] = t2a.iarea / t2a.ageb_area * t2a.pobtot

        tvor_pop = t2a.groupby('tower').Pop.sum().to_frame().reindex(towers_vor.index, fill_value=0)
        tvor_pop.to_csv(fn)

    return tvor_pop


def pts_x_region(rkind, per_mun=False, urb_only=False):
    if rkind == 'mpa':
        zms = region.mpa_all_variants(per_mun, urb_only, to_4326=False)
        t_pts = pts(to_4326=False)
        tXzms = gp.sjoin(t_pts, zms, op='within')
        tXzms.rename(columns={'index_right': zms.index.name}, inplace=True)
        cols = ['gtid', 'CVE_SUN', 'mun_id'] if per_mun else ['gtid', 'CVE_SUN']
        tXzms = tXzms[cols]
    else:
        raise ValueError('rkind', rkind, 'not valid')
    return tXzms


def voronoi_x_region(rkind, to_4326=False):
    path = f'{DIR_INTPL}/tvor_x_{rkind}.csv'

    if os.path.exists(path):
        x_mapping = pd.read_csv(path, index_col=0, dtype=str)
        if rkind == 'mpa': x_mapping.CVE_SUN = x_mapping.CVE_SUN.astype(int)
    else:
        if rkind == 'mgl':
            r = region.localidads(to_4326=to_4326)
        elif rkind == 'mgm':
            r = region.municipalities(to_4326=to_4326)
        elif rkind == 'mpa':
            r = region.mpa_all(to_4326)
        elif rkind == 'mga':
            r = region.agebs(to_4326=to_4326)
        else:
            raise ValueError('rkind', rkind, 'not valid')
        tvor = voronoi(to_4326)
        x_mapping = gp.sjoin(tvor, r)['index_right'].to_frame().reset_index()
        x_mapping.rename(columns={'index': 'gtid', 'index_right': r.index.name}, inplace=True)
        x_mapping.to_csv(path)
    return x_mapping
