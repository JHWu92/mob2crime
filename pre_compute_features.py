import datetime

import geopandas as gp
import pandas as pd
from tinydb import TinyDB

import src.creds as const
import src.ftrs.feature_generator as ftr_gen
import src.ftrs.hotspot as ftr_hs
import src.mex.regions2010 as region
import src.mex.tower as mex_tower
import src.mex_helper as mex_helper
import src.tower_interpolation as tw_int
import src.utils.gis as gis


def get_su(su_type, admin_lvl, admin_id, admin_shape, crs=None):
    if su_type.startswith('grid'):
        index_name, grid_side = su_type.split('-')
        grid_side = int(grid_side)
        gs, rids, cids = gis.poly2grids(admin_shape, grid_side)
        su = gp.GeoDataFrame(list(zip(gs, rids, cids)), columns=['geometry', 'row_id', 'col_id'])
        su[admin_lvl] = admin_id
        su.crs = crs
        su.index.name = index_name
    else:
        raise NotImplementedError(f'su_type="{su_type}" not Implemented')
    return su


def get_hotspots_per_hour(towers_vor, su, tw_footfall, intpl_type, cache_path=None, verbose=0):
    """
    return interpolated footfall for spatial untis
    :param towers_vor: tower vor polygons
    :param su: spatial units
    :param tw_footfall: footfall for towers
    :param intpl_type: type of interpolation method
    :param cache_path: if not None, cahce t2su results, used by specific interpolation function
    :param verbose:
    :return:
    """
    if intpl_type == 'Uni':
        su_footfall = tw_int.interpolate_uni(towers_vor, su, tw_footfall, cache_path, verbose)
    else:
        raise NotImplementedError(f'intpl_type="{intpl_type}" has not been implemented')

    return su_footfall


def main():
    db = TinyDB(const.feature_db)

    # define parameters
    country = 'Mex'
    admin_lvl = 'mun_id'
    boundary_type = 'Urban'
    urb_only = {'Urban': True, 'UrbanRural': False}[boundary_type]
    su_type = 'grid-500'
    intpl_type = 'Uni'
    # footfall hotspots related
    hs_types = ['loubar', 'average'][:1]
    wd_wk = 'wd'  # average weekdays or weekends

    # for compactness indices
    raster_resolution = 100
    raster_use_p_centroid_if_none = True

    # load the geometry of municipality
    print('loading municipalities')
    mgm = region.municipalities(load_pop=True, urb_only=urb_only)
    mex_crs = mgm.crs

    # load cell towers
    print('loading towers and polygons')
    towers = mex_tower.pts().set_index('gtid')
    towers_vor = mex_tower.voronoi()
    towers_vor_in_mgm = mex_tower.voronoi_x_region('mgm')

    # load cached tower average footfall
    # if average_over_observed_day==False
    # average over the number of days in the observation period
    # 75% of the towers are the same as average_over_observed_day==True
    call_direction = 'out+in'
    n_bins = 24  # 24 hours
    print(f'loading footfall for {wd_wk}, n_bins={n_bins}, call_direction={call_direction}')
    aver = mex_helper.stat_tw_dow_aver_hr_uniq_user(call_direction, n_bins=n_bins, average_over_observed_day=True)
    tw_footfall = pd.DataFrame(aver[wd_wk]).T

    redo = False
    for mid, munic in mgm.iterrows():
        admin_id = mid
        print(f'====== working on {admin_lvl}={admin_id}, {datetime.datetime.now()}', end='')
        city_area = munic.geometry.area
        su = get_su(su_type, admin_lvl, mid, munic.geometry, mex_crs)
        tid_oi = towers_vor_in_mgm[towers_vor_in_mgm.mun_id == mid].gtid

        su_footfall = get_hotspots_per_hour(towers_vor.loc[tid_oi], su,
                                            tw_footfall.reindex(tid_oi, fill_value=0),
                                            intpl_type)
        for hotspot_type in hs_types:
            hotspots_per_hour = ftr_hs.keep_hotspot(su_footfall.copy(), hotspot_type)

            ftr_gen.ftr_hs_scale(db, country, admin_lvl, admin_id, boundary_type, su_type, intpl_type,
                                 hotspot_type=hotspot_type, hotspots_per_hour=hotspots_per_hour,
                                 su=su, redo=redo)

            ftr_gen.ftr_compacity(db, country, admin_lvl, admin_id, boundary_type, su_type, intpl_type,
                                  hotspot_type=hotspot_type, hotspots_per_hour=hotspots_per_hour,
                                  su=su, city_area=city_area, redo=redo)

            ftr_gen.ftr_compactness(db, country, admin_lvl, admin_id, boundary_type, su_type, intpl_type,
                                    hotspot_type=hotspot_type, raster_resolution=raster_resolution,
                                    raster_use_p_centroid_if_none=raster_use_p_centroid_if_none,
                                    su=su, hotspots_per_hour=hotspots_per_hour, redo=redo, verbose=0)
        print(f' ~ {datetime.datetime.now()}')


if __name__ == "__main__":
    main()
