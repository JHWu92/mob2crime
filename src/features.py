import src.mex_helper as mex
import src.utils.gis as gis
import numpy as np
import pandas as pd


# TODO
#  all the functions are for mex and grid=1000


def urban_dilatation_index(avg):
    mex_grids = mex.grids('cities', 1000)
    mex_cities = mex.cities()
    areas = mex_cities.to_crs(gis.crs_normalization(mex.AREA_CRS)).geometry.apply(lambda x: x.area)

    dv_cities = {}
    for city, cgrids in mex_grids.groupby('city'):
        sqrt_area = np.sqrt(areas[city])
        grid_dist = gis.polys_centroid_pairwise_dist(cgrids, dist_crs=mex.EQDC_CRS)
        n_grids = len(grid_dist)
        cgrids_avg = avg.loc[cgrids.grid]
        s = cgrids_avg / cgrids_avg.sum()
        s.index = range(n_grids)

        dv = {}
        for t in s.columns:
            st_outer = np.outer(s[t], s[t])
            np.fill_diagonal(st_outer, 0)
            dv[int(t)] = (st_outer * grid_dist).sum() / st_outer.sum() / sqrt_area

        dv_cities[city] = dv

    dv_cities = pd.DataFrame(dv_cities).T
    dv_cities.columns = [f'dv_{c:02}' for c in dv_cities.columns]
    dv_cities['dilatation coefficient'] = dv_cities.max(axis=1) / dv_cities.min(axis=1)
    return dv_cities


def hotspot_stats(avg):
    from collections import defaultdict
    from src.utils import loubar_thres

    mex_grids = mex.grids('cities', 1000)
    mex_cities = mex.cities()
    areas = mex_cities.to_crs(gis.crs_normalization(mex.AREA_CRS)).geometry.apply(lambda x: x.area)

    def keep_hotspot(avg):
        for h in avg:
            arr = avg[h]
            loubar, arr_thres = loubar_thres(arr, is_sorted=False)
            avg[h][avg[h] <= arr_thres] = 0
        #         print(h, loubar, arr_thres)
        return avg

    n_hotspot_cities = {}
    hotspot_stats_cities = defaultdict(dict)
    for city, cgrids in mex_grids.groupby('city'):
        sqrt_area = np.sqrt(areas[city])
        # keep hotspot only, else set to 0
        cgrids_avg = avg.loc[cgrids.grid].copy()
        chotspot = keep_hotspot(cgrids_avg)
        # stats of all hotspots
        n_hotspot_cities[city] = (chotspot != 0).sum()

        # stats based on persistence
        persistence = (chotspot != 0).sum(axis=1)
        permenant = persistence[persistence == 24]
        intermediate = persistence[(persistence < 24) & (persistence >= 7)]
        intermittent = persistence[(persistence < 7) & (persistence >= 1)]

        hotspot_stats_cities['n_per'][city] = len(permenant)
        hotspot_stats_cities['n_med'][city] = len(intermediate)
        hotspot_stats_cities['n_int'][city] = len(intermittent)

        avg_dist = lambda x: gis.polys_centroid_pairwise_dist(cgrids.loc[x.index], dist_crs=mex.EQDC_CRS).sum() / len(
            x) / (len(x) - 1)
        d_per = avg_dist(permenant)
        d_med = avg_dist(intermediate)
        d_int = avg_dist(intermittent)

        hotspot_stats_cities['compacity_coefficient'][city] = d_per / sqrt_area
        hotspot_stats_cities['d_per_med'][city] = d_per / d_med if d_med != 0 else np.nan
        hotspot_stats_cities['d_med_int'][city] = d_med / d_int if d_int != 0 else np.nan

    n_hotspot_cities = pd.DataFrame(n_hotspot_cities).T
    n_hotspot_cities.columns = [f'nhot_{int(c):02}' for c in n_hotspot_cities.columns]
    hotspot_stats_cities = pd.DataFrame(hotspot_stats_cities)
    return n_hotspot_cities,hotspot_stats_cities
