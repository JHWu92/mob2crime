import numpy as np
import pandas as pd

import src.mex_helper as mex
import src.utils.gis as gis


# TODO
#  all the functions are for mex and grid=1000


def urban_dilatation_index(avg, rkind, rname):
    mex_grids = mex.grids(rkind, 1000)
    mex_cities = mex.cities()
    areas = mex_cities.to_crs(gis.crs_normalization(mex.AREA_CRS)).geometry.apply(lambda x: x.area)

    dv_r = {}
    for region, rgrids in mex_grids.groupby(rname):
        sqrt_area = np.sqrt(areas[region])
        grid_dist = gis.polys_centroid_pairwise_dist(rgrids, dist_crs=mex.EQDC_CRS)
        n_grids = len(grid_dist)
        cgrids_avg = avg.loc[rgrids.grid]
        s = cgrids_avg / cgrids_avg.sum()
        s.index = range(n_grids)

        dv = {}
        for t in s.columns:
            st_outer = np.outer(s[t], s[t])
            np.fill_diagonal(st_outer, 0)
            dv[int(t)] = (st_outer * grid_dist).sum() / st_outer.sum() / sqrt_area

        dv_r[region] = dv

    dv_r = pd.DataFrame(dv_r).T
    dv_r.columns = [f'dv_{c:02}' for c in dv_r.columns]
    dv_r['dilatation coefficient'] = dv_r.max(axis=1) / dv_r.min(axis=1)
    return dv_r


def hotspot_stats(avg, rkind, rname):
    from collections import defaultdict
    from src.utils import loubar_thres

    mex_grids = mex.grids(rkind, 1000)
    mex_region = mex.regions(rkind)
    areas = mex_region.to_crs(gis.crs_normalization(mex.AREA_CRS)).geometry.apply(lambda x: x.area)

    def keep_hotspot(avg):
        for h in avg:
            arr = avg[h]
            loubar, arr_thres = loubar_thres(arr, is_sorted=False)
            avg[h][avg[h] <= arr_thres] = 0
        #         print(h, loubar, arr_thres)
        return avg

    n_hotspot_regions = {}
    hotspot_stats_regions = defaultdict(dict)
    for region, rgrids in mex_grids.groupby(rname):
        sqrt_area = np.sqrt(areas[region])
        # keep hotspot only, else set to 0
        cgrids_avg = avg.loc[rgrids.grid].copy()
        chotspot = keep_hotspot(cgrids_avg)
        # stats of all hotspots
        n_hotspot_regions[region] = (chotspot != 0).sum()

        # stats based on persistence
        persistence = (chotspot != 0).sum(axis=1)
        permenant = persistence[persistence == 24]
        intermediate = persistence[(persistence < 24) & (persistence >= 7)]
        intermittent = persistence[(persistence < 7) & (persistence >= 1)]

        hotspot_stats_regions['n_per'][region] = len(permenant)
        hotspot_stats_regions['n_med'][region] = len(intermediate)
        hotspot_stats_regions['n_int'][region] = len(intermittent)

        avg_dist = lambda x: gis.polys_centroid_pairwise_dist(rgrids.loc[x.index], dist_crs=mex.EQDC_CRS).sum() / len(
            x) / (len(x) - 1)
        d_per = avg_dist(permenant)
        d_med = avg_dist(intermediate)
        d_int = avg_dist(intermittent)

        hotspot_stats_regions['compacity_coefficient'][region] = d_per / sqrt_area
        hotspot_stats_regions['d_per_med'][region] = d_per / d_med if d_med != 0 else np.nan
        hotspot_stats_regions['d_med_int'][region] = d_med / d_int if d_int != 0 else np.nan

    n_hotspot_regions = pd.DataFrame(n_hotspot_regions).T
    n_hotspot_regions.columns = [f'nhot_{int(c):02}' for c in n_hotspot_regions.columns]
    hotspot_stats_regions = pd.DataFrame(hotspot_stats_regions)
    return n_hotspot_regions, hotspot_stats_regions
