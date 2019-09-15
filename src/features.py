import numpy as np
import pandas as pd
import os
import src.mex_helper as mex
import src.utils.gis as gis
import datetime


# TODO
#  all the functions are for mex
#  and grid=1000 --> added gside

def fast_dv(rgrids, s, sqrt_area):
    grid_dist = gis.polys_centroid_pairwise_dist(rgrids, dist_crs=mex.EQDC_CRS)
    dv = {}
    for t in s.columns:
        st_outer = np.outer(s[t], s[t])
        np.fill_diagonal(st_outer, 0)
        dv[int(t)] = (st_outer * grid_dist).sum() / st_outer.sum() / sqrt_area
    return dv


def slow_dv(rgrids, s, sqrt_area, rkind, gside, region):
    path = f'stats/urban_dilatation_index/{rkind}-{gside}-{region}.txt'
    if os.path.exists(path):
        dv = eval(open(path).read())
        return dv

    n_grids = len(rgrids)
    dv = {}
    grid_cens = np.array(rgrids.to_crs(mex.EQDC_CRS).geometry.apply(lambda x: x.centroid.coords[0]).tolist())
    for t in s.columns:
        st = s[t].tolist()
        st_sum = 0
        st_grid_dist = 0
        print(f'slow_dv doing t={t}, at time:', datetime.datetime.now())
        for i in range(n_grids):
            sti = st[i]
            gi = grid_cens[i]
            for j in range(i + 1, n_grids):
                gj = grid_cens[j]
                st_ij = sti * st[j]
                st_sum += st_ij
                dist = np.linalg.norm(gi - gj)
                st_grid_dist += st_ij * dist
        dv[t] = st_grid_dist / st_sum / sqrt_area
    with open(path, 'w') as fout:
        fout.write(str(dv))

    return dv


def handle_missing_grid_ids(grid_in_avg, rgrids, region):
    grid_id = set(rgrids.grid)

    grid_not_in_avg = grid_id - grid_in_avg
    if len(grid_not_in_avg) != 0:
        print(':::WARNING:::', region, 'has some grids not intersecting tower vor', len(grid_not_in_avg))
        grid_id = list(grid_id & grid_in_avg)
        rgrids = rgrids[~rgrids.grid.isin(grid_not_in_avg)]

    return rgrids


def urban_dilatation_index(avg, rkind, rname, gside):
    mex_grids = mex.grids(rkind, gside)
    mex_regions = mex.regions(rkind)
    areas = mex_regions.to_crs(gis.crs_normalization(mex.AREA_CRS)).geometry.apply(lambda x: x.area)

    grid_in_avg = set(avg.index)
    dv_r = {}
    for region, rgrids in mex_grids.groupby(rname):
        print('working on',region)
        rgrids = handle_missing_grid_ids(grid_in_avg, rgrids, region)
        grid_id = rgrids.grid
        n_grids = len(grid_id)

        sqrt_area = np.sqrt(areas[region])
        cgrids_avg = avg.loc[grid_id]
        s = cgrids_avg / cgrids_avg.sum()
        s.index = range(n_grids)

        if n_grids < 40000:
            dv = fast_dv(rgrids, s, sqrt_area)
        else:
            print(f'Region {region} has #grids: {n_grids}, using slow_dv function')
            before = datetime.datetime.now()
            dv = slow_dv(rgrids, s, sqrt_area, rkind, gside, region)
            print('it took {} seconds'.format((datetime.datetime.now() - before)).total_seconds())
        dv_r[region] = dv

    dv_r = pd.DataFrame(dv_r).T
    dv_r.columns = [f'dv_{c:02}' for c in dv_r.columns]
    dv_r['dilatation coefficient'] = dv_r.max(axis=1) / dv_r.min(axis=1)
    return dv_r


def hotspot_stats(avg, rkind, rname, gside, hotspot_type):
    from collections import defaultdict
    from src.utils import loubar_thres

    mex_grids = mex.grids(rkind, gside)
    mex_region = mex.regions(rkind)
    areas = mex_region.to_crs(gis.crs_normalization(mex.AREA_CRS)).geometry.apply(lambda x: x.area)

    def keep_hotspot(avg):
        for h in avg:
            arr = avg[h]
            if hotspot_type == 'loubar':
                _, arr_thres = loubar_thres(arr, is_sorted=False)
            elif hotspot_type == 'average':
                arr_thres = np.mean(arr)
            else:
                raise ValueError('hotspot type', hotspot_type, 'not implemented')
            avg[h][avg[h] <= arr_thres] = 0
            # print(h, loubar, arr_thres)
        return avg

    n_hotspot_regions = {}
    hotspot_stats_regions = defaultdict(dict)
    permanent_regions = {}
    persistence_regions = {}
    grid_in_avg = set(avg.index)
    for region, rgrids in mex_grids.groupby(rname):
        rgrids = handle_missing_grid_ids(grid_in_avg, rgrids, region)

        sqrt_area = np.sqrt(areas[region])
        # keep hotspot only, else set to 0
        cgrids_avg = avg.loc[rgrids.grid].copy()
        chotspot = keep_hotspot(cgrids_avg)
        # print(chotspot)
        # stats of all hotspots
        n_hotspot_regions[region] = (chotspot != 0).sum()

        # stats based on persistence
        persistence = (chotspot != 0).sum(axis=1)
        persistence_regions[region] = persistence
        permanent = persistence[persistence == 24]
        permanent_regions[region] = cgrids_avg.loc[permanent.index]
        intermediate = persistence[(persistence < 24) & (persistence >= 7)]
        intermittent = persistence[(persistence < 7) & (persistence >= 1)]

        hotspot_stats_regions['n_per'][region] = len(permanent)
        hotspot_stats_regions['n_med'][region] = len(intermediate)
        hotspot_stats_regions['n_int'][region] = len(intermittent)

        def avg_dist(x):
            if len(x) <= 1:
                return 0
            l = len(x)
            if len(x)>40000:
                print(':::WARNING:::, too many grids, aborted')
                return np.nan
            pair_dist = gis.polys_centroid_pairwise_dist(rgrids.loc[x.index], dist_crs=mex.EQDC_CRS).sum()
            return pair_dist / l / (l - 1)

        d_per = avg_dist(permanent)
        d_med = avg_dist(intermediate)
        d_int = avg_dist(intermittent)

        hotspot_stats_regions['compacity_coefficient'][region] = d_per / sqrt_area
        hotspot_stats_regions['d_per_med'][region] = d_per / d_med if d_med != 0 else np.nan
        hotspot_stats_regions['d_med_int'][region] = d_med / d_int if d_int != 0 else np.nan

    n_hotspot_regions = pd.DataFrame(n_hotspot_regions).T
    n_hotspot_regions.columns = [f'nhot_{int(c):02}' for c in n_hotspot_regions.columns]
    hotspot_stats_regions = pd.DataFrame(hotspot_stats_regions)
    return n_hotspot_regions, hotspot_stats_regions, permanent_regions, persistence_regions
