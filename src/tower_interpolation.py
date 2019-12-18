import os
from itertools import chain

import geopandas as gp
import numpy as np
import pandas as pd

import src.mex as mex
import src.mex.regions2010 as region
import src.mex.tower as tower
import src.utils.gis as gis
import src.utils.idw as idw

DIR_INTPL = 'data/mex_tw_intpl'

PER_MUN_STR = lambda per_mun: 'perMun' if per_mun else 'whole'
URB_ONLY_STR = lambda urb_only: 'urb' if urb_only else 'uNr'


def interpolate_idw(tw_avg, side, per_mun=False, urb_only=False, max_k=10, grids=None, n_bins=24):
    per_mun_str = PER_MUN_STR(per_mun)
    if n_bins == 24:
        path = f'{DIR_INTPL}/interpolate_idw{max_k}_g{side}_{per_mun_str}_{urb_only}.csv.gz'
    else:
        path = f'{DIR_INTPL}/interpolate{n_bins}_idw{max_k}_g{side}_{per_mun_str}_{urb_only}.csv.gz'

    if os.path.exists(path):
        print('interpolate_idw loading existing file', path)
        g_avg = pd.read_csv(path, index_col=0, compression='gzip')
        g_avg.columns = g_avg.columns.astype(str)
        return g_avg

    print('====computing interpolate_idw', per_mun, urb_only)
    print('reading tower points')
    tws = mex.tower.pts()
    zms = region.mpa_all()
    tws_x_zms = gp.sjoin(tws, zms)[['gtid', 'index_right']]

    # allow to pass on grids, without loading it again
    if grids is None:
        print('reading grids file')
        grids = region.mpa_grids(side, per_mun=per_mun, urb_only=urb_only, to_4326=False)

    print('interpolating per hour')
    g_avg = {}

    for h in range(n_bins):
        h = str(h)
        g_avg[h] = _interpolate_idw_per_hour(grids, tws, tws_x_zms, tw_avg[h], max_k)
    df = []
    for h, g in g_avg.items():
        g.columns = [h]
        df.append(g)
    df = pd.concat(df, axis=1)
    df.to_csv(path, compression='gzip')
    return df


def _interpolate_idw_per_hour(grids, tws, tws_x_zms, z, max_k=10):
    # print('computing weight for each SUN')
    gs_avg = []

    for sun in grids.CVE_SUN.unique():
        zm_grids = grids[grids.CVE_SUN == sun]
        zm_tws = tws[tws.gtid.isin(tws_x_zms[tws_x_zms.index_right == sun].gtid)]
        zm_z = z.reindex(zm_tws.gtid, fill_value=0)
        zm_g_coords = np.array(zm_grids.geometry.apply(lambda x: x.centroid.coords[0]).tolist())
        zm_t_coords = np.array(zm_tws.geometry.apply(lambda x: x.coords[0]).tolist())
        idw_tree = idw.tree(zm_t_coords, zm_z)
        k = min(len(zm_t_coords), max_k)
        g_avg = idw_tree(zm_g_coords, k=k)
        scale = zm_z.sum() / g_avg.sum()
        g_avg *= scale
        g_avg = pd.DataFrame(g_avg, index=zm_grids.index)
        gs_avg.append(g_avg)

    gs_avg = pd.concat(gs_avg)
    return gs_avg


def interpolate_stats(tw_avg, t2region, n_bins=24):
    # there are grids without any call throughout the observation period
    #     print('grid_average')
    r_avg = t2region.merge(tw_avg, left_on='tower', right_index=True, how='left')

    for h in range(n_bins):
        h = str(h)
        r_avg[h] = r_avg[h] * r_avg['weight']

    r_avg = r_avg.drop('weight', axis=1).groupby(level=0).sum()  # fillna=0 by default

    return r_avg


def to_mpa_agebs(by='area', return_geom=False):
    assert by in ('area', 'pop'), f'by={by}, it should be either "area" or "pop"'

    path = f'{DIR_INTPL}/tower_to_mpa_agebs_by_area.csv' if by == 'area' else f'{DIR_INTPL}/tower_to_mpa_agebs_by_pop.csv'
    if not return_geom and os.path.exists(path):
        print('to_map_agebs loading existing file', path)
        t2ageb = pd.read_csv(path, index_col=0)
        return t2ageb
    print('computing t2a', by, 'return_geom =', return_geom)
    zms = region.mpa_all()
    mun_ids = sorted(list(set(chain(*zms.mun_ids.apply(lambda x: x.split(','))))))
    zms_agebs = region.agebs(mun_ids=mun_ids)
    tvor = tower.voronoi()
    tvor_x_zm = tower.voronoi_x_region('mpa')
    zms_tvors = tvor.loc[tvor_x_zm.gtid.unique()]

    if by == 'area':  # by area only
        t2ageb = gis.polys2polys(zms_tvors, zms_agebs, 'tower', 'ageb', area_crs=mex.crs, intersection_only=False)

    else:  # by pop
        tvor_x_agebs = tower.voronoi_x_region('mga')
        covered = tvor_x_agebs[tvor_x_agebs.gtid.isin(zms_tvors.index)]
        covered_ageb_ids = covered.ageb_id.unique()
        covered_loc_ids = sorted(set([aid[:9] for aid in covered_ageb_ids]))
        covered_mun_ids = sorted(set([aid[:5] for aid in covered_ageb_ids]))
        covered_agebs = region.agebs(mun_ids=covered_mun_ids, loc_ids=covered_loc_ids)
        t2covered_ageb = gis.polys2polys(zms_tvors, covered_agebs, 'tower', 'ageb', area_crs=mex.crs,
                                         intersection_only=False)

        t2covered_ageb = t2covered_ageb.merge(covered_agebs[['pobtot']], left_on='ageb', right_index=True)

        # ageb area is the sum area covered by towers
        # in case ageb' polgyons are not exactly the same as the official map (happens for localidads)
        # also, the points are bufferred, which adds fake areas.
        ageb_area = t2covered_ageb.groupby('ageb').iarea.sum()
        ageb_area.name = 'ageb_area'

        t2covered_ageb = t2covered_ageb.rename(
            columns={'ageb_area': 'original_ageb_area', 'weight': 'area_weight'}).merge(ageb_area.to_frame(),
                                                                                        left_on='ageb',
                                                                                        right_index=True)

        # iPop is the population of the intersected area between a tower and a ageb
        # within a ageb, the population is assumed to be distributed evenly over space
        # therefore the population is divided proportionally to the intersection area
        t2covered_ageb['iPop'] = t2covered_ageb.pobtot * t2covered_ageb.iarea / t2covered_ageb.ageb_area

        # the total population covered by a tower is the sum of iPop
        tower_cover_pop = t2covered_ageb.groupby('tower').iPop.sum()
        tower_cover_pop.name = 'tower_pop'
        t2covered_ageb = t2covered_ageb.merge(tower_cover_pop.to_frame(), left_on='tower', right_index=True)

        # the weight to distribute tower's users count
        t2covered_ageb['weight'] = t2covered_ageb.iPop / t2covered_ageb.tower_pop
        t2ageb = t2covered_ageb[t2covered_ageb.ageb.isin(zms_agebs.index)]

    t2ageb[['tower', 'ageb', 'weight']].to_csv(path)
    print('returning t2ageb')
    if return_geom:
        return t2ageb

    return t2ageb[['tower', 'ageb', 'weight']]


def to_mpa_grids(side, by='area', per_mun=False, urb_only=False, grids=None):
    assert by in ('area', 'pop'), f'by={by}, it should be either "area" or "pop"'
    path = f'{DIR_INTPL}/tower_to_mpa_g{side}_{PER_MUN_STR(per_mun)}_{URB_ONLY_STR(urb_only)}_{by}.csv'

    if os.path.exists(path):
        print('to_map_grids loading existing file', path)
        t2g = pd.read_csv(path, index_col=0)
        return t2g

    print('computing to_map_grids', by, per_mun, urb_only)
    # allow to pass on grids, without loading it again
    if grids is None:
        grids = region.mpa_grids(side, per_mun=per_mun, urb_only=urb_only, to_4326=False)

    if by == 'area':
        tvor = tower.voronoi()
        tvor_x_zm = tower.voronoi_x_region('mpa')
        zms_tvors = tvor.loc[tvor_x_zm.gtid.unique()]
        t2g = gis.polys2polys(zms_tvors, grids, 'tower', 'grid', area_crs=mex.crs, intersection_only=False)
    else:
        t2ageb_by_pop = to_mpa_agebs('pop', return_geom=True)
        t2ageb_by_pop_rename = t2ageb_by_pop.rename(
            columns={'iPop': 'txa_pop', 'Pop': 'ageb_pop', 'weight': 'w_t2a_bP'})

        txa2g_raw = gis.polys2polys(t2ageb_by_pop, grids, pname1='txa', pname2='grid', area_crs=mex.crs,
                                    intersection_only=False)

        txa2g = txa2g_raw.rename(columns={'iarea': 'txa2g_area', 'weight': 'w_txa2g_bA'})
        txa2g = txa2g.merge(t2ageb_by_pop_rename.drop(['geometry', 'iarea'], axis=1), left_on='txa', right_index=True)
        txa2g['weight'] = txa2g.w_t2a_bP * txa2g.w_txa2g_bA
        txa2g = txa2g[['weight', 'txa', 'ageb', 'tower', 'w_t2a_bP', 'grid', 'w_txa2g_bA',
                       'txa_pop', 'tower_pop', 'txa2g_area', 'txa_area', 'geometry',
                       'pobtot', 'tower_area', 'grid_area', 'ageb_area']]
        t2g = txa2g.groupby(['tower', 'grid']).weight.sum().reset_index()

    t2g[['tower', 'grid', 'weight']].to_csv(path)
    return t2g[['tower', 'grid', 'weight']]


def to_mpa_vors(by='area', per_mun=False, urb_only=False, zms_vors=None):
    assert by in ('area', 'pop'), f'by={by}, it should be either "area" or "pop"'
    path = f'{DIR_INTPL}/tower_to_mpa_vors_{PER_MUN_STR(per_mun)}_{URB_ONLY_STR(urb_only)}_{by}.csv'

    if os.path.exists(path):
        print('to_mpa_vors loading existing file', path)
        t2v = pd.read_csv(path, index_col=0)
        return t2v

    print('computing to_map_vors', by, per_mun, urb_only)
    # allow to pass on zms_vors, without loading it again
    if zms_vors is None:
        zms_vors = region.mpa_vors(per_mun=per_mun, urb_only=urb_only, to_4326=False)

    if by == 'area':
        t2v = zms_vors.reset_index()
    else:
        t2ageb_by_pop = to_mpa_agebs('pop', return_geom=True)
        t2ageb_by_pop_rename = t2ageb_by_pop.rename(
            columns={'iPop': 'txa_pop', 'Pop': 'ageb_pop', 'weight': 'w_t2a_bP'})

        txa2v_raw = gis.polys2polys(t2ageb_by_pop, zms_vors, pname1='txa', pname2='vor', area_crs=mex.crs,
                                    intersection_only=False)

        txa2v = txa2v_raw.rename(columns={'iarea': 'txa2v_area', 'weight': 'w_txa2v_bA'})
        txa2v = txa2v.merge(t2ageb_by_pop_rename.drop(['geometry', 'iarea'], axis=1), left_on='txa', right_index=True)
        txa2v['weight'] = txa2v.w_t2a_bP * txa2v.w_txa2v_bA
        txa2v = txa2v[['weight', 'txa', 'ageb', 'tower', 'w_t2a_bP', 'vor', 'w_txa2v_bA',
                       'txa_pop', 'tower_pop', 'txa2v_area', 'txa_area', 'geometry',
                       'pobtot', 'tower_area', 'vor_area', 'ageb_area']]
        t2v = txa2v.groupby(['tower', 'vor']).weight.sum().reset_index()
        t2v = t2v[t2v.weight > 1e-12]
        t2v[['tower', 'vor', 'weight']].to_csv(path)

    return t2v[['tower', 'vor', 'weight']]
