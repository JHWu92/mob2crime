import os
from itertools import chain

import pandas as pd
import src.mex_helper as mex_helper
import src.mex as mex
import src.mex.regions2010 as region
import src.mex.tower as tower
import src.utils.gis as gis

DIR_INTPL = 'data/mex_tw_intpl'


def interpolate_stats(tw_avg, t2region):
    # there are grids without any call throughout the observation period
    #     print('grid_average')
    r_avg = t2region.merge(tw_avg, left_on='tower', right_index=True, how='left')

    for h in range(24):
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
    if return_geom:
        return t2ageb
    return t2ageb[['tower', 'ageb', 'weight']]


def to_mpa_grids(side, by='area', per_mun=False):
    assert by in ('area', 'pop'), f'by={by}, it should be either "area" or "pop"'
    per_mun_str = 'per_mun_' if per_mun else ''
    path = f'{DIR_INTPL}/tower_to_mpa_g{side}_{per_mun_str}by_area.csv' if by == 'area' \
        else f'{DIR_INTPL}/tower_to_mpa_g{side}_{per_mun_str}by_pop.csv'

    if os.path.exists(path):
        print('to_map_grids loading existing file', path)
        t2g = pd.read_csv(path, index_col=0)
        return t2g


    grids = region.mpa_grids(side, per_mun, to_4326=False)
    if by == 'area':
        tvor = tower.voronoi()
        tvor_x_zm = tower.voronoi_x_region('mpa')
        zms_tvors = tvor.loc[tvor_x_zm.gtid.unique()]
        t2g = gis.polys2polys(zms_tvors, grids, 'tower', 'grid', area_crs=mex.crs, intersection_only=False)
    else:
        t2ageb_by_pop = to_mpa_agebs('pop',return_geom=True)
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
