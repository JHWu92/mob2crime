import os
import sys

if not os.getcwd().endswith('mob2crime'):
    os.chdir('..')
sys.path.insert(0, os.getcwd())

import folium
import datetime
import src.utils.gis as gis
import src.utils.map_vis as mv
import src.mex_helper as mex
import geopandas as gp
import gzip


def tower2loc(loc_buffer):
    t2loc_path = f'data/mex_tower/tower2loc-{loc_buffer}.geojson.gz'
    if os.path.exists(t2loc_path):
        print('reading existing t2loc file')
        t2loc = gp.read_file(f'gzip://{t2loc_path}')
        t2loc = t2loc.set_index('id')
        t2loc.index = t2loc.index.astype(int)
        t2loc.index.name = None
        gis.assign_crs(t2loc, 4326)
        return t2loc

    # =============
    # distribute tower's users count to intersections with localidad by population
    # =============
    print('load tower vor')
    tvor = mex.tower_vor()

    print('load localidads')
    localidad = mex.localidad(loc_buffer, to_crs=4326)
    loc_with_pop = localidad[localidad.Pop > 0]

    print('intersect tower and loc')
    # compute the intersection area between tower and localidad
    t2loc = gis.polys2polys(tvor, loc_with_pop, pname1='tower', pname2='localidad',
                            cur_crs=4326, area_crs=mex.AREA_CRS, intersection_only=False)

    t2loc = t2loc.merge(loc_with_pop[['Pop']], left_on='localidad', right_index=True)

    print('compute weight')
    # localidad area is the sum area covered by towers
    # because Localidads' polgyons are note exactly the same as the official map
    # also, the points are bufferred, which adds fake areas.
    loc_area = t2loc.groupby('localidad').iarea.sum()
    loc_area.name = 'loclidad_area'
    t2loc = t2loc.drop(['localidad_area', 'weight'], axis=1).merge(loc_area.to_frame(), left_on='localidad',
                                                                   right_index=True)

    # iPop is the population of the intersected area between a tower and a localidad
    # within a localidad, the population is assumed to be distributed evenly over space
    # therefore the population is divided proportionally to the intersection area
    t2loc['iPop'] = t2loc.Pop * t2loc.iarea / t2loc.loclidad_area

    # the total population covered by a tower is the sum of iPop
    tower_cover_pop = t2loc.groupby('tower').iPop.sum()
    tower_cover_pop.name = 'tower_pop'
    t2loc = t2loc.merge(tower_cover_pop.to_frame(), left_on='tower', right_index=True)

    # the weight to distribute tower's users count
    t2loc['weight'] = t2loc.iPop / t2loc.tower_pop

    print('saving result')
    with gzip.open(t2loc_path, 'wt') as fout:
        fout.write(t2loc.to_json())
    return t2loc


def tl2grid(t2loc, rkind, grid_side, loc_buffer):
    path = f'data/mex_tower/Tw2Loc2GridByArea-{rkind}-GS{grid_side}-LBf{loc_buffer}.csv'
    if os.path.exists(path):
        print(path, 'exist, skipped')
        return

    grids = mex.grids(RKind, Grid_side)
    print('T2LOC.shape =', t2loc.shape, 'Grids.shape =', grids.shape)
    print('gis.p2p on t2loc and grids', datetime.datetime.now())
    tl2g_raw = gis.polys2polys(t2loc, grids, pname1='tl', pname2='grid',
                               cur_crs=4326, area_crs=mex.AREA_CRS, intersection_only=False)

    print('computing the final weight from tower to grid', datetime.datetime.now())
    tl2g = tl2g_raw.rename(columns={'iarea': 'tl2g_area', 'weight': 'w_tl2g_bA'})
    tl2g = tl2g.merge(t2loc.drop(['geometry', 'iarea'], axis=1), left_on='tl', right_index=True)
    tl2g['weight'] = tl2g.w_t2l_bP * tl2g.w_tl2g_bA
    tl2g = tl2g[['weight', 'tl', 'localidad', 'tower', 'w_t2l_bP', 'grid', 'w_tl2g_bA',
                 'tl_pop', 'tower_pop', 'tl2g_area', 'tl_area', 'geometry',
                 'loc_pop', 'tower_area', 'grid_area', 'loclidad_area']]
    print('saving weights', datetime.datetime.now())
    tl2g.drop('geometry', axis=1).to_csv(path)

    # create example maps for the process
    Tvor['distributed_weight'] = tl2g.groupby('tower').weight.sum()
    Tvor.distributed_weight.fillna(0, inplace=True)
    visualize_selected_towers(RKind, Grid_side)
    visualize_selected_grids(RKind, Grid_side,tl2g)
    return tl2g


def visualize_selected_towers(rkind, gs):
    lon, lat = MEXDF.geometry.centroid.coords[0]

    some_map = folium.Map(location=[lat, lon], zoom_start=8)

    tids = ['32C98855-32C98856-32C98857', '32C99895-32C99896']
    target_tvor = Tvor.loc[tids]
    target_t2loc = T2LOC[T2LOC.tower.isin(tids)]

    mv.geojson_per_row_color_col(gp.GeoDataFrame([MEXDF]), 'mexdf', color='green', some_map=some_map)
    mv.geojson_per_row_color_col(target_tvor.reset_index(), 'tower', tip_cols=['gtid', 'distributed_weight'],
                                 color_col='distributed_weight', some_map=some_map)
    mv.geojson_per_row_color_col(LOCALIDAD.loc[target_t2loc.localidad.unique()].reset_index(), 'localidad',
                                 color='purple', tip_cols=['loc_id', 'Pop'], some_map=some_map)
    mv.geojson_per_row_color_col(target_t2loc.reset_index(), 'tl', color='yellow',
                                 tip_cols=['index', 'localidad', 'tower', 'iPop', 'tower_pop', 'weight'],
                                 some_map=some_map)

    folium.LayerControl().add_to(some_map)
    some_map.save(f'data/mex_tower/tw2loc2grid_selected_tower-{rkind}-GS{gs}.html')


def visualize_selected_grids(rkind, gs,tl2g):
    lon, lat = MEXDF.geometry.centroid.coords[0]
    # select by grid id
    some_map = folium.Map(location=[lat, lon], zoom_start=8)
    gids = [3935, 2533, 561]
    target_tl2g = tl2g[tl2g.grid.isin(gids)]

    tids = target_tl2g.tower.unique()
    target_tvor = Tvor.loc[tids]
    target_t2loc = T2LOC[T2LOC.tower.isin(tids)]

    lids = target_tl2g.localidad.unique()
    target_loc = LOCALIDAD.loc[lids]

    mv.geojson_per_row_color_col(gp.GeoDataFrame([MEXDF]), 'mexdf', color='green', some_map=some_map)
    mv.geojson_per_row_color_col(target_tvor.reset_index(), 'tower', tip_cols=['gtid', 'distributed_weight'],
                                 color_col='distributed_weight', some_map=some_map)
    mv.geojson_per_row_color_col(target_loc.reset_index(), 'localidad', color='purple', tip_cols=['loc_id', 'Pop'],
                                 some_map=some_map)
    mv.geojson_per_row_color_col(target_t2loc.reset_index(), 'tl', color='yellow',
                                 tip_cols=['index', 'localidad', 'tower', 'iPop', 'tower_pop', 'weight'],
                                 some_map=some_map)
    mv.geojson_per_row_color_col(target_tl2g.reset_index(), 'tl2g', color='blue',
                                 tip_cols=['tl', 'localidad', 'tower', 'grid', 'weight'], some_map=some_map)

    folium.LayerControl().add_to(some_map)
    some_map.save(f'data/mex_tower/tw2loc2grid_selected_grids-{rkind}-GS{gs}.html')


Loc_Buffer = 500
print('======getting T2LOC')
T2LOC = tower2loc(Loc_Buffer)
print('======The average sum weight distributed out by all tower is:', T2LOC.groupby('tower').weight.sum().mean())
T2LOC_rename = T2LOC.rename(columns={'iPop': 'tl_pop', 'Pop': 'loc_pop', 'weight': 'w_t2l_bP'})
print('======loading Tvor')
Tvor = mex.tower_vor()

for RKind in ['metropolitans_all']:
    print(f'========rkind={RKind}, loading mexdf and localidad for creating example maps')
    REGION = mex.regions(RKind)
    MEXDF = REGION.iloc[0]
    LOCALIDAD = mex.localidad(Loc_Buffer, to_crs=4326)

    for Grid_side in [500, 1000, 2000]:
        print(f'==========rkind={RKind}, grid side = {Grid_side}', datetime.datetime.now())
        tl2grid(T2LOC_rename, RKind, Grid_side, Loc_Buffer)

