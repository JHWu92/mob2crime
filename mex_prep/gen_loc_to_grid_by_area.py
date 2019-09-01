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


def loc2grid(rkind, grid_side, loc_buffer=500):
    path = f'data/mex_tower/Loc2GridByArea-{rkind}-GS{grid_side}-LBf{loc_buffer}.csv'
    if os.path.exists(path):
        print(path, 'exist, skipped')
        return

    grids = mex.grids(rkind, grid_side)
    localidad = mex.localidad(loc_buffer, to_crs=4326)

    print('running polys2polys', datetime.datetime.now())
    loc2grid = gis.polys2polys(localidad, grids, 'localidad', 'grids',
                               cur_crs=4326, area_crs=mex.AREA_CRS, intersection_only=False)

    print('saving weights', datetime.datetime.now())
    loc2grid[['localidad', 'grids', 'weight', 'iarea']].to_csv(path)

    print('creating example html')
    example = loc2grid[loc2grid.localidad.isin(
        localidad[localidad.CVE_ENT.isin(['09', '12', '14'])].index
    )]
    m = folium.Map(location=[19.381495, -99.139095], zoom_start=6)
    mv.geojson_per_row(grids[grids.grid.isin(example.grids)], 'Grids', some_map=m)
    mv.geojson_per_row(example, 'G2loc', some_map=m,
                       color='green',
                       tip_cols=['localidad', 'grids', 'weight', 'iarea', 'localidad_area', 'grids_area'])
    folium.LayerControl().add_to(m)
    m.save('Loc2Grid-{rkind}-GS{grid_side}-LBf{loc_buffer}-Example.html')
    print('done')


for RKind, Grid_side in [
    ('metropolitans_all', 500),
    ('metropolitans_all', 1000),
    ('metropolitans_all', 2000),
]:
    print(f'rkind={RKind}, grid side = {Grid_side}', datetime.datetime.now())

    loc2grid(RKind, Grid_side, loc_buffer=500)
