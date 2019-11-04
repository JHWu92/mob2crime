import src.tower_interpolation as tw_int
import src.ftrs.hotspot as ftr_hs
import src.mex.regions2010 as region
import src.mex.tower as tower
import src.mex_helper as mex_helper
import src.ftrs.dilatation as dilatation
import pandas as pd
import datetime as dt

PER_MUN_DISPLAY = lambda x: 'PerMun' if x else 'Metro'
URB_ONLY_DISPLAY = lambda x: 'UrbanOnly' if x else 'UrbanRural'
ADMIN_STR = lambda x, y: f'{PER_MUN_DISPLAY(x)}_{URB_ONLY_DISPLAY(y)}'

zms_sort_cols = ['Area', 'Area_urb', 'Area_rur', 'Area_urb_pcnt', 'Area_rur_pcnt', 'pobtot', 'pob_urb', 'pob_rur',
                 'pob_urb_pcnt', 'pob_rur_pcnt']
G_side = 500


def _zm_pops(zms, mg_mappings):
    zms_mun_ids = mg_mappings.mun_id.unique()
    # zms_mgms = region.municipalities(zms_mun_ids)
    zms_mglus = region.locs_urban(zms_mun_ids)
    zms_mglus['Area'] = zms_mglus.area

    zms_mglrs = region.locs_rural(zms_mun_ids, buffer_point=None)
    zms_mglrs['Area'] = zms_mglrs.area

    zms = zms.join(zms_mglus
                   .reset_index(drop=False)
                   .merge(mg_mappings[['loc_id', 'CVE_SUN']].drop_duplicates())
                   .groupby('CVE_SUN')[['pobtot', 'Area']]
                   .sum()
                   .rename(columns={'pobtot': 'pob_urb', 'Area': 'Area_urb'})
                   )
    zms = zms.join(zms_mglrs
                   .reset_index(drop=False)
                   .merge(mg_mappings[['loc_id', 'CVE_SUN']].drop_duplicates())
                   .groupby('CVE_SUN')[['pobtot', 'Area']]
                   .sum()
                   .rename(columns={'pobtot': 'pob_rur', 'Area': 'Area_rur'})
                   )
    zms['Area_urb_pcnt'] = zms.Area_urb / zms.Area
    zms['Area_rur_pcnt'] = zms.Area_rur / zms.Area
    zms['pob_urb_pcnt'] = zms.pob_urb / zms.pobtot
    zms['pob_rur_pcnt'] = zms.pob_rur / zms.pobtot
    return zms


def load_geoms():
    print('loading zms')
    mg_mappings = region.ageb_ids_per_mpa()
    zms = region.mpa_all()
    zms['Area'] = zms.area
    zms = _zm_pops(zms, mg_mappings)

    print('loading agebs')
    zms_mun_ids = mg_mappings.mun_id.unique()
    zms_loc_ids = mg_mappings.loc_id.unique()
    zms_agebs = region.agebs(zms_mun_ids, zms_loc_ids)

    print('loading voronoi')
    tvor = tower.voronoi()
    tvor_x_zms = tower.voronoi_x_region('mpa')
    tvor_x_zms = tvor_x_zms[tvor_x_zms.CVE_SUN.isin(zms.index.astype(str))]
    zms_tvor = tvor.loc[set(tvor_x_zms.gtid)]

    zms_grids = {}
    for per_mun in [False, True]:
        for urb_only in [False, True]:
            print('=' * 20, 'loading grids', per_mun, urb_only, dt.datetime.now())
            zms_grids[(per_mun, urb_only)] = region.mpa_grids(G_side, per_mun, urb_only)
    return zms, zms_agebs, zms_tvor, zms_grids, mg_mappings


def interpolation(zms_grids):
    call_direction = 'out+in'
    aver = mex_helper.stat_tw_dow_aver_hr_uniq_user(call_direction)
    avg_tw = pd.DataFrame(aver['wd']).T

    avg_a = {}
    for by in ['area', 'pop']:
        t2a = tw_int.to_mpa_agebs(by)
        t2a.set_index('ageb', inplace=True)
        avg_a[by] = tw_int.interpolate_stats(avg_tw, t2a)

    avg_g = {}
    for by in ['area', 'pop']:
        for per_mun in [False, True]:
            for urb_only in [False, True]:
                grids = zms_grids[(per_mun, urb_only)]
                t2g = tw_int.to_mpa_grids(G_side, by=by, per_mun=per_mun, urb_only=urb_only, grids=grids)
                t2g.set_index('grid', inplace=True)
                avg_g[(by, per_mun, urb_only)] = tw_int.interpolate_stats(avg_tw, t2g)

    avg_idw = {}
    for per_mun in [False, True]:
        for urb_only in [False, True]:
            grids = zms_grids[(per_mun, urb_only)]
            avg_idw[(per_mun, urb_only)] = tw_int.interpolate_idw(avg_tw, G_side, per_mun=per_mun, urb_only=urb_only,
                                                                  max_k=10, grids=grids)

    return avg_tw, avg_a, avg_g, avg_idw


def compute_dilatation(avg_a, avg_g, avg_idw, zms, zms_agebs, zms_grids):
    # TODO: cannot handle 500*500 grids
    print('computing dv_a')
    dv_a = {}
    for by, avg in avg_a.items():
        area_col = 'Area'
        dv_a[by] = dilatation.dv_for_mpa_ageb(avg, zms, zms_agebs, area_col)

    print('computing dv_g')
    dv_g = {}
    for (by, per_mun, urb_only), avg in avg_g.items():
        # TODO: need to think about whether we want to use urban area when considering only per_mun
        # area_col = 'Area' if not per_mun else 'Area_urb'
        zms_g = zms_grids[(per_mun, urb_only)]
        dv_g[(by, per_mun, urb_only)] = dilatation.dv_for_mpa_grids(avg, zms, zms_g)

    print('computing dv_idw')
    dv_idw = {}
    for key, avg in avg_idw.items():
        area_col = 'Area'
        zms_g = zms_grids[key]
        dv_idw[key] = dilatation.dv_for_mpa_grids(avg, zms, zms_g, area_col)

    return dv_a, dv_g, dv_idw


def compute_hotspot_stats(avg_a, avg_g, avg_idw, zms, zms_agebs, zms_grids, mg_mappings, hotspot_type='loubar'):
    # compute hot stats
    hotspot_type = 'loubar'
    n_hs_a = {}
    comp_coef_a = {}
    for by in ['area', 'pop']:
        for per_mun in [False, True]:
            for urb_only in [False, True]:
                key = (by, per_mun, urb_only)
                print(key, end=' ')
                n, cc = ftr_hs.hs_stats_ageb(
                    avg_a[by], zms, zms_agebs, mg_mappings, per_mun, urb_only, hotspot_type)
                n_hs_a[key] = n
                comp_coef_a[key] = cc

    n_hs_g = {}
    comp_coef_g = {}
    for key, avg in avg_g.items():
        print(key, end=' ')
        by, per_mun, urb_only = key
        zms_g = zms_grids[(per_mun, urb_only)]
        # TODO: is it no need to pass on urb_only to has_stats_grid?
        n, cc = ftr_hs.hs_stats_grid(
            avg, zms, zms_g, per_mun, hotspot_type)
        n_hs_g[key] = n
        comp_coef_g[key] = cc

    n_hs_idw = {}
    comp_coef_idw = {}
    for key, avg in avg_idw.items():
        print(key, end=' ')
        per_mun, urb_only = key
        zms_g = zms_grids[key]
        # TODO: is it no need to pass on urb_only to has_stats_grid?
        n, cc = ftr_hs.hs_stats_grid(
            avg, zms, zms_g, per_mun, hotspot_type)
        n_hs_idw[key] = n
        comp_coef_idw[key] = cc
    return
