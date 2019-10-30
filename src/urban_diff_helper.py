import src.tower_interpolation as tw_int
import src.ftrs.hotspot as ftr_hs
import src.mex.regions2010 as region
import src.mex.tower as tower
import src.mex_helper as mex_helper
import src.ftrs.dilatation as dilatation
import pandas as pd

zms_sort_cols = ['Area', 'Area_urb', 'Area_rur', 'Area_urb_pcnt', 'Area_rur_pcnt', 'pobtot', 'pob_urb', 'pob_rur',
                 'pob_urb_pcnt', 'pob_rur_pcnt']


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
    mg_mappings = region.ageb_ids_per_mpa()
    zms = region.mpa_all()
    zms['Area'] = zms.area
    zms = _zm_pops(zms, mg_mappings)

    zms_mun_ids = mg_mappings.mun_id.unique()
    zms_loc_ids = mg_mappings.loc_id.unique()
    zms_agebs = region.agebs(zms_mun_ids, zms_loc_ids)

    tvor = tower.voronoi()
    tvor_x_zms = tower.voronoi_x_region('mpa')
    tvor_x_zms = tvor_x_zms[tvor_x_zms.CVE_SUN.isin(zms.index.astype(str))]
    zms_tvor = tvor.loc[set(tvor_x_zms.gtid)]

    zms_grids = {}
    for side in [1000, 2000]:
        for per_mun in [False, True]:
            zms_grids[(side, per_mun)] = region.mpa_grids(side, per_mun)
    return zms, zms_agebs, zms_tvor, zms_grids, mg_mappings


def interpolation():
    call_direction = 'out+in'
    aver = mex_helper.stat_tw_dow_aver_hr_uniq_user(call_direction)
    avg_tw = pd.DataFrame(aver['wd']).T

    avg_a = {}
    for by in ['area', 'pop']:
        t2a = tw_int.to_mpa_agebs(by)
        t2a.set_index('ageb', inplace=True)
        avg_a[by] = tw_int.interpolate_stats(avg_tw, t2a)

    avg_g = {}
    for side in [1000, 2000]:
        for by in ['area', 'pop']:
            for per_mun in [False, True]:
                t2g = tw_int.to_mpa_grids(side, by=by, per_mun=per_mun)
                t2g.set_index('grid', inplace=True)
                avg_g[(side, by, per_mun)] = tw_int.interpolate_stats(avg_tw, t2g)

    avg_idw = {}
    for side in [1000, 2000]:
        for per_mun in [False, True]:
            avg_idw[(side, per_mun)] = tw_int.interpolate_idw(avg_tw, side, per_mun=per_mun, max_k=10)

    return avg_tw, avg_a, avg_g, avg_idw


def compute_dilatation(avg_a, avg_g, avg_idw, zms, zms_agebs, zms_grids):
    print('computing dv_a')
    dv_a = {}
    for by, avg in avg_a.items():
        area_col= 'Area'
        dv_a[by] = dilatation.dv_for_mpa_ageb(avg, zms, zms_agebs, area_col)

    print('computing dv_g')
    dv_g = {}
    for (side, by, per_mun), avg in avg_g.items():
        # TODO: need to think about whether we want to use urban area when considering only per_mun
        # area_col = 'Area' if not per_mun else 'Area_urb'
        zms_g = zms_grids[(side, per_mun)]
        dv_g[(side, by, per_mun)] = dilatation.dv_for_mpa_grids(avg, zms, zms_g)

    print('computing dv_idw')
    dv_idw = {}
    for key, avg in avg_idw.items():
        area_col= 'Area'
        zms_g = zms_grids[key]
        dv_idw[key] = dilatation.dv_for_mpa_grids(avg, zms, zms_g, area_col)

    return dv_a, dv_g, dv_idw


def compute_hotspot_stats(avg_a, avg_g, avg_idw, zms, zms_agebs, zms_grids, mg_mapping, hotspot_type='loubar'):
    for by in ['area', 'pop']:
        for per_mun in [False, True]:
            n_hs_average, comp_coef = ftr_hs.hs_stats_ageb(avg_a[by], zms, zms_agebs, mg_mapping, per_mun,
                                                                 hotspot_type)

    return