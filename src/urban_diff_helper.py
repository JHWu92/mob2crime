import datetime
import sys

sys.path.insert(0, '/home/Jiahui/mob2crime')
import src.tower_interpolation as tw_int
import src.ftrs.hotspot as ftr_hs
import src.mex.regions2010 as region
import src.mex.tower as tower
import src.mex_helper as mex_helper
import src.ftrs.dilatation as dilatation
import pandas as pd
import datetime as dt

PER_MUN_DISPLAY = lambda x: 'PerMuni' if x else 'Metro'
URB_ONLY_DISPLAY = lambda x: 'U' if x else 'UR'
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

    zms_sub_vors = {}
    for per_mun in [False, True]:
        for urb_only in [False, True]:
            #         print('=' * 20, 'loading grids', per_mun, urb_only, dt.datetime.now())
            zms_sub_vors[(per_mun, urb_only)] = region.mpa_vors(per_mun, urb_only)
    return zms, zms_agebs, zms_tvor, zms_grids, zms_sub_vors, mg_mappings


def interpolation(zms_grids, zms_sub_vors, n_bins=24, average_over_observed_day=False):
    call_direction = 'out+in'
    # if average_over_observed_day==False
    # average over the number of days in the observation period
    # 75% of the towers are the same as average_over_observed_day==True
    aver = mex_helper.stat_tw_dow_aver_hr_uniq_user(call_direction, n_bins=n_bins,
                                                    average_over_observed_day=average_over_observed_day)
    avg_tw = pd.DataFrame(aver['wd']).T
    assert avg_tw.shape[1] == n_bins

    avg_a = {}
    for by in ['area', 'pop']:
        t2a = tw_int.to_mpa_agebs(by)
        t2a.set_index('ageb', inplace=True)
        avg_a[by] = tw_int.interpolate_stats(avg_tw, t2a, n_bins=n_bins)

    avg_g = {}
    for by in ['area', 'pop']:
        for per_mun in [False, True]:
            for urb_only in [False, True]:
                grids = zms_grids[(per_mun, urb_only)]
                t2g = tw_int.to_mpa_grids(G_side, by=by, per_mun=per_mun, urb_only=urb_only, grids=grids)
                t2g.set_index('grid', inplace=True)
                avg_g[(by, per_mun, urb_only)] = tw_int.interpolate_stats(avg_tw, t2g, n_bins=n_bins)

    avg_idw = {}
    for per_mun in [False, True]:
        for urb_only in [False, True]:
            grids = zms_grids[(per_mun, urb_only)]
            avg_idw[(per_mun, urb_only)] = tw_int.interpolate_idw(avg_tw, G_side, per_mun=per_mun, urb_only=urb_only,
                                                                  max_k=10, grids=grids, n_bins=n_bins)

    avg_vor = {}
    for by in ['area', 'pop']:
        for per_mun in [False, True]:
            for urb_only in [False, True]:
                sub_vors = zms_sub_vors[(per_mun, urb_only)]
                t2v = tw_int.to_mpa_vors(by=by, per_mun=per_mun, urb_only=urb_only, zms_vors=sub_vors)
                t2v.set_index('vor', inplace=True)
                avg_vor[(by, per_mun, urb_only)] = tw_int.interpolate_stats(avg_tw, t2v, n_bins=n_bins)

    return avg_tw, avg_a, avg_g, avg_idw, avg_vor


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


def compute_hotspot_stats(avg_a, avg_g, avg_idw, avg_tw, avg_vor,
                          zms, zms_agebs, zms_grids, zms_sub_vors,
                          mg_mappings, hs_type='loubar', loading=(), area_normalization=False, verbose=0,
                          roll_width=3, chunk_width=4):
    # compute hot stats
    hs_stats_ageb = {}
    if 'ageb' in loading:
        print('=' * 20, 'ageb')
        for by in ['area', 'pop']:
            for per_mun in [False, True]:
                for urb_only in [False, True]:
                    key = (by, per_mun, urb_only)
                    print(key, end=' ')
                    stats = ftr_hs.hs_stats_ageb(avg_a[by], zms, zms_agebs, mg_mappings, by, per_mun, urb_only,
                                                 area_normalized=area_normalization, hotspot_type=hs_type,
                                                 verbose=verbose, roll_width=roll_width, chunk_width=chunk_width)
                    hs_stats_ageb[key] = stats
        print(datetime.datetime.now())

    hs_stats_g = {}
    if 'grid' in loading:
        print('=' * 20, 'grid')
        for key, avg in avg_g.items():
            print(key, end=' ')
            by, per_mun, urb_only = key
            zms_g = zms_grids[(per_mun, urb_only)]
            stats = ftr_hs.hs_stats_grid_or_vor(avg, zms, zms_g, 'grid', by, per_mun, urb_only,
                                                area_normalized=area_normalization, hotspot_type=hs_type,
                                                verbose=verbose, roll_width=roll_width, chunk_width=chunk_width)
            hs_stats_g[key] = stats
        print(datetime.datetime.now())

    hs_stats_idw = {}
    if 'idw' in loading:
        print('=' * 20, 'idw')
        for key, avg in avg_idw.items():
            print(key, end=' ')
            per_mun, urb_only = key
            by = 'idw'
            zms_g = zms_grids[key]
            stats = ftr_hs.hs_stats_grid_or_vor(avg, zms, zms_g, 'grid', by, per_mun, urb_only,
                                                area_normalized=area_normalization, hotspot_type=hs_type,
                                                verbose=verbose, roll_width=roll_width, chunk_width=chunk_width)
            hs_stats_idw[key] = stats
        print(datetime.datetime.now())

    hs_stats_vor = {}
    if 'vor' in loading:
        print('=' * 20, 'vor')
        for key, avg in avg_vor.items():
            print(key, end=' ')
            by, per_mun, urb_only = key
            zms_vor = zms_sub_vors[(per_mun, urb_only)]
            # the comp coef seems to be computed using geometric centroid, not tower location
            stats = ftr_hs.hs_stats_grid_or_vor(avg, zms, zms_vor, 'vor', by, per_mun, urb_only,
                                                area_normalized=area_normalization, hotspot_type=hs_type,
                                                verbose=verbose, roll_width=roll_width, chunk_width=chunk_width)
            hs_stats_vor[key] = stats
        print(datetime.datetime.now())

    hs_stats_tw = {}
    if 'tw' in loading:
        for per_mun in [False, True]:
            for urb_only in [False, True]:
                key = (per_mun, urb_only)
                print(key, end=' ')
                stats = ftr_hs.hs_stats_tw(avg_tw, zms, per_mun, urb_only,
                                           area_normalized=area_normalization, hotspot_type=hs_type,
                                           verbose=verbose, roll_width=roll_width, chunk_width=chunk_width)
                hs_stats_tw[key] = stats
    return hs_stats_ageb, hs_stats_g, hs_stats_idw, hs_stats_vor, hs_stats_tw


if __name__ == "__main__":
    # run this first to cache time consuming intermediate results
    import os

    print(os.getcwd())
    print(datetime.datetime.now())
    LOADING = ('ageb', 'grid', 'idw', 'vor',)
    N_BINS = 24
    AREA_NORMALIZATION = False
    ROLL_WIDTH = 5
    CHUNK_WIDTH = 6
    print('loading', LOADING)
    print('n bins:', N_BINS)
    print('area_normalization:', AREA_NORMALIZATION)
    print(f'roll width={ROLL_WIDTH}, chunk width={CHUNK_WIDTH}')

    raster_use_p_centroid_if_none, average_over_observed_day = False, True
    # raster_use_p_centroid_if_none, average_over_observed_day = True, False
    # make sure only these two combination is allowed
    assert (raster_use_p_centroid_if_none, average_over_observed_day) in [(False, True), (True, False)]

    # the buggy version
    if (raster_use_p_centroid_if_none, average_over_observed_day) == (False, True):
        ftr_hs.MEASURES_DIR = 'data/mex_hotspot_measures_buggy'
        ftr_hs.RASTER_USE_P_CENTROID_IF_NONE = raster_use_p_centroid_if_none
    print('ftr_hs.MEASURES_DIR', ftr_hs.MEASURES_DIR)
    print('ftr_hs.RASTER_USE_P_CENTROID_IF_NONE', ftr_hs.RASTER_USE_P_CENTROID_IF_NONE)

    ZMS, ZMS_AGEBS, ZMS_TVOR, ZMS_GRIDS, ZMS_SUB_VORS, MG_MAPPINGS = load_geoms()

    # cache avg_idw
    AVG_TW, AVG_A, AVG_G, AVG_IDW, AVG_VOR = interpolation(ZMS_GRIDS, ZMS_SUB_VORS, N_BINS,
                                                           average_over_observed_day=average_over_observed_day)
    print(datetime.datetime.now())

    # caches the compactness results
    compute_hotspot_stats(AVG_A, AVG_G, AVG_IDW, AVG_TW, AVG_VOR, ZMS, ZMS_AGEBS, ZMS_GRIDS, ZMS_SUB_VORS, MG_MAPPINGS,
                          loading=LOADING, area_normalization=AREA_NORMALIZATION,
                          roll_width=ROLL_WIDTH, chunk_width=CHUNK_WIDTH)
    print(datetime.datetime.now())
