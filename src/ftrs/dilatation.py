import src.utils.gis as gis
import numpy as np
import pandas as pd
import src.mex.regions2010 as region


def compute_dv(geoms, s, sqrt_area):
    geom_dist = gis.polys_centroid_pairwise_dist(geoms, dist_crs=geoms.crs)
    dv = {}
    for t in s.columns:
        st_outer = np.outer(s[t], s[t])
        np.fill_diagonal(st_outer, 0)
        dv[int(t)] = (st_outer * geom_dist).sum() / st_outer.sum() / sqrt_area
    return dv


def dv_for_mpa_ageb(avg_ageb, zms, zms_agebs, area_col):
    mg_mappings = region.ageb_ids_per_mpa()
    dv_zms = {}
    for sun in zms.index:
        sqrt_area = np.sqrt(zms.loc[sun,area_col])
        zm_ageb_ids = mg_mappings[mg_mappings.CVE_SUN == sun].ageb_id
        zm_agebs = zms_agebs.loc[zm_ageb_ids]
        zm_agebs_avg = avg_ageb.loc[zm_ageb_ids]

        s = zm_agebs_avg / zm_agebs_avg.sum()
        dv = compute_dv(zm_agebs, s, sqrt_area)
        dv_zms[sun] = dv

    dv_zms = pd.DataFrame(dv_zms).T
    dv_zms.columns = [f'dv_{c:02}' for c in dv_zms.columns]
    dv_zms['dilatation coefficient'] = dv_zms.max(axis=1) / dv_zms.min(axis=1)
    return dv_zms


def dv_for_mpa_grids(grids_avg, zms, zms_grids, area_col):
    dv_zms = {}
    for sun in zms.index:
        sqrt_area = np.sqrt(zms.loc[sun, area_col])
        zm_grids = zms_grids[zms_grids.CVE_SUN == sun]
        zm_g_avg = grids_avg.reindex(zm_grids.index, fill_value=0)

        s = zm_g_avg / zm_g_avg.sum()
        dv = compute_dv(zm_grids, s, sqrt_area)
        dv_zms[sun] = dv
    dv_zms = pd.DataFrame(dv_zms).T
    dv_zms.columns = [f'dv_{c:02}' for c in dv_zms.columns]
    dv_zms['dilatation coefficient'] = dv_zms.max(axis=1) / dv_zms.min(axis=1)
    return dv_zms
