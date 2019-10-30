import numpy as np
from src.utils import loubar_thres
import src.utils.gis as gis
import src.mex.regions2010 as region
import pandas as pd


def keep_hotspot(avg, hotspot_type='loubar'):
    for h in avg:
        arr = avg[h]
        # arr can be all 0, which would break the loubar method, and there is no hotspot
        if arr.sum()==0:
            continue

        if hotspot_type == 'loubar':
            _, arr_thres = loubar_thres(arr, is_sorted=False)
        elif hotspot_type == 'average':
            arr_thres = np.mean(arr)
        else:
            raise ValueError('hotspot type', hotspot_type, 'not implemented')
        avg[h][avg[h] <= arr_thres] = 0
        # print(h, loubar, arr_thres)
    return avg


def avg_dist(geoms):
    if len(geoms) <= 1:
        return 0
    l = len(geoms)
    if len(geoms) > 40000:
        print(':::WARNING:::, too many grids, aborted')
        return np.nan
    pair_dist = gis.polys_centroid_pairwise_dist(geoms, dist_crs=geoms.crs).sum()
    return pair_dist / l / (l - 1)


def hs_stats_ageb(avg_a, zms, zms_agebs, mg_mapping, per_mun=False, hotspot_type='loubar'):
    n_hs_average = {}
    comp_coef = {}
    print('working on', end=' ')
    for sun, zm_mapping in mg_mapping.groupby('CVE_SUN'):
        print(sun, end=' ')
        zm = zms.loc[sun]
        zm_a = zms_agebs.loc[zm_mapping.ageb_id].copy()
        zm_avg_a = avg_a.loc[zm_a.index].copy()

        hs = HotSpot(zm_avg_a, zm_a, zm, hotspot_type)
        hs_avg = None

        # TODO: hs_stats can merge, they differ in how to obtain mun_level hotspot
        if per_mun:
            hs_avg = []
            for _, mun in zm_mapping[zm_mapping.Type == 'Urban'].groupby('mun_id'):
                mun_ageb_avg = avg_a.loc[mun.ageb_id].copy()
                if len(mun) < 10:
                    # print(mid, len(mun))
                    continue
                mun_hot = keep_hotspot(mun_ageb_avg, hotspot_type)
                hs_avg.append(mun_hot)
            hs_avg = pd.concat(hs_avg).reindex(zm_a.index, fill_value=0)

        hs.calc_stats(hs_avg)
        n_hs_average[sun] = hs.n_hs_average
        comp_coef[sun] = hs.compacity_coefficient
    print()
    return n_hs_average, comp_coef


def hs_stats_grid(avg_g, zms, zms_grids, per_mun=False, hotspot_type='loubar'):
    n_hs_average = {}
    comp_coef = {}
    print('working on', end=' ')
    for sun in sorted(zms.index):
        print(sun, end=' ')
        zm = zms.loc[sun]
        zm_g = zms_grids[zms_grids.CVE_SUN == sun].copy()
        zm_avg_g = avg_g.reindex(zm_g.index, fill_value=0).copy()
        hs = HotSpot(zm_avg_g, zm_g, zm, hotspot_type)
        hs_avg = None

        # TODO: hs_stats can merge, they differ in how to obtain mun_level hotspot
        if per_mun:
            hs_avg = []
            for _, mun_g in zm_g.groupby('mun_id'):
                mun_avg_g = avg_g.reindex(mun_g.index, fill_value=0).copy()
                # print(sun, mun_g.mun_id.iloc[0],'mun g not in avg', set(mun_g.index) - set(avg_g.index))
                # print('mun_avg_g isnull', mun_avg_g.isnull().sum(), mun_avg_g.shape)
                if len(mun_g) < 10 :
                    continue
                mun_hot = keep_hotspot(mun_avg_g, hotspot_type)
                hs_avg.append(mun_hot)
            hs_avg = pd.concat(hs_avg).reindex(zm_g.index, fill_value=0)

        hs.calc_stats(hs_avg)
        n_hs_average[sun] = hs.n_hs_average
        comp_coef[sun] = hs.compacity_coefficient
    print()
    return n_hs_average, comp_coef


class HotSpot:
    def __init__(self, avg, geoms, cover_region, hotspot_type='loubar'):
        self.sqrt_area = np.sqrt(cover_region.Area)
        self.avg = avg.copy()
        self.geoms = geoms
        self.region = cover_region
        self.hotspot_type = hotspot_type

    def calc_stats(self, hs_avg=None):
        self._get_hs(hs_avg)
        self._n_hs_persistence()
        self._hs_type_by_persistence()
        self._hs_type_distance()

    def _get_hs(self, hs_avg=None):
        if hs_avg is None:
            self.hs_avg = keep_hotspot(self.avg.copy(), self.hotspot_type)
        else:
            self.hs_avg = hs_avg

    def _n_hs_persistence(self):
        self.n_hs = (self.hs_avg != 0).sum(axis=0)
        self.n_hs_average = self.n_hs.mean()
        self.persistence = (self.hs_avg != 0).sum(axis=1)

    def _hs_type_by_persistence(self):
        persistence = self.persistence
        self.hs_permanent = persistence[persistence == 24]
        self.hs_intermediate = persistence[(persistence < 24) & (persistence >= 7)]
        self.hs_intermittent = persistence[(persistence < 7) & (persistence >= 1)]

        self.n_hs_per = len(self.hs_permanent)
        self.n_hs_med = len(self.hs_intermediate)
        self.n_hs_mit = len(self.hs_intermittent)

    def _hs_type_distance(self):
        d_per = avg_dist(self.geoms.loc[self.hs_permanent.index])
        d_med = avg_dist(self.geoms.loc[self.hs_intermediate.index])
        d_mit = avg_dist(self.geoms.loc[self.hs_intermittent.index])

        self.compacity_coefficient = d_per / self.sqrt_area
        self.d_per_med = d_per / d_med if d_med != 0 else np.nan
        self.d_med_mit = d_med / d_mit if d_mit != 0 else np.nan
