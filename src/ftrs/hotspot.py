import numpy as np
from src.utils import loubar_thres
import src.utils.gis as gis
import src.mex.regions2010 as region
import pandas as pd

HOME_HOURS = ['0', '1', '2', '3', '4', '5', '6', '22', '23']  # 10pm - 7am
WORK_HOURS = ['9', '10', '11', '12', '13', '14', '15', '16', '17']  # 9am - 6pm

def keep_hotspot(avg, hotspot_type='loubar'):
    for h in avg:
        arr = avg[h]
        # arr can be all 0, which would break the loubar method, and there is no hotspot
        if arr.sum() == 0:
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


def hs_stats_tw(avg_tw, zms, per_mun=False, urb_only=False, hotspot_type='loubar'):
    import src.mex.tower as tower
    tXzms = tower.pts_x_region('mpa', per_mun, urb_only)
    t_pts = tower.pts().set_index('gtid')
    n_hs_average = {}
    comp_coef = {}
    comp_coef_home = {}
    comp_coef_work = {}

    print('working on', end=' ')
    for sun, zm_mapping in tXzms.groupby('CVE_SUN'):
        print(sun, end=' ')

        zm = zms.loc[sun]
        zm_t = t_pts.loc[zm_mapping.gtid].copy()
        zm_avg_t = avg_tw.reindex(zm_mapping.gtid, fill_value=0).copy()
        hs = HotSpot(zm_avg_t, zm_t, zm, hotspot_type)
        hs_avg = None

        if per_mun:
            # build hs_avg per mun
            hs_avg = []
            for mun_id, mun in zm_mapping.groupby('mun_id'):
                mun_avg_t = zm_avg_t.loc[mun.gtid].copy()

                if len(mun_avg_t) <= 1:
                    print(f'(`{mun_id}`={len(mun_avg_t)} no hotspot)', end=' ')
                    continue
                elif len(mun_avg_t) < 3:
                    # TODO: loubar is likely not working when the array is small, force to use average
                    print(f'(`{mun_id}`={len(mun_avg_t)} using `average`)', end=' ')
                    mun_hot = keep_hotspot(mun_avg_t, hotspot_type='average')
                else:
                    mun_hot = keep_hotspot(mun_avg_t.copy(), hotspot_type)

                hs_avg.append(mun_hot)
            # concat hs_avg, in case hs_avg is empty
            if len(hs_avg) != 0:
                hs_avg = pd.concat(hs_avg).reindex(zm_avg_t.index, fill_value=0)
            else:
                # each mun has just one tower
                print(f'(`zm{sun}` has no mun_hotspot)')
                hs_avg = pd.DataFrame([], index=zm_avg_t.index, columns=zm_avg_t.columns).fillna(0)

        hs.calc_stats(hs_avg)
        n_hs_average[sun] = hs.n_hs_average
        comp_coef[sun] = hs.compacity_coefficient
        comp_coef_home[sun] = hs.comp_coef_home
        comp_coef_work[sun] = hs.comp_coef_work
    print()
    return n_hs_average, comp_coef, comp_coef_home, comp_coef_work


def hs_stats_ageb(avg_a, zms, zms_agebs, mg_mapping, per_mun=False, urb_only=False, hotspot_type='loubar'):
    n_hs_average = {}
    comp_coef = {}
    comp_coef_home = {}
    comp_coef_work = {}
    print('working on', end=' ')
    for sun, zm_mapping in mg_mapping.groupby('CVE_SUN'):
        print(sun, end=' ')
        if urb_only:
            zm_mapping = zm_mapping[zm_mapping.Type == 'Urban']
        zm = zms.loc[sun]
        zm_a = zms_agebs.loc[zm_mapping.ageb_id].copy()
        zm_avg_a = avg_a.loc[zm_a.index].copy()

        hs = HotSpot(zm_avg_a, zm_a, zm, hotspot_type)
        hs_avg = None

        # TODO: hs_stats can merge, they differ in how to obtain mun_level hotspot
        if per_mun:
            hs_avg = []
            for _, mun in zm_mapping.groupby('mun_id'):
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
        comp_coef_home[sun] = hs.comp_coef_home
        comp_coef_work[sun] = hs.comp_coef_work
    print()
    return n_hs_average, comp_coef, comp_coef_home, comp_coef_work


def hs_stats_grid(avg_g, zms, zms_grids, per_mun=False, hotspot_type='loubar'):
    n_hs_average = {}
    comp_coef = {}
    comp_coef_home = {}
    comp_coef_work = {}
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
                if len(mun_g) < 10:
                    continue
                mun_hot = keep_hotspot(mun_avg_g, hotspot_type)
                hs_avg.append(mun_hot)
            hs_avg = pd.concat(hs_avg).reindex(zm_g.index, fill_value=0)

        hs.calc_stats(hs_avg)
        n_hs_average[sun] = hs.n_hs_average
        comp_coef[sun] = hs.compacity_coefficient
        comp_coef_home[sun] = hs.comp_coef_home
        comp_coef_work[sun] = hs.comp_coef_work
    print()
    return n_hs_average, comp_coef, comp_coef_home, comp_coef_work


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
        self.persistence_home = (self.hs_avg[HOME_HOURS] != 0).sum(axis=1)
        self.persistence_work = (self.hs_avg[WORK_HOURS] != 0).sum(axis=1)

    def _hs_type_by_persistence(self):
        persistence = self.persistence
        self.hs_permanent = persistence[persistence == 24]
        self.hs_intermediate = persistence[(persistence < 24) & (persistence >= 7)]
        self.hs_intermittent = persistence[(persistence < 7) & (persistence >= 1)]

        self.n_hs_per = len(self.hs_permanent)
        self.n_hs_med = len(self.hs_intermediate)
        self.n_hs_mit = len(self.hs_intermittent)

        self.hs_permanent_home = self.persistence_home[self.persistence_home==len(HOME_HOURS)]
        self.hs_permanent_work = self.persistence_work[self.persistence_work==len(WORK_HOURS)]
        self.n_hs_per_home = len(self.hs_permanent_home)
        self.n_hs_per_work = len(self.hs_permanent_work)

    def _hs_type_distance(self):
        d_per = avg_dist(self.geoms.loc[self.hs_permanent.index])
        d_med = avg_dist(self.geoms.loc[self.hs_intermediate.index])
        d_mit = avg_dist(self.geoms.loc[self.hs_intermittent.index])

        self.compacity_coefficient = d_per / self.sqrt_area
        self.d_per_med = d_per / d_med if d_med != 0 else np.nan
        self.d_med_mit = d_med / d_mit if d_mit != 0 else np.nan

        d_per_home = avg_dist(self.geoms.loc[self.hs_permanent_home.index])
        d_per_work = avg_dist(self.geoms.loc[self.hs_permanent_work.index])
        self.comp_coef_home = d_per_home / self.sqrt_area
        self.comp_coef_work = d_per_work / self.sqrt_area
