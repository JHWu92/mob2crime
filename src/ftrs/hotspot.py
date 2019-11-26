import json
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import src.utils.gis as gis
from src.ftrs.compactness_index import compacity_coefficient, cohesion, proximity, moment_inertia, \
    mass_moment_inertia
from src.utils import loubar_thres

HOME_HOURS = ['0', '1', '2', '3', '4', '5', '6', '22', '23']  # 10pm - 7am
WORK_HOURS = ['9', '10', '11', '12', '13', '14', '15', '16', '17']  # 9am - 6pm
MEASURES_DIR = 'data/mex_hotspot_measures'

PER_MUN_DISPLAY = lambda x: 'PerMun' if x else 'Metro'
URB_ONLY_DISPLAY = lambda x: 'Urban' if x else 'UrbanRural'
ADMIN_STR = lambda x, y: f'{PER_MUN_DISPLAY(x)}_{URB_ONLY_DISPLAY(y)}'


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


def _handle_geoms_pairdist_input(geoms=None, pair_dist=None):
    if geoms is None and pair_dist is None:
        raise ValueError('either geoms or pair_dist should be passed')
    if geoms is not None and pair_dist is not None:
        print('computing pair distance matrix using geoms, pair_dist passed is ignored')
    if pair_dist is None:
        pair_dist = pair_dist_matrix(geoms)
    return pair_dist


def avg_dist(geoms=None, pair_dist=None):
    if len(geoms) == 1:
        # only 1 geometry, distance is 0
        return 0
    pair_dist = _handle_geoms_pairdist_input(geoms, pair_dist)
    n = len(pair_dist)
    # The pair_dist is not wrong: it is a n*n matrix, the diagonal is zeros, there are n*(n-1) pairs
    return pair_dist.sum() / n / (n - 1)


def avg_dist_square(geoms=None, pair_dist=None):
    pair_dist = _handle_geoms_pairdist_input(geoms, pair_dist)
    n = len(pair_dist)
    # The pair_dist is not wrong: it is a n*n matrix, the diagonal is zeros, there are n*(n-1) pairs
    return (pair_dist ** 2).sum() / n / (n - 1)


def pair_dist_matrix(geoms):
    if len(geoms) <= 1:
        return 0
    if len(geoms) > 40000:
        raise ValueError('pair_dist_matrix:::WARNING:::, too many grids, aborted')
    pair_dist = gis.polys_centroid_pairwise_dist(geoms, dist_crs=geoms.crs)
    return pair_dist


def hs_stats_tw(avg_tw, zms, per_mun=False, urb_only=False, hotspot_type='loubar', verbose=0):
    import src.mex.tower as tower
    tXzms = tower.pts_x_region('mpa', per_mun, urb_only)
    t_pts = tower.pts().set_index('gtid')

    n_hs = {}
    compactness = {}
    print('working on', end=' ')
    for sun, zm_mapping in tXzms.groupby('CVE_SUN'):
        print(sun, end=' ')

        zm = zms.loc[sun]
        zm_t = t_pts.loc[zm_mapping.gtid].copy()
        zm_avg_t = avg_tw.reindex(zm_mapping.gtid, fill_value=0).copy()
        fn_pref = f'{MEASURES_DIR}/tw_{ADMIN_STR(per_mun, urb_only)}_ZM{sun}'
        hs = HotSpot(zm_avg_t, zm_t, zm, hotspot_type, verbose=verbose, fn_pref=fn_pref)
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
        n_hs[sun] = hs.n_hs
        compactness[sun] = hs.compactness
    print()
    return {'n_hs': n_hs, 'compactness': compactness}


def hs_stats_ageb(avg_a, zms, zms_agebs, mg_mapping,
                  by='area', per_mun=False, urb_only=False, hotspot_type='loubar', verbose=0):
    n_hs = {}
    compactness = {}
    print('working on', end=' ')
    for sun, zm_mapping in mg_mapping.groupby('CVE_SUN'):
        if verbose: print('=' * 10, end=' ')
        print(sun, end=' ')
        if urb_only:
            zm_mapping = zm_mapping[zm_mapping.Type == 'Urban']
        zm = zms.loc[sun]
        zm_a = zms_agebs.loc[zm_mapping.ageb_id].copy()
        zm_avg_a = avg_a.loc[zm_a.index].copy()

        fn_pref = f'{MEASURES_DIR}/ageb_{by}_{ADMIN_STR(per_mun, urb_only)}_ZM{sun}'
        hs = HotSpot(zm_avg_a, zm_a, zm, hotspot_type, verbose=verbose, fn_pref=fn_pref)
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
        n_hs[sun] = hs.n_hs
        compactness[sun] = hs.compactness
    print()
    return {'n_hs': n_hs, 'compactness': compactness}


def hs_stats_grid_or_vor(avg_geom, zms, zms_geoms, geom_type='grid', by='area', per_mun=False, urb_only=False,
                         hotspot_type='loubar', verbose=0):
    """grid and vor has the same formats"""
    assert geom_type in ('grid', 'vor')
    n_hs = {}
    compactness = {}
    print('working on', end=' ')
    for sun in sorted(zms.index):
        print(sun, end=' ')
        zm = zms.loc[sun]
        zm_g = zms_geoms[zms_geoms.CVE_SUN == sun].copy()
        zm_avg_g = avg_geom.reindex(zm_g.index, fill_value=0).copy()
        fn_pref = f'{MEASURES_DIR}/{geom_type}_{by}_{ADMIN_STR(per_mun, urb_only)}_ZM{sun}'
        hs = HotSpot(zm_avg_g, zm_g, zm, hotspot_type, verbose=verbose, fn_pref=fn_pref)
        hs_avg = None

        # TODO: hs_stats can merge, they differ in how to obtain mun_level hotspot
        if per_mun:
            hs_avg = []
            for _, mun_g in zm_g.groupby('mun_id'):
                mun_avg_g = avg_geom.reindex(mun_g.index, fill_value=0).copy()
                # print(sun, mun_g.mun_id.iloc[0],'mun g not in avg', set(mun_g.index) - set(avg_g.index))
                # print('mun_avg_g isnull', mun_avg_g.isnull().sum(), mun_avg_g.shape)
                # TODO: I don't remember why set 10 for grid. 10 doesn't work for Vor, sun 15 always < 10 in PerMun True
                if geom_type == 'grid' and len(mun_g) < 10:
                    continue
                if geom_type == 'vor' and len(mun_g) < 2:
                    continue
                mun_hot = keep_hotspot(mun_avg_g, hotspot_type)
                hs_avg.append(mun_hot)

            hs_avg = pd.concat(hs_avg).reindex(zm_g.index, fill_value=0)

        hs.calc_stats(hs_avg)
        n_hs[sun] = hs.n_hs
        compactness[sun] = hs.compactness
    print()
    return {'n_hs': n_hs, 'compactness': compactness}


class HotSpot:
    def __init__(self, avg, geoms, cover_region, hotspot_type='loubar', raster_resolution=100, verbose=0, fn_pref=None):
        self.sqrt_area = np.sqrt(cover_region.Area)
        self.avg = avg.copy()
        self.geoms = geoms
        self.region = cover_region
        self.hotspot_type = hotspot_type
        self.raster_resolution = raster_resolution
        self.verbose = verbose
        self.fn_pref = fn_pref

    def calc_stats(self, hs_avg=None):
        self._get_hs(hs_avg)
        self._number_of_hs()
        self._hs_type_by_persistence()
        self._hs_all_compactness()

    def _get_hs(self, hs_avg=None):
        if self.verbose: print('masking out non hot spot, defined by', self.hotspot_type)
        if hs_avg is None:
            self.hs_avg = keep_hotspot(self.avg.copy(), self.hotspot_type)
        else:
            self.hs_avg = hs_avg

    def _number_of_hs(self):
        if self.verbose: print('computing number of hot spot per hour')
        n_hs_hourly = (self.hs_avg != 0).sum(axis=0)
        n_hs_average = n_hs_hourly.mean()
        self.n_hs = {'average': n_hs_average, 'hourly': n_hs_hourly}

    def _hs_type_by_persistence(self):
        if self.verbose: print('computing persistency and obtaining permanent hot spots')
        persistence = (self.hs_avg != 0).sum(axis=1)
        persistence_home = (self.hs_avg[HOME_HOURS] != 0).sum(axis=1)
        persistence_work = (self.hs_avg[WORK_HOURS] != 0).sum(axis=1)

        self.hs_permanent = persistence[persistence == 24]
        self.n_hs_per = len(self.hs_permanent)

        self.hs_permanent_home = persistence_home[persistence_home == len(HOME_HOURS)]
        self.n_hs_per_home = len(self.hs_permanent_home)

        self.hs_permanent_work = persistence_work[persistence_work == len(WORK_HOURS)]
        self.n_hs_per_work = len(self.hs_permanent_work)

        # self.hs_intermediate = persistence[(persistence < 24) & (persistence >= 7)]
        # self.n_hs_med = len(self.hs_intermediate)
        # self.hs_intermittent = persistence[(persistence < 7) & (persistence >= 1)]
        # self.n_hs_mit = len(self.hs_intermittent)

    def _calc_compactness(self, hs_index, hs_count):
        target_hs = self.geoms.loc[hs_index]
        # TODO: this is a simplify version of density, assuming counts in the shape is uniformly
        #  distributed, but the underlying density is not, the smallest unit of density should
        #  be the interection of vor and ageb
        hs_density = hs_count / target_hs.area
        hs_density.name = 'Density'

        # index: compacity
        hs_pair_d_avg = avg_dist(target_hs)  # distance among hotspots
        comp_coef = compacity_coefficient(hs_pair_d_avg, self.sqrt_area)

        # ----------------
        # rasterize hot spots
        if self.verbose: print(f'raster {len(target_hs)} hot spots with resolution: {self.raster_resolution}m', end='')
        # TODO: no idea on how to choose area_pcnt_thres. Clipping the grids won't fit the MI raster equation.
        #  Not Clipping will bring much extra area.
        #  Rastering first is very time consuming. But I am not sure about the vector form formula
        # raster_rper = gis.gp_polys_to_grids(target_hs, pname=target_hs.index.name, side=self.raster_resolution,
        #                                     no_grid_by_area=True, clip_by_poly=False, area_pcnt_thres=0.2)
        # raster_rper.crs = target_hs.crs
        # raster_rper = raster_rper.merge(hs_density.reset_index())
        # raster_rper['Area'] = raster_rper.area
        # raster_rper['Mass'] = raster_rper.Area * raster_rper.Density
        raster_rper = gis.gp_polys_to_raster_centroids(target_hs, side=self.raster_resolution,
                                                       pname=target_hs.index.name)
        raster_rper = raster_rper.merge(hs_density.reset_index())
        raster_rper['Area'] = self.raster_resolution ** 2
        raster_rper['Mass'] = raster_rper.Area * raster_rper.Density
        if self.verbose: print(' into ', len(raster_rper), 'grids')

        # area centroid and mass centroid
        # TODO: there are some overlapping polygons:
        #  rural agebs with point locations are buffered into circles.
        #  These circles overlap. Causing the following areal centroid isn't equal to cascasd_union.centroid
        rx = raster_rper.centroid.apply(lambda x: x.coords[0][0])
        ry = raster_rper.centroid.apply(lambda x: x.coords[0][1])
        cx = (rx * raster_rper.Area).sum() / raster_rper.Area.sum()
        cy = (ry * raster_rper.Area).sum() / raster_rper.Area.sum()
        raster_centroid = (cx, cy)
        cx = (rx * raster_rper.Mass).sum() / raster_rper.Mass.sum()
        cy = (ry * raster_rper.Mass).sum() / raster_rper.Mass.sum()
        raster_mass_centroid = (cx, cy)
        raster_rper_centroids = raster_rper.centroid.apply(lambda x: x.coords[0]).tolist()
        # if self.verbose: print(f'raster_centroid: {raster_centroid}, raster_mass_centroids: {raster_mass_centroid}')
        # rasterize pairwise and to centroid distance
        # pairwise_dist_square_avg = avg_dist_square(raster_rper)
        pairwise_dist_square_avg = gis.pairwise_dist_average(np.array(raster_rper_centroids))
        d2centroid = cdist(raster_rper_centroids, [raster_centroid])[:, 0]
        d2centroid_avg = d2centroid.mean()
        d2mass_centroid = cdist(raster_rper_centroids, [raster_mass_centroid])[:, 0]
        # ----------------

        # hotspots equivalent circle
        ref_circle_radius = np.sqrt(raster_rper.Area.sum() / np.pi)

        # indexes
        coh = cohesion(ref_circle_radius, pairwise_dist_square_avg)
        prox = proximity(ref_circle_radius, d2centroid_avg)
        nmi = moment_inertia(raster_rper.Area, d2centroid)
        nmmi = mass_moment_inertia(raster_rper, d2mass_centroid)

        return {'comp_coef': comp_coef, 'cohesion': coh, 'proximity': prox, 'NMI': nmi, 'NMMI': nmmi}

    def _hs_all_compactness(self):
        compactness_fn = f'{self.fn_pref}_compactness.json'
        if self.fn_pref and os.path.exists(compactness_fn):
            if self.verbose: print('loading existing compactness at', compactness_fn)
            with open(compactness_fn, 'r') as f:
                self.compactness = json.load(f)
            return

        if self.verbose: print('computing compactness indexes for all day')
        hs_index = self.hs_permanent.index
        hs_count = self.hs_avg.loc[hs_index].mean(axis=1)
        compact_index_all_day = self._calc_compactness(hs_index, hs_count)

        if self.verbose: print('computing compactness indexes for Home time')
        hs_index = self.hs_permanent_work.index
        hs_count = self.hs_avg.loc[hs_index].mean(axis=1)
        compact_index_work = self._calc_compactness(hs_index, hs_count)

        if self.verbose: print('computing compactness indexes for work time')
        hs_index = self.hs_permanent_home.index
        hs_count = self.hs_avg.loc[hs_index].mean(axis=1)
        compact_index_home = self._calc_compactness(hs_index, hs_count)

        if self.verbose: print('computing compactness indexes for hourly')
        compact_index_hourly = []
        for hour in self.hs_avg:
            hs_count_hourly = self.hs_avg[hour]
            hs_count_hourly = hs_count_hourly[hs_count_hourly != 0]
            c_index = self._calc_compactness(hs_count_hourly.index, hs_count_hourly)
            c_index['hour'] = hour
            compact_index_hourly.append(c_index)
        compact_index_hourly = compact_index_hourly

        self.compactness = {'all_day': compact_index_all_day, 'home_time': compact_index_home,
                            'work_time': compact_index_work, 'hourly': compact_index_hourly}
        if self.fn_pref:
            with open(compactness_fn, 'w') as f:
                json.dump(self.compactness, f)
