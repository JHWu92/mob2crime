import json
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import src.utils.gis as gis
from src.ftrs.compactness_index import compacity_coefficient, cohesion, proximity, moment_inertia, \
    mass_moment_inertia
from src.utils import loubar_thres

# 0: 0~1 am
# 6: 6~7 am
# 9: 9~10 am
# 17: 5~6 pm
# 22: 22~23 (10~11pm)
# 23: 23~24 (11~11:59pm)

# 10pm - 7am
HOME_HOURS = {
    4: ['0', '1'],  # 11pm to 11am
    24: ['0', '1', '2', '3', '4', '5', '6', '22', '23'],
    48: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '44', '45', '46', '47']
}
# 9am - 6pm
WORK_HOURS = {
    4: ['2', '3'],  # 11am to 11pm
    24: ['9', '10', '11', '12', '13', '14', '15', '16', '17'],
    48: ['18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35']
}
MEASURES_DIR = 'data/mex_hotspot_measures'
RASTER_USE_P_CENTROID_IF_NONE = True

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
    if geoms is not None:
        # compute pair_dist if geoms is not None
        if pair_dist is not None:
            print('computing pair distance matrix using geoms, pair_dist passed is ignored')
        pair_dist = pair_dist_matrix(geoms)
    else:
        if pair_dist is None:
            raise ValueError('either geoms or pair_dist should be passed')
    return pair_dist


def avg_dist(geoms=None, pair_dist=None):
    if len(geoms) <= 1:
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
    # use the centroid of the geometry to compute distance
    if len(geoms) <= 1:
        return 0
    if len(geoms) > 40000:
        raise ValueError('pair_dist_matrix:::WARNING:::, too many grids, aborted')
    pair_dist = gis.polys_centroid_pairwise_dist(geoms, dist_crs=geoms.crs)
    return pair_dist


def hs_stats_tw(avg_tw, zms, per_mun=False, urb_only=False, area_normalized=False, hotspot_type='loubar', verbose=0,
                roll_width=3, chunk_width=4):
    import src.mex.tower as tower
    tXzms = tower.pts_x_region('mpa', per_mun, urb_only)
    t_pts = tower.pts().set_index('gtid')

    area_norm_str = 'density_' if area_normalized else ''
    if MEASURES_DIR.endswith('buggy'): area_norm_str.replace('_', '')
    n_hs = {}
    compactness = {}
    print('working on', end=' ')
    for sun, zm_mapping in tXzms.groupby('CVE_SUN'):
        print(sun, end=' ')

        zm = zms.loc[sun]
        zm_t = t_pts.loc[zm_mapping.gtid].copy()
        zm_avg_t = avg_tw.reindex(zm_mapping.gtid, fill_value=0).copy()

        if area_normalized:
            zm_avg_t = zm_avg_t.apply(lambda x: x / (zm_t.area / 1000 ** 2))

        fn_pref = f'{area_norm_str}tw_{ADMIN_STR(per_mun, urb_only)}_ZM{sun}'
        hs = HotSpot(zm_avg_t, zm_t, zm, hotspot_type, verbose=verbose, directory=MEASURES_DIR, fn_pref=fn_pref,
                     raster_use_p_centroid_if_none=RASTER_USE_P_CENTROID_IF_NONE,
                     roll_width=roll_width, chunk_width=chunk_width)
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
                  by='area', per_mun=False, urb_only=False, area_normalized=False,
                  hotspot_type='loubar', verbose=0,
                  roll_width=3, chunk_width=4):
    area_norm_str = 'density_' if area_normalized else ''
    if MEASURES_DIR.endswith('buggy'): area_norm_str.replace('_', '')
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
        if area_normalized:
            zm_avg_a = zm_avg_a.apply(lambda x: x / (zm_a.area / 1000 ** 2))
        fn_pref = f'{area_norm_str}ageb_{by}_{ADMIN_STR(per_mun, urb_only)}_ZM{sun}'
        hs = HotSpot(zm_avg_a, zm_a, zm, hotspot_type, verbose=verbose, directory=MEASURES_DIR, fn_pref=fn_pref,
                     raster_use_p_centroid_if_none=RASTER_USE_P_CENTROID_IF_NONE,
                     roll_width=roll_width, chunk_width=chunk_width)
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


def hs_stats_grid_or_vor(avg_geom, zms, zms_geoms, geom_type='grid', by='area',
                         per_mun=False, urb_only=False, area_normalized=False,
                         hotspot_type='loubar', verbose=0, roll_width=3, chunk_width=4):
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
        area_norm_str = ''
        if area_normalized and by != 'idw':
            # cannot normalized for idw, cause idw doesn't depend on area at all
            area_norm_str = 'density_'
            if MEASURES_DIR.endswith('buggy'): area_norm_str.replace('_', '')
            zm_avg_g = zm_avg_g.apply(lambda x: x / (zm_g.area / 1000 ** 2))
        fn_pref = f'{area_norm_str}{geom_type}_{by}_{ADMIN_STR(per_mun, urb_only)}_ZM{sun}'
        hs = HotSpot(zm_avg_g, zm_g, zm, hotspot_type, verbose=verbose, directory=MEASURES_DIR, fn_pref=fn_pref,
                     raster_use_p_centroid_if_none=RASTER_USE_P_CENTROID_IF_NONE,
                     roll_width=roll_width, chunk_width=chunk_width)
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
    def __init__(self, avg, geoms, cover_region, hotspot_type='loubar', raster_resolution=100, verbose=0,
                 directory=None, fn_pref=None, raster_use_p_centroid_if_none=True, roll_width=3, chunk_width=4):
        """
        :param directory: the directory to store data
        :param fn_pref: the prefix str for the file name. If None, no data is cached
        :param raster_use_p_centroid_if_none: some geoms could be smaller than raster_resolution,
            if False, there will be no centroids of rasterization returned for these geoms
            if True, the centroid of each of these geoms is returned as the rasterized centroid.
        """
        self.sqrt_area = np.sqrt(cover_region.Area)
        self.avg = avg.copy()
        self.geoms = geoms
        self.region = cover_region
        self.hotspot_type = hotspot_type
        self.raster_resolution = raster_resolution
        self.raster_use_p_centroid_if_none = raster_use_p_centroid_if_none
        self.verbose = verbose
        self.n_bins = avg.shape[1]

        if roll_width > self.n_bins / 2:
            raise ValueError(f'roll width ({roll_width}) should be no more than 1/2 of n_bins ({self.n_bins})')
        if chunk_width != 0 and self.n_bins % chunk_width != 0:
            raise ValueError(f'chunk width {chunk_width} should be a factor of n_bins ({self.n_bins})')
        self.roll_width = roll_width
        self.chunk_width = chunk_width

        # caching file name prefix
        if fn_pref:  # not None, not '', not 0, etc
            paths = []
            if directory: paths.append(directory)
            if self.n_bins != 24: paths.append(f'tod{self.n_bins}')
            paths.append(fn_pref)
            fn_pref = '/'.join(paths)
        self.fn_pref = fn_pref if fn_pref else None
        if not fn_pref and verbose:
            print('fn_pref is None, will not cache results')

        # measurements
        self.compactness = {}
        self.n_hs = {}

    def calc_stats(self, hs_avg=None):
        self._get_hs(hs_avg)
        self._hs_type_by_persistence()
        self._number_of_hs()
        self._hs_all_compactness()
        self._hs_mass_comp_coef()

        if self.roll_width <= 1:
            if self.verbose: print('roll width <=1, not computing rolling')
        else:
            self._number_of_hs_rolling()
            self._hs_rolling_compactness()
        if self.chunk_width <= 1:
            if self.verbose: print('roll width <=1, not computing rolling')
        else:
            self._number_of_hs_chunking()
            self._hs_chunking_compactness()
            self._hs_chunking_mass_comp_coef()

    def _get_hs(self, hs_avg=None):
        if self.verbose: print('masking out non hot spot, defined by', self.hotspot_type)
        if hs_avg is None:
            self.hs_avg = keep_hotspot(self.avg.copy(), self.hotspot_type)
        else:
            self.hs_avg = hs_avg

    def _rolling_hs_avg(self):
        rstep = int(pd.np.ceil((self.roll_width - 1) / 2))
        lstep = rstep - self.roll_width + 1
        hour_bins = self.hs_avg.columns.tolist()

        for i in range(self.n_bins):
            center = hour_bins[i]
            left = i + lstep
            right = i + rstep
            if left < 0:
                left_bins = hour_bins[left:]
                right_bins = hour_bins[:right + 1]
                rolling_bins = left_bins + right_bins
            elif right >= self.n_bins:
                left_bins = hour_bins[left:]
                right_bins = hour_bins[:(right + 1) - 24]
                rolling_bins = left_bins + right_bins
            else:
                rolling_bins = hour_bins[left:right + 1]
            # print(left, i, right, sorted([int(j) for j in rolling_bins]))

            hs_avg_rolling = self.hs_avg[rolling_bins]
            yield center, hs_avg_rolling

    def _chunking_hs_avg(self):
        hour_bins = self.hs_avg.columns.tolist()
        for i in range(0, self.n_bins, self.chunk_width):
            hour = hour_bins[i]
            chunk_bins = hour_bins[i:i + self.chunk_width]
            # hour = chunk_bins[0] + '-' + chunk_bins[-1]
            hs_avg_chunk = self.hs_avg[chunk_bins]
            yield hour, hs_avg_chunk

    def get_hourly_hs_index(self):
        hourly_hs_index = {}
        for hour in self.hs_avg:
            hourly_hs_index[hour] = self.hs_avg[self.hs_avg[hour] != 0].index.tolist()
        return hourly_hs_index

    def _hs_type_by_persistence(self):
        if self.verbose: print('computing persistency and obtaining permanent hot spots')
        home_hours = HOME_HOURS[self.n_bins]
        work_hours = WORK_HOURS[self.n_bins]

        persistence = (self.hs_avg != 0).sum(axis=1)
        persistence_home = (self.hs_avg[home_hours] != 0).sum(axis=1)
        persistence_work = (self.hs_avg[work_hours] != 0).sum(axis=1)

        self.hs_permanent = persistence[persistence == self.n_bins]
        self.n_hs_per = len(self.hs_permanent)

        self.hs_permanent_home = persistence_home[persistence_home == len(home_hours)]
        self.n_hs_per_home = len(self.hs_permanent_home)

        self.hs_permanent_work = persistence_work[persistence_work == len(work_hours)]
        self.n_hs_per_work = len(self.hs_permanent_work)

        # self.hs_intermediate = persistence[(persistence < 24) & (persistence >= 7)]
        # self.n_hs_med = len(self.hs_intermediate)
        # self.hs_intermittent = persistence[(persistence < 7) & (persistence >= 1)]
        # self.n_hs_mit = len(self.hs_intermittent)

    def _calc_compactness(self, hs_index, hs_count):
        """

        :param hs_index: pull up the geoms by hs_index => target_hs. target_hs are then used to compute compactness
        :param hs_count: hotspot's (mean if not hourly) number of users. Used to compute NMMI
        :return:
        """
        # meaning there isn't any hotspot to be computed
        # return None and exit
        if len(hs_index) == 0:
            print('Len of hs_index==0, no hotspot to be considered')
            return {'comp_coef': None, 'cohesion': None, 'proximity': None, 'NMI': None, 'NMMI': None}

        target_hs = self.geoms.loc[hs_index]

        # this density is to redistribute the mass for computing NMMI
        # TODO: this is a simplify version of density, assuming counts in the shape is uniformly
        #  distributed, but the underlying density is not, the smallest unit of density should
        #  be the interection of vor and ageb
        hs_density = hs_count / target_hs.area
        hs_density.name = 'Density'

        # index: compacity
        # use the centroid of the geometry to compute distance, for Vor, not the tower location
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
                                                       pname=target_hs.index.name,
                                                       use_p_centroid_if_none=self.raster_use_p_centroid_if_none)
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

    def _get_hs_area(self, hs_index):
        if len(hs_index) == 0:
            return 0
        else:
            return self.geoms.loc[hs_index].area.sum() / 1000 / 1000

    def _number_of_hs(self):
        if self.verbose: print('computing number of hot spot per hour')
        n_hs_hourly = (self.hs_avg != 0).sum(axis=0)
        n_hs_average = n_hs_hourly.mean()
        self.n_hs['average'] = n_hs_average

        # self.n_hs['hourly'] = n_hs_hourly

        self.n_hs['all_day'] = {'NHS': self.n_hs_per, 'AHS': self._get_hs_area(self.hs_permanent.index)}
        self.n_hs['home_time'] = {'NHS': self.n_hs_per_home, 'AHS': self._get_hs_area(self.hs_permanent_home.index)}
        self.n_hs['work_time'] = {'NHS': self.n_hs_per_work, 'AHS': self._get_hs_area(self.hs_permanent_work.index)}

        n_a_hs_hourly = []
        for hour in self.hs_avg:
            hs_count_hourly = self.hs_avg[hour]
            hs_count_hourly = hs_count_hourly[hs_count_hourly != 0]
            hs_index = hs_count_hourly.index
            n_a_hs_per_hour = {'hour': hour, 'NHS': len(hs_index), 'AHS': self._get_hs_area(hs_index)}
            n_a_hs_hourly.append(n_a_hs_per_hour)
        self.n_hs['hourly'] = n_a_hs_hourly

    def _number_of_hs_chunking(self):
        if self.verbose: print(f'computing number of hot spot per chunking width={self.chunk_width}')
        avg_n_hs_chunking = []
        for hour, hs_avg_chunking in self._chunking_hs_avg():
            hs_index = hs_avg_chunking[(hs_avg_chunking != 0).sum(axis=1) == self.chunk_width].index
            avg_n_hs_chunking.append({'hour': hour, 'NHS': len(hs_index), 'AHS': self._get_hs_area(hs_index)})
        self.n_hs['chunking'] = avg_n_hs_chunking

    def _number_of_hs_rolling(self):
        if self.verbose: print(f'computing number of hot spot per rolling width={self.roll_width}')
        avg_n_hs_rolling = []
        for hour, hs_avg_rolling in self._rolling_hs_avg():
            hs_index = hs_avg_rolling[(hs_avg_rolling != 0).sum(axis=1) == self.roll_width].index
            avg_n_hs_rolling.append({'hour': hour, 'NHS': len(hs_index), 'AHS': self._get_hs_area(hs_index)})
        self.n_hs['rolling'] = avg_n_hs_rolling

    def _hs_rolling_compactness(self):
        rolling_comp_fn = f'{self.fn_pref}_rolling{self.roll_width}_compactness.json'
        if self.fn_pref and os.path.exists(rolling_comp_fn):
            if self.verbose: print(f'loading existing rolling {self.roll_width} compactness at', rolling_comp_fn)
            with open(rolling_comp_fn, 'r') as f:
                self.compactness['rolling'] = json.load(f)
            return

        if self.verbose: print(f'computing compactness of hot spot per rolling width={self.roll_width}')
        compact_index_rolling = []
        for hour, hs_avg_rolling in self._rolling_hs_avg():
            rolling_persistence = (hs_avg_rolling != 0).sum(axis=1)
            rolling_hs_index = rolling_persistence[rolling_persistence == self.roll_width].index
            rolling_hs_count = hs_avg_rolling.loc[rolling_hs_index].mean(axis=1)
            c_index = self._calc_compactness(rolling_hs_index, rolling_hs_count)
            c_index['hour'] = hour
            compact_index_rolling.append(c_index)

        if self.fn_pref:
            if self.verbose: print(f'dumping rolling {self.roll_width} compactness at', rolling_comp_fn)
            with open(rolling_comp_fn, 'w') as f:
                json.dump(compact_index_rolling, f)

        self.compactness['rolling'] = compact_index_rolling

    def _hs_chunking_compactness(self):
        chunking_comp_fn = f'{self.fn_pref}_chunking{self.chunk_width}_compactness.json'
        if self.fn_pref and os.path.exists(chunking_comp_fn):
            if self.verbose: print(f'loading existing chunking {self.chunk_width} compactness at', chunking_comp_fn)
            with open(chunking_comp_fn, 'r') as f:
                self.compactness['chunking'] = json.load(f)
            return

        if self.verbose: print(f'computing compactness of hot spot per chunking width={self.chunk_width}')
        compact_index_chunking = []
        for hour, hs_avg_chunking in self._chunking_hs_avg():
            chunking_persistence = (hs_avg_chunking != 0).sum(axis=1)
            chunking_hs_index = chunking_persistence[chunking_persistence == self.chunk_width].index
            chunking_hs_count = hs_avg_chunking.loc[chunking_hs_index].mean(axis=1)
            c_index = self._calc_compactness(chunking_hs_index, chunking_hs_count)
            c_index['hour'] = hour
            compact_index_chunking.append(c_index)

        if self.fn_pref:
            if self.verbose: print(f'dumping chunking {self.chunk_width} compactness at', chunking_comp_fn)
            with open(chunking_comp_fn, 'w') as f:
                json.dump(compact_index_chunking, f)

        self.compactness['chunking'] = compact_index_chunking

    def _hs_chunking_mass_comp_coef(self):

        for i, (hour, hs_avg_chunking) in enumerate(self._chunking_hs_avg()):
            chunking_persistence = (hs_avg_chunking != 0).sum(axis=1)
            chunking_hs_index = chunking_persistence[chunking_persistence == self.chunk_width].index
            mass_comp_coef = self.__mass_comp_coef(chunking_hs_index)
            c_index = self.compactness['chunking'][i]
            assert c_index['hour'] == hour
            c_index['mass_comp_coef'] = mass_comp_coef

    def _hs_all_compactness(self):
        compactness_fn = f'{self.fn_pref}_compactness.json'

        if self.fn_pref and os.path.exists(compactness_fn):
            if self.verbose: print('loading existing compactness at', compactness_fn)
            with open(compactness_fn, 'r') as f:
                self.compactness.update(json.load(f))
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

        self.compactness.update({'all_day': compact_index_all_day, 'home_time': compact_index_home,
                                 'work_time': compact_index_work, 'hourly': compact_index_hourly})
        if self.fn_pref:
            if self.verbose: print(f'dumping compactness at', compactness_fn)
            with open(compactness_fn, 'w') as f:
                json.dump(self.compactness, f)

    def __mass_comp_coef(self, hs_index):

        if len(hs_index) == 1:
            mass_comp_coef = 0
        elif len(hs_index) == 0:
            mass_comp_coef = None
        else:
            hs_count = self.hs_avg.loc[hs_index].mean(axis=1)
            hs_count = np.array(hs_count.tolist())
            hs_geoms = self.geoms.loc[hs_index]
            hs_centroid = hs_geoms.geometry.apply(lambda x: x.centroid.coords[:][0]).tolist()
            hs_centroid = np.array(hs_centroid)
            mass_comp_coef = gis.pairwise_dist_mass_average(hs_centroid, hs_count, square=False) / self.sqrt_area
        return mass_comp_coef

    def _hs_mass_comp_coef(self):

        hs_index = self.hs_permanent.index
        self.compactness['all_day']['mass_comp_coef'] = self.__mass_comp_coef(hs_index)

        hs_index = self.hs_permanent_work.index
        self.compactness['home_time']['mass_comp_coef'] = self.__mass_comp_coef(hs_index)

        hs_index = self.hs_permanent_home.index
        self.compactness['work_time']['mass_comp_coef'] = self.__mass_comp_coef(hs_index)

        for i, hour in enumerate(self.hs_avg):
            hs_count_hourly = self.hs_avg[hour]
            hs_count_hourly = hs_count_hourly[hs_count_hourly != 0]
            hs_index = hs_count_hourly.index
            mass_comp_coef = self.__mass_comp_coef(hs_index)
            c_index = self.compactness['hourly'][i]
            assert c_index['hour'] == hour
            c_index['mass_comp_coef'] = mass_comp_coef
