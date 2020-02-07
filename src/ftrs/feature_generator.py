from scipy.spatial.distance import cdist
from tinydb import Query
import numpy as np
import src.utils.gis as gis
import src.ftrs.compactness_index as comp_index

WORK_HOURS = ['9', '10', '11', '12', '13', '14', '15', '16', '17']
HOME_HOURS = ['0', '1', '2', '3', '4', '5', '6', '22', '23']


def get_city_features(db, admin_lvl, admin_id, boundary_type, su_type, intpl):
    query = Query()
    city = db.search((query.admin_id == admin_id)
                     & (query.admin_lvl == admin_lvl)
                     & (query.boundary_type == boundary_type)
                     & (query.su_type == su_type) & (query.intpl == intpl))
    if len(city) == 0:
        # if there is no record
        city_id = db.insert({
            'admin_lvl': admin_lvl,
            'admin_id': admin_id,
            'boundary_type': boundary_type,
            'su_type': su_type,
            'intpl': intpl,
            'features': {}
        })
        city = db.get(doc_id=city_id)
    elif len(city) == 1:
        # there is one record
        city = city[0]
    else:
        raise ValueError(
            f'has multiple records for city with '
            f'admin_lvl={admin_lvl}, admin_id={admin_id}, '
            f'boundary_type={boundary_type}, su_type={su_type}, intpl={intpl}')
    return city


def update_feature(db, new_city):
    # tinydb.write_back
    db.write_back([new_city])


def ftr_hs_scale(db, admin_lvl, admin_id, boundary_type, su_type, intpl,
                 hotspot_type='loubar', su=None, hotspots_per_hour=None, redo=False):
    city = get_city_features(db, admin_lvl, admin_id, boundary_type, su_type, intpl)
    # if updated, new feature is computed, the doc in db needs update.
    updated = False
    # if not complete, no new feature can be computed
    data_complete = (su is not None) & (hotspots_per_hour is not None)

    feature_set_name = f'hs-{hotspot_type}'
    if feature_set_name not in city['features']:
        # this set of feature hasn't been computed
        features = {}
        updated = True
    else:
        features = city['features'][feature_set_name]

    if 'NHS' not in features or redo:
        if data_complete:
            features['NHS'] = number_of_hotspots(hotspots_per_hour)
            updated = True
        else:
            print('NHS is not computed, but the data is incomplete')

    if 'AHS' not in features or redo:
        if data_complete:
            features['AHS'] = area_of_hotspots(hotspots_per_hour, su)
            updated = True
        else:
            print('AHS is not computed, but the data is incomplete')

    if updated:
        print(features)
        city['features'][feature_set_name] = features
        update_feature(db, city)

    return city


def ftr_compacity(db, admin_lvl, admin_id, boundary_type, su_type, intpl,
                  hotspot_type='loubar', su=None, hotspots_per_hour=None,
                  city_area=None, redo=False):
    city = get_city_features(db, admin_lvl, admin_id, boundary_type, su_type, intpl)
    # if updated, new feature is computed, the doc in db needs update.
    updated = False
    # if not complete, no new feature can be computed
    data_complete = (su is not None) & (hotspots_per_hour is not None) & (city_area is not None)

    feature_set_name = f'hs-{hotspot_type}'
    if feature_set_name not in city['features']:
        # this set of feature hasn't been computed
        features = {}
        updated = True
    else:
        features = city['features'][feature_set_name]

    if 'COMP' not in features or redo:
        if data_complete:
            features['COMP'] = compacity_coefficient(city_area, su, hotspots_per_hour)
            updated = True
        else:
            print('COMP is not computed, but the data is incomplete')

    if 'MCOMP' not in features or redo:
        if data_complete:
            features['MCOMP'] = mass_compacity_coefficient(city_area, su, hotspots_per_hour)
            updated = True
        else:
            print('MCOMP is not computed, but the data is incomplete')

    if updated:
        city['features'][feature_set_name] = features
        update_feature(db, city)

    return city


def ftr_compactness(db, admin_id, admin_lvl, boundary_type, su_type, intpl,
                    hotspot_type='loubar', raster_resolution=100, raster_use_p_centroid_if_none=False,
                    su=None, hotspots_per_hour=None, redo=False, verbose=0):
    city = get_city_features(db, admin_id, admin_lvl, boundary_type, su_type, intpl)
    feature_names = ['COHE', 'PROX', 'NMI', 'NMMI']

    # if update, new feature is computed, the doc in db needs update.
    update = False

    feature_set_name = f'hs-{hotspot_type}-rs-{raster_resolution}'
    if feature_set_name not in city['features']:
        # this set of feature hasn't been computed
        features = {}
        update = True
    else:
        features = city['features'][feature_set_name]

    # if there is one feature that hasn't been computed, update = True
    for fname in feature_names:
        if fname not in features or redo:
            # initialize the dictionary for fname
            features[fname] = {'hourly': []}
            update = True

    # if redo, need update
    if redo:
        update = True

    # if not complete, no new feature can be computed
    data_complete = (su is not None) & (hotspots_per_hour is not None)
    if update and not data_complete:
        print('need update, but data is not complete')
        update = False

    if not update:
        return city

    # ==== need update and data is complete
    permanent_index, permanent_index_home, permanent_index_work = permanent_index_three_time_range(hotspots_per_hour)
    average_footfall = hotspots_per_hour.mean(axis=1)

    for time_range_name, index in [('all_day', permanent_index), ('home_hour', permanent_index_home),
                                   ('work_hour', permanent_index_work)]:
        prep = preparation_for_compactness(su.loc[index], average_footfall.loc[index],
                                           raster_resolution, raster_use_p_centroid_if_none, verbose)
        if 'COHE' not in features or redo:
            features['COHE'][time_range_name] = comp_index.cohesion(prep['ref_circle_radius'],
                                                                    prep['pairwise_dist_square_avg'])

        if 'PROX' not in features or redo:
            features['PROX'][time_range_name] = comp_index.proximity(prep['ref_circle_radius'], prep['d2centroid_avg'])

        if 'NMI' not in features or redo:
            features['NMI'][time_range_name] = comp_index.moment_inertia(prep['su_rasterized'].Area, prep['d2centroid'])

        if 'NMMI' not in features or redo:
            features['NMMI'][time_range_name] = comp_index.mass_moment_inertia(prep['su_rasterized'],
                                                                               prep['d2mass_centroid'])

    for hour in hotspots_per_hour:
        hs_hourly = hotspots_per_hour[hour]
        hs_index_hourly = hs_hourly[hs_hourly != 0].index
        prep = preparation_for_compactness(su.loc[hs_index_hourly],
                                           hs_hourly.loc[hs_index_hourly],
                                           raster_resolution,
                                           raster_use_p_centroid_if_none, verbose)
        if 'COHE' not in features or redo:
            features['COHE']['hourly'].append(comp_index.cohesion(prep['ref_circle_radius'],
                                                                  prep['pairwise_dist_square_avg']))

        if 'PROX' not in features or redo:
            features['PROX']['hourly'].append(comp_index.proximity(prep['ref_circle_radius'], prep['d2centroid_avg']))

        if 'NMI' not in features or redo:
            features['NMI']['hourly'].append(comp_index.moment_inertia(prep['su_rasterized'].Area, prep['d2centroid']))

        if 'NMMI' not in features or redo:
            features['NMMI']['hourly'].append(comp_index.mass_moment_inertia(prep['su_rasterized'],
                                                                             prep['d2mass_centroid']))

    city['features'][feature_set_name] = features
    update_feature(db, city)

    return city


def preparation_for_compactness(su, footfall, raster_resolution, raster_use_p_centroid_if_none, verbose=0):
    su_rasterized = gis.gp_polys_to_raster_centroids(
        su, side=raster_resolution, pname=su.index.name,
        use_p_centroid_if_none=raster_use_p_centroid_if_none)
    if verbose:
        print(f'raster {len(su)} hot spots with resolution: {raster_resolution}m into {len(su_rasterized)} grids')

    density = footfall / su.area
    density.name = 'Density'

    su_rasterized = su_rasterized.merge(density.reset_index())
    su_rasterized['Area'] = raster_resolution ** 2
    su_rasterized['Mass'] = su_rasterized.Area * su_rasterized.Density

    rx = su_rasterized.centroid.apply(lambda x: x.coords[0][0])
    ry = su_rasterized.centroid.apply(lambda x: x.coords[0][1])

    cx = (rx * su_rasterized.Area).sum() / su_rasterized.Area.sum()
    cy = (ry * su_rasterized.Area).sum() / su_rasterized.Area.sum()
    raster_centroid = (cx, cy)

    cx_mass = (rx * su_rasterized.Mass).sum() / su_rasterized.Mass.sum()
    cy_mass = (ry * su_rasterized.Mass).sum() / su_rasterized.Mass.sum()
    raster_mass_centroid = (cx_mass, cy_mass)

    raster_rper_centroids = su_rasterized.centroid.apply(lambda x: x.coords[0]).tolist()
    pairwise_dist_square_avg = gis.pairwise_dist_average(np.array(raster_rper_centroids))
    d2centroid = cdist(raster_rper_centroids, np.array([raster_centroid]))[:, 0]
    d2centroid_avg = d2centroid.mean()
    d2mass_centroid = cdist(raster_rper_centroids, np.array([raster_mass_centroid]))[:, 0]

    # hotspots equivalent circle
    ref_circle_radius = np.sqrt(su_rasterized.Area.sum() / np.pi)

    return {
        'su_rasterized': su_rasterized,
        'ref_circle_radius': ref_circle_radius,
        'pairwise_dist_square_avg': pairwise_dist_square_avg,
        'd2centroid_avg': d2centroid_avg,
        'd2centroid': d2centroid,
        'd2mass_centroid': d2mass_centroid
    }


def permanent_index_three_time_range(hotspots_per_hour):
    persistence = (hotspots_per_hour != 0).sum(axis=1)
    permanent_index = persistence[persistence == hotspots_per_hour.shape[1]].index

    persistence_home = (hotspots_per_hour[HOME_HOURS] != 0).sum(axis=1)
    permanent_index_home = persistence_home[persistence_home == len(HOME_HOURS)].index

    persistence_work = (hotspots_per_hour[WORK_HOURS] != 0).sum(axis=1)
    permanent_index_work = persistence_work[persistence_work == len(WORK_HOURS)].index
    return permanent_index, permanent_index_home, permanent_index_work


def number_of_hotspots(hotspots_per_hour):
    permanent_index, permanent_index_home, permanent_index_work = permanent_index_three_time_range(hotspots_per_hour)
    hourly = []
    for x in (hotspots_per_hour != 0).sum().iteritems():
        hourly.append(dict(zip(['hour', 'value'], x)))

    nhs = {
        'all_day': len(permanent_index),
        'home_hour': len(permanent_index_home),
        'work_hour': len(permanent_index_work),
        'hourly': hourly
    }
    return nhs


def area_of_hotspots(hotspots_per_hour, su):
    def get_hs_area(hidx):
        if len(hidx) == 0:
            return 0
        else:
            return su.loc[hidx].area.sum() / 1000 / 1000

    permanent_index, permanent_index_home, permanent_index_work = permanent_index_three_time_range(hotspots_per_hour)
    hourly = []
    for hour in hotspots_per_hour:
        hs_hourly = hotspots_per_hour[hour]
        hs_index_hourly = hs_hourly[hs_hourly != 0].index
        hourly.append({'hour': hour, 'value': get_hs_area(hs_index_hourly)})
    ahs = {
        'all_day': get_hs_area(permanent_index),
        'home_hour': get_hs_area(permanent_index_home),
        'work_hour': get_hs_area(permanent_index_work),
        'hourly': hourly
    }

    return ahs


def compacity_coefficient(city_area, su, hotspots_per_hour):
    permanent_index, permanent_index_home, permanent_index_work = permanent_index_three_time_range(hotspots_per_hour)
    sqrt_area = np.sqrt(city_area)

    hourly = []
    for hour in hotspots_per_hour:
        hs_hourly = hotspots_per_hour[hour]
        hs_index_hourly = hs_hourly[hs_hourly != 0].index
        hourly.append({'hour': hour, 'value': comp_index.comp_coef(su.loc[hs_index_hourly], sqrt_area)})

    comp = {
        'all_day': comp_index.comp_coef(su.loc[permanent_index], sqrt_area),
        'home_hour': comp_index.comp_coef(su.loc[permanent_index_home], sqrt_area),
        'work_hour': comp_index.comp_coef(su.loc[permanent_index_work], sqrt_area),
        'hourly': hourly
    }
    return comp


def mass_compacity_coefficient(city_area, su, hotspots_per_hour):
    permanent_index, permanent_index_home, permanent_index_work = permanent_index_three_time_range(hotspots_per_hour)
    sqrt_area = np.sqrt(city_area)
    average_footfall = hotspots_per_hour.mean(axis=1)

    hourly = []
    for hour in hotspots_per_hour:
        hs_hourly = hotspots_per_hour[hour]
        hs_index_hourly = hs_hourly[hs_hourly != 0].index
        hourly.append({'hour': hour,
                       'value': comp_index.mass_comp_coef(su.loc[hs_index_hourly],
                                                          hs_hourly.loc[hs_index_hourly].values,
                                                          sqrt_area)})

    mcomp = {
        'all_day': comp_index.mass_comp_coef(su.loc[permanent_index],
                                             average_footfall.loc[permanent_index].values,
                                             sqrt_area),
        'home_hour': comp_index.mass_comp_coef(su.loc[permanent_index_home],
                                               average_footfall.loc[permanent_index_home].values,
                                               sqrt_area),
        'work_hour': comp_index.mass_comp_coef(su.loc[permanent_index_work],
                                               average_footfall.loc[permanent_index_work].values,
                                               sqrt_area),
        'hourly': hourly
    }
    return mcomp
