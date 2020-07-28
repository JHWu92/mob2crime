import datetime
import json
from collections import defaultdict
import os
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from tinydb import TinyDB

import src.exp.preprocessing as exp_prep
import src.exp.sklearn_models as exp_sk
import src.ftrs.feature_generator as fgen
import src.mex.crime as mex_crime
import src.mex.regions2010 as mex_region

SEEDS = [86539080, 61388538, 75279011, 19571100, 38799689, 72041214, 44786281, 18776859, 89013112, 76063141,
         107044, 4279302, 9047249, 4510024, 4041726, 3733114, 7011654, 7824036, 9847967, 4532047]


def get_train_test_splits():
    path = 'data/SigSpatial_train_test_split.json'
    if not os.path.exists(path):

        data = pd.read_csv('data/SigSpatial_crime_data_full.csv')
        data_idx = data.index.tolist()
        idx_json = []
        for i, seed in enumerate(SEEDS):
            train_idx, test_idx = train_test_split(data_idx, test_size=0.2, random_state=seed)
            idx_json.append({'ith': str(i), 'seed': seed,
                             'train': sorted(train_idx), 'test': sorted(test_idx)
                             })

        json.dump(idx_json, open(path, 'w'))
    else:
        idx_json = json.load(open(path))
    return idx_json


def get_5fold_split(random_state=0):
    if random_state == 0:
        path = 'data/SigSpatial_5fold_split.json'
    else:
        path = f'data/SigSpatial_5fold_split-rs{random_state}.json'
    if not os.path.exists(path):

        data = pd.read_csv('data/SigSpatial_crime_data_full.csv')
        data_idx = data.index.values.astype(int)
        kf = KFold(5, shuffle=True, random_state=random_state)
        idx_json = []
        for ith, (tridx, teidx) in enumerate(kf.split(data_idx)):
            idx_json.append({
                'ith': ith,
                'seed': 0,
                'train': sorted(data_idx[tridx].tolist()),
                'test': sorted(data_idx[teidx].tolist())
            })

            json.dump(idx_json, open(path, 'w'))
    else:
        idx_json = json.load(open(path))
    return idx_json


def load_crime(type_level=1):
    crimes = mex_crime.sesnsp_crime_counts(2011, type_level)
    mgms = mex_region.municipalities(load_pop=True)
    mun_ids_with_data = sorted(set(crimes.index) & set(mgms.index))
    crimes = crimes.loc[mun_ids_with_data].copy()
    # crimes['pop'] = mgms.loc[mun_ids_with_data].pobtot / 10000
    crimes_rate = crimes.divide(mgms.loc[mun_ids_with_data].pobtot / 10000, axis=0)
    crimes = crimes.rename(columns={'property': '#pro', 'violent': '#vio'})
    crimes_rate = crimes_rate.rename(columns={'property': '%pro', 'violent': '%vio'})
    if type_level == 1:
        crimes = crimes.join(crimes_rate)
    return crimes


def load_feature(db_path, boundary_type, su_type, intpl, hourly=True, mun_ids=None):
    db = TinyDB(db_path)
    cities = fgen.filter_cities(db, 'mun_id', boundary_type=boundary_type, su_type=su_type, intpl=intpl)

    db.close()
    features = fgen.flat_ftr_hotspot(cities, hourly=hourly)
    features = pd.DataFrame(features).set_index('mun_id')
    if mun_ids is not None:
        features = features.loc[mun_ids]
    return features


def train_eval_models(model_names, dataset, random_state=0,
                      random_search_cv=True, n_iter=10, cv=5, n_jobs=None,
                      verbose=0):
    """

    :param model_names:
    :param dataset:
    :param random_state:
    :param random_search_cv:
    :param n_iter: number of hyperparameters combo for randomsearchCV
    :param cv:
    :param n_jobs:
    :param verbose:
    :return:
    """
    train_X, test_X, train_y, test_y = dataset
    results = {}
    for model_name in model_names:
        start_at = datetime.datetime.now()
        model = exp_sk.train_regressor(model_name, train_X, train_y, random_state=random_state,
                                       random_search_cv=random_search_cv, n_iter=n_iter, cv=cv,
                                       n_jobs=n_jobs, verbose=max(0, verbose - 1))
        train_set = exp_sk.evaluate_regressor(model, train_X, train_y, scatter_plot=False, prefix='train_')
        test_set = exp_sk.evaluate_regressor(model, test_X, test_y, scatter_plot=False, prefix='test_')
        end_at = datetime.datetime.now()
        result = {'train_time': (end_at - start_at).total_seconds()}
        if verbose:
            print(f'ran {model_name}, using {result["train_time"]} seconds')
        result.update(test_set)
        result.update(train_set)
        results[model_name] = result
    return results


def exp_sigspatial(debug=False, n_iter=40, n_jobs=4, verbose=0,
                   split='trte', crime_rate=False, random_state=0, scaling=True, add_pop_ftr=False):
    rs_str = f'rs{random_state}-' if random_state != 0 else ''
    scaling_str = 'noScale_' if not scaling else ''
    add_pop_str = 'Pop_' if add_pop_ftr else ''
    print('running exp_sigspatial, property crimes, crime_rate:',
          crime_rate, add_pop_str, scaling_str, random_state)
    model_names = ['ols', 'ridge', 'lasso', 'DTreg', 'SVR', 'linearSVR',
                   'ADAreg', 'GDBreg', 'RFreg', 'BAGreg', 'XGBreg']
    if debug:
        n_iter = 3

    data = pd.read_csv('data/SigSpatial_crime_data_full.csv')
    ftr_col = ['nhs', 'ahs', 'comp', 'mcomp', 'prox', 'cohe', 'nmi', 'nmmi', 'poverty_rate', 'active_unoccupied']
    if add_pop_ftr:
        ftr_col += ['popu']
    ftr = data[ftr_col]
    if crime_rate:
        z = data['pro_cnt'] / data['popu']
        path = f'exp_result/SigSpatial/ml_pred_z_rate_{scaling_str}{rs_str}{split}.json'
    else:
        z = data['pro_cnt']
        path = f'exp_result/SigSpatial/ml_pred_z_{add_pop_str}{scaling_str}{rs_str}{split}.json'
    print('split:', split)
    if split == 'trte':
        idx_json = get_train_test_splits()
    else:
        idx_json = get_5fold_split(random_state)

    pred_result = defaultdict(dict)
    for idx in idx_json:
        print('-------- train test split: ', idx['ith'])
        train_x = ftr.loc[idx['train']]
        train_y = z.loc[idx['train']]
        test_x = ftr.loc[idx['test']]
        if scaling:
            print('scaling')
            train_x, test_x = exp_prep.scaling_data(train_x, test_x, 'StandardScaler')
        for model_name in model_names:
            print('model:', model_name)
            model = exp_sk.train_regressor(model_name, train_x, train_y, random_state=0,
                                           random_search_cv=True, n_iter=n_iter, cv=5,
                                           n_jobs=n_jobs, verbose=max(0, verbose - 1))
            train_pred = model.predict(train_x).tolist()
            test_pred = model.predict(test_x).tolist()
            pred_result[model_name][idx['ith']] = {
                'seed': idx['seed'], 'train_pred': train_pred, 'test_pred': test_pred}

    json.dump(pred_result, open(path, 'w'))


def exp_sigspatial_vio(debug=False, n_iter=40, n_jobs=4, verbose=0,
                       split='trte', crime_rate=False, random_state=0, scaling=True, add_pop_ftr=False):
    rs_str = f'rs{random_state}-' if random_state != 0 else ''
    scaling_str = 'noScale_' if not scaling else ''
    add_pop_str = 'Pop_' if add_pop_ftr else ''
    print('running exp_sigspatial on violent crimes, crime_rate:',
          crime_rate, add_pop_str, scaling_str, random_state)
    model_names = ['ols', 'ridge', 'lasso', 'DTreg', 'SVR', 'linearSVR',
                   'ADAreg', 'GDBreg', 'RFreg', 'BAGreg', 'XGBreg']
    if debug:
        n_iter = 3

    data = pd.read_csv('data/SigSpatial_crime_data_full.csv')
    ftr_col = ['nhs', 'ahs', 'comp', 'mcomp', 'prox', 'cohe', 'nmi', 'nmmi',
               'adult_rate', 'nev_mar_rate', 'male_to_female_ratio',
               'male_to_female_household_ratio', 'foreign_ent_rate',
               'poverty_rate', 'no_spanish_rate']
    if add_pop_ftr:
        ftr_col += ['popu']
    ftr = data[ftr_col]
    if crime_rate:
        z = data['vio_cnt'] / data['popu']
        path = f'exp_result/SigSpatial/ml_pred_z_viorate_{scaling_str}{rs_str}{split}.json'
    else:
        z = data['vio_cnt']
        path = f'exp_result/SigSpatial/ml_pred_z_vio_{add_pop_str}{scaling_str}{rs_str}{split}.json'

    print('split:', split)
    if split == 'trte':
        idx_json = get_train_test_splits()
    else:
        idx_json = get_5fold_split(random_state)

    pred_result = defaultdict(dict)
    for idx in idx_json:
        print('-------- train test split: ', idx['ith'])
        train_x = ftr.loc[idx['train']]
        train_y = z.loc[idx['train']]
        test_x = ftr.loc[idx['test']]
        if scaling:
            print('scaling')
            train_x, test_x = exp_prep.scaling_data(train_x, test_x, 'StandardScaler')

        for model_name in model_names:
            print('model:', model_name)
            model = exp_sk.train_regressor(model_name, train_x, train_y, random_state=0,
                                           random_search_cv=True, n_iter=n_iter, cv=5,
                                           n_jobs=n_jobs, verbose=max(0, verbose - 1))
            train_pred = model.predict(train_x).tolist()
            test_pred = model.predict(test_x).tolist()
            pred_result[model_name][idx['ith']] = {
                'seed': idx['seed'], 'train_pred': train_pred, 'test_pred': test_pred}

    json.dump(pred_result, open(path, 'w'))


def experiments(debug=False, n_iter=40, n_jobs=4, verbose=0):
    settings = {
        0: ('Urban', 'grid-500', 'Uni'),  # Done
        1: ('Urban', 'grid-500', 'Pop'),  # Done
        2: ('UrbanRural', 'grid-500', 'Uni'),  # Done
        3: ('Urban', 'ageb', 'Uni'),  # Done
        4: ('Urban', 'ageb', 'Pop'),
    }

    db_path = 'data/features_database_tmp.json'
    boundary_type, su_type, intpl = settings[4]
    # scaling = 'RobustScaler'
    scaling = 'StandardScaler'
    print(f'boundary_type={boundary_type}, su_type={su_type}, interpolation={intpl}, scaling={scaling}')

    model_names = ['ols', 'ridge', 'lasso', 'DTreg', 'SVR', 'linearSVR', 'ADAreg', 'GDBreg', 'RFreg', 'BAGreg']
    if debug:
        n_iter = 3
        model_names = model_names[:4]

    print('loading crimes')
    crimes = load_crime()

    writer = pd.ExcelWriter(f'exp_result/muni_crime-{boundary_type}-{su_type}-{intpl}-{scaling}.xlsx',
                            engine='xlsxwriter')
    start_at = datetime.datetime.now()
    for hourly in [False, True]:
        hourly_str = 'hourT' if hourly else 'hourF'
        features = load_feature(db_path, boundary_type, su_type, intpl, hourly, mun_ids=crimes.index)
        features = exp_prep.fillna(features)

        for crime_col in crimes:
            sheet_name = f'{crime_col}-{hourly_str}'
            print(f'========current sheet_name={sheet_name}, n sample = {len(features)}')

            y = crimes[crime_col]
            train_X, test_X, train_y, test_y = exp_prep.train_test_split(features, y, random_state=0)
            scaled_train_X, scaled_test_X = exp_prep.scaling_data(train_X, test_X, scaling)
            dataset = (scaled_train_X, scaled_test_X, train_y, test_y)
            res = train_eval_models(model_names, dataset, random_state=0,
                                    random_search_cv=True, n_iter=n_iter, cv=5, n_jobs=n_jobs, verbose=verbose)
            res = pd.DataFrame(res).T.sort_values('test_RMSE')
            res.to_excel(writer, sheet_name=sheet_name)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    end_at = datetime.datetime.now()
    print(start_at, '~', end_at, '=', end_at - start_at)
    print(f'boundary_type={boundary_type}, su_type={su_type}, interpolation={intpl}, scaling={scaling}')


if __name__ == '__main__':
    import sys
    import warnings
    import os

    from sklearn.exceptions import ConvergenceWarning

    # from sklearn.utils.testing import ignore_warnings
    # ConvergenceWarning('ignore')
    # warnings.filterwarnings('ignore', category=ConvergenceWarning)

    if not sys.warnoptions:
        warnings.simplefilter("ignore", category=(ConvergenceWarning, UserWarning))
        os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

    # experiments(debug=False, verbose=1)
    exp_sigspatial(debug=False, n_iter=60, n_jobs=4,
                   split='5fold', crime_rate=False, scaling=True, random_state=0, add_pop_ftr=True)
    exp_sigspatial_vio(debug=False, n_iter=60, n_jobs=4,
                       split='5fold', crime_rate=False, scaling=True, random_state=0, add_pop_ftr=True)
