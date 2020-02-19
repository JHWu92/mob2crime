import src.mex.crime as mex_crime
import src.mex.regions2010 as mex_region
import src.ftrs.feature_generator as fgen
from tinydb import TinyDB
import pandas as pd
import datetime
import src.exp.sklearn_models as exp_sk
import src.exp.preprocessing as exp_prep
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


def load_crime():
    crimes = mex_crime.sesnsp_crime_counts(2011, 1)
    mgms = mex_region.municipalities(load_pop=True)
    mun_ids_with_data = sorted(set(crimes.index) & set(mgms.index))
    crimes = crimes.loc[mun_ids_with_data]
    crimes_rate = crimes.divide(mgms.loc[mun_ids_with_data].pobtot / 10000, axis=0)
    crimes = crimes.rename(columns={'property': '#pro', 'violent': '#vio'})
    crimes_rate = crimes_rate.rename(columns={'property': '%pro', 'violent': '%vio'})
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


@ignore_warnings(category=ConvergenceWarning)
def train_eval_models(model_names, dataset, random_state=0,
                      random_search_cv=True, n_iter=10, cv=5, n_jobs=None,
                      verbose=0):
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


def experiments(debug=False, n_iter=40, n_jobs=4, verbose=0):
    settings = {
        0: ('Urban', 'grid-500', 'Uni'),
        1: ('Urban', 'grid-500', 'Pop'),
        2: ('UrbanRural', 'grid-500', 'Uni'),
        3: ('Urban', 'ageb', 'Uni'),
        4: ('Urban', 'ageb', 'Pop'),
    }

    db_path = 'data/features_database.json'
    boundary_type, su_type, intpl = settings[2]
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
    experiments(debug=False, verbose=1)
