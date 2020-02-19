import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn import linear_model, svm, tree, ensemble, neural_network
# from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
from sklearn.model_selection import RandomizedSearchCV
# from sklearn.utils.testing import ignore_warnings

import src.exp.parameters_space as param_space

try:
    import xgboost
except ImportError:
    xgboost = None

models_hyperparam_space = {
    'ols': {},
    'ridge': {'alpha': np.logspace(0, 2, 10)},
    'lasso': {'alpha': np.logspace(0, 2, 10)},
    'DTreg': param_space.sk_decision_tree,
    'RFreg': param_space.sk_random_forest(),
    'ADAreg': param_space.sk_adaboost,
    'BAGreg': param_space.sk_bagging,
    'GDBreg': param_space.sk_gradient_boost,
    'SVR': param_space.sk_svr,
    'linearSVR': param_space.sk_linear_svr,
    'XGBreg': {},
}


def sk_models(reg=True, cls=True, stoplist=()):
    """
    return sk models with names by regression and/or classification.
    """
    reg_models = {
        'ols': linear_model.LinearRegression,
        'ridge': linear_model.Ridge,
        'lasso': linear_model.Lasso,
        'DTreg': tree.DecisionTreeRegressor,
        'RFreg': ensemble.RandomForestRegressor,
        'ADAreg': ensemble.AdaBoostRegressor,
        'BAGreg': ensemble.BaggingRegressor,
        'GDBreg': ensemble.GradientBoostingRegressor,
        'SVR': svm.SVR,
        'linearSVR': svm.LinearSVR,
        'MLPreg': neural_network.MLPRegressor,
    }

    cls_models = {
        'logistics': linear_model.LogisticRegression,
        'DTcls': tree.DecisionTreeClassifier,
        'RFcls': ensemble.RandomForestClassifier,
        'ADAcls': ensemble.AdaBoostClassifier,
        'BAGcls': ensemble.BaggingClassifier,
        'GDBcls': ensemble.GradientBoostingClassifier,
        'SVM': svm.SVC,
        'linearSVM': svm.LinearSVC,
        'MLPcls': neural_network.MLPClassifier,
        # 'GNBcls'   : naive_bayes.GaussianNB,  # doesn't accept sparse matrix
    }

    if xgboost is not None:
        reg_models['XGBreg'] = xgboost.sklearn.XGBRegressor
        cls_models['XGBcls'] = xgboost.sklearn.XGBClassifier

    models = {}
    if reg:
        for name in stoplist: reg_models.pop(name, None)
        models['reg'] = reg_models
    if cls:
        for name in stoplist: cls_models.pop(name, None)
        models['cls'] = cls_models
    return models


def regression_scoring(true, pred, prefix=''):
    return {
        f'{prefix}RMSE': mean_squared_error(y_true=true, y_pred=pred, squared=False),
        f'{prefix}MAE': mean_absolute_error(y_true=true, y_pred=pred),
        f'{prefix}Exp_var': explained_variance_score(y_true=true, y_pred=pred),
        f'{prefix}r2': r2_score(y_true=true, y_pred=pred),
        f'{prefix}pear_cor': pearsonr(true, pred)[0],
        f'{prefix}spea_cor': spearmanr(true, pred)[0],
        f'{prefix}kend_cor': kendalltau(true, pred)[0],
    }


# @ignore_warnings(category=ConvergenceWarning)
def train_regressor(model_name, train_x, train_y, random_state=None, random_search_cv=False,
                    n_iter=10, cv=5, n_jobs=2, verbose=0):
    if model_name in ('ols', 'SVR'):
        # OLS has no random state, no tunning
        model = sk_models(cls=False)['reg'][model_name]()
    else:
        model = sk_models(cls=False)['reg'][model_name](random_state=random_state)

        if random_search_cv:
            if verbose: print('running random search cv')
            hyper_space = models_hyperparam_space[model_name]
            rcv = RandomizedSearchCV(model, hyper_space, n_iter=n_iter, cv=cv, n_jobs=n_jobs,
                                     scoring='neg_mean_squared_error', refit=False, random_state=random_state)
            rcv.fit(train_x, train_y)
            model.set_params(**rcv.best_params_)

    if verbose: print('fitting model', model_name)
    model.fit(train_x, train_y)
    return model


def evaluate_regressor(model, test_x, test_y, scatter_plot=False, prefix=''):
    pred_y = model.predict(test_x)
    if scatter_plot:
        pd.DataFrame(zip(pred_y, test_y), columns=['pred', 'true']).plot(kind='scatter', x='pred', y='true')
    return regression_scoring(test_y, pred_y, prefix=prefix)
