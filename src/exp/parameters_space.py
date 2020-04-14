import numpy as np


def sk_random_forest():
    """ source:
    https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    """
    # Number of trees in random forest
    n_estimators = [5, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    return {'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap}


n_estimators = [5, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900]
c_s = [1.e-04, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1.e+02]
gamma_s = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
learning_rate = [1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.]
max_depth = [None, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

params_xgb = {
    'booster': ['dart', 'gbtree'],
    'verbosity': [0],
    'n_jobs': [1],

    # related to conservative/overfit
    'learning_rate': learning_rate,
    'gamma': [0, 1, 10, 50, 100, 200, 1000],
    'max_depth': max_depth,
    'min_child_weight': [0, 1, 10, 50, 100, 200, 1000],
    'subsample': [0.1, 0.2, 0.3, 0.5, 0.8],
    'colsample_bytree': [0.2, 0.4, 0.6, 0.8, 1.0],
    'n_estimators': n_estimators
    # maybe for booster=gblinear only
    # 'reg_lambda': [0.1, 1.0, 10.0, 100.],
    # 'reg_alpha': [0.1, 1, 10., 100.],
}

sk_adaboost = {
    'n_estimators': n_estimators,
    'learning_rate': learning_rate,
    'loss': ['linear', 'square', 'exponential']
}

sk_bagging = {
    'n_estimators': n_estimators,
    'max_features': [.1, .3, .6, .9, 1.],
    'bootstrap': [True, False]
}

sk_decision_tree = {
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': max_depth,
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],

}

sk_gradient_boost = {
    'n_estimators': n_estimators,
    'max_features': ['auto', 'sqrt', 'log2'],
    'learning_rate': learning_rate,
    'max_depth': max_depth,
    'subsample': [0.5, 0.7, 1.0],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

sk_svr = [
    {'kernel': ['rbf'], 'C': c_s, 'gamma': gamma_s},
    {'kernel': ['sigmoid'], 'C': c_s, 'gamma': gamma_s},
    {'kernel': ['poly'], 'C': c_s, 'gamma': gamma_s, 'degree': [2, 3]},
]

sk_linear_svr = {
    'C': c_s,
    'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
    # 'epsilon': [0, 0.1, 1], # the value of this parameter depends on the scale of the target variable y. If unsure, set epsilon=0.
}
