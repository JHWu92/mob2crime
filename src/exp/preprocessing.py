import sklearn.preprocessing as sk_prep


def train_test_split(X, y, random_state=0):
    train_X = X.sample(frac=0.8, random_state=random_state)
    train_y = y.loc[train_X.index]
    test_index = sorted(set(X.index) - set(train_X.index))
    test_X = X.loc[test_index]
    test_y = y.loc[test_index]
    return train_X, test_X, train_y, test_y


def fillna(features):
    for col in features.columns:
        if col.split('-')[1] in ['COMP', 'MCOMP', 'COHE', 'PROX', 'NMI', 'NMMI']:
            # NaN in these features means there isn't any hotspots, hence no feature is computed.
            # here I use 0 for NaN
            features[col] = features[col].fillna(0)
    return features


def scaling_data(train_x, test_x, scaling='RobustScaler'):
    if isinstance(scaling, str):
        assert scaling in ('StandardScaler', 'MinMaxScaler', 'RobustScaler')
        scaler = getattr(sk_prep, scaling)
    else:
        # assuming scaling is a sklearn scaler
        scaler = scaling

    scaler = scaler().fit(train_x)
    scaled_train_x = scaler.transform(train_x)
    scaled_test_x = scaler.transform(test_x)
    return scaled_train_x, scaled_test_x
