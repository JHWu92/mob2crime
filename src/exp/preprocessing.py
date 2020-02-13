import sklearn.preprocessing as sk_prep


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
