import numpy as np
from copy import copy


def loubar_thres(arr, is_sorted=False):
    if not is_sorted:
        arr = copy(arr)
        arr = sorted(arr)

    lonrenz_y = arr.cumsum() / arr.sum()
    lonrenz_y = np.insert(lonrenz_y, 0, 0)
    x_axis = np.arange(lonrenz_y.size)/(lonrenz_y.size-1)
    slope = (lonrenz_y[-1] - lonrenz_y[-2]) / (x_axis[-1] - x_axis[-2])
    loubar = (slope - 1) / slope
    arr_thres = arr[int(np.ceil((lonrenz_y.size - 1) * loubar) - 1)]

    return loubar, arr_thres