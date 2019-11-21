import numpy as np


def compacity_coefficient(pairwise_distance_average, sqrt_area):
    return pairwise_distance_average / sqrt_area


def cohesion(ref_circle_radius, pairwise_dist_square_avg):
    # cohesion = average distance-square of reference circle / avg distance-square of the shape
    # I don't have the formula for reference circle, but based on random sample points, the avg
    # is radius ** 2
    return ref_circle_radius ** 2 / pairwise_dist_square_avg


def proximity(ref_circle_radius, d2centroid_avg):
    # average distance to a circle is 2R/3
    return 2 * ref_circle_radius / 3 / d2centroid_avg


def moment_inertia(raster_areas, d2centroid):
    # moment of inertia, hs_raster_area = s ** 2
    # MI = I (MI of a grid to its own centroid) + distance square to centroid * its own area
    mi = (raster_areas ** 2) / 6 + d2centroid ** 2 * raster_areas
    mi = mi.sum()
    mi_ref = raster_areas.sum() ** 2 / 2 / np.pi  # A^2 / 2pi
    nmi = mi_ref / mi
    return nmi


def mass_moment_inertia(raster_rper, d2mass_centroid):
    # for raster representation, area of each grid are the same
    if raster_rper.Area.nunique() != 1:
        area = raster_rper.Area
        aiaj = np.outer(raster_rper.Area.values, raster_rper.Area.values)
    else:
        area = raster_rper.Area.unique()[0]
        aiaj = area ** 2

    # mass moment of inertia
    # IGs = density * IG (the mi of the grid to its own centroid)
    # md2s = distance square to mass centroid * its own mass
    # mmi = sum of IGs+md2s
    IGs = raster_rper.Density * area ** 2 / 6
    md2s = raster_rper.Mass * d2mass_centroid ** 2
    mmi = (IGs + md2s).sum()

    density = raster_rper.Density
    di = density[np.newaxis, :]
    dj = density[:, np.newaxis]
    min_pij_aij = (di - dj) < 0
    min_pij_aij = min_pij_aij * di + (1 - min_pij_aij) * dj
    min_pij_aij = 2 * min_pij_aij * aiaj
    sum_min_pij_aij = (min_pij_aij.sum() - np.trace(min_pij_aij).sum()) / 2
    density_area2 = (density * area ** 2).sum()
    mmi_ref = (density_area2 + sum_min_pij_aij) / 2 / np.pi
    nmmi = mmi_ref / mmi
    return nmmi
