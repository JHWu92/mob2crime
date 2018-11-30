import geopandas as gp
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    from math import radians, cos, sin, asin, sqrt

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def voronoi_finite_polygons_2d(vor, radius=None):
    """Reconstruct infinite Voronoi regions in a
    2D diagram to finite regions.
    Source:
    [https://stackoverflow.com/a/20678647/1595060](https://stackoverflow.com/a/20678647/1595060)
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()
    # Construct a map containing all ridges for a
    # given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points,
                                  vor.ridge_vertices):
        all_ridges.setdefault(
            p1, []).append((p2, v1, v2))
        all_ridges.setdefault(
            p2, []).append((p1, v1, v2))
    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue
        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue
            # Compute the missing endpoint of an
            # infinite ridge
            t = vor.points[p2] - \
                vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = vor.points[[p1, p2]]. \
                mean(axis=0)
            direction = np.sign(
                np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + \
                        direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        # Sort region counterclockwise.
        vs = np.asarray([new_vertices[v]
                         for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(
            vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[
            np.argsort(angles)]
        new_regions.append(new_region.tolist())
    return new_regions, np.asarray(new_vertices)


def vor2gp(vor, radius=None, dataframe=False, lonlat_bounded=True):
    regions, vertices = voronoi_finite_polygons_2d(vor, radius=radius)
    polys = []
    for r in regions:
        p = Polygon(vertices[r])
        if lonlat_bounded:
            p = p.intersection(box(-180, -90, 180, 90))
        polys.append(p)
    if dataframe:
        return gp.GeoDataFrame(polys, columns=['geometry'])
    return gp.GeoSeries(polys)


def lonlats2vorpolys(lonlats, radius=None, dataframe=False, lonlat_bounded=True):
    vor = Voronoi(lonlats)
    return vor2gp(vor, radius, dataframe, lonlat_bounded)


def polys2polys(polys1, polys2, pname1='poly1', pname2='poly2', cur_epsg=None, area_epsg=None, intersection_only=True):
    """Compute the weights of from polygons 1 to polygons 2,
    So that the statistics in polys1 can be transferred to polys2

    If intersection_only:
        Weight(i,j) = Area(polys1i in polys2j) / Area(polys1i in polys2)
    Else:
        Weight(i,j) = Area(polys1i in polys2j) / Area(polys1i)

    :param polys1: GeoDataFrame
        polygons with statistics to distributed over the other polygons
    :param polys2: GeoDataFrame
        polygons to get statistics from polys1
    :param pname1: column name for the index of polys1 in the output
    :param pname2: column name for the index of polys2 in the output
    :param cur_epsg: the current epsg of polys1 and polys2
    :param area_epsg: the epsg for the area computation

    :return: pd.DataFrame(columns=[pname1, pname2, 'weight'])
        the mapping from polys1 to polys2
    """

    do_crs_transform = True

    # make sure CRS is set correctly
    if cur_epsg is None and polys1.crs is None and polys2.crs is None:
        if area_epsg is None:
            do_crs_transform = False
            print("No current epsg is specified. Area is computed directed in the current coordinates")
        else:
            raise ValueError('area epsg is specified, but the polygons have no CRS')

    if do_crs_transform:
        if area_epsg is None:
            raise ValueError(
                'Need to do area transform, but area is not specified. '
                f"cur_epsg is {cur_epsg}, polys1.crs is {polys1.crs}, polys2.crs is {polys2.crs}"
            )
        if polys1.crs is None: polys1.crs = {'init': 'epsg:%d' % cur_epsg, 'no_defs': True}
        if polys2.crs is None: polys2.crs = {'init': 'epsg:%d' % cur_epsg, 'no_defs': True}

    # get intersections between polys1 and polys2
    ps1tops2 = gp.sjoin(polys1, polys2)
    itxns = []
    for li, row in ps1tops2.iterrows():
        itxn = polys2.loc[row.index_right].geometry.intersection(polys1.loc[li].geometry)
        itxns.append({pname1: li, pname2: row.index_right, 'geometry': itxn})
    itxns = gp.GeoDataFrame(itxns)

    # get area of the intersections
    if do_crs_transform:
        itxns.crs = polys1.crs
        itxns_for_area = itxns.to_crs(epsg=area_epsg)
    else:
        itxns_for_area = itxns
    itxns['iarea'] = itxns_for_area.geometry.apply(lambda x: x.area)
    itxns.drop(itxns[itxns['iarea'] == 0].index, inplace=True)

    # compute the weight
    if intersection_only:
        polys1_area = itxns.groupby(pname1).apply(lambda x: x['iarea'].sum()).to_frame()
    else:
        polys1_area = polys1.to_crs(epsg=area_epsg).geometry.apply(lambda x: x.area).to_frame()
        polys1_area.index.name = pname1
    polys1_area = polys1_area
    polys1_area.columns = [pname1 + '_area']
    polys1_area.reset_index(inplace=True)
    itxns = itxns.merge(polys1_area)
    itxns['weight'] = itxns['iarea'] / itxns[pname1 + '_area']
    return gp.pd.DataFrame(itxns[[pname1, pname2, 'weight']])
