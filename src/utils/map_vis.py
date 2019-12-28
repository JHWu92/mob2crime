import datetime

import folium
import pandas as pd


def geojson_per_row_color_col(gpdf, name, color='blue', tip_cols=None, some_map=None, color_col=None):
    """
    $('path').attr({'fill-opacity':1, 'stroke':'#000000', 'stroke-width':1})
    """
    if color_col:
        from branca.colormap import linear
        cmap = {
            'red': linear.Reds_09.scale(),
            'blue': linear.Blues_09.scale(),
            'green': linear.Greens_09.scale(),
            'orange': linear.Oranges_09.scale(),
            'purple': linear.Purples_09.scale(),
            'grey': linear.Greys_09.scale(),
        }[color]
        colors = gpdf[color_col].apply(cmap)
    else:
        colors = [color] * len(gpdf)
    feature_group = folium.FeatureGroup(name=name)
    for i, row in enumerate(gpdf.itertuples()):
        geom = row.geometry
        # get boundaries from geom
        if geom.type in ['Polygon', 'MultiPolygon']:
            if geom.boundary.type == 'MultiLineString':
                lines = geom.boundary
            else:
                lines = [geom.boundary]
        else:
            lines = []
            for part in geom:
                if part.type == 'Point':
                    continue
                elif part.type == 'LineString':
                    lines.append(part)
                else:
                    lines.append(part.boundary)
        clr = colors[i]
        for line in lines:
            tip = '<br>'.join(
                ['%s: %s' % (col, getattr(row, col)) for col in tip_cols]) if tip_cols is not None else name

            folium.Polygon(locations=[(lat, lon) for lon, lat in line.coords], color=clr, fill_color=clr,
                           tooltip=tip, popup=tip).add_to(feature_group)
    if some_map is not None:
        feature_group.add_to(some_map)
    return feature_group

def geojson_per_row(gpdf, name, color='blue', tip_cols=None, some_map=None):
    feature_group = folium.FeatureGroup(name=name)
    for row in gpdf.itertuples():
        geom = row.geometry
        # get boundaries from geom
        if geom.type in ['Polygon', 'MultiPolygon']:
            if geom.boundary.type == 'MultiLineString':
                lines = geom.boundary
            else:
                lines = [geom.boundary]
        else:
            lines = []
            for part in geom:
                if part.type == 'Point':
                    continue
                elif part.type == 'LineString':
                    lines.append(part)
                else:
                    lines.append(part.boundary)
        # if row.geometry.boundary.type == 'MultiLineString':
        #     lines = row.geometry.boundary
        # else:
        #     lines = [row.geometry.boundary]

        for line in lines:
            tip = '<br>'.join(
                ['%s: %s' % (col, getattr(row, col)) for col in tip_cols]) if tip_cols is not None else name
            folium.Polygon(locations=[(lat, lon) for lon, lat in line.coords], color=color, fill_color=color,
                           tooltip=tip, popup=tip).add_to(feature_group)
    if some_map is not None:
        feature_group.add_to(some_map)
    return feature_group


def point_per_row(gpdf, name, tip_cols=None, some_map=None):
    feature_group = folium.FeatureGroup(name=name)
    for row in gpdf.itertuples():
        tip = '<br>'.join(['%s: %s' % (col, getattr(row, col)) for col in tip_cols]) if tip_cols is not None else name
        folium.Marker((row.geometry.y, row.geometry.x), tooltip=tip, popup=tip).add_to(feature_group)
    if some_map is not None:
        feature_group.add_to(some_map)
    return feature_group


def time_slider_choropleth(gpolys, values, dates, mini=None, maxi=None, dstr_format='%Y-%m-%d', color_per_day=False):
    """
    :param gpolys: GeoDataFrame with the shape
    :param values: {index_of_gpolys: [values per day]}
    :param dates: list of dates (str) as the same format as dstr_format
    :param mini: min value across all values, if None, min(values) is used
    :param maxi: max value across all values, if None, max(values) is used
    :param dstr_format: format of dates str
    :param color_per_day: whether the color percentage/min/max is computed for each day. Default False
    :return: a layer that can call add_to(folium_map)
    """
    from folium.plugins import TimeSliderChoropleth
    from branca.colormap import linear
    cmap = linear.Reds_09.scale()

    dates = [str(int(datetime.datetime.strptime(t, dstr_format).timestamp())) for t in dates]
    values = pd.DataFrame(values)
    assert len(dates) == values.shape[0]

    if maxi is None:
        maxi = values.max().max()
    if mini is None:
        mini = values.min().min()

    if color_per_day:
        colors = values.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1).applymap(cmap)
    else:
        colors = values.applymap(lambda x: (x - mini) / (maxi - mini)).applymap(cmap)
    colors.index = dates

    styledict = {}
    for i in colors:
        styledict[str(i)] = {d: {'color': c, 'opacity': 0.8} for d, c in colors[i].iteritems()}
    return TimeSliderChoropleth(gpolys.to_json(), styledict=styledict)
