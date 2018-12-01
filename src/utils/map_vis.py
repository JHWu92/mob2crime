import folium
import pandas as pd
import datetime

def geojson_per_row(gpdf, name, color='blue', tip_cols=None, some_map=None):
    feature_group = folium.FeatureGroup(name=name)
    for row in gpdf.itertuples():

        if row.geometry.boundary.type == 'MultiLineString':
            lines = row.geometry.boundary
        else:
            lines = [row.geometry.boundary]

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


def time_slider_choropleth(gpolys, values, dates=None, mini=None, maxi=None, dstr_format='%Y-%m-%d'):
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

    colors = values.applymap(lambda x: (x - mini) / (maxi - mini)).applymap(cmap)
    colors.index = dates

    styledict = {}
    for i in colors:
        styledict[str(i)] = {d: {'color': c, 'opacity': 0.8} for d, c in colors[i].iteritems()}
    return TimeSliderChoropleth(gpolys.to_json(), styledict=styledict)