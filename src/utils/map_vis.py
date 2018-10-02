import folium

def geojson_per_row(gpdf, name, color='blue', tip_cols=None, some_map=None):

    feature_group = folium.FeatureGroup(name=name)
    for row in gpdf.itertuples():

        if row.geometry.boundary.type=='MultiLineString':
            lines = row.geometry.boundary
        else:
            lines= [row.geometry.boundary]
            
        for line in lines:
            print
            tip = '<br>'.join(['%s: %s' % (col, getattr(row, col)) for col in tip_cols]) if tip_cols is not None else name
            folium.Polygon(locations=[(lat,lon) for lon,lat in line.coords], color=color, fill_color=color, tooltip=tip, popup=tip).add_to(feature_group)
    if some_map is not None:
        feature_group.add_to(some_map)
    return feature_group