# +proj=lcc +lat_1=17.5 +lat_2=29.5 +lat_0=12 +lon_0=-102 +x_0=2500000 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs
# it is said to be equal area: https://gis.stackexchange.com/questions/234075/crs-for-calculating-areas-in-mexico
# it works good in buffer (/data_explore/Mexico Explore CRS)
crs = {'proj': 'lcc',
       'lat_1': 17.5,
       'lat_2': 29.5,
       'lat_0': 12,
       'lon_0': -102,
       'x_0': 2500000,
       'y_0': 0,
       'ellps': 'GRS80',
       'units': 'm',
       'no_defs': True}

clat, clon = 19.381495, -99.139095