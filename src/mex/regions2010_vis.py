import os
import sys
from itertools import chain

import folium

import src.mex as mex
import src.mex.regions2010 as region
import src.utils.map_vis as mvis

if not os.getcwd().endswith('mob2crime'):
    os.chdir('..')
sys.path.insert(0, os.getcwd())

zms = region.mpa_all(to_4326=True)
mun_ids = list(chain(*zms.mun_ids.apply(lambda x: x.split(',')).tolist()))
mgm_zms = region.municipalities(mun_ids, to_4326=True)
mglu_zms = region.mpa_urban_per_municipality(to_4326=True)
mglr_zms = region.locs_rural(mun_ids, to_4326=True)

print('building map')
m = folium.Map(location=[mex.clat, mex.clon], zoom_start=5)
mvis.geojson_per_row(zms, name='metro', tip_cols=['NOM_SUN', 'CVE_SUN', 'pobtot'], color='grey', some_map=m)
mvis.geojson_per_row(mgm_zms, name='mgm', tip_cols=['NOM_MUN', 'mun_id'], color='yellow', some_map=m)
mvis.geojson_per_row(mglu_zms, name='mglu_zms', tip_cols=['CVE_SUN', 'NOM_LOC', 'mun_id', 'loc_id', 'pobtot'],
                     color='green', some_map=m)
mvis.geojson_per_row(mglr_zms, name='mglr_zms', tip_cols=['NOM_LOC', 'mun_id', 'loc_id', 'pobtot'], color='green',
                     some_map=m)

folium.LayerControl().add_to(m)
m.save('maps/Mex zms to localidads.html')
