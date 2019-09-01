import os
import sys

if not os.getcwd().endswith('mob2crime'):
    os.chdir('..')
sys.path.insert(0, os.getcwd())
print(os.getcwd())


import pandas as pd
import geopandas as gp
import folium

import src.utils.gis as gis
import src.utils.map_vis as mv
import src.mex_helper as mex

# ==============
# load files
# ==============
# tower vor
tvor = mex.tower_vor()

# population file
population = pd.read_csv('data/mexico/Localidades-population.csv')
population['loc_id'] = population['Clave de localidad'].apply(lambda x: f'{x:09}')
population['CVE_ENT'] = population['Clave entidad'].apply(lambda x: f'{x:02}')

# add population to localidade location
LUR = gp.read_file('data/mexico/inegi2018/Marco_Geoestadistico_Integrado_diciembre_2018/conjunto de datos/01_32_l.shp')
LUR['loc_id'] = LUR.CVE_ENT + LUR.CVE_MUN + LUR.CVE_LOC
LUR4326 = LUR.to_crs(epsg=4326)
LUR4326 = LUR4326.merge(population[['loc_id', 'Población total']], how='left')
LUR4326 = LUR4326.rename(columns={'Población total': 'Pop'})

LPR = gp.read_file(
    'data/mexico/inegi2018/Marco_Geoestadistico_Integrado_diciembre_2018/conjunto de datos/01_32_lpr.shp')
LPR['loc_id'] = LPR.CVE_ENT + LPR.CVE_MUN + LPR.CVE_LOC
LPRBf = LPR.copy()
LPRBf.geometry = LPRBf.buffer(500)
LPRBf4326 = LPRBf.to_crs(epsg=4326)
LPRBf4326 = LPRBf4326.merge(population[['loc_id', 'Población total']], how='left')
LPRBf4326 = LPRBf4326.rename(columns={'Población total': 'Pop'})
LPRBf4326_Ponly = LPRBf4326[~LPRBf4326.loc_id.isin(LUR4326.loc_id)]

# ==============
# verify the quality of the match between localidad and population
# ==============
# most localidad popluation has geographic information
df = pd.DataFrame([
    LUR4326.groupby('CVE_ENT')['Pop'].sum(),
    LPRBf4326_Ponly.groupby('CVE_ENT')['Pop'].sum(),
    population.groupby('CVE_ENT')['Población total'].sum()
]).T
df.columns = ['pop_poly', 'pop_point', 'pop_all']
df['pop_shap'] = df.pop_poly + df.pop_point
df['poly_all'] = df.pop_poly / df.pop_all
df['shap_all'] = df.pop_shap / df.pop_all
print('localidad with geographic information', (df.shap_all > 0.95).mean(), 'min percentage', df.shap_all.min())

# areas that has population > 0
LUR_pop = LUR4326[LUR4326.Pop > 0].drop(['CVEGEO', 'CVE_LOC'], axis=1)
LPR_pop = LPRBf4326_Ponly[LPRBf4326_Ponly.Pop > 0].drop(['CVE_LOC', 'CVE_AGEB', 'CVE_MZA', 'PLANO', 'CVEGEO'], axis=1)
LPR_pop['AMBITO'] = 'Rural'
# merge two sources
L_pop = pd.concat([LUR_pop, LPR_pop[LUR_pop.columns]], ignore_index=True).set_index('loc_id')

# =============
# distribute tower's users count by population
# =============

# compute the intersection area between tower and localidad
t2loc = gis.polys2polys(tvor, L_pop, pname1='tower', pname2='localidad', cur_crs=4326, area_crs=mex.AREA_CRS,
                        intersection_only=False)
t2loc = t2loc.merge(L_pop[['Pop']], left_on='localidad', right_index=True)

# localidad area is the sum area covered by towers
loc_area = t2loc.groupby('localidad').iarea.sum()
loc_area.name = 'loclidad_area'
t2loc = t2loc.drop(['localidad_area', 'weight'], axis=1).merge(loc_area.to_frame(), left_on='localidad', right_index=True)

# iPop is the population of the intersected area between a tower and a localidad
# within a localidad, the population is assumed to be distributed evenly over space
# therefore the population is divided proportionally to the intersection area
t2loc['iPop'] = t2loc.Pop * t2loc.iarea / t2loc.loclidad_area

# the total population covered by a tower is the sum of iPop
tower_cover_pop = t2loc.groupby('tower').iPop.sum()
tower_cover_pop.name = 'tower_pop'
t2loc = t2loc.merge(tower_cover_pop.to_frame(), left_on='tower', right_index=True)

# the weight to distribute tower's users count
t2loc['weight'] = t2loc.iPop / t2loc.tower_pop

t2loc[
    ['localidad', 'tower', 'weight', 'iarea', 'tower_area', 'loclidad_area', 'Pop', 'iPop', 'tower_pop']
].to_csv('data/mex_tower/TVorByLocPop.csv')

# an example map of t2loc
m = folium.Map(location=[19.381495, -99.139095], zoom_start=6)
mv.geojson_per_row(tvor, 'Tower', some_map=m)
mv.geojson_per_row(t2loc[t2loc.localidad.isin(population[population.CVE_ENT.isin(['01', '02'])].loc_id)], 'T2loc',
                some_map=m, color='green', tip_cols=['localidad', 'tower', 'weight', 'Pop', 'tower_pop', 'iPop'])
# geojson_per_row(target_area, 'poly', color='green', some_map=m)
# geojson_per_row(target_area2, 'circle', color='yellow', some_map=m)
folium.LayerControl().add_to(m)
m.save('data/mex_tower/VorByPop.html')
