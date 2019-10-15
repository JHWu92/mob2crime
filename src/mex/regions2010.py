import glob
import os
import sys

import geopandas as gp
import pandas as pd
from shapely.ops import cascaded_union

import src.mex as mex
import src.mex.census2010 as census

if not os.getcwd().endswith('mob2crime'):
    os.chdir('..')
sys.path.insert(0, os.getcwd())

DIR_CenGeo = 'data/mexico/geography-socioeconomics/2010CensusGeography'
DIR_ZM = 'data/mexico/geography-socioeconomics/2010MetropolitanAreas'


def filter_mun_ids(gpdf, mun_ids):
    if mun_ids is not None:
        gpdf = gpdf[gpdf.mun_id.isin(mun_ids)].copy()
    return gpdf


def filter_loc_ids(gpdf, loc_ids):
    if loc_ids is not None:
        gpdf = gpdf[gpdf.loc_id.isin(loc_ids)].copy()
    return gpdf


def states(to_4326=False):
    mge = gp.read_file(f'{DIR_CenGeo}/national_macro/mge2010v5_0/estados.shp')
    if to_4326:
        mge = mge.to_crs(epsg=4326)
    return mge


def country(to_4326=False):
    mge = states(to_4326=False)
    ctry = cascaded_union(mge.geometry)
    ctry = gp.GeoDataFrame([[ctry]], columns=['geometry'])
    ctry.crs = mge.crs
    if to_4326:
        ctry = ctry.to_crs(epsg=4326)
    return ctry


def municipalities(mun_ids=None, to_4326=False):
    mgm = gp.read_file(f'{DIR_CenGeo}/national_macro/mgm2010v5_0/municipios.shp')
    mgm['mun_id'] = mgm.CVE_ENT + mgm.CVE_MUN
    mgm = filter_mun_ids(mgm, mun_ids)
    if to_4326:
        mgm = mgm.to_crs(epsg=4326)
    return mgm


def locs_urban(mun_ids=None, loc_ids=None, to_4326=False):
    mglu = gp.read_file(f'{DIR_CenGeo}/national_macro/mglu2010v5_0/poligonos_urbanos.shp')
    mglu['mun_id'] = mglu.CVE_ENT + mglu.CVE_MUN
    mglu['loc_id'] = mglu.CVE_ENT + mglu.CVE_MUN + mglu.CVE_LOC
    mglu = filter_mun_ids(mglu, mun_ids)
    mglu = filter_loc_ids(mglu, loc_ids)
    if to_4326:
        mglu = mglu.to_crs(epsg=4326)
    # add population
    pop_mglu = census.pop_loc_urban()
    mglu = mglu.merge(pop_mglu[['loc_id', 'pobtot']], on='loc_id', how='left')
    return mglu


def locs_rural(mun_ids=None, loc_ids=None, to_4326=False, buffer_point=500, mglr_only=False):
    # mglr in this file has only (lat,lon) points
    if not mglr_only:
        mgar_pls = _agebs_mzas('Rural', 'Ageb', mun_ids, loc_ids, to_4326)

    mglr_pts = gp.read_file(f'{DIR_CenGeo}/national_macro/mglr2010v5_0/localidades_rurales.shp')
    mglr_pts['mun_id'] = mglr_pts.CVE_ENT + mglr_pts.CVE_MUN
    mglr_pts['loc_id'] = mglr_pts.CVE_ENT + mglr_pts.CVE_MUN + mglr_pts.CVE_LOC
    mglr_pts = filter_mun_ids(mglr_pts, mun_ids)
    mglr_pts = filter_loc_ids(mglr_pts, loc_ids)
    # remove the pts that has polygons
    if not mglr_only:
        mglr_pts = mglr_pts[~mglr_pts.loc_id.isin(mgar_pls.loc_id)].copy()

    if buffer_point:
        mglr_pts.geometry = mglr_pts.buffer(buffer_point)
    if to_4326:
        mglr_pts = mglr_pts.to_crs(epsg=4326)

    if not mglr_only:
        mglr = pd.concat([mgar_pls, mglr_pts], ignore_index=True, sort=False)
    else:
        mglr = mglr_pts

    # add population
    pop_mglr = census.pop_loc_rural()
    mglr = mglr.merge(pop_mglr[['loc_id', 'pobtot']], on='loc_id', how='left')
    mglr.pobtot = mglr.pobtot.fillna(0).astype(int)
    return mglr


def agebs_rural():
    return locs_rural()


def agebs_urban(mun_ids=None, loc_ids=None, to_4326=False):
    mgau = _agebs_mzas('Urban', 'Ageb', mun_ids, loc_ids, to_4326)
    pop_mgau = census.pop_ageb_urban()
    mgau = mgau.merge(pop_mgau[['ageb_id', 'pobtot']], on='ageb_id', how='left')
    return mgau


def mzas_urban(mun_ids=None, loc_ids=None, to_4326=False):
    # mzas crs is 4326 (saved by "Mexico 2010 Census stats and basemap 2Organize.ipynb")
    mgmzu = _agebs_mzas('Urban', 'Mza', mun_ids, loc_ids, to_4326)
    pop_mgmzu = census.pop_mza_urban()
    mgmzu.merge(pop_mgmzu[['mza_id', 'pobtot']], on='mza_id', how='left')
    return mgmzu


def _agebs_mzas(urb_or_rur, ageb_or_mza, mun_ids=None, loc_ids=None, to_4326=False):
    # ageb crs is 4326 (saved by "Mexico 2010 Census stats and basemap 2Organize.ipynb")
    assert urb_or_rur in ('Urban', 'Rural')
    assert ageb_or_mza in ('Ageb', 'Mza')
    if mun_ids is None:
        mun_ids = [fn[-16:-11] for fn in glob.glob(f'{DIR_CenGeo}/{urb_or_rur}{ageb_or_mza}/*')]

    mg = []
    for mun_id in mun_ids:
        path = f'{DIR_CenGeo}/{urb_or_rur}{ageb_or_mza}/{mun_id}.geojson.gz'
        if not os.path.exists(path):
            print(f'{urb_or_rur}{ageb_or_mza}/{mun_id} not exists')
            continue
        ageb = gp.read_file(f'gzip://{DIR_CenGeo}/{urb_or_rur}{ageb_or_mza}/{mun_id}.geojson.gz')
        mg.append(ageb)
    mg = pd.concat(mg, ignore_index=True, sort=False)

    mg['CVE_ENT'] = mg.CVEGEO.apply(lambda x: x[:2])
    mg['CVE_MUN'] = mg.CVEGEO.apply(lambda x: x[2:5])
    mg['CVE_LOC'] = mg.CVEGEO.apply(lambda x: x[5:9])
    mg['mun_id'] = mg.CVEGEO.apply(lambda x: x[:5])
    mg['loc_id'] = mg.CVEGEO.apply(lambda x: x[:9])
    if ageb_or_mza == 'Ageb':
        mg['CVE_AGEB'] = mg.CVEGEO.apply(lambda x: x[-4:])
        mg.rename(columns={'NOMLOC': 'NOM_LOC', 'CVEGEO': 'ageb_id'}, inplace=True)
        mg = mg.reindex(
            columns=['CVE_ENT', 'CVE_MUN', 'CVE_LOC', 'CVE_AGEB', 'NOM_LOC', 'geometry', 'mun_id', 'loc_id', 'ageb_id'])

    else:
        mg['CVE_MZA'] = mg.CVEGEO.apply(lambda x: x[13:])
        mg['CVE_AGEB'] = mg.CVEGEO.apply(lambda x: x[9:13])
        mg.rename(columns={'NOMLOC': 'NOM_LOC', 'CVEGEO': 'mza_id'}, inplace=True)
    mg = filter_loc_ids(mg, loc_ids)
    if not to_4326:
        mg = mg.to_crs(mex.crs)
    return mg


def mpa_all(to_4326=False):
    """
    Source: http://www.conapo.gob.mx/es/CONAPO/Datos_Abiertos_Delimitacion_de_Zonas_Metropolitanas
    The shape and population haven't been checked against census 2010.
    :return: metropolitan areas sorted by population, in mex.crs
    """
    zm = gp.read_file(f'{DIR_ZM}/ZM_2010.shp')

    # this file have better NOM_SUN
    zm_meta = pd.read_csv(f'{DIR_ZM}/ZM_00-10_info.csv')
    zm_meta = zm_meta[zm_meta['AÑO'] == 2010]

    suns = []

    for cve_sun, sun in zm.groupby('CVE_SUN'):
        geometry = cascaded_union(sun.geometry.tolist())
        pob = sun.POB_2010.sum()
        zm_name = zm_meta[zm_meta.CVE_ZM == cve_sun].ZM.values[0].replace('Zona metropolitana de ', '').replace(
            'Zona metropolitana del ', '')
        suns.append({'NOM_SUN': zm_name, 'CVE_SUN': cve_sun, 'pobtot': pob,
                     'mun_ids': ','.join(sun.CVE_MUN1.tolist()), 'geometry': geometry})

    suns = gp.GeoDataFrame(suns)
    suns.crs = zm.crs
    suns = suns.sort_values('pobtot', ascending=False)
    if to_4326:
        suns = suns.to_crs(epsg=4326)
    return suns


def mpa_urban_per_municipality(to_4326=False):
    zm_meta = pd.read_csv(f'{DIR_ZM}/ZM_00-10_info.csv')
    zm_meta = zm_meta[zm_meta['AÑO'] == 2010]
    zm_meta['mun_id'] = zm_meta.CVE_MUN.apply(lambda x: f'{x:05}')
    mglu_zm = locs_urban(zm_meta.mun_id.tolist(), to_4326=to_4326)
    mglu_zm = mglu_zm.merge(zm_meta[['mun_id', 'CVE_ZM']]).rename(columns={'CVE_ZM': 'CVE_SUN'})
    return mglu_zm
