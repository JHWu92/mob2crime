import sys

import pandas as pd

sys.path.insert(0, '../../')
folder_census = 'data/mexico/geography-socioeconomics/2010Census'


def pop_mza_urban():
    pop_mgzu = pd.read_csv(f'{folder_census}/urban_mza_pop.csv.gz',
                           dtype={'entidad': str, 'mun': str, 'loc': str, 'ageb': str, 'mza': str})
    pop_mgzu['mza_id'] = pop_mgzu.entidad + pop_mgzu.mun + pop_mgzu[
        'loc'] + pop_mgzu.ageb + pop_mgzu.mza
    return pop_mgzu


def pop_ageb_urban():
    pop_mgau = pd.read_csv(f'{folder_census}/urban_ageb_pop.csv',
                           dtype={'entidad': str, 'mun': str, 'loc': str, 'ageb': str, 'mza': str})
    pop_mgau['ageb_id'] = pop_mgau.entidad + pop_mgau.mun + pop_mgau['loc'] + pop_mgau.ageb
    return pop_mgau


def pop_loc_rural():
    pop_mglr = pd.read_csv(f'{folder_census}/rural_loc_pop.csv.gz',
                           dtype={'entidad': str, 'mun': str, 'loc': str, 'mza': str}, index_col=0)
    pop_mglr['loc_id'] = pop_mglr.entidad + pop_mglr.mun + pop_mglr['loc']
    return pop_mglr


def pop_loc_urban():
    pop_mgau = pop_ageb_urban()
    pop_mglu = pop_mgau.groupby(['entidad', 'mun', 'loc']).pobtot.sum().reset_index()
    pop_mglu['loc_id'] = pop_mglu.entidad + pop_mglu.mun + pop_mglu['loc']
    return pop_mglu
