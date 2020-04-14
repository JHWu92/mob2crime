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


def poverty_munic(kind='all', rate=True):
    """if rate, return percentage"""
    kinds = ['poverty', 'lack_food', 'poverty_lack_food']
    if kind == 'all':
        kind = kinds
    else:
        assert kind in kinds
        kind = [kind]

    if rate:
        kind = [k + '_rate' for k in kind]
    else:
        kind = [k + '_pop' for k in kind]

    df = pd.read_excel(
        'data/mexico/geography-socioeconomics/poverty-lack-food-2010.xlsx',
        dtype={
            'CVE_ENT': str,
            'CVE_MUN': str
        })
    df = df.set_index('CVE_MUN')
    df.index.name = 'mun_id'
    return df[kind]


def econ_active_unoccupied(rate=True):
    """if rate, return percentage"""
    df = pd.read_excel(
        'data/mexico/geography-socioeconomics/econ-active-population-2010.xlsx',
        dtype={
            'CVE_MUN': str
        }).set_index('CVE_MUN')
    df.index.name = 'mun_id'
    if rate:
        df['active_unoccupied'] = df['active_unoccupied'] / df['active_total']
    return df[['active_unoccupied']] * 100


def income_indigenous_group():
    """return df with columns:
    aver_income: average income,
    PI_type: type of municipality by Population of Indigenous
    """

    df = pd.read_excel('data/mexico/geography-socioeconomics/income-indigenous-group-2010.xlsx',
                       dtype={'CVE_MUN': str}).set_index('CVE_MUN')
    df.index.name = 'mun_id'
    return df[['aver_income', 'PI_type']]
