import pandas as pd

import src.creds as const

# (MODALITY, TIPO, SUBTIPO): (type level 1, type level 2)
# level 1: battery/assualt, burglary, homicide, kidnapping, robbery,
# sexual offense, theft, theft from a vehicle, theft of cattle
# level 2: violent, property
sesnsp_types_of_crimes = {
    ('DELITOS PATRIMONIALES', 'ABUSO DE CONFIANZA', 'ABUSO DE CONFIANZA'): ('', ''),
    ('DELITOS PATRIMONIALES', 'DAÑO EN PROPIEDAD AJENA', 'DAÑO EN PROPIEDAD AJENA'): ('', ''),
    ('DELITOS PATRIMONIALES', 'DESPOJO', 'CON VIOLENCIA'): ('robbery', 'violent'),
    ('DELITOS PATRIMONIALES', 'DESPOJO', 'SIN DATOS'): ('theft', 'property'),
    ('DELITOS PATRIMONIALES', 'DESPOJO', 'SIN VIOLENCIA'): ('theft', 'property'),
    ('DELITOS PATRIMONIALES', 'EXTORSION', 'EXTORSION'): ('', ''),
    ('DELITOS PATRIMONIALES', 'FRAUDE', 'FRAUDE'): ('', ''),
    ('DELITOS SEXUALES (VIOLACION)', 'VIOLACION', 'VIOLACION'): ('sexual offense', 'violent'),
    ('HOMICIDIOS', 'CULPOSOS', 'CON ARMA BLANCA'): ('homicide', 'violent'),
    ('HOMICIDIOS', 'CULPOSOS', 'CON ARMA DE FUEGO'): ('homicide', 'violent'),
    ('HOMICIDIOS', 'CULPOSOS', 'OTROS'): ('homicide', 'violent'),
    ('HOMICIDIOS', 'CULPOSOS', 'SIN DATOS'): ('homicide', 'violent'),
    ('HOMICIDIOS', 'DOLOSOS', 'CON ARMA BLANCA'): ('homicide', 'violent'),
    ('HOMICIDIOS', 'DOLOSOS', 'CON ARMA DE FUEGO'): ('homicide', 'violent'),
    ('HOMICIDIOS', 'DOLOSOS', 'OTROS'): ('homicide', 'violent'),
    ('HOMICIDIOS', 'DOLOSOS', 'SIN DATOS'): ('homicide', 'violent'),
    ('LESIONES', 'CULPOSAS', 'CON ARMA BLANCA'): ('battery/assualt', 'violent'),
    ('LESIONES', 'CULPOSAS', 'CON ARMA DE FUEGO'): ('battery/assualt', 'violent'),
    ('LESIONES', 'CULPOSAS', 'OTROS'): ('battery/assualt', 'violent'),
    ('LESIONES', 'CULPOSAS', 'SIN DATOS'): ('battery/assualt', 'violent'),
    ('LESIONES', 'DOLOSAS', 'CON ARMA BLANCA'): ('battery/assualt', 'violent'),
    ('LESIONES', 'DOLOSAS', 'CON ARMA DE FUEGO'): ('battery/assualt', 'violent'),
    ('LESIONES', 'DOLOSAS', 'OTROS'): ('battery/assualt', 'violent'),
    ('LESIONES', 'DOLOSAS', 'SIN DATOS'): ('battery/assualt', 'violent'),
    ('OTROS DELITOS', 'AMENAZAS', 'AMENAZAS'): ('battery/assualt', 'violent'),
    ('OTROS DELITOS', 'ESTUPRO', 'ESTUPRO'): ('sexual offense', 'violent'),
    ('OTROS DELITOS', 'OTROS SEXUALES', 'OTROS SEXUALES'): ('sexual offense', 'violent'),
    ('OTROS DELITOS', 'RESTO DE LOS DELITOS (OTROS)', 'RESTO DE LOS DELITOS (OTROS)'): (
        '', ''),
    ('PRIV. DE LA LIBERTAD (SECUESTRO)', 'SECUESTRO', 'SECUESTRO'): ('kidnapping', 'violent'),
    ('ROBO COMUN', 'CON VIOLENCIA', 'A CASA HABITACION'): ('burglary', 'property'),
    ('ROBO COMUN', 'CON VIOLENCIA', 'A NEGOCIO'): ('robbery', 'violent'),
    ('ROBO COMUN', 'CON VIOLENCIA', 'A TRANSEUNTES'): ('robbery', 'violent'),
    ('ROBO COMUN', 'CON VIOLENCIA', 'A TRANSPORTISTAS'): ('robbery', 'violent'),
    ('ROBO COMUN', 'CON VIOLENCIA', 'DE VEHICULOS'): ('theft from a vehicle', 'property'),
    ('ROBO COMUN', 'CON VIOLENCIA', 'OTROS'): ('robbery', 'violent'),
    ('ROBO COMUN', 'CON VIOLENCIA', 'SIN DATOS'): ('robbery', 'violent'),
    ('ROBO COMUN', 'SIN VIOLENCIA', 'A CASA HABITACION'): ('burglary', 'property'),
    ('ROBO COMUN', 'SIN VIOLENCIA', 'A NEGOCIO'): ('robbery', 'violent'),
    ('ROBO COMUN', 'SIN VIOLENCIA', 'A TRANSEUNTES'): ('robbery', 'violent'),
    ('ROBO COMUN', 'SIN VIOLENCIA', 'A TRANSPORTISTAS'): ('robbery', 'violent'),
    ('ROBO COMUN', 'SIN VIOLENCIA', 'DE VEHICULOS'): ('theft from a vehicle', 'property'),
    ('ROBO COMUN', 'SIN VIOLENCIA', 'OTROS'): ('robbery', 'violent'),
    ('ROBO COMUN', 'SIN VIOLENCIA', 'SIN DATOS'): ('robbery', 'violent'),
    ('ROBO DE GANADO (ABIGEATO)', 'ABIGEATO', 'ABIGEATO'): ('theft of cattle', 'property'),
    ('ROBO EN CARRETERAS', 'CON VIOLENCIA', 'A AUTOBUSES'): ('theft from a vehicle', 'property'),
    ('ROBO EN CARRETERAS', 'CON VIOLENCIA', 'A CAMIONES DE CARGA'): (
        'theft from a vehicle', 'property'),
    ('ROBO EN CARRETERAS', 'CON VIOLENCIA', 'A VEHICULOS PARTICULARES'): (
        'theft from a vehicle', 'property'),
    ('ROBO EN CARRETERAS', 'CON VIOLENCIA', 'OTROS'): ('theft from a vehicle', 'property'),
    ('ROBO EN CARRETERAS', 'CON VIOLENCIA', 'SIN DATOS'): ('theft from a vehicle', 'property'),
    ('ROBO EN CARRETERAS', 'SIN VIOLENCIA', 'A AUTOBUSES'): ('theft from a vehicle', 'property'),
    ('ROBO EN CARRETERAS', 'SIN VIOLENCIA', 'A CAMIONES DE CARGA'): (
        'theft from a vehicle', 'property'),
    ('ROBO EN CARRETERAS', 'SIN VIOLENCIA', 'A VEHICULOS PARTICULARES'): (
        'theft from a vehicle', 'property'),
    ('ROBO EN CARRETERAS', 'SIN VIOLENCIA', 'OTROS'): ('theft from a vehicle', 'property'),
    ('ROBO EN CARRETERAS', 'SIN VIOLENCIA', 'SIN DATOS'): ('theft from a vehicle', 'property'),
    ('ROBO EN INSTITUCIONES BANCARIAS', 'CON VIOLENCIA', 'A BANCOS'): ('', ''),
    ('ROBO EN INSTITUCIONES BANCARIAS', 'CON VIOLENCIA', 'A CASA DE BOLSA'): ('', ''),
    ('ROBO EN INSTITUCIONES BANCARIAS', 'CON VIOLENCIA', 'A CASA DE CAMBIO'): ('', ''),
    ('ROBO EN INSTITUCIONES BANCARIAS', 'CON VIOLENCIA', 'A EMPRESA DE TRASLADO DE VALORES'): (
        '', ''),
    ('ROBO EN INSTITUCIONES BANCARIAS', 'CON VIOLENCIA', 'OTROS'): ('', ''),
    ('ROBO EN INSTITUCIONES BANCARIAS', 'CON VIOLENCIA', 'SIN DATOS'): ('', ''),
    ('ROBO EN INSTITUCIONES BANCARIAS', 'SIN VIOLENCIA', 'A BANCOS'): ('', ''),
    ('ROBO EN INSTITUCIONES BANCARIAS', 'SIN VIOLENCIA', 'A CASA DE BOLSA'): ('', ''),
    ('ROBO EN INSTITUCIONES BANCARIAS', 'SIN VIOLENCIA', 'A CASA DE CAMBIO'): ('', ''),
    ('ROBO EN INSTITUCIONES BANCARIAS', 'SIN VIOLENCIA', 'A EMPRESA DE TRASLADO DE VALORES'): (
        '', ''),
    ('ROBO EN INSTITUCIONES BANCARIAS', 'SIN VIOLENCIA', 'OTROS'): ('', ''),
    ('ROBO EN INSTITUCIONES BANCARIAS', 'SIN VIOLENCIA', 'SIN DATOS'): ('', ''),
}


def get_crime_type(type_in_sesnsp, level=0):
    return sesnsp_types_of_crimes[(type_in_sesnsp['MODALIDAD'], type_in_sesnsp['TIPO'], type_in_sesnsp['SUBTIPO'])][
        level]


def sesnsp_crime_counts(year, type_level):
    sesnsp = pd.read_csv(f'{const.crime_dir}/IDM_mar19.csv', encoding='1252')
    sesnsp.INEGI = sesnsp.INEGI.apply(lambda x: f'{x:05}')
    crimes = sesnsp[sesnsp['AÑO'] == year].copy()
    crimes['crime_type'] = crimes.apply(lambda x: get_crime_type(x, type_level), axis=1)
    crimes['annual_count'] = crimes[
        ['ENERO', 'FEBRERO', 'MARZO', 'ABRIL', 'MAYO', 'JUNIO', 'JULIO', 'AGOSTO', 'SEPTIEMBRE', 'OCTUBRE', 'NOVIEMBRE',
         'DICIEMBRE']].sum(axis=1)
    crimes = crimes[crimes.crime_type != ''].groupby(
        ['INEGI', 'crime_type']).annual_count.sum().to_frame()
    crimes = crimes.reset_index().pivot(index='INEGI', columns='crime_type', values='annual_count')
    crimes.index.name = 'mun_id'
    crimes.columns.name = None
    return crimes
