import pandas as pd
import numpy as np
import re
import pycountry

def load_data(url):
    """
    Carga el archivo de Excel desde la URL proporcionada.
    """
    return pd.read_excel(url)

def clean_data(df):
    """
    Realiza la limpieza inicial de los datos:
    - Elimina filas con años anteriores a 1700.
    - Elimina columnas y filas con demasiados valores nulos.
    - Elimina columnas que no aportan información útil.
    """
    # Eliminar filas anteriores a 1700
    df = df[df['Year'] >= 1700]

    # Eliminar columnas con más del 50% de valores faltantes
    missing_values_threshold = 0.5
    df = df.loc[:, df.isnull().mean() < missing_values_threshold]

    # Eliminar filas con más de 5 valores faltantes
    df = df.dropna(thresh=len(df.columns) - 5)
    
    # Eliminar columnas no útiles
    columns_to_drop = ['pdf', 'href formula', 'href', 'Case Number.1', 'Case Number', 'original order', 'Source', 'Name', 'Species ', 'Location', 'Injury', 'Year', 'Time', 'Type']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Renombrar columna 'Unnamed: 11' por 'Deceased'
    df.rename(columns={"Unnamed: 11": "Deceased"}, inplace=True)

    return df

def standardize_country(df):
    """
    Estandariza los nombres de los países:
    - Reemplaza minúsculas por mayúsculas.
    - Elimina espacios y caracteres especiales.
    - Usa un diccionario de mapeo para países no reconocidos.
    """
    # Reemplazar minúsculas por mayúsculas y limpiar espacios
    df['Country'] = df['Country'].str.lower().str.strip().replace('?', '')

    # Función para estandarizar países usando pycountry
    def estandarizar_pais(pais):
        try:
            return pycountry.countries.lookup(pais).name
        except LookupError:
            return 'Unknown'

    df['standarized_country'] = df['Country'].apply(estandarizar_pais)

    # Diccionario de mapeo para países no reconocidos
    mapeo_paises = {
        'burma': 'Myanmar', 'reunion': 'Reunion', 'north pacific ocean': 'North Pacific Ocean',
        'okinawa': 'Japan', 'turkey': 'Turkey', 'st helena, british overseas territory': 'United Kingdom',
        'falkland islands': 'United Kingdom', 'andaman / nicobar islands': 'India', 'new britain': 'Papua New Guinea',
        'england': 'United Kingdom', 'st martin': 'Saint Martin', 'scotland': 'United Kingdom',
        'maldive islands': 'Maldives', 'the balkans': 'Balkans', 'atlantic ocean': 'Atlantic Ocean',
        'ceylon': 'Sri Lanka', 'pacific ocean': 'Pacific Ocean', 'north atlantic ocean': 'North Atlantic Ocean',
        'italy / croatia': 'Italy', 'caribbean sea': 'Caribbean', 'admiralty islands': 'Papua New Guinea',
        'russia': 'Russia', 'micronesia': 'Micronesia', 'persian gulf': 'Persian Gulf', 'iran / iraq': 'Iran',
        'tobago': 'Trinidad and Tobago', 'south atlantic ocean': 'South Atlantic Ocean', 'new guinea': 'Papua New Guinea',
        'turks & caicos': 'Turks and Caicos Islands', 'san domingo': 'Dominican Republic',
        'british isles': 'United Kingdom', 'red sea': 'Red Sea', 'indian ocean': 'Indian Ocean',
        'trinidad & tobago': 'Trinidad and Tobago', 'columbia': 'Colombia', 'egypt / israel': 'Egypt',
        'cape verde': 'Cape Verde', 'west indies': 'Caribbean', 'crete': 'Greece', 'nevis': 'Saint Kitts and Nevis',
        'british west indies': 'Caribbean', 'mid atlantic ocean': 'Mid Atlantic Ocean', 'central pacific': 'Central Pacific Ocean',
        'british new guinea': 'Papua New Guinea', 'diego garcia': 'British Indian Ocean Territory',
        'st. martin': 'Saint Martin', 'curacao': 'Curaçao', 'grand cayman': 'Cayman Islands',
        'gulf of aden': 'Gulf of Aden', 'southwest pacific ocean': 'Southwest Pacific Ocean',
        'johnston island': 'United States', 'western samoa': 'Samoa', 'st. maartin': 'Saint Martin',
        'united arab emirates (uae)': 'United Arab Emirates', 'northern arabian sea': 'Northern Arabian Sea',
        'red sea / indian ocean': 'Red Sea', 'solomon islands / vanuatu': 'Solomon Islands',
        'south pacific ocean': 'South Pacific Ocean', 'azores': 'Portugal', 'bay of bengal': 'Bay of Bengal',
        'tasman sea': 'Tasman Sea', 'netherlands antilles': 'Curaçao', 'palestinian territories': 'Palestinian Territories',
        'st kitts / nevis': 'Saint Kitts and Nevis', 'mid-pacific ocean': 'Mid-Pacific Ocean',
        'south china sea': 'China', 'java': 'Indonesia', 'antigua': 'Antigua and Barbuda',
        'equatorial guinea / cameroon': 'Equatorial Guinea',
    }

    df['standarized_country'] = df['Country'].map(mapeo_paises).fillna(df['standarized_country'])

    return df

def standardize_state(df):
    """
    Estandariza los nombres de los estados:
    - Reemplaza minúsculas por mayúsculas.
    - Elimina espacios y valores vacíos.
    """
    df['State'] = df['State'].str.upper().str.strip().replace('', 'Unknown')
    return df

def standardize_activity(df):
    """
    Estandariza las actividades de los ataques:
    - Usa expresiones regulares para clasificar las actividades.
    """
    df['Activity'] = df['Activity'].fillna('Unspecific').str.lower().str.strip()
    
    def standardize_activity(activity):
        if re.search(r'swim|swimming', activity):
            return 'swimming'
        elif re.search(r'surf|surfing', activity):
            return 'surfing'
        elif re.search(r'dive|diving|snorkeling', activity):
            return 'diving'
        elif re.search(r'fish|fishing', activity):
            return 'fishing'
        elif re.search(r'kayak|canoe|paddle', activity):
            return 'kayaking'
        elif re.search(r'board|boat', activity):
            return 'boarding'
        elif re.search(r'wade|wading', activity):
            return 'wading'
        else:
            return 'Unspecific'
    
    df['Activity'] = df['Activity'].apply(standardize_activity)
    return df

def standardize_sex(df):
    """
    Estandariza la información de sexo:
    - Usa diccionario de reemplazos y convierte a mayúsculas.
    """
    df['Sex'] = df['Sex'].fillna('Unknown').str.strip()
    dic_remplazos = {'M x 2': 'M', 'lli': 'M', 'N': 'M', '.': 'Unknown'}
    df['Sex'] = df['Sex'].replace(dic_remplazos).str.upper()
    return df

def standardize_age(df):
    """
    Convierte la columna de edad a formato numérico.
    """
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    return df

def standardize_deceased(df):
    """
    Estandariza la información de fallecidos:
    - Usa diccionario de reemplazos y convierte a mayúsculas.
    """
    dic_remplazos_Deceased = {'M': 'UNKNOWN', 'F': 'UNKNOWN', 'Nq': 'UNKNOWN', 'Y x 2': 'Y'}
    df['Deceased'] = df['Deceased'].str.strip().replace(dic_remplazos_Deceased).str.upper().fillna('UNKNOWN')
    return df

def clean_date(df):
    """
    Limpia y estandariza la columna de fechas:
    - Reemplaza caracteres especiales.
    - Extrae mes y año.
    """
    df['Date'] = df['Date'].str.strip().replace(' ', '-').replace('.', '-')
    df['Date'] = df['Date'].str.replace('^(Reported-|Late-|Early-)', '', regex=True)
    
    def quitar_dia(x):
        parts = x.split('-')
        if len(parts) == 3:
            return '-'.join(parts[1:])  # Elimina el día
        return x
    
    df['Date'] = df['Date'].apply(lambda x: quitar_dia(x) if isinstance(x, str) else x)
    
    
    # Convertir a datetime
    df['Date'] = pd.to_datetime(df['Date'], format ='%b-%Y', errors='coerce')
    
    # Extraer mes y año
    df['Month'] = df['Date'].dt.strftime('%B')
    df['Year'] = df['Date'].dt.year
    df['Decade'] = (df['Year'] // 10) * 10

    # Determinar estación
    hemisferio_norte = [
        "United States", "Canada", "Bahamas", "Mexico", "Costa Rica", "Cuba", "Jamaica", "Belize", "Guatemala", 
        "El Salvador", "Honduras", "Nicaragua", "Panama", "Dominican Republic", "Puerto Rico", "Trinidad and Tobago", 
        "Saint Kitts and Nevis", "Saint Martin", "Barbados", "Dominica", "Martinique", "Grenada", "Haiti", "Turkey", 
        "Cyprus", "Israel", "Jordan", "Iraq", "Lebanon", "Syrian Arab Republic", "Palestinian Territories", 
        "Saudi Arabia", "United Arab Emirates", "Kuwait", "Oman", "Qatar", "Bahrain", "Georgia", "Russia", "Ukraine", 
        "Belarus", "Poland", "Slovakia", "Hungary", "Romania", "Bulgaria", "Greece", "Albania", "North Macedonia", 
        "Kosovo", "Serbia", "Montenegro", "Bosnia and Herzegovina", "Croatia", "Slovenia", "Italy", "Malta", "France", 
        "Andorra", "Monaco", "Luxembourg", "Belgium", "Netherlands", "Denmark", "Norway", "Sweden", "Iceland", 
        "Ireland", "United Kingdom", "Spain"
    ]

    hemisferio_sur = [
        "South Africa", "Australia", "New Zealand", "Fiji", "Samoa", "Papua New Guinea", "Solomon Islands", 
        "Vanuatu", "Tuvalu", "Kiribati", "Nauru", "American Samoa", "Tonga", "Comoros", "Seychelles", "Mauritius", 
        "Madagascar", "Mozambique", "Tanzania", "Kenya", "Uganda", "Rwanda", "Burundi", "Zambia", "Zimbabwe", 
        "Botswana", "Namibia", "Angola", "Democratic Republic of the Congo", "Congo", "Gabon", "Equatorial Guinea", 
        "Sao Tome and Principe", "Maldives", "Sri Lanka", "India", "Bangladesh", "Myanmar", "Timor-Leste", "Falkland Islands", 
        "Argentina", "Chile", "Peru", "Bolivia", "Paraguay", "Uruguay", "Brazil", "Suriname", "Guyana"
    ]

    def get_season(month, country):
        if country in hemisferio_sur:
            # Hemisferio sur
            if month in ['December', 'January', 'February']:
                return 'Summer'
            elif month in ['March', 'April', 'May']:
                return 'Autumn'
            elif month in ['June', 'July', 'August']:
                return 'Winter'
            elif month in ['September', 'October', 'November']:
                return 'Spring'
        elif country in hemisferio_norte:
            # Hemisferio norte
            if month in ['December', 'January', 'February']:
                return 'Winter'
            elif month in ['March', 'April', 'May']:
                return 'Spring'
            elif month in ['June', 'July', 'August']:
                return 'Summer'
            elif month in ['September', 'October', 'November']:
                return 'Autumn'
        return 'Unknown'

    df['Season'] = df.apply(lambda row: get_season(row['Month'], row['standarized_country']), axis=1)
    df['Season'].fillna('Unknown', inplace=True)

    return df

def main(url):
    df = load_data(url)
    df = clean_data(df)
    df = standardize_country(df)
    df = standardize_state(df)
    df = standardize_activity(df)
    df = standardize_sex(df)
    df = standardize_age(df)
    df = standardize_deceased(df)
    df = clean_date(df)
    
    # Verificar y eliminar duplicados
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Exportar a CSV
    df.to_csv('clean_data.csv', index=False)
    
    return df


