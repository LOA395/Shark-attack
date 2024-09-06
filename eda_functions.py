import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración global de Seaborn para unificar los colores y estilos de los gráficos
sns.set(style="whitegrid", palette="deep")

def cargar_datos(filepath):
    """Carga el archivo CSV en un DataFrame y convierte la columna 'Date' a tipo datetime."""
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def tasa_mortalidad(df):
    """Calcula y visualiza la tasa de mortalidad."""
    fallecidos = df[df['Deceased'] == 'Y'].shape[0]
    total_incidentes = df[df['Deceased'] != 'UNKNOWN'].shape[0]
    porcentaje_fallecidos = (fallecidos / total_incidentes) * 100
    porcentaje_no_fallecidos = ((total_incidentes - fallecidos) / total_incidentes) * 100
    labels = ['Fallecidos', 'No Fallecidos']
    sizes = [porcentaje_fallecidos, porcentaje_no_fallecidos]
    colors = ['#FF6F61', '#6B8E23']  # Colores unificados
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    plt.title('Porcentaje de Fallecidos y No Fallecidos')

def ataques_por_actividad(df):
    """Visualiza el número de ataques por actividad y resultado de fallecimiento."""
    df_sin_unknown = df[df['Deceased'].isin(['Y', 'N'])]
    activity_attacks = df_sin_unknown.groupby(['Activity', 'Deceased'], as_index=False).size().rename(columns={'size': 'Number of Attacks'})
    plt.figure(figsize=(12, 8))
    sns.barplot(data=activity_attacks, y='Activity', x='Number of Attacks', hue='Deceased', palette={'Y': '#FF6F61', 'N': '#6B8E23'})  # Colores unificados
    plt.title('Número de Ataques por Actividad')
    plt.xlabel('Número de Ataques')
    plt.ylabel('Actividad')
    plt.legend(title='Fallecimiento', labels=['No', 'Sí'])

def tendencia_incidentes_por_decada(df):
    """Visualiza la tendencia de incidentes por década."""
    plt.figure(figsize=(10, 6))
    df_decade = df.dropna(subset=['Decade'])
    df_decade['Decade'] = df_decade['Decade'].astype(int)
    sns.countplot(data=df_decade, x='Decade', palette='deep')  # Uso de la paleta global
    plt.title('Número de Incidentes por Década')
    plt.xticks(rotation=45)

def incidentes_por_mes(df):
    """Visualiza el número de incidentes por mes."""
    meses_ordenados = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    df['Month'] = pd.Categorical(df['Month'], categories=meses_ordenados, ordered=True)
    paleta_colores_azul = sns.color_palette("deep", n_colors=len(df['Month'].unique()))  # Paleta unificada
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Month', palette=paleta_colores_azul)
    plt.title('Número de Incidentes por Mes')
    plt.xticks(rotation=45)

def incidentes_por_estacion(df):
    """Visualiza el número de incidentes por estación del año."""
    df_filtrado = df[df['Season'] != 'Unknown']
    incidentes_por_estacion = df_filtrado['Season'].value_counts()
    plt.figure(figsize=(10, 6))
    bars = plt.bar(incidentes_por_estacion.index, incidentes_por_estacion, color=sns.color_palette("deep", len(incidentes_por_estacion)))  # Colores unificados
    for bar in bars:
        yval = bar.get_height()
        porcentaje = (yval / incidentes_por_estacion.sum()) * 100
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05 * yval, f'{yval} ({porcentaje:.1f}%)', ha='center', va='bottom')
    plt.xlabel('Estación')
    plt.ylabel('Número de Incidentes')
    plt.title('Número de Incidentes por Estación')
    plt.xticks(rotation=45)

def incidentes_por_genero(df):
    """Visualiza el número de incidentes por género."""
    incidentes_por_genero = df['Sex'].value_counts()
    porcentaje_por_genero = (incidentes_por_genero / incidentes_por_genero.sum()) * 100
    plt.figure(figsize=(8, 8))
    plt.pie(incidentes_por_genero, labels=[f'{gen}' for gen, porcentaje in zip(incidentes_por_genero.index, porcentaje_por_genero)], autopct='%1.1f%%', colors=sns.color_palette("deep", len(incidentes_por_genero)), startangle=140)  # Colores unificados
    plt.title('Número de Incidentes por Género')

def incidentes_por_pais(df, top_n=10):
    """Visualiza el número de incidentes en los principales países, ordenados por frecuencia."""
    # Obtener los países con más incidentes
    top_countries = df['standarized_country'].value_counts().head(top_n)
    ordered_countries = top_countries.index
    
    # Filtrar el DataFrame para solo incluir los países principales
    filtered_df = df[df['standarized_country'].isin(ordered_countries)]
    
    # Configurar el tamaño de la figura
    plt.figure(figsize=(12, 8))
    
    # Crear el gráfico de barras ordenado
    sns.countplot(data=filtered_df, x='standarized_country', 
                  order=ordered_countries, 
                  palette=sns.color_palette("deep", top_n))  # Colores unificados
    
    # Título y rotación de etiquetas
    plt.title(f'Frecuencia de Incidentes en los {top_n} Principales Países')
    plt.xticks(rotation=45)

def incidentes_por_rango_de_edad(df):
    """Visualiza el porcentaje de incidentes por rango de edad."""
    bins = [0, 18, 30, 40, 50, 60, 100]
    labels = ['0-18', '19-30', '31-40', '41-50', '51-60', '60+']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    incidentes_por_edad = df['AgeGroup'].value_counts().sort_index()
    porcentaje_por_edad = (incidentes_por_edad / incidentes_por_edad.sum()) * 100

    plt.figure(figsize=(12, 8))
    bars = sns.barplot(x=incidentes_por_edad.index, y=porcentaje_por_edad, palette='deep')  # Paleta unificada

    # Añadir los porcentajes sobre las barras
    for bar in bars.patches:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2, 
            yval + 0.5, 
            f'{yval:.1f}%',  # Mostrar solo el porcentaje
            ha='center', 
            va='bottom'
        )

    plt.xlabel('Rango de Edad')
    plt.ylabel('Porcentaje de Incidentes')
    plt.title('Porcentaje de Incidentes por Rango de Edad')
    plt.xticks(rotation=45)
    
def distribucion_edad_por_actividad(df):
    """Visualiza la distribución de edades para diferentes actividades."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Activity', y='Age', palette=sns.color_palette("deep", 8))  # Paleta unificada
    plt.title('Distribución de Edad por Actividad')
    plt.xlabel('Actividad')
    plt.ylabel('Edad')
    plt.xticks(rotation=45)
    plt.show()

def densidad_edad_por_actividad(df):
    """Visualiza la densidad de edad para diferentes actividades."""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='Age', hue='Activity', fill=True, palette=sns.color_palette("deep", 8))  # Paleta unificada
    plt.title('Densidad de Edad por Actividad')
    plt.xlabel('Edad')
    plt.ylabel('Densidad')
    plt.xticks(rotation=45)
    plt.show()

def actividad_por_genero(df):
    """Visualiza la distribución de actividades según el género con un gráfico de barras apiladas."""
    # Filtrar datos para excluir valores UNKNOWN en la columna 'Sex'
    df_filtrado = df[df['Sex'] != 'UNKNOWN']
    
    plt.figure(figsize=(12, 8))
    
    # Crear una tabla de contingencia
    actividad_genero = df_filtrado.groupby(['Activity', 'Sex']).size().unstack().fillna(0)
    
    # Crear el gráfico de barras apiladas
    actividad_genero.plot(kind='bar', stacked=True, colormap='Set2', figsize=(12, 8))  # Paleta unificada
    plt.title('Distribución de Actividades por Género')
    plt.xlabel('Actividad')
    plt.ylabel('Número de Incidentes')
    plt.xticks(rotation=45)
    plt.legend(title='Género', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
