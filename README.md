# Airbnb-analysis-NYC-2025
# Radiografía del turismo cooperativo: Nueva York 2025
## Importamos las librerías necesarias
import pandas as pd
import numpy as np
import seaborn as sns
import folium
import geopandas as gpd
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
## Leemos el dataset y vemos como está compuesto
df = pd.read_csv('listings.csv')
df.head()
df.info()
df.describe()
### Hacemos recuento de los valores nulos y los imputamos o eliminamos, ya que el propósito de esta tarea es la visualizacion y no la creación de modelos
df.isnull().sum()
df.dropna(subset=['name', 'host_name'], inplace=True)

df['price'].fillna(df['price'].median(), inplace=True)

df['reviews_per_month'].fillna(0, inplace=True)
df['last_review'].fillna(0, inplace=True)

df['license'].fillna(0, inplace=True)

print(df.isnull().sum())
df.head()
## Visualización
### Creamos un treemap que permita observar el recuento de airbnb por cada barrio de Nueva York, además, se puede acceder al distrito de cada barrio y en el siguiente nivel observar el tipo de habitación

### Destacan los barrios de Manhattan y los de Brooklyn como los dos principales. El distrito con más viviendas vacacionales es Bedford-Stuyvesant con 2.663 viviendas
import plotly.express as px


df['count_airbnb'] = df.groupby(['neighbourhood_group', 'neighbourhood'])['id'].transform('count')


fig = px.treemap(df, 
                 path=[px.Constant("Nueva York"),'neighbourhood_group', 'neighbourhood', 'room_type'], 
                 title="Distribución de Airbnb por barrio y distrito", 
                 color='count_airbnb', 
                 hover_data=['count_airbnb', 'room_type'],
                 color_continuous_scale='Viridis',
                 labels={'count_airbnb': 'Recuento de Airbnb', 
                         'neighbourhood_group': 'Barrio Principal', 
                         'neighbourhood': 'Subbarrio'})
fig.update_layout(margin=dict(t=40, l=20, r=20, b=20)) 
fig.show()


### Observamos el precio medio por barrio, destacan Manhattan y Brooklyn
import matplotlib.pyplot as plt
import seaborn as sns

precio_medio_barrio = df.groupby('neighbourhood_group')['price'].mean().sort_values(ascending=False)

sns.set(style="whitegrid")

plt.figure(figsize=(14, 7))
barplot = sns.barplot(x=precio_medio_barrio.index,
                      y=precio_medio_barrio.values,
                      palette='pastel',
                      edgecolor='black')

plt.title("Precio medio por barrio en Nueva York", fontsize=18, weight='bold', pad=20)
plt.xlabel("Barrio", fontsize=14, labelpad=10)
plt.ylabel("Precio Medio ($)", fontsize=14, labelpad=10)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

for i, value in enumerate(precio_medio_barrio.values):
    plt.text(i, value + 5, f"${value:.0f}", ha='center', va='bottom', fontsize=11)

sns.despine()

plt.tight_layout()
plt.show()



### Podemos ir más allá del barrio y conocer cuales son los distritos más caros para los turistas en Nueva York
precio_medio_barrio = df.groupby('neighbourhood')['price'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(12,7))
sns.barplot(x=precio_medio_barrio.values, y=precio_medio_barrio.index, palette='magma')
plt.title('10 distritos más caros en Nueva York en 2025', fontsize=16)
plt.xlabel('Precio medio ($)')
plt.ylabel('Distrito')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

### Observamos el número de apartamentos según la propiedad, la vivienda entera y la habitación privada son las que más destacan
apartamentos_por_tipo = df['room_type'].value_counts()

apartamentos_por_tipo = apartamentos_por_tipo.sort_values(ascending=True)

pastel_palette = sns.color_palette("pastel", len(apartamentos_por_tipo))

plt.figure(figsize=(12, 8))
bars = apartamentos_por_tipo.plot(kind='barh', color=pastel_palette)
plt.title("Número de Apartamentos por Tipo de Propiedad", fontsize=16)
plt.xlabel("Número de Apartamentos", fontsize=12)
plt.ylabel("Tipo de Propiedad", fontsize=12)
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()
### La vivienda vacacional ha empezado un debate en las ciudades, muchas de ellas se encuentran a favor y otras en contra, sin embargo un elemento común es el de necesitar una licencia para ejercer la actividad. Otra medida que han tomado muchas ciudades es el de alquilar la propiedad por un período mínimo de días, en la mayoría de casos se sitúa sobre los 30 días. Ésta es la fecha que ha tomado la ciudad de Nueva York.

### A continuación, se observa un gráfico que permite al lector saber que propiedades tienen licencia y cuales no, además del tipo de propiedad.
import plotly.express as px

df = pd.read_csv('listings.csv')

df['tiene_licencia'] = df['license'].apply(lambda x: 'Sí' if pd.notnull(x) else 'No')


fig = px.sunburst(
    df,
    path=['tiene_licencia', 'neighbourhood_group', 'room_type'],
    values=None,  
    color='tiene_licencia',
    color_discrete_map={'Sí': 'green', 'No': 'red'},
    title="Distribución de licencias de Airbnb en Nueva York 2025"
)

fig.show()

### A continuación, mostramos los anfitriones con más viviendas en Airbnb, desconocemos si se trata de personas físicas o jurídicas
conteo = df['host_name'].value_counts().head(10).sort_values(ascending=False)
top_hosts = conteo.reset_index()
top_hosts.columns = ['host_name', 'num_props']

sns.set(style="whitegrid", font_scale=1.1)
colors = sns.color_palette("rocket", len(top_hosts))

plt.figure(figsize=(12, 7))
ax = sns.barplot(
    x='num_props', 
    y='host_name', 
    data=top_hosts, 
    palette=colors,
    linewidth=1.5,
    edgecolor="black"
)

for i, (value, name) in enumerate(zip(top_hosts['num_props'], top_hosts['host_name'])):
    ax.text(value + 0.5, i, f"{value}", va='center', fontsize=12, color='black')

plt.title("10 anfitriones con más propiedades", fontsize=18, weight='bold')
plt.xlabel("Número de propiedades", fontsize=14)
plt.ylabel("Nombre del anfitrión", fontsize=14)
plt.tight_layout()
plt.show()


## Visualización espacial
### Uno de los aspectos fundamentales cuando se trata de la vivienda vacacional, es el análisis espacial, ya que la mayoría de las viviendas tienden a situarse en los principales puntos turísticos de las ciudades, lo que provoca que esas viviendas se eliminen de la oferta para residentes. Como los turistas se encuentran dispuestos a pagar un precio mayor por noche en el corto plazo, los tenedores de la vivienda eligen ofertar en el corto plazo en vez del largo, a continuación, se encuentra un mapa interactivo con las viviendas y su ubicación. Los límites de cada distrito y barrio se han obtenido del Geojson que facilita la plataforma. Cada tipo de propiedad tiene un color específico. Se puede acceder a los datos de cada vivienda. El mapa está diseñado para que muestre un color rojo cuando más viviendas hay y verde cuando menos.

### Como se comentó anteriormente los barrios de Manhattan y Brooklyn son los que tienen más viviendas vacacionales.

### Para ello emplearemos folium que permite visualizar mapas a partir de OpenStreetMap.
import folium
from folium.plugins import MarkerCluster

map_nyc = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

marker_cluster = MarkerCluster().add_to(map_nyc)

def get_color(room_type):
    if room_type == 'Entire home/apt':
        return 'green'
    elif room_type == 'Private room':
        return 'blue'
    elif room_type == 'Shared room':
        return 'red'
    elif room_type == 'Hotel room':
        return 'purple'
    else:
        return 'gray'  
        
for idx, row in df.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"<b>{row['name']}</b><br>"
              f"Barrio: {row['neighbourhood']}<br>"
              f"Tipo: {row['room_type']}<br>"
              f"Precio: {row['price']}$",
        icon=folium.Icon(color=get_color(row['room_type']), icon='home')
    ).add_to(marker_cluster)

map_nyc


### Este mapa es similar al anterior pero con la densidad de propiedades en el mapa.
from folium.plugins import HeatMap

map_heat = folium.Map(location=[40.7128, -74.0060], zoom_start=11)


heat_data = df[['latitude', 'longitude']].dropna().values.tolist()


HeatMap(heat_data, radius=8, blur=15, min_opacity=0.5).add_to(map_heat)

map_heat

pip install dash

### Para concluir este mapa permite al lector filtrar por distrito y por rango de precio, similar a lo que podría verse en una aplicación o web de viajes.

### Para este ejercicio vamos a importar dash que permite hacer los siguientes tipos de aplicaciones y visualizaciones.
import dash
from dash import dcc, html, Output, Input
import plotly.express as px
import pandas as pd
import numpy as np

df = pd.read_csv("listings.csv")

q_low = df['price'].quantile(0.01)
q_high = df['price'].quantile(0.99)
df = df[(df['price'] >= q_low) & (df['price'] <= q_high)]

app = dash.Dash(__name__)

min_price = int(df['price'].min())
max_price = int(df['price'].max())
step = int((max_price - min_price) / 6)
marks = {p: f"{p:,}€" for p in range(min_price, max_price+1, step)}

app.layout = html.Div([
    html.H1("Precios de Airbnb en Nueva York", style={'textAlign': 'center', 'fontWeight': 'bold'}),
    
    html.Div([
        html.Label("Selecciona distrito:"),
        dcc.Dropdown(
            id='barrio_dropdown',
            options=[{'label': b, 'value': b} for b in sorted(df['neighbourhood'].unique())],
            placeholder="Todos los barrios",
            multi=True
        ),
        
        html.Label("Selecciona tipo de propiedad:"),
        dcc.Dropdown(
            id='habitacion_dropdown',
            options=[{'label': h, 'value': h} for h in sorted(df['room_type'].unique())],
            placeholder="Todos los tipos",
            multi=True
        ),

        html.Label("Selecciona rango de precio ($):"),
        dcc.RangeSlider(
            id='precio_slider',
            min=min_price,
            max=max_price,
            step=10,
            value=[min_price, max_price],
            marks=marks,
            tooltip={"placement": "bottom", "always_visible": True}
        ),
    ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px', 'verticalAlign': 'top'}),

    html.Div([
        dcc.Graph(id='mapa_airbnb')
    ], style={'width': '68%', 'display': 'inline-block', 'padding': '20px', 'verticalAlign': 'top'})
])

@app.callback(
    Output('mapa_airbnb', 'figure'),
    Input('barrio_dropdown', 'value'),
    Input('habitacion_dropdown', 'value'),
    Input('precio_slider', 'value')
)

def actualizar_mapa(barrio, habitacion, precio_range):
    df_filtrado = df.copy()

    if barrio:
        df_filtrado = df_filtrado[df_filtrado['neighbourhood'].isin(barrio)]
    if habitacion:
        df_filtrado = df_filtrado[df_filtrado['room_type'].isin(habitacion)]
    
    df_filtrado = df_filtrado[
        (df_filtrado['price'] >= precio_range[0]) & (df_filtrado['price'] <= precio_range[1])
    ]

    fig = px.scatter_mapbox(
        df_filtrado,
        lat="latitude",
        lon="longitude",
        color="price",
        size="price",
        color_continuous_scale="inferno_r",
        size_max=15,
        zoom=11,
        hover_data=["name", "neighbourhood", "price", "room_type"]
    )

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
    return fig

if __name__ == '__main__':
    app.run(debug=True)


## Conclusiones
### La vivienda vacacional ha supuesto un cambio en la tendencia de los viajes, pero no solo ha cambiado el negocio, sino que también ha transformado las ciudades, lo que antes eran barrios residenciales tranquilos, ahora se llenan de maletas y turistas que quieren hacer turismo como si fueran residentes.
### El impacto que genera la vivienda vacacional no lo sufren los turistas que incluso se ven beneficiados con precios más bajos que los que venían ofreciendo los hoteles. El problema lo sufren los residentes que ven como se sustrae de la oferta del alquiler de largo plazo miles de viviendas en grandes ciudades a las que antes si podían acceder.

### La reducción de la oferta cuando se mantiene la demanda produce un incremento de los precios, ya que la elasticidad-precio de la vivienda tiende a ser inelástica y los residentes tienden a estar dispuestos a pagar precios más altos.

### De ahí que muchas ciudades principalmente europeas como estadounidenses hayan optado por expedir licencias, limitar su número y pedir un mínimo de días, generalmente de 30 días, lo que hace que muchos inquilinos se replanteen el alquiler vacacional.

### Otro de los problemas que acarrea la vivienda vacacional es que muchas empresas o fondos de inversión adquieren múltiples viviendas como así se observa en el gráfico anterior de los 10 propietarios con más viviendas vacacionales.
