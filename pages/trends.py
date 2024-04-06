import pandas as pd
import geopandas as gpd
from dash import dcc, html
import numpy as np
import plotly.express as px


def extract_season_year(premiered):
    if premiered == 'UNKNOWN':
        return None, None
    else:
        season, year = premiered.split()
        return season, int(year)


def number_of_anime_released_per_year():
    df_anime = pd.read_csv(r"D:\data\anime_cleaned.csv")
    #count row of df_anime'
    # print(df_anime.shape)
    # scores = df_anime['Score'][df_anime['Score'] != 'Unknown']
    # scores = scores.astype('float')
    # score_mean = round(scores.mean(), 2)
    # #delete rows where ranked column is unknown
    # df_anime['Premiered'] = df_anime['Premiered'][df_anime['Premiered'] != 'Unknown']
    # df_anime.drop(df_anime[df_anime['Premiered'] == 'Unknown'].index, inplace=True)
    # # df_anime['Ranked'] = df_anime['Ranked'][df_anime['Ranked'] != 'Unknown']
    # df_anime['Score'] = df_anime['Score'].replace('Unknown', score_mean)
    # df_anime['Score'] = df_anime['Score'].astype('float64')
    # df_anime = df_anime[df_anime['Type'] != 'Unknown']
    # #save into csv
    # df_anime.to_csv(r"D:\data\anime_cleaned.csv", index=False)
    print(df_anime.shape)

    season_year = df_anime['Premiered'].map(extract_season_year)
    premiered_season = season_year.apply(lambda x: x[0])
    premiered_Year = season_year.apply(lambda x: x[1])
    filtered_premiered_season = premiered_season.dropna()
    fitered_premiered_year = premiered_Year.dropna()
    year_count = fitered_premiered_year.value_counts().sort_index()
    #delete rows where premiered column is unknown
    print(year_count)
    return year_count


# Load Data
data = pd.read_csv(r"D:\data\anime_popularity.csv")

# ... (Your code for loading data, etc.)

fig = px.choropleth(data, locations="iso_alpha",
                    color="Anime",
                    hover_data=["name","Anime", "Genre"],
                    color_continuous_scale="plasma",
                    projection="orthographic")

fig.update_layout(margin=dict(l=60, r=60, t=50, b=50),
                  geo=dict(bgcolor= 'rgba(2,1,50,0)',
                           showframe=True,oceancolor='deepskyblue'),
                  )  # Customize this color
# ... (Rest of your code)


fig1 = px.line(x=number_of_anime_released_per_year().index, y=number_of_anime_released_per_year().values, labels={'x': 'Year', 'y': 'Number of Anime Released'})
# Assuming Dash setup exists ...
layout = html.Div([dcc.Graph(figure=fig),
                   #adding a graph to show the number of anime released per year
                   html.H1('Number of Anime Released per Year'),
                   html.P('This is the content of the trends page.'),
                     dcc.Graph(figure=fig1)
                   ])
