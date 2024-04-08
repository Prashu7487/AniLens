import pandas as pd
import geopandas as gpd
from dash import dcc, html, Input, Output, callback
import numpy as np
import plotly.express as px
import os


def extract_season_year(premiered):
    if premiered == 'UNKNOWN':
        return None, None
    else:
        season, year = premiered.split()
        return season, int(year)


def number_of_anime_released_per_year():
    df_anime = pd.read_csv(os.path.abspath('./data/anime_cleaned.csv'))
    # count row of df_anime'
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

    season_year = df_anime['Premiered'].map(extract_season_year)
    premiered_season = season_year.apply(lambda x: x[0])
    premiered_Year = season_year.apply(lambda x: x[1])
    filtered_premiered_season = premiered_season.dropna()
    fitered_premiered_year = premiered_Year.dropna()
    year_count = fitered_premiered_year.value_counts().sort_index()
    # delete rows where premiered column is unknown
    return year_count


# Load Data
data = pd.read_csv(os.path.abspath('./data/anime_popularity.csv'))

# ... (Your code for loading data, etc.)

fig = px.choropleth(data, locations="iso_alpha",
                    color="Anime",
                    hover_data=["name", "Anime", "Genre"],
                    color_continuous_scale="plasma",
                    projection="orthographic")

fig.update_layout(margin=dict(l=60, r=60, t=50, b=50),
                  geo=dict(bgcolor='rgba(2,1,50,0)',
                           showframe=True, oceancolor='deepskyblue'),
                  )  # Customize this color
# ... (Rest of your code)


fig1 = px.line(x=number_of_anime_released_per_year().index, y=number_of_anime_released_per_year().values,
               labels={'x': 'Year', 'y': 'Number of Anime Released'})
# Assuming Dash setup exists ...
layout = html.Div([dcc.Graph(figure=fig),
                   # adding a graph to show the number of anime released per year
                   html.H1('Number of Anime Released per Year'),
                   html.P('This is the content of the trends page.'),
                   dcc.Graph(figure=fig1),
                   dcc.RadioItems(
                       id='radio-items',
                       options=[
                           {'label': 'Completed', 'value': 'completed'},
                           {'label': 'Watching', 'value': 'watching'},
                           {'label': 'On hold', 'value': 'on_hold'},
                           {'label': 'Plan to watch', 'value': 'plan_to_watch'},
                           {'label': 'Dropped', 'value': 'dropped'},
                           {'label': 'Favorite', 'value': 'favorite'},
                       ],
                       value="completed",
                       labelStyle={'display': 'inline-block'}
                   ),
                   dcc.Graph(id='bar_graph')
                   ])


@callback(
    Output('bar_graph', 'figure'),
    Input('radio-items', 'value')
)
def update_pie_chart(selected_value):
    # load data using os module
    data = pd.read_csv(os.path.abspath('./data/anime_cleaned.csv'))
    map = {'completed': 'Completed', 'watching': 'Watching', 'on_hold': 'On-Hold', 'plan_to_watch': 'Plan to Watch',
           'dropped': 'Dropped', 'favorite': 'Favorites'}
    # filter data based on selected value and find percentage of each map value

    filtered_data_status = data[['Name', 'Completed', 'Watching', 'On-Hold', 'Plan to Watch', 'Dropped','Favorites']]

    # index,value= filtered_data_status['Name'], (filtered_data_status[map[selected_value]] / filtered_data_status[
    #     ['Completed', 'Watching', 'On-Hold', 'Plan to Watch', 'Dropped']].sum(axis=1)) * 100
    # map_index = {index[i]: value[i] for i in range(len(index))}
    # map_fav = {index[i]: filtered_data_status['Favorites'][i] for i in range(len(index))}
    # if selected_value == 'favorite':
    #     # sort based on value inside map_fav
    #     map_fav = dict(sorted(map_fav.items(), key=lambda item: item[1], reverse=True))
    #     # get top 10 anime
    #     map_fav = dict(list(map_fav.items())[:10])
    #     figure = px.bar(x=list(map_fav.keys()), y=list(map_fav.values()), title=f'Top 10 Anime based on {map[selected_value]}')
    # else:
    #     #sort based on value inside map_index
    #     map_index = dict(sorted(map_index.items(), key=lambda item: item[1], reverse=True))
    #     #get top 10 anime
    #     map_index = dict(list(map_index.items())[:10])
    #     figure = px.bar(x=list(map_index.keys()), y=list(map_index.values()), title=f'Top 10 Anime based on {map[selected_value]}')
    # return figure
    #sort based on selected value and get top 10 anime
    filtered_data_status = filtered_data_status.sort_values(by=map[selected_value], ascending=False)
    filtered_data_status = filtered_data_status.head(10)
    figure = px.bar(x=filtered_data_status['Name'], y=filtered_data_status[map[selected_value]], title=f'Top 10 Anime based on {map[selected_value]}')
    return figure