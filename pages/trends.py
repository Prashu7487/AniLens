import pandas as pd
import geopandas as gpd
from dash import dcc, html, Input, Output, callback
import numpy as np
import plotly.express as px
import os
from dash import html, dcc, callback


def adaptation_source_trends():
    # Load Data
    data = pd.read_csv(os.path.abspath('./data/anime_cleaned.csv'))
    season_year = data['Premiered'].map(extract_season_year)
    premiered_season = season_year.apply(lambda x: x[0])
    premiered_year = season_year.apply(lambda x: x[1])
    data['Premiered'] = premiered_year
    data['Premiered'] = data['Premiered'].astype('int')
    data = data[data['Premiered'] <= 2020]
    data = data[data['Premiered'] >= 2015]
    data = data[data['Source'] != 'Unknown']
    data = data[data['Source'].apply(lambda x: x in ['Manga', 'Light Novel', 'Other', 'Game', 'Original'])]
    # Create a bar chart which shows the source of anime adaptations
    data = data.groupby(['Source', 'Premiered']).size().reset_index(name='Count')
    adaptation_fig = px.bar(data, x='Premiered', y='Count', color='Source', title='Adaptation Source Trends')
    return adaptation_fig


def seasonal_anime_ratings():
    # Load Data
    data = pd.read_csv(os.path.abspath('./data/anime_cleaned.csv'))
    # Create a bar chart which shows avg score in each season for last five years
    season_year = data['Premiered'].map(extract_season_year)
    premiered_season = season_year.apply(lambda x: x[0])
    premiered_year = season_year.apply(lambda x: x[1])
    data['Premiered'] = premiered_year
    data['Season'] = premiered_season
    data['Score'] = data[data['Score'] != 'Unknown']['Score'].astype('float')
    # convert premiered column to int
    data['Premiered'] = data['Premiered'].astype('int')
    # get last 5 years data
    data = data[data['Premiered'] <= 2020]
    data = data[data['Premiered'] >= 2015]
    data = data.groupby(['Premiered', 'Season']).agg({'Score': 'mean'}).reset_index()
    seasonal_fig = px.bar(data, x='Premiered', y='Score', color='Season', title='Seasonal Anime Ratings',
                          color_continuous_scale='plasma')
    return seasonal_fig


def top_10_anime_based_on_most_genre():
    # Load Data
    data = pd.read_csv(os.path.abspath('./data/anime_cleaned.csv'))
    # Create a bar chart which shows have most typeof genre
    data['Genres'] = data['Genres'].apply(lambda x: x.split(', '))
    data = data.explode('Genres')
    # plot should be name vs how many genres it has
    data = data.groupby('Name').size().reset_index(name='Count')
    data = data.sort_values(by='Count', ascending=False)
    data = data.head(10)
    genre_fig = px.bar(data, x='Name', y='Count', title='Top 10 Anime based on Most Genre')
    return genre_fig


def ep_duration_vs_popularity_scatter_plot():
    # Load Data
    data = pd.read_csv(os.path.abspath('./data/anime_cleaned.csv'))
    # Create a scatter plot
    scatter_fig = px.scatter(data, x='Episodes', y='Popularity', title='Episode Duration vs Popularity',
                             labels={'Episodes': 'Number of Episodes', 'Popularity': 'Popularity'})
    return scatter_fig


def rating_evoution_over_decade():
    # Load Data
    data = pd.read_csv(os.path.abspath('./data/anime_cleaned.csv'))
    season_year = data['Premiered'].map(extract_season_year)
    premiered_year = season_year.apply(lambda x: x[1])
    data['Premiered'] = premiered_year
    # convert premiered column to int
    data['Premiered'] = data['Premiered'].astype('int')
    # create bins for decades
    bins = [1960, 1970, 1980, 1990, 2000, 2010, 2020]
    labels = ['1960-1970', '1970-1980', '1980-1990', '1990-2000', '2000-2010', '2010-2020']
    data['Decade'] = pd.cut(data['Premiered'], bins=bins, labels=labels, right=False)
    # group data based on rated
    # remove unknown values
    data = data[data['Rating'] != 'Unknown']
    data = data.groupby(['Decade', 'Rating']).size().reset_index(name='Count')
    # create a pivot table
    data = data.pivot(index='Decade', columns='Rating', values='Count')
    data = data.fillna(0)
    # create a line plot and each point should have different icon
    symbol_sequence = ['circle', 'square', 'diamond', 'cross']
    decade_fig = px.line(data, x=data.index, y=data.columns, title='Rating Evolution Over Decade', markers=True,
                         symbol="Rating")
    return decade_fig


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
data_popularity = pd.read_csv(os.path.abspath('./data/anime_popularity.csv'))
anime_data = pd.read_csv(os.path.abspath('./data/anime_cleaned.csv'))
anime = pd.read_csv(os.path.abspath('./data/anime.csv'))
list_of_all_genres = anime['Genres'].apply(lambda x: x.split(', ')).explode().unique().tolist()

# ... (Your code for loading data, etc.)

fig = px.choropleth(data_popularity, locations="iso_alpha",
                    color="Anime",
                    hover_data=["name", "Anime", "Genre"],
                    color_continuous_scale="plasma",
                    projection="orthographic")

fig.update_layout(margin=dict(l=60, r=60, t=50, b=50),
                  geo=dict(bgcolor='rgba(2,1,50,0)',
                           showframe=True, oceancolor='deepskyblue'),
                  )  # Customize this color
# ... (Rest of your code)


# fig1 = px.line(x=number_of_anime_released_per_year().index, y=number_of_anime_released_per_year().values,
#                labels={'x': 'Year', 'y': 'Number of Anime Released'})
# Assuming Dash setup exists ...
layout = html.Div([
    html.Div(
        [dcc.Graph(figure=fig),
         ]),
    html.Div([
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
    ]),
    html.Div([
        html.Div([dcc.Dropdown(
                id='genre-dropdown',
                options=list_of_all_genres,
                value=['Action'],
                multi=True
            ),
            dcc.Graph(id='genre-graph'), ],style={'width': '50%','border':'1px solid #ccc','padding':'10px'}),
        dcc.Graph(figure=rating_evoution_over_decade()),
    ],style={'display': 'flex', 'justify-content': 'space-around','flex-direction':'row','border':'1px solid #ccc','padding':'10px'}),
    html.Div([

    ]),
    html.Div([
        dcc.Graph(figure=adaptation_source_trends()),
    ]),
    html.Div([
        dcc.Graph(figure=seasonal_anime_ratings()),
    ]),
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

    filtered_data_status = data[['Name', 'Completed', 'Watching', 'On-Hold', 'Plan to Watch', 'Dropped', 'Favorites']]

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
    # sort based on selected value and get top 10 anime
    filtered_data_status = filtered_data_status.sort_values(by=map[selected_value], ascending=False)
    filtered_data_status = filtered_data_status.head(10)
    figure = px.bar(x=filtered_data_status['Name'], y=filtered_data_status[map[selected_value]],
                    title=f'Top 10 Anime based on {map[selected_value]}')
    return figure


@callback(
    Output('genre-graph', 'figure'),
    Input('genre-dropdown', 'value')
)
def update_chart(selected_value):
    # load data using os module
    print(selected_value)
    data = pd.read_csv(os.path.abspath('./data/anime_cleaned.csv'))
    # calculate number of releases selcted genres according to year
    season_year = data['Premiered'].map(extract_season_year)
    premiered_year = season_year.apply(lambda x: x[1])
    data['Premiered'] = premiered_year
    # convert selected value to list
    # if selected value is in particular anime's genres add it to list
    # find all anime sthat have selected value in their genres
    # below is wrong
    data = data[data['Genres'].apply(lambda x: all(genre in x for genre in selected_value))]
    # get number of releases per selected value according to year
    data = data.groupby(['Premiered']).size().reset_index(name='Number of Releases')
    figure = px.line(data, x='Premiered', y='Number of Releases', title='Number of Releases per Genre')
    return figure
