import pandas as pd
import geopandas as gpd
from dash import dcc, html, Input, Output, callback,State
import numpy as np
import plotly.express as px
import dash_bootstrap_components as dbc
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
    adaptation_fig.update_layout(bargap=0.6)
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
    seasonal_fig.update_layout(bargap=0.6)
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
    data = data.groupby(['Decade', 'Rating'], observed=False).size().reset_index(name='Count')
    # create a pivot table
    data = data.pivot(index='Decade', columns='Rating', values='Count')
    data = data.fillna(0)
    # create a line plot and each point should have different icon
    symbol_sequence = ['circle', 'square', 'diamond', 'cross']
    decade_fig = px.line(data, x=data.index, y=data.columns, title='Rating Evolution Over Decade', markers=True,
                         symbol="Rating",labels={'Decade': 'Decade', 'value': 'Number of Releases'})
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
                    color_continuous_scale="Viridis",
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
    dbc.Container(
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H1("Anime Status Dashboard", className='card-title', style={'marginBottom': '20px'}),
                        dbc.RadioItems(
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
                            inline=True,  # Make the radio buttons appear horizontally
                            labelStyle={'display': 'inline-block', 'margin': '0px 10px 10px 0px',
                                        'verticalAlign': 'middle'}
                        ),
                        dcc.Graph(
                            id='bar_graph',
                            config={'displayModeBar': False},
                            style={'border': '1px solid #ccc', 'borderRadius': '10px', 'margin': '10px 0px',
                                   'boxShadow': '0 4px 8px rgba(0,0,0,0.1)', 'background-color': '#f8f9fa',
                                   'overflow': 'hidden'}  # Add overflow:hidden to ensure rounded corners are visible
                        )
                    ],
                    className='card-text',
                )
            ],
            className='shadow p-3 mb-5 bg-white rounded',
            style={'margin': '20px'}  # Adjust outer margin as needed
        )
    ),
    html.Div(style={'margin': '5%'}, children=[
        dbc.Card(style={ 'padding': '10px', 'width': '40%'}, children=[
            dbc.CardBody([
                html.H5("Genre Selection", className="card-title"),
                dcc.Dropdown(
                    id='genre-dropdown',
                    options=list_of_all_genres,
                    value=['Action'],
                    multi=True,
                    className="form-control",
                ),
                dcc.Graph(id='genre-graph', className="mt-3"),
            ]),
        ]),
        dbc.Card(style={'width':'50%', 'padding': '10px'}, children=[
            dbc.CardBody([
                html.H5("Rating Evolution Over Decade", className="card-title"),
                dcc.Graph(
                    id='rating-evolution-graph',
                    figure=rating_evoution_over_decade(),
                    config={'displayModeBar': False},  # Disable Plotly's mode bar
                ),
            ]),
        ]),
    ], className="d-flex justify-content-around flex-row"),
    html.Div(style={'margin': '20px 10%'}, children=[
        dbc.Card(style={'margin-bottom': '20px'}, children=[
            dbc.CardHeader(html.H5("Adaptation Source Trends")),
            dbc.CardBody([
                dcc.Graph(
                    id='adaptation-graph',
                    figure=adaptation_source_trends(),
                    config={'displayModeBar': False},  # Disable Plotly's mode bar
                ),
            ]),
        ]),
        dbc.Card(style={'margin-bottom': '20px'}, children=[
            dbc.CardHeader(html.H5("Seasonal Anime Ratings")),
            dbc.CardBody([
                dcc.Graph(
                    id='seasonal-ratings-graph',
                    figure=seasonal_anime_ratings(),
                    config={'displayModeBar': False},  # Disable Plotly's mode bar
                ),
            ]),
        ]),
    ]),
    html.Div(style={'margin': '0 10%'}, children=[
        dbc.Card(style={'margin-bottom': '20px'}, children=[
            dbc.CardHeader(html.H5("Genre Selection")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Genre:"),
                        dcc.Dropdown(
                            id='genre-dropdown_1',
                            options=list_of_all_genres,
                            value='Action',
                            clearable=False,
                            className="form-control",
                        ),
                    ]),
                    dbc.Col([
                        html.Label("Controversy"),
                        dcc.Dropdown(
                            id='genre-dropdown_2',
                            options=[
                                {"label": "Most Controversial Anime", "value": "Most Controversial Anime"},
                                {"label": "Least Controversial Anime", "value": "Least Controversial Anime"},
                            ],
                            value='Most Controversial Anime',
                            clearable=False,
                            className="form-control",
                        ),
                    ]),
                ]),
                dbc.Button("Submit", id='submit-val', n_clicks=0, color="primary", className="mt-3"),
            ]),
        ]),
        dbc.Card(children=[
            dbc.CardHeader(html.H5("Controversial Anime Graph")),
            dbc.CardBody([
                dcc.Graph(
                    id='contro-graph',
                    style={'width': '100%', 'height': '400px'},  # Fixed width and height
                ),
            ]),
        ]),
    ])
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
    # Load data using os module
    data = pd.read_csv(os.path.abspath('./data/anime_cleaned.csv'))

    # Calculate number of releases for selected genres according to year
    season_year = data['Premiered'].map(extract_season_year)  # Assuming extract_season_year is defined
    premiered_year = season_year.apply(lambda x: x[1])
    data['Premiered'] = premiered_year

    # Convert selected_value to list if it's not already a list
    selected_value = [selected_value] if not isinstance(selected_value, list) else selected_value

    # Filter data for anime with selected genres
    data = data[data['Genres'].apply(lambda x: all(genre in x for genre in selected_value))]

    # Get number of releases per selected genre according to year
    data = data.groupby(['Premiered']).size().reset_index(name='Number of Releases')

    # Create an interactive line chart using plotly express with customizations
    figure = px.line(data, x='Premiered', y='Number of Releases', title='Number of Releases per Genre',
                     color_discrete_sequence=px.colors.qualitative.Pastel,
                     labels={'Premiered': 'Year', 'Number of Releases': 'Number of Releases'})

    # Adjust legend position and add grid lines
    figure.update_layout(legend=dict(x=0.02, y=0.98), xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='lightgray'),
                         yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='lightgray'))

    return figure

@callback(
    Output('contro-graph', 'figure'),
    [State('genre-dropdown_1', 'value'),
    State('genre-dropdown_2', 'value')],
    Input('submit-val', 'n_clicks')
)

def update_contro(n_clicks, genre_1, genre_2):
    Anime_df = pd.read_csv(os.path.abspath('./data/anime_cleaned.csv'))
    Anime_df['Score-10'] = Anime_df['Score-10'].str.replace('Unknown', '0')
    Anime_df['Score-9'] = Anime_df['Score-9'].str.replace('Unknown', '0')
    Anime_df['Score-8'] = Anime_df['Score-8'].str.replace('Unknown', '0')
    Anime_df['Score-7'] = Anime_df['Score-7'].str.replace('Unknown', '0')
    Anime_df['Score-6'] = Anime_df['Score-6'].str.replace('Unknown', '0')
    Anime_df['Score-5'] = Anime_df['Score-5'].str.replace('Unknown', '0')
    Anime_df['Score-4'] = Anime_df['Score-4'].str.replace('Unknown', '0')
    Anime_df['Score-3'] = Anime_df['Score-3'].str.replace('Unknown', '0')
    Anime_df['Score-2'] = Anime_df['Score-2'].str.replace('Unknown', '0')
    Anime_df['Score-1'] = Anime_df['Score-1'].str.replace('Unknown', '0')
    Anime_df['Score-10'] = Anime_df['Score-10'].astype(float)
    Anime_df['Score-9'] = Anime_df['Score-9'].astype(float)
    Anime_df['Score-8'] = Anime_df['Score-8'].astype(float)
    Anime_df['Score-7'] = Anime_df['Score-7'].astype(float)
    Anime_df['Score-6'] = Anime_df['Score-6'].astype(float)
    Anime_df['Score-5'] = Anime_df['Score-5'].astype(float)
    Anime_df['Score-4'] = Anime_df['Score-4'].astype(float)
    Anime_df['Score-3'] = Anime_df['Score-3'].astype(float)
    Anime_df['Score-2'] = Anime_df['Score-2'].astype(float)
    Anime_df['Score-1'] = Anime_df['Score-1'].astype(float)
    Anime_df['Score'] = Anime_df['Score'].str.replace('Unknown', '0')
    Anime_df['Score'] = Anime_df['Score'].astype(float)
    Anime_df['Total_Votes'] = (
            Anime_df['Score-10'] + Anime_df['Score-9'] + Anime_df['Score-8'] + Anime_df['Score-7'] + Anime_df[
        'Score-6'] +
            Anime_df['Score-5'] + Anime_df['Score-4'] + Anime_df['Score-3'] + Anime_df['Score-2'] + Anime_df[
                'Score-1'])
    Anime_df['Total_Votes'] = Anime_df['Total_Votes'].astype(int)
    multiplication_factors = {'Score-10': 10,
                              'Score-9': 9,
                              'Score-8': 8,
                              'Score-7': 7,
                              'Score-6': 6,
                              'Score-5': 5,
                              'Score-4': 4,
                              'Score-3': 3,
                              'Score-2': 2,
                              'Score-1': 1
                              }

    for col, factor in multiplication_factors.items():
        Anime_df[col + '_multiplied'] = Anime_df[col] * factor - Anime_df['Score']
    Anime_df['SD'] = Anime_df[['Score-10_multiplied',
                               'Score-9_multiplied',
                               'Score-8_multiplied',
                               'Score-7_multiplied',
                               'Score-6_multiplied',
                               'Score-5_multiplied',
                               'Score-4_multiplied',
                               'Score-3_multiplied',
                               'Score-2_multiplied',
                               'Score-1_multiplied'
                               ]].std(axis=1)

    df_split = Anime_df['Genres'].str.split(', ', expand=True)

    # Creating a new DataFrame by stacking the split values
    df_split = df_split.stack().reset_index(level=1, drop=True).rename('Genre')

    # Merging the original DataFrame with the split DataFrame based on the index
    Anime_df = Anime_df.drop('Genres', axis=1).merge(df_split, left_index=True, right_index=True)

    Anime_df['Genre'] = Anime_df['Genre'].str.replace(' ', '')

    Genres = Anime_df['Genre'].unique()

    Anime_df = Anime_df.sort_values(by=['Genre'])

    grouped = Anime_df.groupby('Genre')

    # Iterate over the groups and create separate dataframes
    for category, group in grouped:
        globals()['df_' + category] = group
        globals()['df_' + category] = globals()['df_' + category][globals()['df_' + category]['Total_Votes'] > 100000]
        globals()['df_' + category].loc[:, 'SD'] = globals()['df_' + category]['SD'] / globals()['df_' + category][
            'Total_Votes']

    Genre_list = 'df_'+list_of_all_genres[genre_2]

    if genre_1 == 'Most Controversial Anime':
        temp_df1 = globals()[Genre_list].sort_values(by=['SD'], ascending=False)
        top10_Name = temp_df1.head(10)['Name'].tolist()
        top10_SD = temp_df1.head(10)['SD'].tolist()

        temp_df2 = globals()[Genre_list][globals()[Genre_list]['Score'] > 8]
        temp_df2 = temp_df2.sort_values(by=['SD'], ascending=False)
        top10_Name_8 = temp_df2.head(10)['Name'].tolist()
        top10_SD_8 = temp_df2.head(10)['SD'].tolist()
    else:
        temp_df1 = globals()[Genre_list].sort_values(by=['SD'], ascending=False)
        top10_Name = temp_df1.tail(10)['Name'].tolist()
        top10_Name.reverse()
        top10_SD = temp_df1.tail(10)['SD'].tolist()
        top10_SD.reverse()

        temp_df2 = globals()[Genre_list][globals()[Genre_list]['Score'] > 7]
        temp_df2 = temp_df2.sort_values(by=['SD'], ascending=False)
        top10_Name_8 = temp_df2.tail(10)['Name'].tolist()
        top10_Name_8.reverse()
        top10_SD_8 = temp_df2.tail(10)['SD'].tolist()
        top10_SD_8.reverse()
    genre_val = list_of_all_genres[genre_2]
    #do not show label on x axis
    new_label = []
    #add first letter of each word in name
    for name in top10_Name:
        temp = name.split()
        # if name has no spaces then add full one
        if len(temp) == 1:
            new_label.append(name)
        else:
            new_label.append(''.join([x[0] for x in temp]))
    # create list with short names and long names
    new = {'Short Name': new_label, 'Long Name': top10_Name,'val':top10_SD}
    fig = px.bar(new, x='Short Name', y='val', title=f'{genre_val} Anime',hover_data='Long Name',labels={'x':'Anime','y':'SD'})
    #show abrevated names on xaxis
    fig.update_layout(bargap=0.6)
    return fig

