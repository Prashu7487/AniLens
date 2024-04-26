import os
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from wordcloud import WordCloud
from anilens_functions.overview_function import (get_studio_contributions,
                                                 most_watched_anime_by_studio,
                                                 avg_watched_and_popularity_per_year_per_studio,
                                                 visualize_anime_count_by_rating_pie_interactive)

######################################################################################### Global_vars
data_path = os.path.abspath('./data/anime.csv')
anime_df = pd.read_csv(data_path)
anime_df = anime_df[['Name', 'Score', 'Episodes', 'Genres', 'Completed', 'Premiered', 'Popularity', 'Studios']]

############################################################################################################################## Component 1
most_watched = anime_df[['Name', 'Score', 'Episodes', 'Genres', 'Completed']]
most_watched = most_watched[(anime_df['Score'] != 'Unknown') & (anime_df['Genres'] != 'Unknown')]
list_of_all_genres = most_watched['Genres'].apply(lambda x: x.split(', ')).explode().unique().tolist()

component_1 = dbc.Card(
    [
        dbc.CardHeader(html.H2("Most Watched Anime",
                               style={'text-align': 'center', 'marginTop': '20px', 'marginBottom': '20px'})),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H4("Select Score:", style={'fontSize': '14px'}),
                    dcc.Slider(
                        min=0, max=10, value=5, marks=None, id='score-slider',
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.H4("Select Genre:", style={'fontSize': '14px', 'marginTop': '50px'}),
                    dcc.Dropdown(
                        options=[{'label': genre, 'value': genre} for genre in list_of_all_genres],
                        multi=True, id='genre-dropdown', value=['Action', 'Comedy'], clearable=False
                    ),
                    html.H4("Select Number of Episodes:", style={'fontSize': '14px', 'marginTop': '60px'}),
                    dcc.RangeSlider(
                        min=1, max=3057, step=1, value=[100, 300], id='episodes-slider', marks=None,
                        tooltip={"placement": "bottom", "always_visible": True,
                                 "style": {"color": "LightSteelBlue", "fontSize": "10px"}}
                    ),
                    dbc.Button('Update', id='update-button', n_clicks=0, color="primary",
                               style={'display': 'block', 'margin': 'auto', 'marginTop': '80px'}),
                    # html.H4("Most watched Animes (in selected genres) having...",
                    #         style={'fontSize': '14px', 'marginTop': '40px'}),
                    # html.H4("Rating >= selected", style={'fontSize': '14px'}),
                    # html.H4("Number of episodes in selected range", style={'fontSize': '14px'}),
                ], width=5),
                dbc.Col([dcc.Graph(id='most-watched-graph', figure={})], width=6)
            ])
        ]),
    ],
)


@callback(
    Output('most-watched-graph', 'figure'),
    [Input('update-button', 'n_clicks')],
    [State('score-slider', 'value'),
     State('genre-dropdown', 'value'),
     State('episodes-slider', 'value')]
)
def update_graph(n_clicks, selected_score, selected_genres, selected_episodes_range):
    def is_valid_episode(episode_str):
        if episode_str == 'Unknown':
            return True
        try:
            episode_int = int(episode_str)
            return selected_episodes_range[0] <= episode_int <= selected_episodes_range[1]
        except ValueError:
            return False

    def truncate_labels(label):
        if len(label) > 15:
            return label[:12] + '...'  # Truncate to 17 chars + ellipsis
        else:
            return label

    filtered_df = most_watched[
        (most_watched['Score'].astype(float) >= selected_score) &
        (most_watched['Genres'].apply(lambda x: all(genre in x for genre in selected_genres))) &
        (most_watched['Episodes'].apply(is_valid_episode))
        ]

    filtered_df = filtered_df.sort_values(by='Completed', ascending=False).head(15)
    filtered_df['Name'] = filtered_df['Name'].apply(truncate_labels)

    fig = px.bar(
        filtered_df,
        x='Name',
        y='Completed',
        labels={'Name': 'Anime Name', 'Completed': 'Watched'},
        color='Completed',
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        margin=dict(l=100, r=0, t=0, b=0),
        width=800,
        height=500,
        coloraxis_showscale=False
    )

    return fig


############################################################################################################################## Component 2

"""
    anime_cleaned is data after removing unknowns from Premiered
"""
data_path = os.path.abspath('./data/anime_cleaned.csv')
data = pd.read_csv(data_path)  #this data is used in component_3
data['Premiered'] = data['Premiered'].str.split().str[-1].astype(int)

# this is filtered data specially for comp_2
pop_year = data[['Name', 'Premiered', 'Popularity']]

component_2 = dbc.Card([
    dbc.CardHeader(html.H2("Most Popular Anime in Year Range",
                           style={'marginTop': '20px', 'marginBottom': '20px', 'textAlign': 'center'})),
    dbc.CardBody([
        html.H4("Select Year Range:",
                style={'fontSize': '14px', 'marginTop': '20px', 'textAlign': 'center'}),
        html.Div([
            dcc.RangeSlider(min=1963, max=2020, value=[1975, 2000], step=1,
                            id='year-range-slider', allowCross=False,
                            className='my-range-slider', marks=None,
                            tooltip={
                                "placement": "bottom",
                                "always_visible": True,
                                "style": {"color": "LightSteelBlue", "fontSize": "10px"},
                            })
        ], style={'paddingLeft': '200px', 'paddingRight': '200px', 'marginBottom': '25px'}),
        html.Div([
            html.Div([
                dcc.Graph(id='popularity-word-cloud', figure={}),
            ], style={'width': '50%'}),
            html.Div([
                dcc.Graph(id='release-per-year', figure={}),
            ], style={'width': '50%'})
        ], style={'display': 'flex', 'flex-direction': 'row', 'width': '100%',
                  'margin': 'auto', 'justify-content': 'space-between'})
    ])
])


@callback(
    [Output('popularity-word-cloud', 'figure'),
     Output('release-per-year', 'figure')],
    [Input('year-range-slider', 'value')]
)
def update_graph(selected_years_range):
    filtered_data = pop_year[
        (pop_year['Premiered'] <= selected_years_range[1]) &
        (pop_year['Premiered'] >= selected_years_range[0])
        ]

    year_bar = go.Figure()
    year_bar.add_trace(go.Bar(
        x=filtered_data['Premiered'].value_counts().values,
        y=filtered_data['Premiered'].value_counts().index,
        orientation='h',
        marker_color=filtered_data['Premiered'].value_counts().index
    ))
    year_bar.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
    )

    filtered_data = filtered_data.sort_values(by='Premiered', ascending=False).head(150)
    text_data = ' '.join(
        word * weight for word, weight in zip(filtered_data['Name'], filtered_data['Premiered'])
    )

    # Generate Word Cloud
    wordcloud = WordCloud(
        background_color='white', collocations=False, colormap='gist_earth', height=700, width=1100,
        max_words=100, max_font_size=100).generate(text_data)

    cloud = px.imshow(wordcloud.to_array(), aspect='fill')
    cloud.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_showticklabels=False,
        yaxis_showticklabels=False,
    )
    cloud.update_traces(hoverinfo='skip')
    return cloud, year_bar


############################################################################################################################## Component 3

component_3 = dbc.Card([
    dbc.CardHeader(html.H2("Studio Analysis",
                           style={'marginTop': '20px', 'marginBottom': '20px', 'textAlign': 'center'})),
    html.Div([
                    dcc.RangeSlider(min=1963, max=2020, value=[1975, 2000],
                                    id='year-range-slider-3', marks=None, step=1, allowCross=False,
                                    tooltip={"placement": "bottom", "always_visible": True})
                ], style={'paddingLeft': '200px', 'paddingRight': '200px', 'marginTop': '25px', 'marginBottom': '25px'}),
    dbc.CardBody([
            dbc.CardBody([
                dbc.Row([  # Row to hold both graphs
                    dbc.Col([
                        dcc.Graph(id='studio-donut-graph-3')
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(id='top-anime-graph-3')
                    ], width=6)
                ])
            ])
        ]),
    html.H3("Analysis of avg Watched and avg Popularity for these Top contributing studios:",
            style={'fontSize': '20px', 'marginTop': '20px', 'textAlign': 'center'}),
    dbc.Row([  # Row for the avg completed(watched) and popularity
            dbc.Col([
                dcc.Graph(id='studio-stats-avg-watched-graph-3')
            ], width=6, style={'margin': 'auto'}),
            dbc.Col([
                dcc.Graph(id='studio-stats-avg-pop-graph-3')
            ], width=6, style={'margin': 'auto'})
        ])
    ])


@callback(
    Output('studio-donut-graph-3', 'figure'),  # Updated Output
    Output('top-anime-graph-3', 'figure'),  # Updated Output
    Output('studio-stats-avg-watched-graph-3', 'figure'),  # Updated Output
    Output('studio-stats-avg-pop-graph-3', 'figure'),
    Input('year-range-slider-3', 'value')  # Updated Input
)
def update_studio_graphs(year_range):
    filtered_df = data   # data is the filtered data in component_2
    studio_counts = get_studio_contributions(filtered_df, year_range)
    top_studio_list = studio_counts['Studios'].tolist()

    # Dict of Top animes their watch counts
    most_watched_stats = most_watched_anime_by_studio(top_studio_list, filtered_df, year_range)

    # adding studio name in dict
    studios_avgwatch_avgpop = avg_watched_and_popularity_per_year_per_studio(
        top_studio_list, filtered_df, year_range
    )
    year_list = list(range(year_range[0], year_range[1] + 1))

    studio_colors = px.colors.qualitative.Plotly[:len(studio_counts)]
    color_dict = dict(zip(studio_counts['Studios'], studio_colors))  # Main color mapping used for all plots

    # Donut Graph - Create a color dictionary and get colors for legend
    fig_donut = px.pie(studio_counts, values='Percent', names='Studios', hole=0.4,
                       title='Top 7 Contributing Studios in the given year range',
                       color='Studios', color_discrete_map=color_dict)

    #  Horizontal Histogram - Updated for labels above bars
    fig_top_anime = px.bar(most_watched_stats, x='watch_counts', y='Anime_names',
                           orientation='h', title='Top watched Anime of respective Studios', color=top_studio_list,
                           color_discrete_map=color_dict)
    fig_top_anime.update_xaxes(title='Watch Counts')  # Update x-axis label
    fig_top_anime.update_yaxes(title='Most watched anime per studio')

    # Plot average watched
    fig_stats_avgwatched = go.Figure()
    for studio, result_list in studios_avgwatch_avgpop.items():
        fig_stats_avgwatched.add_trace(go.Scatter(x=year_list, y=result_list['avg_watched'],
                                                  mode='lines+markers', name=studio, showlegend=False,
                                                  line=dict(color=color_dict[studio])))
    fig_stats_avgwatched.update_layout(title='Average Watched Over Time', xaxis_title='Time',
                                       yaxis_title='Average Watched')

    # Plot average popularity
    fig_stats_avgpop = go.Figure()
    for studio, result_list in studios_avgwatch_avgpop.items():
        fig_stats_avgpop.add_trace(go.Scatter(x=year_list, y=result_list['avg_pop'],
                                              mode='lines+markers', name=studio, line=dict(color=color_dict[studio])))
    fig_stats_avgpop.update_layout(title='Average Popularity Over Time', xaxis_title='Time',
                                   yaxis_title='Average Popularity')

    return fig_donut, fig_top_anime, fig_stats_avgwatched, fig_stats_avgpop

############################################################################################################################## Component 4


filtered_df4 = data
anime_per_genre = filtered_df4['Genres'].str.split(', ').explode().value_counts()
anime_per_genre.sort_values(ascending=False, inplace=True)

# Calculate percentage
total_anime = anime_per_genre.sum()
genre_percentage = (anime_per_genre / total_anime) * 100

# Creating treemap visualizations for distributions of anime genre
genreDistribution = go.Figure(data=[go.Treemap(
    labels=anime_per_genre.index.tolist(),
    parents=[''] * len(anime_per_genre),
    values=genre_percentage.values,
    branchvalues='total'
)])

card_1 = dbc.Card(
    [
        dbc.CardHeader(html.H3(children='Anime Genre Distribution across data'), style={'textAlign': 'center'}),
        dbc.CardBody(dcc.Graph(figure=genreDistribution))
    ],
    style={"width": "100%"}
)
animeRatingtypeDistribution = visualize_anime_count_by_rating_pie_interactive(filtered_df4)

# Create card for animeRatingDistribution plot
card_2 = dbc.Card(
    [
        dbc.CardHeader(html.H3(children='Anime Rating type Distribution'), style={'textAlign': 'center'}),
        dbc.CardBody(dcc.Graph(figure=animeRatingtypeDistribution))
    ],
    style={"width": "100%"}
)

# Layout with 50/50 split
component_4 = html.Div([
    dbc.Row([
        dbc.Col(card_1, width=6),  # First column
        dbc.Col(card_2, width=6),   # Second column
    ])
])

#########################################################################################  Home_Layout


layout = html.Div([
    html.Div(component_1, style={'marginBottom': '20px'}),
    html.Div(component_2, style={'marginBottom': '20px'}),
    html.Div(component_3, style={'marginBottom': '20px'}),
    html.Div(component_4),
], style={'margin': 'auto', 'width': '100%', 'padding': '20px'})

