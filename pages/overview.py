import os
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from wordcloud import WordCloud, STOPWORDS

######################################################################################### Global_vars
data_path = os.path.abspath('./data/anime.csv')
anime_df = pd.read_csv(data_path)
anime_df = anime_df[['Name', 'Score', 'Episodes', 'Genres', 'Completed', 'Premiered', 'Popularity', 'Studios']]

#########################################################################################  Component 1
most_watched = anime_df[['Name', 'Score', 'Episodes', 'Genres', 'Completed']]
most_watched = most_watched[(anime_df['Score'] != 'Unknown') & (anime_df['Genres'] != 'Unknown')]
list_of_all_genres = most_watched['Genres'].apply(lambda x: x.split(', ')).explode().unique().tolist()

# component_1 = dbc.Row([
#     html.H2("Most Watched Anime",
#             style={'marginTop': '50px', 'marginBottom': '100px'}),
#     dbc.Col([
#         html.Div([  # Wrap slider in a Div with margin
#             dcc.Slider(0, 10, 1, value=5, marks=None, id='score-slider',
#                        tooltip={"placement": "bottom", "always_visible": True})
#         ], style={'marginTop': '50px', 'marginBottom': '100px'}),  # Add margin-bottom
#         html.Div([  # Wrap dropdown in a Div
#             dcc.Dropdown(
#                 options=[{'label': genre, 'value': genre} for genre in list_of_all_genres],
#                 multi=True, id='genre-dropdown', value=['Action', 'Comedy'], clearable=False
#             )
#         ], style={'paddingLeft': '15px', 'paddingRight': '15px', 'marginTop': '50px', 'marginBottom': '100px'}),
#         html.Div([  # Wrap range slider in a Div
#             dcc.RangeSlider(min=1, max=3057, step=1, value=[100, 300], id='episodes-slider', marks=None,
#                             tooltip={
#                                 "placement": "bottom",
#                                 "always_visible": True,
#                                 "style": {"color": "LightSteelBlue", "fontSize": "10px"},
#                             })
#         ], style={'marginBottom': '30px'}),
#         html.Button('Update', id='update-button', n_clicks=0,
#                     style={'display': 'block', 'margin': 'auto'}),
#     ], width=6),
#     dbc.Col([
#         dcc.Graph(id='most-watched-graph', figure={})
#     ], width=6)
# ])
component_1 = dbc.Card([  # Enclose components within a card
    dbc.CardHeader(html.H2("Most Watched Anime", style={'marginTop': '50px', 'marginBottom': '100px', 'textAlign': 'center'})),
    dbc.CardBody([  # CardBody for the rest of the content
        dbc.Row([
            dbc.Col([
                html.Div([  # Wrap slider in a Div with margin
                    dcc.Slider(0, 10, 1, value=5, marks=None, id='score-slider',
                               tooltip={"placement": "bottom", "always_visible": True})
                ], style={'marginTop': '50px', 'marginBottom': '100px'}),  # Add margin-bottom
                html.Div([  # Wrap dropdown in a Div
                    dcc.Dropdown(
                        options=[{'label': genre, 'value': genre} for genre in list_of_all_genres],
                        multi=True, id='genre-dropdown', value=['Action', 'Comedy'], clearable=False
                    )
                ], style={'paddingLeft': '15px', 'paddingRight': '15px', 'marginTop': '50px', 'marginBottom': '100px'}),
                html.Div([  # Wrap range slider in a Div
                    dcc.RangeSlider(min=1, max=3057, step=1, value=[100, 300], id='episodes-slider', marks=None,
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                        "style": {"color": "LightSteelBlue", "fontSize": "10px"},
                                    })
                ], style={'marginBottom': '30px'}),
                html.Button('Update', id='update-button', n_clicks=0,
                            style={'display': 'block', 'margin': 'auto'}),
            ], width=5),
            # dbc.Col([
            #     # dcc.Graph(id='most-watched-graph', figure={}),
            #     dcc.Graph(id='most-watched-graph', figure={}, config={'responsive': True})
            # ], width=6)
            dbc.Col([
                html.Div(dcc.Graph(id='most-watched-graph', figure={}),
                         style={'overflow-x': 'scroll'}) # Enable x-axis scrolling
            ], width=7)

        ])
    ])
])

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
        if len(label) > 20:
            return label[:17] + '...'  # Truncate to 17 chars + ellipsis
        else:
            return label

    filtered_df = most_watched[
        (most_watched['Score'].astype(float) >= selected_score) &
        (most_watched['Genres'].apply(lambda x: all(genre in x for genre in selected_genres))) &
        (most_watched['Episodes'].apply(is_valid_episode))
        ]

    filtered_df = filtered_df.sort_values(by='Completed', ascending=False).head(13)
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


#########################################################################################  Component 2

"""
    anime_cleaned is data after removing unknowns from Premiered
"""
data_path = os.path.abspath('./data/anime_cleaned.csv')
data = pd.read_csv(data_path)  #this data is used in component_3
data['Premiered'] = data['Premiered'].str.split().str[-1].astype(int)

pop_year = data[['Name', 'Premiered', 'Popularity']]   #this is filtered data specially for comp_2

#  refining premiered col to contain only year as per this use case
# pop_year['Premiered'] = pop_year['Premiered'].str.split().str[-1].astype(int)

# component_2 = html.Div([
#     html.Div(html.H2("Most Popular Anime in year range"),
#              style={'marginTop': '70px', 'marginBottom': '30px'}),
#     html.Div([
#         dcc.RangeSlider(min=1963, max=2020, value=[1975, 2000], step=1,
#                         id='year-range-slider', allowCross=False,
#                         className='my-range-slider', marks=None
#                         , tooltip={
#                 "placement": "bottom",
#                 "always_visible": True,
#                 "style": {"color": "LightSteelBlue", "fontSize": "10px"},
#             }, )
#     ], style={'paddingLeft': '200px', 'paddingRight': '200px', 'marginBottom': '25px'}),
#
#     html.Div([
#         html.Div([
#             dcc.Graph(id='popularity-word-cloud', figure={}),
#         ], style={'width': '45%', 'border': 'solid', 'border-color': 'black'}),
#         html.Div([
#             dcc.Graph(id='release-per-year', figure={}),
#         ], style={'width': '45%', 'border': 'solid', 'border-color': 'black'})
#     ], style={'display': 'flex', 'flex-direction': 'row', 'border': 'solid',
#               'border-color': 'black', 'width': '90%', 'margin': 'auto', 'justify-content': 'space-between'})
#
# ])

component_2 = dbc.Card([
    dbc.CardHeader(html.H2("Most Popular Anime in Year Range", style={'marginTop': '70px', 'marginBottom': '30px', 'textAlign': 'center'})),
    dbc.CardBody([
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
            ], style={'width': '50%', 'height': '700px'}),
            html.Div([
                dcc.Graph(id='release-per-year', figure={}),
            ], style={'width': '45%'})
        ], style={'display': 'flex', 'flex-direction': 'row', 'width': '90%', 'margin': 'auto', 'justify-content': 'space-between'})
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
        background_color='white', collocations=False, colormap='gist_earth',
        max_words=100, max_font_size=100).generate(text_data)

    cloud = px.imshow(wordcloud.to_array(), aspect='fill')
    cloud.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_showticklabels=False,
        yaxis_showticklabels=False,
    )
    cloud.update_traces(hoverinfo='skip')
    return cloud, year_bar


######################################################################################### Component3


# component_3 = dbc.Row([
#     html.H2("Studio Analysis",
#             style={'marginTop': '50px', 'marginBottom': '100px'}),
#     dbc.Col([
#         html.Div([
#             dcc.RangeSlider(min=1963, max=2020, value=[1975, 2000],
#                             id='year-range-slider-3', marks=None,  # Changed ID
#                             tooltip={"placement": "bottom", "always_visible": True})
#         ], style={'marginBottom': '50px'})
#     ], width=12),  # Make slider full-width
#     dbc.Col([
#         dcc.Graph(id='studio-donut-graph-3')  # Changed ID
#     ], width=6),
#     dbc.Col([
#         dcc.Graph(id='top-anime-graph-3')  # Changed ID
#     ], width=6),
#     dbc.Row([  # Center-align the next row
#         dbc.Col([
#             dcc.Graph(id='studio-stats-graph-3')  # Changed ID
#         ], width=12, style={'width': '1500px', 'align-items': 'center', 'justify-content': 'center'})
#     ])
# ])
component_3 = dbc.Card([
    dbc.CardHeader(html.H2("Studio Analysis", style={'marginTop': '50px', 'marginBottom': '100px', 'textAlign': 'center'})),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.RangeSlider(min=1963, max=2020, value=[1975, 2000],
                                    id='year-range-slider-3', marks=None,
                                    tooltip={"placement": "bottom", "always_visible": True})
                ], style={'marginBottom': '50px'})
            ], width=12),  # Make slider full-width
        ]),
        dbc.CardBody([
            # dbc.CardHeader("Studio Analysis Graphs"),  # Optional header for the entire card
            dbc.CardBody([
                dbc.Row([  # Row to hold both graphs
                    dbc.Col([
                        dcc.Graph(id='studio-donut-graph-3')
                    ], width=6),  # Adjust widths as needed
                    dbc.Col([
                        dcc.Graph(id='top-anime-graph-3')
                    ], width=6)
                ])
            ])
        ]),

        dbc.Row([  # Row for the larger graph, with adjustments
            dbc.Col([
                dcc.Graph(id='studio-stats-graph-3')
            ], width=12, style={'margin': 'auto'})  # Center and remove fixed width
        ])
    ])
])

@callback(
    Output('studio-donut-graph-3', 'figure'),  # Updated Output
    Output('top-anime-graph-3', 'figure'),  # Updated Output
    Output('studio-stats-graph-3', 'figure'),  # Updated Output
    Input('year-range-slider-3', 'value')  # Updated Input
)
def update_studio_graphs(year_range):
    def get_studio_contributions(df, year_range):
        studio_cnts = (df[df['Premiered'].between(*year_range)]
                       .explode('Studios')
                       .groupby('Studios')
                       .size()
                       .reset_index(name='Count')
                       .sort_values('Count', ascending=False)  # Sort to get top
                       .head(15)  # Keep only the top 15 studios
                       )
        studio_cnts['Percent'] = studio_cnts['Count'] / studio_cnts['Count'].sum() * 100
        return studio_cnts

    def get_top_anime_by_studio(df, year_range):
        studios_top_anime = (df[df['Premiered'].between(*year_range)]
                             .groupby(['Studios', 'Name'])
                             ['Completed']
                             .max()
                             .reset_index()
                             .sort_values(['Studios', 'Completed'], ascending=False))

        # Get top 15 studios first
        top_studios = studios_top_anime['Studios'].unique()[:15]
        return studios_top_anime[studios_top_anime['Studios'].isin(top_studios)].groupby('Studios').head(1)

    def get_studio_stats(df, year_range):
        studio_avg = (df[df['Premiered'].between(*year_range)]
                      .groupby('Studios')[['Completed', 'Popularity']]
                      .mean()
                      .reset_index())
        return studio_avg

    filtered_df = data
    studio_counts = get_studio_contributions(filtered_df, year_range)
    top_anime = get_top_anime_by_studio(filtered_df, year_range)
    studio_stats = get_studio_stats(filtered_df, year_range)

    # Donut Graph - Create a color dictionary and get colors for legend
    studio_colors = px.colors.qualitative.Plotly[:len(studio_counts)]  # Truncate to top 15
    color_dict = dict(zip(studio_counts['Studios'], studio_colors))  # Main color mapping

    fig_donut = px.pie(studio_counts, values='Percent', names='Studios', hole=0.4,
                       title='Studio Contributions (Top 15)',
                       color='Studios', color_discrete_map=color_dict)
    # ...

    # Horizontal Histogram - Updated for labels above bars
    fig_top_anime = px.bar(top_anime, x='Completed', y='Studios', orientation='h',
                           title='Top Anime by Studio (Top 15)',
                           color='Studios', color_discrete_map=color_dict)
    fig_top_anime.update_traces(textposition='outside')
    fig_top_anime.update_layout(showlegend=True,
                                uniformtext_minsize=8, uniformtext_mode='hide')

    # ...

    fig_stats = px.line(studio_stats, x='Studios', y=['Completed', 'Popularity'],
                        title='Average Metrics by Studio (Top 15)', log_y=True)  # Log scale on y-axis
    fig_stats.update_layout(height=800)
    # ...

    return fig_donut, fig_top_anime, fig_stats

#########################################################################################  Component 4

# Load your dataset (replace with the actual path to your file)
df = pd.read_csv('./data/anime.csv')

# Calculate the number of anime per genre
anime_per_genre = df['Genres'].str.split(', ').explode().value_counts()

anime_per_genre.sort_values(ascending=False, inplace=True)
anime_per_genre = anime_per_genre[:10]
# Create the Pie Chart
genreDistribution = px.pie(anime_per_genre,
             values=anime_per_genre.values,
             names=anime_per_genre.index,
             title='Distribution of Anime Genres')

# component_4 = html.Div(children=[
#     html.H1(children='Anime Genre Distribution'),
#     dcc.Graph(
#         figure=genreDistribution
#     )
# ])

card_1 = dbc.Card(
    [
        dbc.CardHeader(html.H3(children='Anime Genre Distribution')),
        dbc.CardBody(dcc.Graph(figure=genreDistribution))
    ],
    style={"width": "100%"}
)

#########################################################################################  Component 4: Card
#drop those rows where either episodes or duration is missing

df = df.dropna(subset=['Episodes', 'Duration'])
df['Total Length (min)'] = df.apply(lambda row: int(row['Duration'].split()[0]) * int(row['Episodes']) if row['Duration'] != 'Unknown' and row['Episodes'] != 'Unknown' else None, axis=1)

# Create the line graph
animeLengthDistribution = px.line(df, x='MAL_ID', y='Total Length (min)', title='Anime Length Distribution')




# Create card for duration line graph
card_2 = dbc.Card(
    [
        dbc.CardHeader(html.H3(children='Anime Length Distribution')),
        dbc.CardBody(dcc.Graph(figure=animeLengthDistribution))
    ],
    style={"width": "100%"}
)

# Layout with 50/50 split
component_4 = html.Div([
    dbc.Row([
        dbc.Col(card_1, width=6),  # 6 columns for 50% width
        dbc.Col(card_2, width=6)
    ])
])

#########################################################################################  Home_Layout


layout = dbc.Container([
    component_1,
    component_2,
    component_3,
    component_4

], fluid=True, style={'margin': 'auto', 'width': '100%'})

# def searchAndPrintAnime(df, mal_id):
#     anime = df[df['MAL_ID'] == mal_id]
#     for col in anime.columns:
#         print(f"{col}: {anime[col].values[0]}")
#
#
#
# searchAndPrintAnime(df, 6277)
