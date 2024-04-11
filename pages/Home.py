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

component_1 = dbc.Row([
    html.H2("Most Watched Anime",
            style={'marginTop': '50px', 'marginBottom': '100px'}),
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
    ], width=6),
    dbc.Col([
        dcc.Graph(id='most-watched-graph', figure={})
    ], width=6)
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


data_path = os.path.abspath('./data/anime_cleaned.csv')
pop_year = pd.read_csv(data_path)
pop_year = pop_year[['Name', 'Premiered', 'Popularity']]

#  refining premiered col to contain only year as per this use case
pop_year['Premiered'] = pop_year['Premiered'].str.split().str[-1].astype(int)

component_2 = html.Div([
    html.Div(html.H2("Most Popular Anime in year range"),
             style={'marginTop': '70px', 'marginBottom': '30px'}),
    html.Div([
        dcc.RangeSlider(min=1963, max=2020, value=[1975, 2000], step=1,
                        id='year-range-slider', allowCross=False,
                        className='my-range-slider', marks=None
                        , tooltip={
                "placement": "bottom",
                "always_visible": True,
                "style": {"color": "LightSteelBlue", "fontSize": "10px"},
            }, )
    ], style={'paddingLeft': '200px', 'paddingRight': '200px', 'marginBottom': '25px'}),

    html.Div([
        html.Div([
            dcc.Graph(id='popularity-word-cloud', figure={}),
        ], style={'width': '45%', 'border': 'solid', 'border-color': 'black',}),
        html.Div([
            dcc.Graph(id='release-per-year', figure={}),
        ], style={'width': '45%', 'border': 'solid', 'border-color': 'black'})
    ], style={'display': 'flex', 'flex-direction': 'row', 'border': 'solid',
              'border-color': 'black', 'width': '90%', 'margin': 'auto','justify-content':'space-between'})  # Added align-items

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

#########################################################################################  Home_Layout


layout = dbc.Container([
    component_1,
    component_2
], fluid=True, style={'margin': 'auto', 'width': '100%'})
