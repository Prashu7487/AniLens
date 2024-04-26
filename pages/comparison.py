import dash
import os
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc, callback
from dash import html, Output, Input, State, dcc
import dash_bootstrap_components as dbc
import pandas as pd

data_path = os.path.abspath('./data/anime.csv')  # Get absolute path
data = pd.read_csv(data_path)


##################################################################################################
# **Comparision**
##################################################################################################

def page_layout(anime1, anime2):
    # =====================================================================variables only===============
    def max(list):
        max = list[0]
        for i in list:
            if i > max:
                max = i
        return max

    def fMax(int1, int2):
        if int1 > int2:
            return int1
        else:
            return int2

    def get_anime_data(anime_name):
        anime_data = data[data['Name'] == anime_name]
        return anime_data

    # read the row from data where name is anime1
    anime1_data = get_anime_data(anime1)
    anime2_data = get_anime_data(anime2)
    list_of_all_genres = data['Genres'].apply(lambda x: x.split(', ')).explode().unique().tolist()
    # slice animal1_data to get the row till the column till the "Favorites" comes
    anime1_data_first = anime1_data.iloc[:, :anime1_data.columns.get_loc('Favorites') + 1]
    anime2_data_first = anime2_data.iloc[:, :anime2_data.columns.get_loc('Favorites') + 1]
    # print(type(anime1_data))

    pieChart_data_anime1 = anime1_data.iloc[0,
                           anime1_data.columns.get_loc('Watching'):anime1_data.columns.get_loc('Plan to Watch') + 1]
    pieChart_data_anime2 = anime2_data.iloc[0,
                           anime2_data.columns.get_loc('Watching'):anime2_data.columns.get_loc('Plan to Watch') + 1]

    pieChart_data_anime1 = pd.Series(pieChart_data_anime1)
    pieChart_data_anime2 = pd.Series(pieChart_data_anime2)
    # print(pieChart_data_anime1)

    # pie_chart =
    score_anime1 = anime1_data.iloc[0, anime1_data.columns.get_loc('Score-10'):]
    score_anime2 = anime2_data.iloc[0, anime2_data.columns.get_loc('Score-10'):]
    # print(score_anime1)

    score = ['Score-10', 'Score-9', 'Score-8', 'Score-7', 'Score-6', 'Score-5', 'Score-4', 'Score-3', 'Score-2',
             'Score-1']
    score_anime1 = pd.Series(score_anime1, index=score)
    score_anime2 = pd.Series(score_anime2, index=score)

    # prepare y
    score_anime1 = score_anime1.values
    score_anime1 = [float(i) for i in score_anime1]

    score_anime2 = score_anime2.values
    score_anime2 = [float(i) for i in score_anime2]

    fig = go.Figure([go.Bar(x=score, y=score_anime1, name=anime1), go.Bar(x=score, y=score_anime2, name=anime2)])
    fig.update_layout(
        barmode='group',
        yaxis_range=[0, (fMax(max(score_anime1), max(score_anime2))) * 1.1]
    )

    # Color Mapping Dictionary
    color_map = {'Watching': 'rgb(255, 0, 0)',  # Red
                 'Completed': 'rgb(0, 255, 0)',  # Green
                 'On-Hold': 'rgb(0, 0, 255)',  # Blue
                 'Dropped': 'rgb(255, 255, 0)',  # Yellow
                 'Plan to Watch': 'rgb(255, 0, 255)'}  # Purple
    # ==========================================================================================================
    comparison_layout = html.Div([
        html.Div([
            html.Div([
                html.H1(children='Anime Comparison', className='display-4',
                        style={'color': '#007bff', 'text-align': 'center', 'margin-bottom': '30px'}),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(f"{anime1}", style={'background-color': '#007bff', 'color': 'white'}),
                            dbc.CardBody([
                                html.Div([
                                    dbc.ListGroup([
                                        dbc.ListGroupItem(
                                            f"{anime1_data_first[col].name} : {anime1_data_first.iloc[0][col]}")
                                        for col in anime1_data_first.columns
                                    ])
                                ])
                            ])
                        ], style={'margin-bottom': '30px'})
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(f"{anime2}", style={'background-color': '#007bff', 'color': 'white'}),
                            dbc.CardBody([
                                html.Div([
                                    dbc.ListGroup([
                                        dbc.ListGroupItem(anime2_data_first.iloc[0][col])
                                        for col in anime1_data_first.columns
                                    ])
                                ])
                            ])
                        ], style={'margin-bottom': '30px'})
                    ], width=6)
                ])
            ], className='container', style={'padding': '20px', 'margin-bottom': '30px'}),

            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(f"Watching Status: {anime1}",
                                           style={'background-color': '#007bff', 'color': 'white'}),
                            dbc.CardBody([
                                dcc.Graph(
                                    id='anime1-pie-chart',
                                    figure=px.pie(
                                        pieChart_data_anime1,
                                        values=pieChart_data_anime1.values,
                                        names=pieChart_data_anime1.index,
                                        title='Anime 1 Viewing Status',
                                        color=[color_map[label] for label in pieChart_data_anime1.index.tolist()[:5]]
                                    )
                                )
                            ])
                        ])
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(f"Watching Status: {anime2}",
                                           style={'background-color': '#007bff', 'color': 'white'}),
                            dbc.CardBody([
                                dcc.Graph(
                                    id='anime2-pie-chart',
                                    figure=px.pie(
                                        pieChart_data_anime2,
                                        values=pieChart_data_anime2.values,
                                        names=pieChart_data_anime2.index,
                                        title='Anime 2 Viewing Status',
                                        color=[color_map[label] for label in pieChart_data_anime1.index.tolist()[:5]]
                                    )
                                )
                            ])
                        ])
                    ], width=6)
                ])
            ], style={'margin-bottom': '30px', 'padding': '20px'})
        ], style={'padding': '20px', 'margin-bottom': '30px'}),

        dbc.Card([
            dbc.CardHeader("Score Comparison",
                           style={'background-color': '#007bff', 'color': 'white','text-align': 'center','font-size':'20px'}),
            # Card Header
            dbc.CardBody([  # Card body for content
                html.Div([
                    dcc.Graph(figure=fig)
                ], style={'width': '90%', 'margin': 'auto'}),
            ]),
            dbc.CardBody([], style={'width': '100%'}),  # Empty div to retain margins
        ], style={'margin': '20px', 'margin-bottom': '30px'})
    ])

    return comparison_layout


##################################################################################################
# **Popup Construction**
##################################################################################################

comparison_page = html.Div(id='main-container')

modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Enter Anime Names"), style={'background-color': '#007bff', 'color': 'white'}),
        dbc.ModalBody(
            [
                dbc.Input(id='anime1', type='text', placeholder='Name of Anime 1', style={'margin-bottom': '10px'}),
                dbc.Input(id='anime2', type='text', placeholder='Name of Anime 2', style={'margin-bottom': '10px'}),
                html.Div(id='validation-message', style={'margin-bottom': '10px'}),
                html.Div(dbc.Button("Compare", id='compare-button', n_clicks=0, color='primary'),
                         style={'text-align': 'center', 'margin-bottom': '10px'}),
            ],
            style={'padding': '20px'}
        ),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-button", className="ms-auto", color='secondary')
        ),
    ],
    id="modal",
    is_open=True,  # Initially open
    centered=True,
    style={'padding': '20px'}
)


@callback(
    Output('modal', 'is_open'),
    Output('main-container', 'children'),
    Output('validation-message', 'children'),
    Input('compare-button', 'n_clicks'),
    Input('close-button', 'n_clicks'),
    State('anime1', 'value'),
    State('anime2', 'value'),
    State('modal', 'is_open')
)
def compare_animes(compare_clicks, close_clicks, anime1name, anime2name, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return True, None, "Enter the names"  # Initial state

    def search_anime(anime_name):
        anime_data = data[data['Name'] == anime_name]
        if not anime_data.empty:
            return True  # Return the index
        else:
            return False

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'compare-button':
        if search_anime(anime1name) and search_anime(anime2name):
            return False, page_layout(anime1name, anime2name), "Redirecting..."
        else:
            return True, None, 'One or both animes are invalid (NA)'
    elif trigger_id == 'close-button':
        return False, None, ''
    else:
        return True, None, ''


##################################################################################################
#  Main Layout** ##################################################################################
##################################################################################################
# **Layout Integration**


layout = html.Div([
    modal,
    comparison_page,
])
