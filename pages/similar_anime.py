import dash_bootstrap_components as dbc
from dash import html
import os

import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import joblib
import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf
from dash import html
from tensorflow import keras
from tensorflow.keras.layers import Activation, BatchNormalization, Input, Embedding, \
    Dot, Dense, Flatten
from tensorflow.keras.models import Model
from dash import html
MIN_LR = 1e-5  # minimum learning rate
MAX_LR = 5e-4  # maximum learning rate
BS = 10000  # batch_size
EPOCHS = 10
USER_EMB_SIZE = 128
ANIME_EMB_SIZE = 128
EMBEDDING_SIZES = (USER_EMB_SIZE, ANIME_EMB_SIZE)

abs_path = os.path.abspath("./data/")
# print(abs_path)
ENCODER_PATH = os.path.join(abs_path, 'encoder_dicts.joblib')

# ENCODER_PATH = 'encoder_dicts.joblib'
encoders_dict = joblib.load(ENCODER_PATH)
# print(encoders_dict.keys())

anime_id_to_idx = encoders_dict['anime_id_to_idx']
anime_idx_to_id = encoders_dict['anime_idx_to_id']
user_id_to_idx = encoders_dict['user_id_to_idx']
user_idx_to_id = encoders_dict['user_idx_to_id']

n_users, n_animes = len(user_id_to_idx), len(anime_id_to_idx)
# n_users, n_animes

K = keras.backend


class RecommenderModel(Model):
    # Based on https://keras.io/examples/structured_data/collaborative_filtering_movielens/
    def __init__(self, n_users, n_animes, embedding_sizes, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.n_users = n_users
        self.n_animes = n_animes
        self.embedding_sizes = embedding_sizes
        self.user_embedding = Embedding(
            name='user_embedding',
            input_dim=n_users,
            output_dim=embedding_sizes[0],
        )
        self.anime_embedding = Embedding(
            name='anime_embedding',
            input_dim=n_animes,
            output_dim=embedding_sizes[1],
        )
        self.dot_layer = Dot(name='dot_product', normalize=True, axes=1)
        self.layers_ = [
            Flatten(),

            #             bias is not needed when using BatchNorm
            Dense(128, activation=activation, use_bias=False, kernel_initializer='he_normal'),
            BatchNormalization(),
            Dense(64, activation=activation, use_bias=False, kernel_initializer='he_normal'),
            BatchNormalization(),
            Dense(1, use_bias=False, kernel_initializer='he_normal'),
            BatchNormalization(),
            Activation('sigmoid'),
        ]

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        anime_vector = self.anime_embedding(inputs[:, 1])
        x = self.dot_layer([user_vector, anime_vector])
        for layer in self.layers_:
            x = layer(x)
        return x


def get_model():
    # print("[INFO] Using Subclassing API Model")
    K.clear_session()

    model = RecommenderModel(n_users, n_animes, EMBEDDING_SIZES)
    model.compile(loss='binary_crossentropy', metrics=['mae', 'mse'], optimizer='adam')

    return model


K = keras.backend
K.clear_session()


def RecommenderNet():
    # from Chaitanya's notebook https://www.kaggle.com/chaitanya99/recommendation-system-cf-anime/notebook
    embedding_size = 128

    user = Input(name='user', shape=[1])
    user_embedding = Embedding(name='user_embedding',
                               input_dim=n_users,
                               output_dim=embedding_size)(user)

    anime = Input(name='anime', shape=[1])
    anime_embedding = Embedding(name='anime_embedding',
                                input_dim=n_animes,
                                output_dim=embedding_size)(anime)

    # normalize=True will generate cosine similarity output
    # axes=2 to perform matrix multiplication on embedding axes
    x = Dot(name='dot_product', normalize=True, axes=2)([user_embedding, anime_embedding])
    x = Flatten()(x)

    x = Dense(1, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("sigmoid")(x)

    model = Model(inputs=[user, anime], outputs=x)
    model.compile(loss='binary_crossentropy', metrics=["mae", "mse"], optimizer='Adam')

    return model


def get_functional_model():
    # print("[INFO] Using Functional API Model")
    model = RecommenderNet()
    return model


model = get_functional_model()

WEIGHTS_PATH = os.path.abspath("./recomendation/weights.h5")
# print(WEIGHTS_PATH)


def load_trained_model():
    model = get_model()
    _ = model(tf.ones((1, 2)))
    model.load_weights(WEIGHTS_PATH)
    return model


model = load_trained_model()


def extract_weights(name, model):
    weight_layer = model.get_layer(name)
    weights = weight_layer.get_weights()[0]
    # because Dot layer was using normalize=True..?
    weights = weights / np.linalg.norm(weights, axis=1).reshape((-1, 1))
    return weights


anime_weights = extract_weights('anime_embedding', model)
user_weights = extract_weights('user_embedding', model)

# os.chdir('/')
path = os.path.join(abs_path, 'anime.csv')
# print(path)
anime_df = pd.read_csv(path)
anime_df.sort_values('Name', inplace=True)

all_anime_types = set(anime_df['Type'].unique())


def check_anime_types(types):
    types = set([types]) if isinstance(types, str) else set(types)
    if types.issubset(all_anime_types):
        return
    else:
        for anime_type in types:
            if anime_type not in all_anime_types:
                raise Exception(f'Anime type "{anime_type}" is not valid!')


def get_anime_rows(df, anime_query, exact_name=False, types=None):
    df = df.copy()
    if isinstance(anime_query, int):
        df = df[df.MAL_ID == anime_query]
    else:
        if exact_name:
            # get exact name
            df = df[df.Name == anime_query]
        else:
            df = df[df.Name.str.contains(anime_query, case=False, regex=False)]

    if types:
        check_anime_types(types)
        df = df[df.Type.isin(types)]

    return df


# pd.reset_option('all')
pd.set_option("max_colwidth", None)


def get_recommendation(anime_query, k=10, exact_name=False, types=None):
    if types:
        check_anime_types(types)
    anime_rows = get_anime_rows(anime_df, anime_query,
                                exact_name=exact_name)
    if len(anime_rows) == 0:
        raise Exception(f'Anime not found for {anime_query}')
    anime_row = anime_rows.iloc[[0]]
    anime_id = anime_row.MAL_ID.values[0]
    anime_name = anime_row.Name.values[0]
    anime_idx = anime_id_to_idx.get(anime_id)

    weights = anime_weights
    distances = np.dot(weights, weights[anime_idx])

    sorted_dists_ind = np.argsort(distances)[::-1]

    # print(f'Recommending animes for {anime_name}')
    # display(anime_row.loc[:, 'MAL_ID': 'Aired'])

    anime_list = []
    # [1:] to skip the first row for anime_query
    for idx in sorted_dists_ind[1:]:
        similarity = distances[idx]
        anime_id = anime_idx_to_id.get(idx)
        anime_row = anime_df[anime_df.MAL_ID == anime_id]
        anime_type = anime_row.Type.values[0]
        if types and anime_type not in types:
            continue
        anime_name = anime_row.Name.values[0]
        score = anime_row.Score.values[0]
        genre = anime_row.Genres.values[0]

        anime_list.append({"Anime_id": anime_id, "Name": anime_name,
                           "Similarity": similarity, "Score": score,
                           "Type": anime_type, "Genre": genre
                           })
        if len(anime_list) == k:
            # enough number of recommendations
            break
    rec_df = pd.DataFrame(anime_list)
    return rec_df


#################################################################################################################
# **Data Reading**
#################################################################################################################
data = pd.read_csv('data/anime.csv')


#################################################################################################################
# **Page Layout**
#################################################################################################################
def page_layout(anime):
    similarAnime = get_recommendation(anime, k=10)
    similarAnime = similarAnime[['Name', 'Similarity']]
    similarity = similarAnime['Similarity']
    similarity = similarity * 100
    similarity = similarity.tolist()
    similarAnime = similarAnime.drop(columns=['Similarity'])
    similarAnime = similarAnime.to_dict('records')
    similarAnime = [anime['Name'] for anime in similarAnime]

    # Preprocess similarity for edge weights
    max_similarity = max(similarity)
    edge_weights = [max_similarity - score for score in similarity]

    # Create a NetworkX graph
    G = nx.Graph()
    G.add_node(anime, pos=(0, 0))  # Central node

    for i, similar in enumerate(similarAnime):
        G.add_node(similar)
        G.add_edge(anime, similar, weight=edge_weights[i])

    # Calculate positions with force-directed layout
    # Calculate positions with force-directed layout
    pos = nx.spring_layout(G)

    # Normalize positions (if needed)
    xmin = min(x for x, y in pos.values())
    xmax = max(x for x, y in pos.values())
    ymin = min(y for x, y in pos.values())
    ymax = max(y for x, y in pos.values())

    def normalize(val, minval, maxval):
        return (val - minval) / (maxval - minval)

    # Construct nodes and edges for Dash Cytoscape
    elements = [
        {'data': {'id': name, 'label': name},
         'position': {'x': normalize(x, xmin, xmax), 'y': normalize(y, ymin, ymax)}}
        for name, (x, y) in pos.items()
    ]

    elements += [
        {'data': {'source': source, 'target': target, 'weight': weight}}
        for source, target, weight in G.edges(data='weight')
    ]

    print(elements)



    return html.Div([
        cyto.Cytoscape(
            id='cytoscape',
            elements=elements,
            layout={'name': 'cose'},  # Adjust layout as needed
            style={'width': '100%', 'height': '600px'},

            stylesheet=[
            {
                'selector': 'node',
                'style': {
                    'background-color': '#4287f5',  # Your shade of blue
                    'label': 'data(label)',
                    'font-size': '10px',  # Smaller font size
                    'text-wrap': 'wrap',  # Allow text wrapping
                    'text-max-width': '80px'  # Limit node label width
                }
            },
            {
                'selector': 'edge',
                'style': {
                    'line-color': '#ccc',  # Subtle edge color
                    'width': '2px'  # Adjust edge thickness
                }
            }
        ]



        )
    ])


#################################################################################################################
#################################################################################################################
# **Layout**
#################################################################################################################
#################################################################################################################
similar_page = html.Div(id='main-container-similar')

pop_up = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Enter Anime Name")),
        dbc.ModalBody(
            [
                dbc.Input(id='anime', type='text', placeholder='Name of Anime'),
                html.Div(id='validation-message-similar'),
                dbc.Button("Search", id='search-button', n_clicks=0)
            ]
        ),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-button-similar", className="ms-auto")
        ),
    ],
    id="pop_up",
    is_open=True,  # Initially open
    centered=True,
)
from dash import callback, Output, Input, State
@callback(
    Output('pop_up', 'is_open'),
    Output('main-container-similar', 'children'),
    Output('validation-message-similar', 'children'),
    Input("search-button", 'n_clicks'),
    Input("close-button-similar", 'n_clicks'),
    State('anime', 'value'),
    State('pop_up', 'is_open')
)
def search_animes(compare_clicks, close_clicks, anime, is_open):
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

    if trigger_id == 'search-button':
        if search_anime(anime):
            return False, page_layout(anime), "Redirecting..."
        else:
            return True, None, 'One or both animes are invalid (NA)'
    elif trigger_id == 'close-button-similar':
        return False, None, ''
    else:
        return True, None, ''


#################################################################################################################
#################################################################################################################
# ** Main Layout **
#################################################################################################################
#################################################################################################################


layout = html.Div([
    similar_page,
    pop_up
])
