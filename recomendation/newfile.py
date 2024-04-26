# from dash import html
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import math
# import joblib
# import os
#
# import tensorflow as tf
#
# from tensorflow import keras
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import Add, Activation, Lambda, BatchNormalization, Concatenate, Dropout, Input, Embedding, \
#     Dot, Reshape, Dense, Flatten
#
# MIN_LR = 1e-5  # minimum learning rate
# MAX_LR = 5e-4  # maximum learning rate
# BS = 10000  # batch_size
# EPOCHS = 10
# USER_EMB_SIZE = 128
# ANIME_EMB_SIZE = 128
# EMBEDDING_SIZES = (USER_EMB_SIZE, ANIME_EMB_SIZE)
#
# abs_path = os.path.abspath(os.path.dirname(__file__))
# ENCODER_PATH = os.path.join(abs_path, 'encoder_dicts.joblib')
#
# # ENCODER_PATH = 'encoder_dicts.joblib'
# encoders_dict = joblib.load(ENCODER_PATH)
# # print(encoders_dict.keys())
#
# anime_id_to_idx = encoders_dict['anime_id_to_idx']
# anime_idx_to_id = encoders_dict['anime_idx_to_id']
# user_id_to_idx = encoders_dict['user_id_to_idx']
# user_idx_to_id = encoders_dict['user_idx_to_id']
#
# n_users, n_animes = len(user_id_to_idx), len(anime_id_to_idx)
# n_users, n_animes
#
# K = keras.backend
#
#
# class RecommenderModel(Model):
#     # Based on https://keras.io/examples/structured_data/collaborative_filtering_movielens/
#     def __init__(self, n_users, n_animes, embedding_sizes, activation='relu', **kwargs):
#         super().__init__(**kwargs)
#         self.n_users = n_users
#         self.n_animes = n_animes
#         self.embedding_sizes = embedding_sizes
#         self.user_embedding = Embedding(
#             name='user_embedding',
#             input_dim=n_users,
#             output_dim=embedding_sizes[0],
#         )
#         self.anime_embedding = Embedding(
#             name='anime_embedding',
#             input_dim=n_animes,
#             output_dim=embedding_sizes[1],
#         )
#         self.dot_layer = Dot(name='dot_product', normalize=True, axes=1)
#         self.layers_ = [
#             Flatten(),
#
#             #             bias is not needed when using BatchNorm
#             Dense(128, activation=activation, use_bias=False, kernel_initializer='he_normal'),
#             BatchNormalization(),
#             Dense(64, activation=activation, use_bias=False, kernel_initializer='he_normal'),
#             BatchNormalization(),
#             Dense(1, use_bias=False, kernel_initializer='he_normal'),
#             BatchNormalization(),
#             Activation('sigmoid'),
#         ]
#
#     def call(self, inputs):
#         user_vector = self.user_embedding(inputs[:, 0])
#         anime_vector = self.anime_embedding(inputs[:, 1])
#         x = self.dot_layer([user_vector, anime_vector])
#         for layer in self.layers_:
#             x = layer(x)
#         return x
#
#
# def get_model():
#     print("[INFO] Using Subclassing API Model")
#     K.clear_session()
#
#     model = RecommenderModel(n_users, n_animes, EMBEDDING_SIZES)
#     model.compile(loss='binary_crossentropy', metrics=['mae', 'mse'], optimizer='adam')
#
#     return model
#
#
# K = keras.backend
# K.clear_session()
#
#
# def RecommenderNet():
#     # from Chaitanya's notebook https://www.kaggle.com/chaitanya99/recommendation-system-cf-anime/notebook
#     embedding_size = 128
#
#     user = Input(name='user', shape=[1])
#     user_embedding = Embedding(name='user_embedding',
#                                input_dim=n_users,
#                                output_dim=embedding_size)(user)
#
#     anime = Input(name='anime', shape=[1])
#     anime_embedding = Embedding(name='anime_embedding',
#                                 input_dim=n_animes,
#                                 output_dim=embedding_size)(anime)
#
#     # normalize=True will generate cosine similarity output
#     # axes=2 to perform matrix multiplication on embedding axes
#     x = Dot(name='dot_product', normalize=True, axes=2)([user_embedding, anime_embedding])
#     x = Flatten()(x)
#
#     x = Dense(1, kernel_initializer='he_normal')(x)
#     x = BatchNormalization()(x)
#     x = Activation("sigmoid")(x)
#
#     model = Model(inputs=[user, anime], outputs=x)
#     model.compile(loss='binary_crossentropy', metrics=["mae", "mse"], optimizer='Adam')
#
#     return model
#
#
# def get_functional_model():
#     print("[INFO] Using Functional API Model")
#     model = RecommenderNet()
#     return model
#
#
# model = get_functional_model()
#
# WEIGHTS_PATH = os.path.join(abs_path, 'weights.h5')
#
#
# def load_trained_model():
#     model = get_model()
#     print('Calling model to load layers...')
#     _ = model(tf.ones((1, 2)))
#     model.load_weights(WEIGHTS_PATH)
#     print('Loaded weights.')
#     return model
#
#
# model = load_trained_model()
#
#
# def extract_weights(name, model):
#     weight_layer = model.get_layer(name)
#     weights = weight_layer.get_weights()[0]
#     # because Dot layer was using normalize=True..?
#     weights = weights / np.linalg.norm(weights, axis=1).reshape((-1, 1))
#     return weights
#
#
# anime_weights = extract_weights('anime_embedding', model)
# user_weights = extract_weights('user_embedding', model)
#
# os.chdir('../')
# print(os.getcwd())
# path = os.path.abspath('./data/anime.csv')
# anime_df = pd.read_csv(path)
# anime_df.sort_values('Name', inplace=True)
#
# all_anime_types = set(anime_df['Type'].unique())
#
#
# def check_anime_types(types):
#     types = set([types]) if isinstance(types, str) else set(types)
#     if types.issubset(all_anime_types):
#         return
#     else:
#         for anime_type in types:
#             if anime_type not in all_anime_types:
#                 raise Exception(f'Anime type "{anime_type}" is not valid!')
#
#
# def get_anime_rows(df, anime_query, exact_name=False, types=None):
#     df = df.copy()
#     if isinstance(anime_query, int):
#         df = df[df.MAL_ID == anime_query]
#     else:
#         if exact_name:
#             # get exact name
#             df = df[df.Name == anime_query]
#         else:
#             df = df[df.Name.str.contains(anime_query, case=False, regex=False)]
#
#     if types:
#         check_anime_types(types)
#         df = df[df.Type.isin(types)]
#
#     return df
#
#
# # pd.reset_option('all')
# pd.set_option("max_colwidth", None)
#
#
# def get_recommendation(anime_query, k=10, exact_name=False, types=None):
#     if types:
#         check_anime_types(types)
#     anime_rows = get_anime_rows(anime_df, anime_query,
#                                 exact_name=exact_name)
#     if len(anime_rows) == 0:
#         raise Exception(f'Anime not found for {anime_query}')
#     anime_row = anime_rows.iloc[[0]]
#     anime_id = anime_row.MAL_ID.values[0]
#     anime_name = anime_row.Name.values[0]
#     anime_idx = anime_id_to_idx.get(anime_id)
#
#     weights = anime_weights
#     distances = np.dot(weights, weights[anime_idx])
#
#     sorted_dists_ind = np.argsort(distances)[::-1]
#
#     print(f'Recommending animes for {anime_name}')
#     # display(anime_row.loc[:, 'MAL_ID': 'Aired'])
#
#     anime_list = []
#     # [1:] to skip the first row for anime_query
#     for idx in sorted_dists_ind[1:]:
#         similarity = distances[idx]
#         anime_id = anime_idx_to_id.get(idx)
#         anime_row = anime_df[anime_df.MAL_ID == anime_id]
#         anime_type = anime_row.Type.values[0]
#         if types and anime_type not in types:
#             continue
#         anime_name = anime_row.Name.values[0]
#         score = anime_row.Score.values[0]
#         genre = anime_row.Genres.values[0]
#
#         anime_list.append({"Anime_id": anime_id, "Name": anime_name,
#                            "Similarity": similarity, "Score": score,
#                            "Type": anime_type, "Genre": genre
#                            })
#         if len(anime_list) == k:
#             # enough number of recommendations
#             break
#     rec_df = pd.DataFrame(anime_list)
#     return rec_df
#
#
# print(get_recommendation('Naruto', k=10))