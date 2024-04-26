import dash
import dash_bootstrap_components as dbc
from dash import html
import os
from PIL import Image

imagePath = './assests/anime_background.jpg'

imagePath = os.path.abspath(imagePath)

pil_image = Image.open(imagePath)

layout = html.Div([
    html.Div([
        html.Img(src=pil_image,
                 style={'width': '100%', 'height': '100%', 'position': 'relative', 'z-index': 1}),
        html.H1("AniLens",
                style={'position': 'absolute', 'z-index': 2, 'top': '50%', 'left': '50%',
                       'transform': 'translate(-50%, -50%)', 'color': 'white', 'font-size': '20em'})
    ])
])

