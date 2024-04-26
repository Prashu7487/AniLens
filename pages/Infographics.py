import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc

# Load the data from the CSV file
df = pd.read_csv("data/anime.csv")

# Explode the 'Genres' column to split multiple genres into individual rows
df['Genres'] = df['Genres'].str.split(', ')
df_exploded = df.explode('Genres')

# Group by genre and count occurrences, then sort by count
top_genres = df_exploded['Genres'].value_counts().nlargest(10)

# Create a dataframe from the genre counts
genre_counts_df = pd.DataFrame({'Genre': top_genres.index, 'Count': top_genres.values})

# Calculate quartiles and interquartile range for outlier detection in 'Members'
Q1 = df['Members'].quantile(0.25)
Q3 = df['Members'].quantile(0.75)
IQR = Q3 - Q1

# Filter outliers
filter_condition = (df['Members'] >= Q1 - 1.5 * IQR) & (df['Members'] <= Q3 + 1.5 * IQR)
df_filtered = df.loc[filter_condition]

df = df[df['Score'] != 'Unknown']
# print(df['Score'])
# Create scatter plot of 'Score' versus 'Members' with a best-fitted line
scatter_fig = px.scatter(df, x='Members', y='Score', title='Scatter Plot of Score vs Community Members')

# Create pie chart for distribution of average favorites among top anime sources
source_favorites = df.groupby('Source')['Favorites'].mean().reset_index()
top_sources = source_favorites.sort_values(by='Favorites', ascending=False).head(10)
pie_fig = px.pie(top_sources, values='Favorites', names='Source',
                 title='Distribution of Average Favorites Among Top Anime Sources')

# Create box plot for 'Score' by 'Type'
box_fig = px.box(df, x='Type', y='Score', title='Box Plot of Score by Type', points=False)

# Create box plot for 'Members' by 'Type'
box_members_fig = px.box(df_filtered, x='Type', y='Members', title='Box Plot of Members by Type')

# Create treemap visualization for top 10 anime genres
treemap_fig = px.treemap(genre_counts_df, path=['Genre'], values='Count', title='Top 10 Anime Genres')


# Defining the  layout of this page with margins and text
layout = dbc.Container([
    html.H3("These static graphs are for analysis purposes."
            " Each graph reveals common patterns in the data or provides insights about the user behavior.",
            style={'marginTop': '50px','marginBottom': '100px', 'textAlign': 'center'}),
    dbc.Row([
        dbc.Col(html.Div(dcc.Graph(figure=pie_fig)), width=6),
        dbc.Col(html.Div(dcc.Graph(figure=box_fig)), width=6),
    ], style={'margin-bottom': '80px'}),
    dbc.Row([
        dbc.Col(html.Div(dcc.Graph(figure=box_members_fig)), width=6),
        dbc.Col(html.Div(dcc.Graph(figure=treemap_fig)), width=6),
    ], style={'margin-bottom': '80px'}),
    dbc.Row([
        dbc.Col(html.Div(dcc.Graph(figure=scatter_fig)), width=12),
    ], style={'margin-bottom': '200px'}),
])
