"""
    This file contains functions that are used to generate overview data for the dashboard
"""

import pandas as pd
import plotly.express as px
import numpy as np


def get_studio_contributions(df, year_range):
    """
    Returns the top 7 studios based on the number of anime they have produced in the given year range
    """
    studio_cnts = (df[df['Premiered'].between(*year_range)]
                   .explode('Studios')
                   .groupby('Studios')
                   .size()
                   .reset_index(name='Count')
                   .sort_values('Count', ascending=False)  # Sort to get top
                   .head(7)  # Keep only the top 7 studios
                   )
    studio_cnts['Percent'] = studio_cnts['Count'] / studio_cnts['Count'].sum() * 100
    return studio_cnts


def most_watched_anime_by_studio(studio_names, df, year_range):
    names = []
    watchcount = []
    for studio_name in studio_names:
        filtered_df = df[df['Studios'] == studio_name]
        filtered_df = filtered_df[filtered_df['Premiered'].between(*year_range)]
        if len(filtered_df) == 0:
            continue
        most_watched_anime = filtered_df.sort_values(by='Completed', ascending=False).iloc[0]
        names.append(most_watched_anime['Name'])
        watchcount.append(most_watched_anime['Completed'])
    return {'Anime_names': names, 'watch_counts': watchcount}


def avg_watched_and_popularity_per_year_per_studio(studio_names, df, year_range):
    """
        Studios that will be passes in this function will not have empty watched count
    """
    result = {}
    filtered_df = df[df['Premiered'].between(*year_range)]
    for studio_name in studio_names:
        studio_filtered_df = filtered_df[filtered_df['Studios'] == studio_name]
        if not studio_filtered_df.empty:
            avg_watched_list = []
            avg_popularity_list = []
            for year in range(year_range[0], year_range[1]+1):
                year_filtered = studio_filtered_df[studio_filtered_df['Premiered'] == year]
                if not year_filtered.empty:
                    avg_watched = int(year_filtered['Completed'].mean())
                    avg_popularity = int(year_filtered['Popularity'].mean())
                    avg_watched_list.append(avg_watched)
                    avg_popularity_list.append(avg_popularity)
                else:
                    avg_watched_list.append(0)
                    avg_popularity_list.append(0)
            result[studio_name] = {'avg_watched': avg_watched_list, 'avg_pop': avg_popularity_list}
    return result


# Function to visualize the number of anime for each rating type using an interactive pie chart
def visualize_anime_count_by_rating_pie_interactive(anime_data):
    rating_counts = anime_data['Rating'].value_counts()
    rating_counts_df = pd.DataFrame({'Rating': rating_counts.index, 'Count': rating_counts.values})
    custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    fig = px.pie(rating_counts_df, values='Count', names='Rating', title='Distribution of Anime for Each Rating Type',
                 color_discrete_sequence=custom_colors)
    fig.update_traces(textinfo='percent+label', hoverinfo='label+percent+value')
    return fig

