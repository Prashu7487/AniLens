#  Premiered Popularity Name Studios
import pandas as pd
import os

data_path = os.path.abspath('../data/anime_cleaned.csv')
pop_year = pd.read_csv(data_path)
pop_year['Premiered'] = pop_year['Premiered'].str.split().str[-1].astype(int)
# pop_year = pop_year[['Name', 'Premiered', 'Popularity']]
# pop_year['Premiered'] = pop_year['Premiered'].str.split().str[-1].astype(int)
print(pop_year.columns)
# print(pop_year['Premiered'][2:10])

# anime_per_genre = pop_year['Genres'].str.split(', ').explode().value_counts()
# print(anime_per_genre.head())

