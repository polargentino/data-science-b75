"""
Explicación:

Gráfico de dispersión: Muestra la relación entre danceability (bailabilidad) y 
energy (energía) de las pistas. Puedes observar si existe alguna correlación 
entre estas dos características.
Histograma: Muestra la distribución de la track_popularity (popularidad de las 
pistas). Puedes ver qué tan comunes son las pistas con diferentes niveles de popularidad.
Gráfico de cajas: Compara la track_popularity entre diferentes playlist_genre 
(géneros de playlist). Puedes observar si hay géneros que tienden a tener pistas 
más populares que otros.
Gráfico de lineas: Muestra la relación entre loudness y energy ordenando primero 
los datos por loudness.
Consideraciones:

Este código te da una idea de cómo puedes explorar tu dataset actual.
Puedes adaptar estos ejemplos y explorar otras relaciones entre las columnas de
 tu archivo spotify.csv.
Si encuentras un dataset de Spotify con la información de reproducciones diarias, 
podrás seguir el tutorial original.

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta del archivo spotify.csv en tu directorio de descargas
spotify_filepath = "~/Downloads/spotify.csv"

# Carga del conjunto de datos
spotify_data = pd.read_csv(spotify_filepath)

# Inspección de los datos (opcional)
print(spotify_data.head().to_markdown(index=False, numalign="left", stralign="left"))

# 1. Gráfico de dispersión (Scatter Plot): Relación entre danceability y energy
plt.figure(figsize=(10, 6))
sns.scatterplot(x='danceability', y='energy', data=spotify_data)
plt.title('Relación entre Danceability y Energy')
plt.xlabel('Danceability')
plt.ylabel('Energy')
plt.show()

# 2. Gráfico de distribución (Histograma): Popularidad de las pistas
plt.figure(figsize=(10, 6))
sns.histplot(spotify_data['track_popularity'], bins=20, kde=True)
plt.title('Distribución de la Popularidad de las Pistas')
plt.xlabel('Popularidad de la Pista')
plt.ylabel('Frecuencia')
plt.show()

# 3. Gráfico de cajas (Box Plot): Popularidad por género de playlist
plt.figure(figsize=(12, 6))
sns.boxplot(x='playlist_genre', y='track_popularity', data=spotify_data)
plt.title('Popularidad de las Pistas por Género de Playlist')
plt.xlabel('Género de Playlist')
plt.ylabel('Popularidad de la Pista')
plt.show()

# 4. Relación entre loudness y energy con un grafico de lineas
plt.figure(figsize=(10, 6))
sns.lineplot(x='loudness', y='energy', data=spotify_data.sort_values(by='loudness'))
plt.title('Relación entre Loudness y Energy')
plt.xlabel('Loudness')
plt.ylabel('Energy')
plt.show()

# Salidas: | track_id               | track_name                                            | track_artist     | track_popularity   | track_album_id         | track_album_name                                      | track_album_release_date   | playlist_name   | playlist_id            | playlist_genre   | playlist_subgenre   | danceability   | energy   | key   | loudness   | mode   | speechiness   | acousticness   | instrumentalness   | liveness   | valence   | tempo   | duration_ms   |
# |:-----------------------|:------------------------------------------------------|:-----------------|:-------------------|:-----------------------|:------------------------------------------------------|:---------------------------|:----------------|:-----------------------|:-----------------|:--------------------|:---------------|:---------|:------|:-----------|:-------|:--------------|:---------------|:-------------------|:-----------|:----------|:--------|:--------------|
# | 6f807x0ima9a1j3VPbc7VN | I Don't Care (with Justin Bieber) - Loud Luxury Remix | Ed Sheeran       | 66                 | 2oCs0DGTsRO98Gh5ZSl2Cx | I Don't Care (with Justin Bieber) [Loud Luxury Remix] | 2019-06-14                 | Pop Remix       | 37i9dQZF1DXcZDD7cfEKhW | pop              | dance pop           | 0.748          | 0.916    | 6     | -2.634     | 1      | 0.0583        | 0.102          | 0                  | 0.0653     | 0.518     | 122.036 | 194754        |
# | 0r7CVbZTWZgbTCYdfa2P31 | Memories - Dillon Francis Remix                       | Maroon 5         | 67                 | 63rPSO264uRjW1X5E6cWv6 | Memories (Dillon Francis Remix)                       | 2019-12-13                 | Pop Remix       | 37i9dQZF1DXcZDD7cfEKhW | pop              | dance pop           | 0.726          | 0.815    | 11    | -4.969     | 1      | 0.0373        | 0.0724         | 0.00421            | 0.357      | 0.693     | 99.972  | 162600        |
# | 1z1Hg7Vb0AhHDiEmnDE79l | All the Time - Don Diablo Remix                       | Zara Larsson     | 70                 | 1HoSmj2eLcsrR0vE9gThr4 | All the Time (Don Diablo Remix)                       | 2019-07-05                 | Pop Remix       | 37i9dQZF1DXcZDD7cfEKhW | pop              | dance pop           | 0.675          | 0.931    | 1     | -3.432     | 0      | 0.0742        | 0.0794         | 2.33e-05           | 0.11       | 0.613     | 124.008 | 176616        |
# | 75FpbthrwQmzHlBJLuGdC7 | Call You Mine - Keanu Silva Remix                     | The Chainsmokers | 60                 | 1nqYsOef1yKKuGOVchbsk6 | Call You Mine - The Remixes                           | 2019-07-19                 | Pop Remix       | 37i9dQZF1DXcZDD7cfEKhW | pop              | dance pop           | 0.718          | 0.93     | 7     | -3.778     | 1      | 0.102         | 0.0287         | 9.43e-06           | 0.204      | 0.277     | 121.956 | 169093        |
# | 1e8PAfcKUYoKkxPhrHqw4x | Someone You Loved - Future Humans Remix               | Lewis Capaldi    | 69                 | 7m7vv9wlQ4i0LFuJiE2zsQ | Someone You Loved (Future Humans Remix)               | 2019-03-05                 | Pop Remix       | 37i9dQZF1DXcZDD7cfEKhW | pop              | dance pop           | 0.65           | 0.833    | 1     | -4.672     | 1      | 0.0359        | 0.0803         | 0                  | 0.0833     | 0.725     | 123.976 | 189052        |
