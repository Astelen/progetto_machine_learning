import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.decomposition import PCA

# Carica il dataset
df1 = pd.read_csv('songs.csv')

# Crea variabili dummy per la colonna 'playlist_genre'
dummies = pd.get_dummies(df1["playlist_genre"])

# Seleziona un sottoinsieme delle colonne e rimuovi alcune colonne specifiche
df_subset = df1.iloc[:, 4:18]
df_subset = df_subset.drop(['track_album_release_date', 'playlist_genre'], axis=1)

# Combina il sottoinsieme del dataframe con le variabili dummy
df = pd.concat([df_subset, dummies], ignore_index=True)

# Sostituisci i valori NaN con 0
df = df.fillna(0)

# Crea classi di popolarità basate su intervalli di valori
bins = np.arange(0, 1.1, 0.1)
df['popularityclass'] = pd.cut(df['track_popularity'], bins=bins, labels=False, include_lowest=True)

# Rimuovi la colonna 'track_popularity'
df = df.drop(['track_popularity'], axis=1)

# Converti le classi categoriali in etichette numeriche
label_encoder = LabelEncoder()
df['popularityclass'] = label_encoder.fit_transform(df['popularityclass'])

# Seleziona le colonne appropriate per X e y
X = df.drop('popularityclass', axis=1)
y = df['popularityclass']

# Scala le caratteristiche
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividi il dataset in set di addestramento e di test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Applica PCA per ridurre la dimensionalità
pca_spotify = PCA(n_components=2)
X_pca = pca_spotify.fit_transform(X_scaled)
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Definisci i migliori parametri trovati tramite GridSearchCV
best_params = {
    'criterion': 'entropy',
    'max_depth': 10,
    'max_features': 'sqrt',
    'min_samples_leaf': 4,
    'min_samples_split': 2,
    'splitter': 'random',
    'random_state': 42
}

# Inizializza e addestra il modello DecisionTreeClassifier con i migliori parametri
clf = DecisionTreeClassifier(**best_params)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Valuta la performance del modello
accuratezza = accuracy_score(y_test, y_pred)
print(f"Accuratezza con DTC dopo GridSearch di: {accuratezza}")

# Prevedi la classe di popolarità per una nuova canzone
new_song = np.array([[0.5, 0.3, 0.7, 0.2, 0.1, 0.4, 0.6, 0.8, 0.9, 0.3, 0.2, 0.5, 0.7, 0.6, 0.4, 0.3, 0.2]])
new_song_scaled = scaler.transform(new_song)
predicted_popularity_class = clf.predict(new_song_scaled)
print(f"Classe di popolarità prevista per la nuova canzone: {predicted_popularity_class[0]}")

# Calcola la media e la deviazione standard per ogni colonna
media_array = np.array([])
devst_array = np.array([])  
for colonna in X.columns:
    media = X[colonna].mean()
    media_array = np.append(media_array, media)
    devst = X[colonna].std()
    devst_array = np.append(devst_array, devst)

print("Media array", media_array)
print("Devst array", devst_array)

# Crea una lista per i titoli delle colonne di input
lista_titoli_input2 = df.columns[0:17]
lista_canzone_input = []

# Richiedi all'utente di inserire i valori per ogni colonna
for titolo in range(len(lista_titoli_input2)):
    valore = float(input(("Inserisci il valore di ", lista_titoli_input2[titolo])))
    lista_canzone_input.append(valore)

print(lista_canzone_input)

# Scala i valori di input
array_canzone_input = np.array(lista_canzone_input)
lista_valore_scaled = []
for valore, media, devst in zip(array_canzone_input, media_array, devst_array):
    valore_scaled = (valore - media) / devst
    lista_valore_scaled.append(valore_scaled)

array_valore_scaled = np.array(lista_valore_scaled).reshape(1, -1)
print(array_valore_scaled)

# Prevedi la classe di popolarità per la canzone di input
predicted_popularity_class = clf.predict(new_song_scaled)
print(f"Classe di popolarità prevista per la nuova canzone: {predicted_popularity_class[0]}")

# Crea un box plot per la popolarità delle tracce per genere musicale
fig = px.box(df1, x="playlist_genre", y="track_popularity").update_layout(xaxis_title="Generi musicali", yaxis_title="Popolarita'")
fig.show()

# Crea un grafico a barre per la ballabilità per genere musicale
df_dancability_genere = df1.groupby("playlist_genre")["danceability"].mean().sort_values(ascending=False)
df_dancability_genere.plot(kind='bar', figsize=(12,8), color='cyan')
plt.title("Ballabilita' per genere")
plt.xlabel('Genere musicale')
plt.ylabel("Ballabilita'")
plt.xticks(rotation=90)
plt.show()

# Crea un grafico a barre per l'energia per genere musicale
df_energy_genere = df1.groupby("playlist_genre")["energy"].mean().sort_values(ascending=False)
df_energy_genere.plot(kind='bar', figsize=(12,8), color='cyan')
plt.title("Energia per genere")
plt.xlabel('Genere musicale')
plt.ylabel("Energia")
plt.xticks(rotation=90)
plt.show()

# Crea un grafico a barre per la liveness per genere musicale
df_liveness_genere = df1.groupby("playlist_genre")["liveness"].mean().sort_values(ascending=False)
df_liveness_genere.plot(kind='bar', figsize=(12,8), color='cyan')
plt.title("Liveness per genere")
plt.xlabel('Genere musicale')
plt.ylabel("Liveness")
plt.xticks(rotation=90)
plt.show()

# Crea un grafico a barre per la valence per genere musicale
df__valence_genere = df1.groupby("playlist_genre")["valence"].mean().sort_values(ascending=False)
df__valence_genere.plot(kind='bar', figsize=(12,8), color='cyan')
plt.title("Valence per genere")
plt.xlabel('Genere musicale')
plt.ylabel("Valence")
plt.xticks(rotation=90)
plt.show()

# Crea una mappa di calore delle correlazioni tra le variabili numeriche
corr_matrix = df.select_dtypes(include=['number']).corr()
fig = px.imshow(corr_matrix,
                text_auto=True, 
                aspect="auto", 
                color_continuous_scale="RdBu",  
                labels=dict(color="Correlazione"))

fig.update_layout(title="Mappa di Calore Interattiva delle Correlazioni", 
                xaxis_title="Variabili",
                yaxis_title="Variabili")

fig.show()