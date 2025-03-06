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

# Funzione per caricare il dataset
def carica_dataset():
    global df1
    df1 = pd.read_csv('songs.csv')
    print('Dataset caricato con successo!')
    print('Ecco il dataset originale:')
    print("Ecco le prime 5 righe del dataset: \n", df1.head())
    print("Ecco le info del dataset: \n", df1.info())
    print("Ecco la descrizone del dataset: \n", df1.describe())

# Funzione per preparare il dataset
def prepara_dataset():
    global df, dummies, df_subset
    dummies = pd.get_dummies(df1['playlist_genre'])
    df_subset = df1.iloc[:, 4:18]
    df_subset = df_subset.drop(['track_album_release_date', 'playlist_genre'], axis=1)
    df = pd.concat([df_subset, dummies], ignore_index=True)
    df = df.fillna(0)
    print('Dataset preparato con successo!')

# Funzione per creare le classi di popolarità
def crea_classi_popolarita():
    global df
    bins = np.arange(0, 1.1, 0.1)
    df['popularityclass'] = pd.cut(df['track_popularity'], bins=bins, labels=False, include_lowest=True)
    df = df.drop(['track_popularity'], axis=1)
    label_encoder = LabelEncoder()
    df['popularityclass'] = label_encoder.fit_transform(df['popularityclass'])
    print('Classi di popolarità create con successo!')

# Funzione per il preprocessing dei dati
def preprocessing_dati():
    global X, y, X_scaled, X_train, X_test, y_train, y_test
    X = df.drop('popularityclass', axis=1)
    y = df['popularityclass']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print('Preprocessing dei dati completato!')

# Funzione per l'addestramento del modello
def addestra_modello():
    global clf
    best_params = {
        'criterion': 'entropy',
        'max_depth': 10,
        'max_features': 'sqrt',
        'min_samples_leaf': 4,
        'min_samples_split': 2,
        'splitter': 'random',
        'random_state': 42
    }
    clf = DecisionTreeClassifier(**best_params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuratezza = accuracy_score(y_test, y_pred)
    print(f'Accuratezza con DTC dopo GridSearch di: {accuratezza}')

# Funzione per la previsione
def previsione_nuova_canzone():
    new_song = np.array([[0.5, 0.3, 0.7, 0.2, 0.1, 0.4, 0.6, 0.8, 0.9, 0.3, 0.2, 0.5, 0.7, 0.6, 0.4, 0.3, 0.2]])
    scaler = StandardScaler()
    new_song_scaled = scaler.fit_transform(new_song)
    predicted_popularity_class = clf.predict(new_song_scaled)
    print(f'Classe di popolarità prevista per la nuova canzone: {predicted_popularity_class[0]}')

# Funzione per la visualizzazione dei grafici
def visualizza_grafici():
    fig = px.box(df1, x='playlist_genre', y='track_popularity').update_layout(xaxis_title='Generi musicali', yaxis_title='Popolarita')
    fig.show()
    df_dancability_genere = df1.groupby('playlist_genre')['danceability'].mean().sort_values(ascending=False)
    df_dancability_genere.plot(kind='bar', figsize=(12, 8), color='cyan')
    plt.title('Ballabilita per genere')
    plt.xlabel('Genere musicale')
    plt.ylabel('Ballabilita')
    plt.xticks(rotation=90)
    plt.show()

# Menu per la selezione delle funzioni
def menu():
    while True:
        print('\nMenu:')
        print('1. Carica il dataset')
        print('2. Prepara il dataset')
        print('3. Crea classi di popolarità')
        print('4. Preprocessing dei dati')
        print('5. Addestra il modello')
        print('6. Previsione nuova canzone')
        print('7. Visualizza grafici')
        print('0. Esci')
        scelta = input("Seleziona un'opzione: ")
        if scelta == '1':
            carica_dataset()
        elif scelta == '2':
            prepara_dataset()
        elif scelta == '3':
            crea_classi_popolarita()
        elif scelta == '4':
            preprocessing_dati()
        elif scelta == '5':
            addestra_modello()
        elif scelta == '6':
            previsione_nuova_canzone()
        elif scelta == '7':
            visualizza_grafici()
        elif scelta == '0':
            print('Uscita dal programma')
            break
        else:
            print('Scelta non valida')

# Avvio del menu
menu()
