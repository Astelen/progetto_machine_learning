import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Funzione per caricare il dataset
def carica_dataset():
    global df
    df = pd.read_csv('progetto_machine_learning/songs.csv')
    print('Dataset caricato con successo!')
    print('Ecco il dataset originale:')
    print("Ecco le prime 5 righe del dataset: \n", df.head())
    print("Ecco le info del dataset: \n", df.info())
    print("Ecco la descrizone del dataset: \n", df.describe())

# Funzione per preparare il dataset
def prepara_dataset():
    global df_subset
    df_subset = df.iloc[:, 4:18].drop(columns=['track_album_release_date', 'playlist_genre'], errors='ignore')
    df_subset = df_subset.dropna()
    print("Ecco le prime 5 righe del dataset preparato: ", df_subset.head())
    print('Dataset preparato con successo!')

# Funzione per creare le classi di popolarità
def crea_classi_popolarita():
    global X, y
    X = df_subset.drop(columns=['track_popularity'])
    y = df_subset['track_popularity']
    print("Classe popolarità: ", y)
    print('Classi di popolarità create con successo!')

# Funzione per il preprocessing dei dati
def preprocessing_dati():
    global X_train, X_test, y_train, y_test, y_binned
    kbins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    y_binned = kbins.fit_transform(y.values.reshape(-1, 1)).astype(int).flatten()
    X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2, random_state=42, stratify=y_binned)

# Funzione per l'addestramento del modello
def addestra_modello():
    global gb_clf, best_gb_clf
    param_grid = {
        'n_estimators': [50, 100, 200,],
        'learning_rate': [ 0.01,0.1,0.2],
        'max_depth': [3,5],
    }
    gb_clf = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(gb_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_gb_clf = GradientBoostingClassifier(**best_params, random_state=42)
    best_gb_clf.fit(X_train, y_train)
    y_pred_best = best_gb_clf.predict(X_test)
    accuracy_best = accuracy_score(y_test, y_pred_best)
    classification_rep_best = classification_report(y_test, y_pred_best)
    print("Accuratezza: ", accuracy_best)
    print("Classification report: ", classification_rep_best)

# Funzione per la previsione
def previsione_nuova_canzone():
    best_gb_clf = joblib.load('progetto_machine_learning/GradientBoostingClassifier_model.pkl')
    lista_titoli_input2 = ["danceability (da 0.0 a 1.0) ","energy (da 0.0 a 1.0) ","key (da 0.0 a 1.0) ","loudness (dB, da 0.0 a 1.0) ) ","mode (0 per tonalità minore, 1 per tonalità maggiore) ","speechiness (da 0.0 a 1.0) ","acousticness (da 0.0 a 1.0) ","liveness (da 0.0 a 1.0) ","valence (da 0.0 a 1.0) ","tempo (BPM, da 0.0 a 1.0) ","duration_ms (ms)"]
    lista_canzone_input = []
    for titolo in range(len(lista_titoli_input2)):
        valore = float(input(("Inserisci il valore di ", lista_titoli_input2[titolo])))
        lista_canzone_input.append(valore)
    array_canzone = np.array(lista_canzone_input).reshape(1, -1)
    predicted_popularity_class = best_gb_clf.predict(array_canzone)
    print(f"Classe di popolarità prevista per la nuova canzone: {predicted_popularity_class[0]}")

# Funzione per la visualizzazione dei grafici
def visualizza_grafici():
    while True:
        print('1. Matrice di correlazione')
        print('2. Istogramma tra ballabilità e genere')
        print('3. Istogramma tra energia e genere')
        print('4. Istogramma tra possibilità di performance live e genere')
        print('5. Istogramma tra positività e genere')
        print('6. Boxplot sulla popolarità dei generi musicali')
        print('7. Pairplot tra tutte le features')
        print('8. Esci dal menù grafici')
        scelta_grafico = input()
        if scelta_grafico == "1":
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
            continue
        elif scelta_grafico == "2":
            df_dancability_genere = df.groupby("playlist_genre")["danceability"].mean().sort_values(ascending=False)
            df_dancability_genere.plot(kind='bar', figsize=(12,8), color='cyan')
            plt.title("Ballabilita' per genere")
            plt.xlabel('Genere musicale')
            plt.ylabel("Ballabilita'")
            plt.xticks(rotation=90)
            plt.show()
            continue
        elif scelta_grafico == "3":
            df_energy_genere = df.groupby("playlist_genre")["energy"].mean().sort_values(ascending=False)
            df_energy_genere.plot(kind='bar', figsize=(12,8), color='cyan')
            plt.title("Energia per genere")
            plt.xlabel('Genere musicale')
            plt.ylabel("Energia")
            plt.xticks(rotation=90)
            plt.show()
            continue
        elif scelta_grafico == "4":
            df_liveness_genere = df.groupby("playlist_genre")["liveness"].mean().sort_values(ascending=False)
            df_liveness_genere.plot(kind='bar', figsize=(12,8), color='cyan')
            plt.title("Possibilità di performance live per genere")
            plt.xlabel('Genere musicale')
            plt.ylabel("Possibilità di performance live")
            plt.xticks(rotation=90)
            plt.show()
            continue
        elif scelta_grafico == "5":
            df__valence_genere = df.groupby("playlist_genre")["valence"].mean().sort_values(ascending=False)
            df__valence_genere.plot(kind='bar', figsize=(12,8), color='cyan')
            plt.title("Positività per genere")
            plt.xlabel('Genere musicale')
            plt.ylabel("Positività")
            plt.xticks(rotation=90)
            plt.show()
            continue
        elif scelta_grafico == "6":
            fig = px.box(df, x="playlist_genre", y="track_popularity").update_layout(xaxis_title="Generi musicali", yaxis_title="Popolarità")
            fig.show()
            continue
        elif scelta_grafico == "7":
            numeric_df = df.select_dtypes(include=['number'])
            if 'track_popularity' not in numeric_df.columns:
                numeric_df['track_popularity'] = df['track_popularity']
            plt.figure(figsize=(8, 5))
            sns.pairplot(numeric_df, hue='track_popularity', palette='coolwarm', corner=True)
            plt.show()
        elif scelta_grafico == "8":
            print("Prosegui con le tue scelte.")
            break
        else:
            print("Scelta non valida.")

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
