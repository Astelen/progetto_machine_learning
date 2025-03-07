import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, homogeneity_score
import joblib

df1 = pd.read_csv('progetto_machine_learning/songs.csv')
# print(df.head())
# print(df.info())
# print(df.describe())
# Verifica i valori unici nella colonna 'playlist_genre'
print(df1['playlist_genre'].unique())
dummies = pd.get_dummies(df1["playlist_genre"])

df_subset = df1.iloc[:, 4:18]
df_subset = df_subset.drop(['track_album_release_date', 'playlist_genre'], axis=1)
# print(df_subset.head())
# print(df_subset.info())

#df = pd.concat([df_subset, dummies], ignore_index=True)
df = pd.concat([df_subset, dummies])
##Bisogna convertire i NaN in 0
df = df.fillna(0)
# print(df.info())
# print(df.head())
# print(df.tail())

# Fare delle classi su popularity
bins = np.arange(0, 1.1, 0.1)
df['popularityclass'] = pd.cut(df['track_popularity'], bins=bins, labels=False, include_lowest=True)

# Rimuovi la colonna 'track_popularity'
df = df.drop(['track_popularity'], axis=1)

# Converti le classi categoriali in etichette numeriche
label_encoder = LabelEncoder()
df['popularityclass'] = label_encoder.fit_transform(df['popularityclass'])

# print(df.info())
# print(df.head())
# print(df.tail())

# Seleziona le colonne appropriate per X e y
X = df.drop('popularityclass', axis=1)
y = df['popularityclass']

print(df.info())
print(df.head())
print(df.tail())

righe_classe_5 = df[df['popularityclass'] == 5]
print(f"Riga 1 con classe 5 {righe_classe_5.iloc[0,:]}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Decision tree model
# clf = DecisionTreeClassifier(random_state = 42)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# #Valuta la performance del modello
# accuratezza = accuracy_score(y_test, y_pred)
# print(f"Accuratezza di: {accuratezza}")
# #Accuratezza 0.62

# #Definisci i parametri da ottimizzare
# param_grid = {
#     'criterion': ['gini', 'entropy'],
#     'splitter': ['best', 'random'],
#     'max_depth': [None, 10, 20, 30, 40, 50],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': [None, 'auto', 'sqrt', 'log2']
# }

# #Inizializza il modello
# clf = DecisionTreeClassifier(random_state=42)

# #Configura GridSearchCV
# grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# #Esegui GridSearchCV
# grid_search.fit(X_train, y_train)

# #Stampa i migliori parametri trovati
# print(f"I migliori parametri trovati sono: {grid_search.best_params_}")

# #Usa il miglior modello trovato per fare predizioni
# best_clf = grid_search.best_estimator_
# y_pred = best_clf.predict(X_test)

# #Valuta la performance del modello
# accuratezza = accuracy_score(y_test, y_pred)
# print(f"Accuratezza di: {accuratezza}")
# #Accuratezza 0.72

#Gradient Boosting model
gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_train, y_train)
predictions = gbc.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: ", accuracy)
#Accuracy 0.67

y_train_pred = gbc.predict(X_train)
y_test_pred = gbc.predict(X_test)

print("Unique predicted classes in training:", np.unique(y_train_pred))
print("Unique predicted classes in test:", np.unique(y_test_pred))

# param_grid_gbc = {
#     'n_estimators': [100, 200, 300],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 4, 5],
#     'subsample': [0.8, 0.9, 1.0],
#     'max_features': ['auto', 'sqrt', 'log2']
# }

# # Inizializza il modello
# gbc = GradientBoostingClassifier(random_state=42)

# # Configura GridSearchCV
# grid_search_gbc = GridSearchCV(estimator=gbc, param_grid=param_grid_gbc, cv=5, n_jobs=-1, verbose=2)

# # Esegui GridSearchCV
# grid_search_gbc.fit(X_train, y_train)

# # Stampa i migliori parametri trovati
# print(f"I migliori parametri trovati sono: {grid_search_gbc.best_params_}")

# # Usa il miglior modello trovato per fare predizioni
# best_gbc = grid_search_gbc.best_estimator_
# y_pred_gbc = best_gbc.predict(X_test)

# # Valuta la performance del modello
# accuratezza_gbc = accuracy_score(y_test, y_pred_gbc)
# print(f"Accuratezza di: {accuratezza_gbc}")
# # Accuratezza 0.72 (0.7166666666666667)




###La Regressione Lineare non e' efficace perche' ho una r2 sempre troppo bassa! Commento tutto.
# # Inizializziamo il modello con parametri di default
# xgb_regressor = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

# # Addestriamo il modello
# xgb_regressor.fit(X_train, y_train)
# y_pred = xgb_regressor.predict(X_test)

# # Calcoliamo il Mean Squared Error (MSE) e R^2 Score
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error: {mse}")
# print(f"R^2 Score: {r2}")
# # Mean Squared Error: 2.241279363632202
# # R^2 Score: 0.12010574340820312

# param_grid = {
#     'n_estimators': [50, 100, 200],  # Numero di alberi
#     'learning_rate': [0.01, 0.1, 0.2],  # Velocità di apprendimento
#     'max_depth': [3, 5, 7],  # Profondità massima degli alberi
#     'subsample': [0.8, 1.0],  # Percentuale di dati usati per ogni albero
#     'colsample_bytree': [0.8, 1.0]  # Percentuale di feature usate per ogni albero
# }

# # Inizializziamo il modello
# xgb_regressor_grid = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

# # Eseguiamo GridSearchCV per trovare la combinazione migliore
# grid_search = GridSearchCV(xgb_regressor_grid, param_grid, cv=3, scoring="r2", n_jobs=-1)
# grid_search.fit(X_train, y_train)

# # Miglior set di parametri trovato
# best_params = grid_search.best_params_

# #Addestramenyo XGBoost
# best_xgb = xgb.XGBRegressor(**best_params, objective="reg:squarederror", random_state=42)
# best_xgb.fit(X_train, y_train)

# # Facciamo previsioni
# y_pred_best = best_xgb.predict(X_test)
# ####Se metto un array dentro X_test posso stimare una canzone futura!

# # Calcoliamo le metriche aggiornate
# mse_best = mean_squared_error(y_test, y_pred_best)
# r2_best = r2_score(y_test, y_pred_best)
# print(f"Mean Squared Error: {mse_best}")
# print(f"R^2 Score: {r2_best}")
# # Mean Squared Error: 1.9250766038894653
# # R^2 Score: 0.24424248933792114

# #Grafico previsioni reali VS previsioni
# plt.figure(figsize=(8,6))
# plt.scatter(y_test, y_pred_best, color="blue", alpha=0.5, label="Previsioni")
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--", color="red", label="Perfetta corrispondenza")
# plt.xlabel("Valori Reali")
# plt.ylabel("Valori Predetti")
# plt.title("XGBoost - Previsioni vs. Valori Reali")
# plt.legend()
# plt.grid(True)
# plt.show()


###PCA per ridurre le dimensioni per vedere se aumenta la precisione
pca_spotify = PCA(n_components=2)
X_pca = pca_spotify.fit_transform(X_scaled)
# plt.figure(figsize=(12, 6))
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=50, cmap='viridis')
# plt.title('Componenti della PCA')
# plt.xlabel('Componente 1')
# plt.ylabel('Componente 2')
# plt.colorbar(label="Classi Iris")
# plt.show()

X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)


#Commentato per non rilanciare la griglia
#Decision tree model
# clf = DecisionTreeClassifier(random_state = 42)
# clf.fit(X_train_pca, y_train)
# y_pred = clf.predict(X_test_pca)

# #Valuta la performance del modello
# accuratezza = accuracy_score(y_test, y_pred)
# print(f"Accuratezza con DTC di: {accuratezza}")
# #Accuratezza 0.62

# #Definisci i parametri da ottimizzare
# param_grid = {
#     'criterion': ['gini', 'entropy'],
#     'splitter': ['best', 'random'],
#     'max_depth': [3, 5, 10],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': [None, 'auto', 'sqrt', 'log2']
# }

# #Inizializza il modello
# clf = DecisionTreeClassifier(random_state=42)

# #Configura GridSearchCV
# grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# #Esegui GridSearchCV
# grid_search.fit(X_train, y_train)

#Stampa i migliori parametri trovati
# print(f"I migliori parametri trovati sono: {grid_search.best_params_}")
# #I migliori parametri trovati sono: {'criterion': 'entropy', 'max_depth': 3, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'splitter': 'random'}

# #Usa il miglior modello trovato per fare predizioni
# best_clf = grid_search.best_estimator_
# y_pred = best_clf.predict(X_test)

# #Valuta la performance del modello
# accuratezza = accuracy_score(y_test, y_pred)
# print(f"Accuratezza con DTC dopo GridSearch di: {accuratezza}")
# #Accuratezza 0.7194444444444444


# ####
# #Gradient Boosting model
# gbc = GradientBoostingClassifier()
# gbc.fit(X_train_pca, y_train)
# predictions = gbc.predict(X_test_pca)

# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy GBC: ", accuracy)
# #Accuracy 0.675
# y_train_pred = gbc.predict(X_train)
# y_test_pred = gbc.predict(X_test)


# param_grid_gbc = {
#     'n_estimators': [100, 200, 300],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 4, 5],
#     'subsample': [0.8, 0.9, 1.0],
#     'max_features': ['auto', 'sqrt', 'log2']
# }

# # Inizializza il modello
# gbc = GradientBoostingClassifier(random_state=42)

# # Configura GridSearchCV
# grid_search_gbc = GridSearchCV(estimator=gbc, param_grid=param_grid_gbc, cv=5, n_jobs=-1, verbose=2)

# # Esegui GridSearchCV
# grid_search_gbc.fit(X_train, y_train)

# # Stampa i migliori parametri trovati
# print(f"I migliori parametri trovati sono: {grid_search_gbc.best_params_}")
# # I migliori parametri trovati sono: {'learning_rate': 0.01, 'max_depth': 3, 'max_features': 'sqrt', 'n_estimators': 100, 'subsample': 0.8}

# # Usa il miglior modello trovato per fare predizioni
# best_gbc = grid_search_gbc.best_estimator_
# y_pred_gbc = best_gbc.predict(X_test)

# # Valuta la performance del modello
# accuratezza_gbc = accuracy_score(y_test, y_pred_gbc)
# print(f"Accuratezza GBC dopo GridSearch di: {accuratezza_gbc}")
# # Accuratezza 0.7166666666666667


##Modello modificato con parametri migliori
#Decision tree model
best_params = {
    'criterion': 'entropy',
    'max_depth': 3,
    'max_features': 'sqrt',
    'min_samples_leaf': 4,
    'min_samples_split': 2,
    'splitter': 'random'
}

# Aggiungi random_state al dizionario best_params
best_params['random_state'] = 42

clf = DecisionTreeClassifier(**best_params)

# Addestra il modello
clf.fit(X_train, y_train)
# Fai le predizioni
y_pred = clf.predict(X_test)

#Valuta la performance del modello
accuratezza = accuracy_score(y_test, y_pred)
print(f"Accuratezza con DTC dopo GridSearch di: {accuratezza}")
#Accuratezza 0.7194444444444444

#Salvo i parametri del modello in un file esterno tramite joblib
joblib.dump(clf, 'decision_tree_model.pkl')

#Input per valutare popolarita' nuove canzoni
# Creiamo un array di input per stimare la popolarità di una nuova canzone
# Assicurati che l'array di input abbia lo stesso numero di caratteristiche e sia scalato
new_song = np.array([[0.5, 0.3, 0.7, 0.2, 0.1, 0.4, 0.6, 0.8, 0.9, 0.3, 0.2, 0.5, 0.7, 0.6, 0.4, 0.3, 0.2]])  # Esempio di input

# Scala l'array di input utilizzando lo stesso StandardScaler
new_song_scaled = scaler.transform(new_song)

# Utilizza il modello addestrato per fare previsioni sull'array di input
predicted_popularity_class = clf.predict(new_song_scaled)

print(f"Classe di popolarità prevista per la nuova canzone: {predicted_popularity_class[0]}")

print(df1.iloc[0,:])

#####Per inserimento con input utente
##Normalizzare tramite fornire media e dev. st.
#Trovo media e dev. st. per ogni colonna, poi credo un array e prendo 
##valore medio per ogni variabile
media_array = np.array([])
devst_array = np.array([])  
for colonna in X.columns:
    media = X[colonna].mean()
    media_array = np.append(media_array, media)
    devst = X[colonna].std()
    devst_array = np.append(devst_array, devst)

print("Media array", media_array)
print("Devst array", devst_array)

##Input con tanti valori quante sono le colonne poi le appendo nell'array (17 input)
lista_titoli_input2 = df.columns[0:17]
#print("Ecco la lista dei titoli2", lista_titoli_input2)
#lista_titoli_input = ["danceability","energy","key","loudness","mode","speechiness","acousticness","liveness","valence","tempo","duration_ms"]

lista_canzone_input = []

#Aggiunta valori della nuova canzone in una lista poi trasformata in array
for titolo in range(len(lista_titoli_input2)):
    valore = float(input(("Inserisci il valore di ", lista_titoli_input2[titolo])))
    lista_canzone_input.append(valore)

print(lista_canzone_input)

#Standardizzazione dei valori
array_canzone_input = np.array(lista_canzone_input)
lista_valore_scaled = []
for valore, media, devst in zip(array_canzone_input, media_array, devst_array):
    valore_scaled = (valore - media) / devst
    lista_valore_scaled.append(valore_scaled)

array_valore_scaled = np.array(lista_valore_scaled).reshape(1, -1)

print(array_valore_scaled)

#### Predirre la popolarita' ma funziona solo nel file dove ho gia' addestrato il modello
# predicted_popularity_class = clf.predict(new_song_scaled)

# print(f"Classe di popolarità prevista per la nuova canzone: {predicted_popularity_class[0]}")

###Valori da inserire in file esterno per prova_modello esportato con joblib
media_array = [3.11697168e-01, 3.20387500e-01, 2.42222222e-01, 3.63362868e-01,
    2.87222222e-01, 5.75038710e-02, 1.12954897e-01, 8.00252125e-02,
    2.59497698e-01, 2.07354403e-01, 1.05686254e+05, 8.88888889e-03,
    6.94444444e-02, 2.37777778e-01, 4.38888889e-02, 1.02222222e-01,
    3.77777778e-02,]
devst_array = [3.37805860e-01, 3.45108749e-01, 3.38480829e-01, 3.71921669e-01,
    4.52591888e-01, 1.10221078e-01, 2.03181785e-01, 1.26751393e-01,
    3.09137652e-01, 2.54380590e-01, 1.10416969e+05, 9.38870259e-02,
    2.54279049e-01, 4.25840640e-01, 2.04904807e-01, 3.03024509e-01,
    1.90711361e-01]

lista_titoli_input = ["danceability (da 0.0 a 1.0) ","energy (da 0.0 a 1.0) ","key (da 0.0 a 1.0) ","loudness (dB, da 0.0 a 1.0) ) ","mode (0 per tonalità minore, 1 per tonalità maggiore) ","speechiness (da 0.0 a 1.0) ","acousticness (da 0.0 a 1.0) ","liveness (da 0.0 a 1.0) ","valence (da 0.0 a 1.0) ","tempo (BPM, da 0.0 a 1.0) ","duration_ms (ms) ", "Edm (0 se non Edm, 1 se Edm) ", "Latin (0 se non Latin, 1 se Latin) ", "Pop (0 se non Pop, 1 se Pop) ", "R&b (0 se non R&b, 1 se R&b) ", "Rap (0 se non Rap, 1 se Rap) ", "Rock (0 se non Rock, 1 se Rock) "]

##Codice per caricare il modello salvato
# Carica il modello salvato
clf = joblib.load('decision_tree_model.pkl')

# Utilizza il modello addestrato per fare previsioni sull'array di input
predicted_popularity_class = clf.predict(array_valore_scaled)

print(f"Classe di popolarità prevista per la nuova canzone: {predicted_popularity_class[0]}")




# Valutare quale caratteristica influenza di piu' la popolarita'
# Esplorazione dei dati

# Box plot 
fig = px.box(df1, x="playlist_genre", y="track_popularity").update_layout(xaxis_title="Generi musicali", yaxis_title="Popolarita'")
fig.show()

#Grafico a barre
df_dancability_genere = df1.groupby("playlist_genre")["danceability"].mean().sort_values(ascending=False)
df_dancability_genere.plot(kind='bar', figsize=(12,8), color='cyan')
plt.title("Ballabilita' per genere")
plt.xlabel('Genere musicale')
plt.ylabel("Ballabilita'")
plt.xticks(rotation=90)
plt.show()

#Grafico a barre
df_energy_genere = df1.groupby("playlist_genre")["energy"].mean().sort_values(ascending=False)
df_energy_genere.plot(kind='bar', figsize=(12,8), color='cyan')
plt.title("Energia per genere")
plt.xlabel('Genere musicale')
plt.ylabel("Energia")
plt.xticks(rotation=90)
plt.show()

#Grafico a barre
df_liveness_genere = df1.groupby("playlist_genre")["liveness"].mean().sort_values(ascending=False)
df_liveness_genere.plot(kind='bar', figsize=(12,8), color='cyan')
plt.title("Liveness per genere")
plt.xlabel('Genere musicale')
plt.ylabel("Liveness")
plt.xticks(rotation=90)
plt.show()

#Grafico a barre
df__valence_genere = df1.groupby("playlist_genre")["valence"].mean().sort_values(ascending=False)
df__valence_genere.plot(kind='bar', figsize=(12,8), color='cyan')
plt.title("Valence per genere")
plt.xlabel('Genere musicale')
plt.ylabel("Valence")
plt.xticks(rotation=90)
plt.show()


# Selezionare solo colonne numeriche e calcolare la matrice di correlazione
corr_matrix = df.select_dtypes(include=['number']).corr()

# Creare una heatmap interattiva con Plotly
fig = px.imshow(corr_matrix,
                text_auto=True, 
                aspect="auto", 
                color_continuous_scale="RdBu",  # Usa una colormap valida
                labels=dict(color="Correlazione"))

fig.update_layout(title="Mappa di Calore Interattiva delle Correlazioni", 
                xaxis_title="Variabili",
                yaxis_title="Variabili")

# Mostrare la heatmap interattiva
fig.show()


# #Pair Plot inutile, non da' informazioni utili. Bocciato.
# # Creare un pair plot per analizzare le relazioni tra variabili numeriche
# plt.figure(figsize=(8, 5))
# sns.pairplot(df.select_dtypes(include=['number']), hue='popularityclass', palette='coolwarm', corner=True)
# plt.show()





import joblib
import numpy as np

media_array = [3.11697168e-01, 3.20387500e-01, 2.42222222e-01, 3.63362868e-01,
    2.87222222e-01, 5.75038710e-02, 1.12954897e-01, 8.00252125e-02,
    2.59497698e-01, 2.07354403e-01, 1.05686254e+05, 8.88888889e-03,
    6.94444444e-02, 2.37777778e-01, 4.38888889e-02, 1.02222222e-01,
    3.77777778e-02,]
devst_array = [3.37805860e-01, 3.45108749e-01, 3.38480829e-01, 3.71921669e-01,
    4.52591888e-01, 1.10221078e-01, 2.03181785e-01, 1.26751393e-01,
    3.09137652e-01, 2.54380590e-01, 1.10416969e+05, 9.38870259e-02,
    2.54279049e-01, 4.25840640e-01, 2.04904807e-01, 3.03024509e-01,
    1.90711361e-01]

lista_titoli_input = ["danceability (da 0.0 a 1.0) ","energy (da 0.0 a 1.0) ","key (da 0.0 a 1.0) ","loudness (dB, da 0.0 a 1.0) ) ","mode (0 per tonalità minore, 1 per tonalità maggiore) ","speechiness (da 0.0 a 1.0) ","acousticness (da 0.0 a 1.0) ","liveness (da 0.0 a 1.0) ","valence (da 0.0 a 1.0) ","tempo (BPM, da 0.0 a 1.0) ","duration_ms (ms) ", "Edm (0 se non Edm, 1 se Edm) ", "Latin (0 se non Latin, 1 se Latin) ", "Pop (0 se non Pop, 1 se Pop) ", "R&b (0 se non R&b, 1 se R&b) ", "Rap (0 se non Rap, 1 se Rap) ", "Rock (0 se non Rock, 1 se Rock) "]

lista_canzone_input = []

#Aggiunta valori della nuova canzone in una lista poi trasformata in array
for titolo in range(len(lista_titoli_input)):
    valore = float(input(("Inserisci il valore di ", lista_titoli_input[titolo])))
    lista_canzone_input.append(valore)

print(lista_canzone_input)

array_valore_scaled = np.array(lista_canzone_input).reshape(1, -1)

# #Standardizzazione dei valori
# array_canzone_input = np.array(lista_canzone_input)
# lista_valore_scaled = []
# for valore, media, devst in zip(array_canzone_input, media_array, devst_array):
#     valore_scaled = (valore - media) / devst
#     lista_valore_scaled.append(valore_scaled)

# array_valore_scaled = np.array(lista_valore_scaled).reshape(1, -1)

print(array_valore_scaled)

##Codice per caricare il modello salvato
# Carica il modello salvato
clf = joblib.load('decision_tree_model.pkl')

# Utilizza il modello addestrato per fare previsioni sull'array di input
predicted_popularity_class = clf.predict(array_valore_scaled)

print(f"Classe di popolarità prevista per la nuova canzone: {predicted_popularity_class[0]}")