import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import KBinsDiscretizer

# Carica il dataset
df = pd.read_csv("progetto_machine_learning/songs.csv")

# Crea un box plot per la popolarità delle tracce per genere musicale
fig = px.box(df, x="playlist_genre", y="track_popularity").update_layout(xaxis_title="Generi musicali", yaxis_title="Popolarita'")
fig.show()

# Crea un grafico a barre per la ballabilità per genere musicale
df_dancability_genere = df.groupby("playlist_genre")["danceability"].mean().sort_values(ascending=False)
df_dancability_genere.plot(kind='bar', figsize=(12,8), color='cyan')
plt.title("Ballabilita' per genere")
plt.xlabel('Genere musicale')
plt.ylabel("Ballabilita'")
plt.xticks(rotation=90)
plt.show()

# Crea un grafico a barre per l'energia per genere musicale
df_energy_genere = df.groupby("playlist_genre")["energy"].mean().sort_values(ascending=False)
df_energy_genere.plot(kind='bar', figsize=(12,8), color='cyan')
plt.title("Energia per genere")
plt.xlabel('Genere musicale')
plt.ylabel("Energia")
plt.xticks(rotation=90)
plt.show()

# Crea un grafico a barre per la liveness per genere musicale
df_liveness_genere = df.groupby("playlist_genre")["liveness"].mean().sort_values(ascending=False)
df_liveness_genere.plot(kind='bar', figsize=(12,8), color='cyan')
plt.title("Liveness per genere")
plt.xlabel('Genere musicale')
plt.ylabel("Liveness")
plt.xticks(rotation=90)
plt.show()

# Crea un grafico a barre per la valence per genere musicale
df__valence_genere = df.groupby("playlist_genre")["valence"].mean().sort_values(ascending=False)
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

#Select relevant columns
df_subset = df.iloc[:, 4:18].drop(columns=['track_album_release_date', 'playlist_genre'], errors='ignore')

#Drop rows with missing values
df_subset = df_subset.dropna()

#Define the target variable (popularity) and features
X = df_subset.drop(columns=['track_popularity'])
y = df_subset['track_popularity']

#Discretize the target variable into bins
kbins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
y_binned = kbins.fit_transform(y.values.reshape(-1, 1)).astype(int).flatten()

#Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2, random_state=42, stratify=y_binned)

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200,],
    'learning_rate': [ 0.01,0.1,0.2],
    'max_depth': [3,5],
}

#Initialize the Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(random_state=42)

#Perform Grid Search with cross-validation
grid_search = GridSearchCV(gb_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

##Get the best parameters
best_params = grid_search.best_params_

#Train the model with the best parameters
best_gb_clf = GradientBoostingClassifier(**best_params, random_state=42)
best_gb_clf.fit(X_train, y_train)

#Make predictions
y_pred_best = best_gb_clf.predict(X_test)

#Evaluate the model
accuracy_best = accuracy_score(y_test, y_pred_best)
classification_rep_best = classification_report(y_test, y_pred_best)
print("Accuratezza: ", accuracy_best)

# Crea una lista per i titoli delle colonne di input
lista_titoli_input2 = df_subset.columns[1:17]
lista_canzone_input = []

# Richiedi all'utente di inserire i valori per ogni colonna
for titolo in range(len(lista_titoli_input2)):
    valore = float(input(("Inserisci il valore di ", lista_titoli_input2[titolo])))
    lista_canzone_input.append(valore)

print(lista_canzone_input)
array_canzone = np.array(lista_canzone_input).reshape(1, -1)

# Prevedi la classe di popolarità per la canzone di input
predicted_popularity_class = best_gb_clf.predict(array_canzone)
print(f"Classe di popolarità prevista per la nuova canzone: {predicted_popularity_class[0]}")
