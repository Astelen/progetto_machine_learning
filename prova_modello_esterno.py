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

#Standardizzazione dei valori
array_canzone_input = np.array(lista_canzone_input)
lista_valore_scaled = []
for valore, media, devst in zip(array_canzone_input, media_array, devst_array):
    valore_scaled = (valore - media) / devst
    lista_valore_scaled.append(valore_scaled)

array_valore_scaled = np.array(lista_valore_scaled).reshape(1, -1)

print(array_valore_scaled)

##Codice per caricare il modello salvato
# Carica il modello salvato
clf = joblib.load('decision_tree_model.pkl')

# Utilizza il modello addestrato per fare previsioni sull'array di input
predicted_popularity_class = clf.predict(array_valore_scaled)

print(f"Classe di popolarità prevista per la nuova canzone: {predicted_popularity_class[0]}")