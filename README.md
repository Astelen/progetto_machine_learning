Progetto Music Popularity Prediction

Questo progetto è stato sviluppato da un team di 3 persone, con l'obiettivo di []. Ogni membro del team ha contribuito a una parte specifica del codice.

Team Members and Contributions
[Biagio Saccone] – Pulizia finale del codice
Biagio ha gestito la pulizia finale del codice, ottimizzando le funzioni e migliorando la leggibilità. Ha assicurato che il codice fosse ben strutturato e privo di errori, mantenendo una logica chiara e una buona gestione delle variabili globali. Si è occupato di testare e debug le funzioni per garantire che il flusso del programma fosse coerente

[Federica Acciarino] – Creazione del menu
Federica ha progettato e sviluppato un'interfaccia utente basata su menu, dove l'utente può selezionare diverse funzionalità, come caricare il dataset, preparare i dati, addestrare il modello e visualizzare grafici. Il menu consente una navigazione fluida attraverso le operazioni del programma, facilitando l'interazione dell'utente.

[Sara Corsetti] – Esportazione del modello con Joblib
Sara si è occupata della parte di machine learning, implementando un modello di Gradient Boosting per prevedere la popolarità delle canzoni. Dopo l'addestramento del modello, ha gestito l'esportazione del modello tramite Joblib, permettendo di salvare il modello addestrato in un file che può essere riutilizzato senza la necessità di riaddestrarlo.

Features
Caricamento del Dataset: Caricamento e visualizzazione delle prime righe del dataset musicale.
Preprocessing dei Dati: Preparazione dei dati con l'eliminazione di valori nulli e la selezione delle colonne pertinenti.
Creazione delle Classi di Popolarità: Creazione di classi per la popolarità delle canzoni, segmentando il target in base a range di valori.
Addestramento del Modello: Utilizzo del modello di Gradient Boosting per predire la popolarità delle canzoni.
Previsione della Popolarità per una Nuova Canzone: Input manuale delle caratteristiche di una canzone per ottenere la previsione della sua popolarità.
Visualizzazione Grafica: Possibilità di visualizzare vari grafici interattivi e statici per esplorare le relazioni tra le variabili.

Tecnologie utilizzate
Python 3.x
Pandas: Per la gestione dei dati.
NumPy: Per operazioni matematiche.
Scikit-learn: Per la creazione del modello di machine learning (Gradient Boosting).
Joblib: Per esportare il modello.
Matplotlib & Plotly: Per la visualizzazione dei grafici.
Seaborn: Per la creazione di pairplot e altre visualizzazioni.

Notes
Il progetto è stato sviluppato utilizzando un approccio di lavoro in team, con ognuno dei membri responsabile per una parte specifica.
Se riscontri bug o hai domande, apri una issue nel repository.

Author
[Biagio Saccone, Federica Acciarino, Sara Corsetti]
