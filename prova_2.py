from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

#Load the dataset
file_path = "progetto_machine_learning/songs.csv"
df = pd.read_csv(file_path)

#Select relevant columns
df_subset = df.iloc[:, 4:18].drop(columns=['track_album_release_date', 'playlist_genre'], errors='ignore')

#Drop rows with missing values
df_subset = df_subset.dropna()

#Define the target variable (popularity) and features
X = df_subset.drop(columns=['track_popularity'])
y = df_subset['track_popularity']

#Discretize the target variable into bins
kbins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
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

print("Accuracy: ", accuracy_best)


