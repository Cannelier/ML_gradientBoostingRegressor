# coding: utf-8
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

#valeurs catégoriques renommées en : 'Etudiant' => 0, 'Ouvrier' => 1...}
def encoding(df, col_str):
    col_str_dicts = {}
    for col in col_str: #col = Marque, col = CSP..
        col_str_dicts[col] = {}
        i=0
        for value in df[col].unique():  # {col = "Marque"  => X[col].unique() = [Peugeot, Toyota...]
            col_str_dicts[col].update({value: i})
            i+=1
    for col in col_str_dicts: #pour chaque colonne
        df[col] = df[col].map(col_str_dicts[col]) #remplace les valeurs dans le dataframe
    print("map de remplacement:",col_str_dicts)
    return df

# ----------------- data cleaning + adding features ------------------------
def dataprep(csvfile):
    with open(csvfile, "r+") as file:
        X = pd.read_csv(file, sep=",")
    del X['index']
    #Age et Prime Mensuelle comportent des valeurs NaN : remplacer par moyenne
    X['Age'] = X['Age'].fillna(X['Age'].mean())
    X['Prime mensuelle'] = X['Prime mensuelle'].fillna(X['Prime mensuelle'].mean())
    #features (suppose CRM Bonus Malus et Prime connectées)
    X['CRMxBonMal'] = X['Coefficient bonus malus']*X['Score CRM']*X['Prime mensuelle']
    X = encoding(X, ['Categorie socio professionnelle','Type de vehicule','Marque'])
    return (X)


# ----------- train & test ------------------
csvTrain = "datasets/labeled_dataset_axaggpdsc.csv"
X = dataprep(csvTrain)
Y = pd.Series(X['Benefice net annuel'])
del X['Benefice net annuel']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)
params = {
    'max_depth':3, #chaque arbre a 2 branches
    'n_estimators' : 30, # 100 arbres dans le modèle
    'learning_rate' : 0.5,    #hyperparameter pour éviter d'overfit
    'random_state' : 42,
    'loss' : 'ls'}
gradient = GradientBoostingRegressor(**params)
gradient.fit(X_train, Y_train)
Y_pred = gradient.predict(X_test)
print(mean_squared_error(Y_test, Y_pred))

# --------- Graph interprétation ---------------
# Comparaison test train
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
for i, Y_pred in enumerate(gradient.staged_predict(X_test)):
    test_score[i] = gradient.loss_(Y_test, Y_pred) #Un point pour chaque prédiction Y
plt.figure(figsize=(5, 4))
plt.subplot(1, 1, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, gradient.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
plt.show()


#Poids des variables
feature_importance = gradient.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(6, 4))
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center') #histogrammes décroissants
plt.yticks(pos, X.keys()[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

# ---------------- predire et exporter les valeurs scoring  ----------------
csvScore = "datasets/scoring_dataset_axaggpdsc.csv"
X = dataprep(csvScore)
Y = gradient.predict(X)

with open(csvScore,"r+") as file: #fichier non encodé
    X = pd.read_csv(file,delimiter=",")
X['benefice'] = Y
X.to_csv('result.csv', index=False)