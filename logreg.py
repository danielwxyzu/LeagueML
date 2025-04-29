import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

df = pd.read_csv('Data/teamdfwithfeatures')
for col in df.columns:
    print(col)

X = df.drop(columns=['result', 'leaguetype', 'gameid', 'league',
'split',
'playoffs',
'date',
'game',
'participantid',
'side',
'position',
'teamname',
'teamid',
'ban1',
'ban2',
'ban3',
'ban4',
'ban5',
'gamelength'])
y = df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_pred_probs = logreg.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred_probs))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': abs(logreg.coef_[0])
}).sort_values(by='coefficient', ascending=False)

print(feature_importance)