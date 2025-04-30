import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import joblib

df = pd.read_csv('Data/teamdfwithfeatures')
for col in df.columns:
    print(col)

#X = df.drop(columns = ['result', 'leaguetype', 'gameid', 'league',
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
#X = df[['recent_form', 'gamestateat15', 'weighted_objective_diff']]
#X = df[['recent_form', 'weighted_objective_diff', 'gspd', 'earned gpm']]
X = df[['recent_form', 'weighted_objective_diff', 'gspd', 'team kpm']]
#X = df[['recent_form', 'weighted_objective_diff', 'gspd', 'dragonsoul']]
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

'''
Precision: True Positives / (True Positives + False Positives)
Recall: True Positives / (True Positives + False Negatives)
F1 (balance between the above): 2 x (Precision x Recall) / (Precision + Recall)
Support: # of samples in that class
Accuracy: Corret Pred / Total Pred
Macro Avg: Avg of P, R, F1 across classes
Weighted Avg: Macro Avg weighed by # of samples in each class
'''

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': abs(logreg.coef_[0])
}).sort_values(by='coefficient', ascending=False).head(10)
print(feature_importance)

joblib.dump(logreg, 'league_win_predictor.pkl')