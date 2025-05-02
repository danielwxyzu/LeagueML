import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

teamdf = pd.read_csv('Data/teamdata')

# calculate rolling average of most recent 5 games
teamdf['date'] = pd.to_datetime(teamdf['date'])
teamdf['recent_form'] = (
    teamdf.sort_values(['teamname', 'date'])
      .groupby('teamname')['result']
      .transform(lambda x: x.shift(1).rolling(window=5).mean()))
teamdf['recent_form'] = teamdf['recent_form'].fillna(teamdf['result'])

# calculate weighted_objective_diff
teamdf['dragonsoul'] = (teamdf['dragons'] == 4).astype(int)

weights = {
    'dragonsoul': 0.36,  
    'elders': 0.075,
    'barons': 0.62,
    'inhibitors': 0.76,
    'heralds': 0.24,
    'towers': 0.89,
    'turretplates': 0.26,
    'firsttothreetowers': 0.49,
    'firstmidtower': 0.40,
    'firsttower': 0.32,
    'firstbaron': 0.56,
    'firstherald': 0.22
}

# create weighted objective difference
teamdf['weighted_objective_diff'] = (
    (weights['firstbaron'] * teamdf['firstbaron']) +
    (weights['dragonsoul'] * teamdf['dragonsoul']) +
    (weights['elders'] * (teamdf['elders'] - teamdf['opp_elders'])) +
    (weights['barons'] * (teamdf['barons'] - teamdf['opp_barons'])) +
    (weights['towers'] * (teamdf['towers'] - teamdf['opp_towers'])) +
    (weights['inhibitors'] * (teamdf['inhibitors'] - teamdf['opp_inhibitors']))
)

# Select and scale features
features = ['recent_form', 'weighted_objective_diff', 'gspd', 'team kpm']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(teamdf[features])

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Create final dataset
final_df = pd.DataFrame(scaled_features, columns=features)
final_df['result'] = teamdf['result']
final_df['teamname'] = teamdf['teamname']

# Save final dataset
final_df.to_csv('Data/teamdf_model_ready.csv', index=False)


