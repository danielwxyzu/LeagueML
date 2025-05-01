import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

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

temp_binary_columns = ['dragonsoul', 'firsttothreetowers', 'firstmidtower', 'firsttower', 'firstherald', 'firstbaron'] 
continuous_columns = [col for col in teamdf.select_dtypes('number').columns if col not in temp_binary_columns]

scaler = StandardScaler()
scaled_continuous = scaler.fit_transform(teamdf[continuous_columns])

scaled_df = teamdf.copy()
scaled_df[continuous_columns] = scaled_continuous  

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

# create weighted objective difference, those not appearing here were irrelevant to performance
scaled_df['weighted_objective_diff'] = (
    (weights['firstbaron'] * scaled_df['firstbaron']) +
    (weights['dragonsoul'] * scaled_df['dragonsoul']) +
    (weights['elders'] * (scaled_df['elders'] - scaled_df['opp_elders'])) +
    (weights['barons'] * (scaled_df['barons'] - scaled_df['opp_barons'])) +
    (weights['towers'] * (scaled_df['towers'] - scaled_df['opp_towers'])) +
    (weights['inhibitors'] * (scaled_df['inhibitors'] - scaled_df['opp_inhibitors']))
)
teamdf['weighted_objective_diff'] = scaled_df['weighted_objective_diff']

# export into a new csv with new features
teamdf.to_csv('teamdfwithfeatures', index = False)