import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

teamdf = pd.read_csv('Data/teamdata')

'''
What features can we add to the df which would provide better insight into w/l probability?
- Rolling averages i.e. most recent 5 games
- Multi kills metric
- Objective control metric
- Gamestate @ 15 or @ 10

Maybe the multi kills metric can get collapsed into a micro game score 
and the obj control can get collapsed into a macro game score
'''
for col in teamdf.columns:
    print(col)

# calculate rolling average of most recent 5 games
teamdf['date'] = pd.to_datetime(teamdf['date'])
teamdf['recent_form'] = (
    teamdf.sort_values(['teamname', 'date'])
      .groupby('teamname')['result']
      .transform(lambda x: x.shift(1).rolling(window=5).mean()))
teamdf['recent_form'] = teamdf['recent_form'].fillna(teamdf['result'])

'''teamdf['date'] = pd.to_datetime(teamdf['date'])
teamdf = teamdf.sort_values(['teamname', 'date'])

def weighted_rolling_mean(result_series, gspd_series, window=5):
    weighted_avgs = []
    
    for idx in range(len(result_series)):
        start_idx = max(0, idx - window)
        result_window = result_series[start_idx:idx]
        gspd_window = gspd_series[start_idx:idx]
        
        if len(result_window) == 0:
            weighted_avgs.append(np.nan)
        else:
            weights = np.abs(gspd_window) + 1e-6
            weighted_avg = np.average(result_window, weights=weights)
            weighted_avgs.append(weighted_avg)
    
    return pd.Series(weighted_avgs, index=result_series.index)

teamdf['recent_form'] = (
    teamdf.groupby('teamname')
          .apply(lambda group: weighted_rolling_mean(
              group['result'].shift(1),
              group['gspd'].shift(1),
              window=5))
          .reset_index(level=0, drop=True)
)

gspd_max = teamdf['gspd'].abs().max()

def adjusted_result(row):
    scale = row['gspd'] / gspd_max
    return 0.5 + 0.5 * scale

teamdf['recent_form'] = teamdf['recent_form'].fillna(
    teamdf.apply(adjusted_result, axis=1)
)


teamdf['recent_form'].describe()

test = teamdf['result'].corr(teamdf['recent_form'])
print(test)

0.466 with adjusted weights
0.752 without adjusted weights'''

# multikills metric
teamdf['weighted_multikill'] = (
    1 * teamdf['doublekills'] +
    2 * teamdf['triplekills'] +
    3 * teamdf['quadrakills'] +
    4 * teamdf['pentakills']
)

# game state at 10/15
substring = 'at15'
substring2 = 'at10'
for col in teamdf.columns:
    if substring in col or substring2 in col:
        print(col)

cols_to_scale = [
'golddiffat10',
'xpdiffat10',
'csdiffat10',

'killsat10',
'opp_killsat10',

'assistsat10',
'opp_assistsat10',

'deathsat10',
'opp_deathsat10',
#
'golddiffat15',
'xpdiffat15',
'csdiffat15',

'killsat15',
'opp_killsat15',

'assistsat15',
'opp_assistsat15',

'deathsat15',
'opp_deathsat15'
]
scaler = StandardScaler()
scaled = scaler.fit_transform(teamdf[cols_to_scale])

scaled_df = teamdf.copy()
scaled_df[cols_to_scale] = scaled

scaled_df['killsdiffat10'] = scaled_df['killsat10'] - scaled_df['opp_killsat10']
scaled_df['assistdiffat10'] = scaled_df['assistsat10'] - scaled_df['opp_assistsat10']
scaled_df['deathdiffat10'] = scaled_df['opp_deathsat10'] - scaled_df['deathsat10']

scaled_df['killsdiffat15'] = scaled_df['killsat15'] - scaled_df['opp_killsat15']
scaled_df['assistdiffat15'] = scaled_df['assistsat15'] - scaled_df['opp_assistsat15']
scaled_df['deathdiffat15'] = scaled_df['opp_deathsat15'] - scaled_df['deathsat15']

scaled_df[['killsdiffat10', 'assistdiffat10', 'deathdiffat10', 'killsdiffat15', 'assistdiffat15', 'deathdiffat15']].describe

scaled_df['gamestateat10'] = (scaled_df['golddiffat10'] +
                              scaled_df['xpdiffat10'] +
                              scaled_df['csdiffat10'] +
                              scaled_df['killsdiffat10'] +
                              scaled_df['assistdiffat10'] +
                              scaled_df['deathdiffat10'])
scaled_df['gamestateat15'] = (scaled_df['golddiffat15'] +
                              scaled_df['xpdiffat15'] +
                              scaled_df['csdiffat15'] +
                              scaled_df['killsdiffat15'] +
                              scaled_df['assistdiffat15'] +
                              scaled_df['deathdiffat15'])
teamdf['gamestateat10'] = scaled_df['gamestateat10']
teamdf['gamestateat15'] = scaled_df['gamestateat15']


# what if you scaled everything and recalculated? would there be a significant difference?
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

# Create weighted objective difference
scaled_df['weighted_objective_diff'] = (
    #(weights['firstmidtower'] * scaled_df['firstmidtower']) +
    #(weights['firsttothreetowers'] * scaled_df['firsttothreetowers']) +
    #(weights['firsttower'] * scaled_df['firsttower']) +
    (weights['firstbaron'] * scaled_df['firstbaron']) +
    #(weights['firstherald'] * scaled_df['firstherald']) +
    (weights['dragonsoul'] * scaled_df['dragonsoul']) +
    (weights['elders'] * (scaled_df['elders'] - scaled_df['opp_elders'])) +
    (weights['barons'] * (scaled_df['barons'] - scaled_df['opp_barons'])) +
    #(weights['heralds'] * (scaled_df['heralds'] - scaled_df['opp_heralds'])) +
    (weights['towers'] * (scaled_df['towers'] - scaled_df['opp_towers'])) +
    #(weights['turretplates'] * (scaled_df['turretplates'] - scaled_df['opp_turretplates'])) +
    (weights['inhibitors'] * (scaled_df['inhibitors'] - scaled_df['opp_inhibitors']))
)
teamdf['weighted_objective_diff'] = scaled_df['weighted_objective_diff']

# for visualizing corr to adjust weights
selected_columns = ['firstherald', 'firstbaron', 'firsttower', 'dragonsoul', 'firsttothreetowers', 'firstmidtower', 'elders', 'barons', 'heralds', 'towers', 'turretplates', 'inhibitors', 'result']
target = 'result'
top_n = 20
corr_matrix = teamdf[selected_columns].corr(numeric_only=True)

top_corr = corr_matrix[target].sort_values(ascending=False).head(top_n)
subset = corr_matrix.loc[top_corr.index, top_corr.index]

plt.figure(figsize=(12,10))
ax = sns.heatmap(
    subset,
    annot=True,
    cmap='coolwarm',
    linewidths=0.5
)

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.title(f'{top_n} Correlations w/ {target}')
plt.tight_layout()
plt.show()

# for visualizing corr after adding features
target = 'result'
top_n = 20
corr_matrix = teamdf.corr(numeric_only=True)

top_corr = corr_matrix[target].sort_values(ascending=False).head(top_n)
subset = corr_matrix.loc[top_corr.index, top_corr.index]

plt.figure(figsize=(12,10))
ax = sns.heatmap(
    subset,
    annot=True,
    cmap='coolwarm',
    linewidths=0.5
)

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.title(f'{top_n} Correlations w/ {target}')
plt.tight_layout()
plt.show()

# export into a new csv with new features
teamdf.to_csv('teamdfwithfeatures', index = False)