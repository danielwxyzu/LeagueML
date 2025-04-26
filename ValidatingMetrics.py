'''
Seperate python file for the sake of organization and not having to run a large python file
Evaluating linearity assumptions of data for feature engineering purposes
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

teamdf = pd.read_csv('Data/teamdata')

# rolling avg metric
teamdf['date'] = pd.to_datetime(teamdf['date'])
teamdf['recent_form'] = (
    teamdf.sort_values(['teamname', 'date'])
      .groupby('teamname')['result']
      .transform(lambda x: x.shift(1).rolling(window=5).mean()))
teamdf['recent_form'] = teamdf['recent_form'].fillna(teamdf['result'])

# multikills metric (failed linearity test)
teamdf['weighted_multikill'] = (
    1 * teamdf['doublekills'] +
    2 * teamdf['triplekills'] +
    3 * teamdf['quadrakills'] +
    4 * teamdf['pentakills']
)

# objective control metric
# can be improved because I don't think individual dragons matter until a team gets soul or soul point
teamdf['weighted_objective_diff'] = (
    (teamdf['dragons'] - teamdf['opp_dragons']) +
    (teamdf['elders'] - teamdf['opp_elders']) * 2 +
    (teamdf['heralds'] - teamdf['opp_heralds']) +
    (teamdf['barons'] - teamdf['opp_barons']) * 2 +
    (teamdf['towers'] - teamdf['opp_towers']) +
    (teamdf['turretplates'] - teamdf['opp_turretplates']) +
    (teamdf['inhibitors'] - teamdf['opp_inhibitors']) * 2
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


# calculating log odds and checking for linearity with new features
teamdf['bin'] = pd.qcut(teamdf['gamestateat15'], q=10)
bin_stats = teamdf.groupby('bin')['result'].agg(['mean']).rename(columns={'mean': 'win_rate'})
bin_stats['log_odds'] = np.log(bin_stats['win_rate'] / (1 - bin_stats['win_rate']))

bin_centers = teamdf.groupby('bin')['gamestateat15'].mean()
plt.plot(bin_centers, bin_stats['log_odds'], marker='o')
plt.xlabel('Gamestate @ 15 (bin center)')
plt.ylabel('Log-Odds of Win')
plt.title('Linearity Check: Gamestate@15 vs Log-Odds of Win')
plt.grid(True)
plt.show()
#______________________________________________________________________________________________

teamdf['bin'] = pd.qcut(teamdf['gamestateat10'], q=10)
bin_stats = teamdf.groupby('bin')['result'].agg(['mean']).rename(columns={'mean': 'win_rate'})
bin_stats['log_odds'] = np.log(bin_stats['win_rate'] / (1 - bin_stats['win_rate']))

bin_centers = teamdf.groupby('bin')['gamestateat10'].mean()
plt.plot(bin_centers, bin_stats['log_odds'], marker='o')
plt.xlabel('Gamestate @ 10 (bin center)')
plt.ylabel('Log-Odds of Win')
plt.title('Linearity Check: Gamestate@10 vs Log-Odds of Win')
plt.grid(True)
plt.show()
#______________________________________________________________________________________________

teamdf['bin'] = pd.qcut(teamdf['recent_form'], q=10)
bin_stats = teamdf.groupby('bin')['result'].agg(['mean']).rename(columns = {'mean':'win_rate'})
bin_stats['log_odds'] = np.log(bin_stats['win_rate'] / (1 - bin_stats['win_rate']))

bin_centers = teamdf.groupby('bin')['recent_form'].mean()
plt.plot(bin_centers, bin_stats['log_odds'], marker='o')
plt.xlabel('Rolling Avg of Recent 5 Games (bin center)')
plt.ylabel('Log-Odds of Win')
plt.title('Linearity Check: Recent Performance vs Log-Odds of Win')
plt.grid(True)
plt.show()
#______________________________________________________________________________________________

teamdf['bin'] = pd.qcut(teamdf['weighted_multikill'], q=10, duplicates='drop')
bin_stats = teamdf.groupby('bin')['result'].agg(['mean']).rename(columns = {'mean':'win_rate'})
bin_stats['log_odds'] = np.log(bin_stats['win_rate'] / (1 - bin_stats['win_rate']))

bin_centers = teamdf.groupby('bin')['weighted_multikill'].mean()
plt.plot(bin_centers, bin_stats['log_odds'], marker='o')
plt.xlabel('Weighted Multikills (bin center)')
plt.ylabel('Log-Odds of Win')
plt.title('Linearity Check: Weighted Multikill vs Log-Odds of Win')
plt.grid(True)
plt.show()
#______________________________________________________________________________________________

teamdf['bin'] = pd.qcut(teamdf['weighted_objective_diff'], q=10)
bin_stats = teamdf.groupby('bin')['result'].agg(['mean']).rename(columns = {'mean':'win_rate'})
bin_stats['log_odds'] = np.log(bin_stats['win_rate'] / (1 - bin_stats['win_rate']))

bin_centers = teamdf.groupby('bin')['weighted_objective_diff'].mean()
plt.plot(bin_centers, bin_stats['log_odds'], marker='o')
plt.xlabel('Weighted Objective Difference (bin center)')
plt.ylabel('Log-Odds of Win')
plt.title('Linearity Check: Weighted Objective Diff vs Log-Odds of Win')
plt.grid(True)
plt.show()
#______________________________________________________________________________________________
