import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

'''
Data Points
player: 10,230 x 89 = 910,740
team: 2,046 x 111 = 227,106
total: 1,137,846
'''

playerdf = pd.read_csv('Data/playerdata') 
teamdf = pd.read_csv('Data/teamdata') 

# vizualize distribution of data first? plot frequency of the result column (W/L) based on team?

list_of_teams = teamdf['teamname'].unique()
list_of_players = playerdf['playername'].unique()

# initialize df's to find w's and l's
team_wl_record = pd.DataFrame(columns=['names', 'w', 'l', 'wr%'])
team_wl_record['names'] = list_of_teams
team_wl_record['w'] = 0
team_wl_record['l'] = 0
team_wl_record['wr%'] = 0.00

player_wl_record = pd.DataFrame(columns=['names', 'w', 'l', 'wr%'])
player_wl_record['names'] = list_of_players
player_wl_record['w'] = 0
player_wl_record['l'] = 0
player_wl_record['wr%'] = 0.00

# define functions (could probably simplify into just one)
def calc_team_wr(result_col):
    for idx in result_col.index:
        team_name = teamdf.loc[idx, 'teamname']
        result = teamdf.loc[idx, 'result']

        team_idx = team_wl_record.index[team_wl_record['names'] == team_name].tolist()[0]

        if result == 1:
            team_wl_record.loc[team_idx, 'w'] += 1
        else:
            team_wl_record.loc[team_idx, 'l'] += 1

    team_wl_record['wr%'] = team_wl_record['w']/(team_wl_record['w']+team_wl_record['l'])

def calc_player_wr(result_col):
    for idx in result_col.index:
        player_name = playerdf.loc[idx, 'playername']
        result = playerdf.loc[idx, 'result']

        player_idx = player_wl_record.index[player_wl_record['names'] == player_name].tolist()[0]

        if result == 1:
            player_wl_record.loc[player_idx, 'w'] += 1
        else:
            player_wl_record.loc[player_idx, 'l'] += 1

    player_wl_record['wr%'] = player_wl_record['w']/(player_wl_record['w']+player_wl_record['l'])

calc_team_wr(teamdf['result'])
calc_player_wr(playerdf['result'])

team_wl_record.head()
player_wl_record.head()
team_wl_record.describe()
player_wl_record.describe()


# check distributions
team_wl_record['wr%'].plot.hist(bins = 7, alpha = 0.5, density = False)
plt.ylabel('Frequency')
plt.xlabel('WR%')
plt.title('Distribution of Team WR%')
plt.show()
player_wl_record['wr%'].plot.hist(bins = 7, alpha = 0.5, density = False)
plt.ylabel('Frequency')
plt.xlabel('WR%')
plt.title('Distribution of Player WR%')
plt.show()

'''
plot corr mx of broader df's where target is result
not using .abs() because a lot of columns are just opposites 
i.e. barons vs opp_barons which have the same corr but in opposite directions

focusing on team performance for now
'''
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

# bottom correlations
target = 'result'
bot_n = 20
corr_matrix = teamdf.corr(numeric_only=True)

bot_corr = corr_matrix[target].sort_values(ascending=True).head(bot_n)
subset = corr_matrix.loc[bot_corr.index, bot_corr.index]

plt.figure(figsize=(12,10))
ax = sns.heatmap(
    subset,
    annot=True,
    cmap='RdBu',
    linewidths=0.5
)

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.title(f'{bot_n} Correlations w/ {target}')
plt.tight_layout()
plt.show()

#split teamdf into other cuts such as tiers (minor/major)
majorleague = teamdf[teamdf['leaguetype'] == 'major'].copy()
minorleague = teamdf[teamdf['leaguetype'] == 'minor'].copy()

target = 'result'
top_n = 20
corr_matrix = majorleague.corr(numeric_only=True)

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
plt.title(f'Top {top_n} Correlations w/ {target} Major League Teams')
plt.tight_layout()

target = 'result'
bot_n = 20
corr_matrix = majorleague.corr(numeric_only=True)

bot_corr = corr_matrix[target].sort_values(ascending=True).head(bot_n)
subset = corr_matrix.loc[bot_corr.index, bot_corr.index]

plt.figure(figsize=(12,10))
ax = sns.heatmap(
    subset,
    annot=True,
    cmap='RdBu',
    linewidths=0.5
)

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.title(f'Bottom {bot_n} Correlations w/ {target} for Major League Teams')
plt.tight_layout()

target = 'result'
top_n = 20
corr_matrix = minorleague.corr(numeric_only=True)

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
plt.title(f'Top {top_n} Correlations w/ {target} for Minor League Teams')
plt.tight_layout()

target = 'result'
bot_n = 20
corr_matrix = minorleague.corr(numeric_only=True)

bot_corr = corr_matrix[target].sort_values(ascending=True).head(bot_n)
subset = corr_matrix.loc[bot_corr.index, bot_corr.index]

plt.figure(figsize=(12,10))
ax = sns.heatmap(
    subset,
    annot=True,
    cmap='RdBu',
    linewidths=0.5
)

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.title(f'Bottom {bot_n} Correlations w/ {target} For Minor League Teams')
plt.tight_layout()
plt.show()