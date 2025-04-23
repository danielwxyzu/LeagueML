import pandas as pd
import numpy as np

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

# multikills metric
teamdf['weighted_multikill'] = (
    1 * teamdf['doublekills'] +
    2 * teamdf['triplekills'] +
    3 * teamdf['quadrakills'] +
    4 * teamdf['pentakills']
)

# objective control metric
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
    if substring or substring2 in col:
        print(col)

'''
calculate metric for absolute gamestate @ 15
- find diff between necessary items and then scale all items then add them?
- in order to scale we can do standardization (z-score normalization) 
        - only for lin models that have normally distributed data
- or min-max scaling for distance related models like KNN
- no need to scale for random forest, could be useful to scale the feature anyways
'''

