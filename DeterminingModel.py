'''
Seperate python file for the sake of organization and not having to run a large python file
Evaluating linearity assumptions of data for feature engineering purposes

To validate:
- Multi kills metric
- Objective control metric
- Gamestate @ 15 or @ 10
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

teamdf = pd.read_csv('Data/teamdata')

# rolling avg metric
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

fig, axs = plt.subplots(3, 1, figsize=(8, 10))

axs[0].scatter(teamdf['result'], teamdf['recent_form'], color='blue')
axs[0].set_title('Recent Form vs Result')
axs[0].set_xticks([0, 1])

axs[1].scatter(teamdf['result'], teamdf['weighted_multikill'], color='green')
axs[1].set_title('Multi Kill vs Result')
axs[1].set_xticks([0, 1])

axs[2].scatter(teamdf['result'], teamdf['weighted_objective_diff'], color='red')
axs[2].set_title('Obj Diff vs Result')
axs[2].set_xticks([0, 1])

plt.tight_layout()
plt.show()

# only weighted obj diff metric passes linearity assumption

