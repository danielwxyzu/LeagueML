import pandas as pd
import numpy as np
from datetime import datetime

teamdf = pd.read_csv('Data/teamdata')

'''
What features can we add to the df which would provide better insight into w/l probability?
- Rolling averages i.e. most recent 5 games
- Multi kills metric
- Objective control metric
- Gamestate @ 15 or @ 10
'''
for col in teamdf.columns:
    print(col)

# calculate performance of most recent 5 games

fivelatestindex = sorted(range(len(teamdf['date'])), key=lambda i: teamdf['date'][i], reverse=True)[:5]
print(fivelatestindex)



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

