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
for col in teamdf.columns:
    print(col)
