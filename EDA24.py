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
# use .loc to select count the times a team or player has won 

list_of_teams = teamdf['teamname'].unique()
list_of_players = playerdf['playername'].unique()

def count_wins(df):
    w_count = 0
    l_count = 0
    for i in df.columns:
        return