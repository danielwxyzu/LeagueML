import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

'''
Questions:
- How does the data look when split between tiers?
- How does the data look when graphed across time? i.e. seasons?
- What are the average winrates and other key metrics?
- Distribution of teams along winrate (remove teams with less than x games played)
- Natural data experiment: what are +1 stdev above avg on winrate much better at? what are -1 stdev teams much worse at? What doesn't matter?
'''

df = pd.read_csv('cleanedcompdataset.csv')

#descriptive stats
df.describe()

#prelim corr mx
corr_matrix = df.corr(numeric_only = True)
plt.figure(figsize=(12, 10))
ax = sns.heatmap(
    corr_matrix,
    annot=False,
    cmap='coolwarm',
    linewidths=0.5,
    xticklabels=corr_matrix.columns,
    yticklabels=corr_matrix.columns
)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.title('Corr Heatmap')
plt.tight_layout()
plt.show()

#prelim WR distribution
df['Win rate'].plot.hist(bins = 7, alpha = 0.5, density = False)
plt.ylabel('Frequency')
plt.xlabel('WR%')
plt.title('Total Distribution of WR%')
plt.show()

#by tiers
tier1_df = df[df['Tier'] == 1]
tier2_df = df[df['Tier'] == 2]
tier3_df = df[df['Tier'] == 3]

tier1_df['Win rate'].plot.hist(bins = 7, alpha = 0.5)
plt.xlim(0, 1)
plt.ylabel('Frequency')
plt.xlabel('WR%')
plt.title('Distribution of Tier 1 WR%')
plt.show()

tier2_df['Win rate'].plot.hist(bins = 7, alpha = 0.5)
plt.xlim(0, 1)
plt.ylabel('Frequency')
plt.xlabel('WR%')
plt.title('Distribution of Tier 2 WR%')
plt.show()

tier3_df['Win rate'].plot.hist(bins = 7, alpha = 0.5)
plt.xlim(0, 1)
plt.ylabel('Frequency')
plt.xlabel('WR%')
plt.title('Distribution of Tier 3 WR%')
plt.show()

df.columns