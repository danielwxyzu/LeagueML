import pandas as pd
import numpy as np

df = pd.read_excel('NewCompData.xlsx')
df.head()
df.shape
df.columns
df.dtypes

cleandf = df.replace('-', np.nan)

cleandf.fillna({
    'Region':'NA',
    'FB%':cleandf['FB%'].median(),
    'FT%':cleandf['FT%'].median(),
    'FOS%':cleandf['FOS%'].median(),
    'VGPG':cleandf['VGPG'].median(),
    'HER%':cleandf['HER%'].median(),
    'ATAKHAN%':cleandf['ATAKHAN%'].median(),
    'DRA@15':cleandf['DRA@15'].median(),
    'TD@15':cleandf['TD@15'].median(),
    'GD@15':cleandf['GD@15'].median(),
    'PPG':cleandf['PPG'].median(),
    'NASHPG':cleandf['NASHPG'].median(),
    'NASH%':cleandf['NASH%'].median(),
    'DPM':cleandf['DPM'].median(),
    'WPM':cleandf['WPM'].median(),
    'VWPM':cleandf['VWPM'].median(),
    'WCPM':cleandf['WCPM'].median()
}, inplace = True)

#create new column splitting teams between tier 1 and teir 2 with outside list of league members
def subtract_list(listone, listtwo):
    return [element for element in listone if element not in listtwo]

all_lpl_teams = list(cleandf.loc[cleandf['Region']=='CN', 'Name'].unique())
all_lck_teams = list(cleandf.loc[cleandf['Region']=='KR', 'Name'].unique())
all_ltan_teams = list(cleandf.loc[cleandf['Region']=='NA', 'Name'].unique())
all_lec_teams = ['Fnatic', 'G2 Esports', 'GIANTX', 'Karmine Corp', 'Movistar KOI', 
                 'Rogue', 'SK Gaming', 'Team BDS', 'Team Heretics', 'Team Vitality', 
                 'MAD Lions KOI', 'Astralis', 'Excel Esports', 'KOI', 'MAD Lions', 'Misfits Gaming',
                 'Excel eSports', 'Misfits', 'Splyce', 'Giants', 'H2K', 'H2K Gaming', 'H2k-Gaming', 'Vitality']

tier1_and_2 = all_ltan_teams+all_lck_teams+all_lec_teams+all_lpl_teams
otherteams = subtract_list(list(cleandf['Name'].unique()), tier1_and_2)

def assign_tier(name):
    if name in all_lck_teams or name in all_lpl_teams:
        return 1
    elif name in all_lec_teams or name in all_ltan_teams:
        return 2
    else:
        return 3
	
cleandf['Tier'] = cleandf['Name'].apply(assign_tier)
cleandf['Season'] = cleandf['Season'].str.replace('S', '', regex = False)
cleandf['Season'] = list(map(int, cleandf['Season']))

dfiltered = cleandf[cleandf['Games'] >= 10]
dfiltered = cleandf[cleandf['Win rate'] > 0]

dfiltered.to_csv('cleanedcompdataset.csv', index=False)