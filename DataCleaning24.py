import pandas as pd
import numpy as np

# import 2024 pro game excel sheet, R:12276 C:124, total data: 1,522,224
df = pd.read_excel('Comprehensive2024Data/2024_LoL_esports_match_data_from_OraclesElixir.xlsx')
print(df.isna().sum().to_string())

# initialize a seperate df for cleaning, also check col names for cleaning context
cleandf = df
for col in cleandf.columns:
    print(col)

# Make sure to make booleans(cols with 1s and 0s) 50/50 before below
substring = 'first'
substring2 = 'result'
substring3 = 'playoffs'
bool_cols = [item for item in cleandf.columns if substring or substring2 or substring3 in item]
print(bool_cols)

for col in cleandf.columns:
    if col in bool_cols:
        cleandf[col] = cleandf[col].fillna(
            pd.Series(np.random.choice([1, 0], size = cleandf[col].isna().sum()),
                index = cleandf[col][cleandf[col].isna()].index)
        )

# begin cleaning data by first replacing remaining nan with either median values for numerical cols or 'missing'
def fill_na_by_type(dataframe):
    for col in dataframe.columns:
        if np.issubdtype(dataframe[col].dtype, np.number):
            dataframe[col].fillna(dataframe[col].median(), inplace = True)
        else:
            dataframe[col].fillna('missing', inplace = True)

fill_na_by_type(cleandf)

# assign major and minor league labels in a new column
def assign_mm_league(league):
    if league in ['LCK', 'LPL', 'LEC', 'LCS']:
        return 'major'
    else:
        return 'minor'
cleandf['leaguetype'] = cleandf['league'].apply(assign_mm_league)

# remove useless cols and then split into multiple df's for analysis
cleandf = cleandf.drop(['datacompleteness', 'url', 'patch', 'year'], axis = 1)

# df for team related analysis

# df for player related analysis


