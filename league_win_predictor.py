import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Data/teamdfwithfeatures')

scaled_features = ['recent_form', 'gspd', 'team kpm']

scaler = StandardScaler()
scaler.fit(df[scaled_features])  

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

#_________________________________________________________

df = pd.read_csv('Data/teamdfwithfeatures') 
teamdf = df[['result', 'teamname', 'recent_form', 'weighted_objective_diff', 'gspd', 'team kpm']]

with open('league_win_predictor.pkl', 'rb') as f:
    model = pickle.load(f)

# --- Load your scaler ---
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# --- Define which features need scaling ---
scaled_features = ['recent_form', 'gspd', 'team kpm']  # Features you scaled during training
unscaled_features = ['weighted_objective_diff']        # Feature you should NOT rescale

# --- Function to get a team's features ---
def prepare_features(team_name):
    team_row = teamdf[teamdf['teamname'] == team_name]
    
    if team_row.empty:
        print(f"Team '{team_name}' not found in the data!")
        return None
    
    # Select features
    to_scale = team_row[scaled_features]
    to_leave = team_row[unscaled_features]
    
    # Scale necessary features
    scaled_part = scaler.transform(to_scale)
    
    # Combine scaled + unscaled into a single feature array
    final_features = np.concatenate([scaled_part.flatten(), to_leave.values.flatten()])
    
    return final_features

# --- Main Loop for Prediction ---
while True:
    print("\n--- Win Probability Predictor ---")
    team1_name = input("Enter Team 1 name (or type 'quit' to exit): ")
    if team1_name.lower() == 'quit':
        break
    team2_name = input("Enter Team 2 name: ")

    # Get features for both teams
    team1_features = prepare_features(team1_name)
    team2_features = prepare_features(team2_name)

    if team1_features is None or team2_features is None:
        continue  # If invalid input, restart

    # Create matchup features (team1 - team2)
    matchup_features = team1_features - team2_features

    # Predict probability
    win_probability = model.predict_proba([matchup_features])[0][1]  # probability of class 1 (win)

    # Display result
    print(f"\nPredicted chance of {team1_name} winning against {team2_name}: {win_probability * 100:.2f}%\n")