import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import StandardScaler

# Load processed dataset
df = pd.read_csv('Data/teamdf_model_ready.csv')

# These are the features the model was trained on
features = ['recent_form', 'weighted_objective_diff', 'gspd', 'team kpm']

# Load trained model
model = joblib.load('league_win_predictor.pkl')

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Function to get a team's features
def get_team_features(team_name):
    # Try exact match first
    team_data = df[df['teamname'] == team_name]
    
    # If no exact match, try case-insensitive match
    if team_data.empty:
        team_data = df[df['teamname'].str.lower() == team_name.lower()]
    
    if team_data.empty:
        print(f"Team '{team_name}' not found in the dataset.")
        print("Available teams include:", df['teamname'].unique())
        return None

    # Get the most recent data point
    most_recent = team_data.iloc[-1]
    
    # Get features in the correct order and format
    team_features = pd.DataFrame([most_recent[features]], columns=features)
    return team_features

# Main prediction loop
while True:
    print("\n--- Win Probability Predictor ---")
    team1_name = input("Enter Team 1 name (or 'quit' to exit): ")
    if team1_name.lower() == 'quit':
        break

    team2_name = input("Enter Team 2 name: ")

    team1_feat = get_team_features(team1_name)
    team2_feat = get_team_features(team2_name)

    if team1_feat is None or team2_feat is None:
        continue

    # Calculate feature difference
    matchup_features = pd.DataFrame(team1_feat.values - team2_feat.values, columns=features)

    # Predict and display
    probability = model.predict_proba(matchup_features)[0][1]
    print(f"\n{team1_name} has a {probability * 100:.2f}% chance of beating {team2_name}.\n")
