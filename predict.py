import sys
import pickle
import pandas as pd

# Load trained model
with open("career_model.pkl", "rb") as file:
    model, encoder = pickle.load(file)

# Get user input from Node.js request
personality, interests, work_preferences = sys.argv[1:]

# Encode user input
input_data = pd.DataFrame({
    "personality_type": [personality],
    "interests": [interests],
    "work_preferences": [work_preferences]
})

input_data["personality_encoded"] = encoder.transform(input_data["personality_type"])
input_data["interests_encoded"] = encoder.transform(input_data["interests"])
input_data["work_pref_encoded"] = encoder.transform(input_data["work_preferences"])

X_input = input_data[["personality_encoded", "interests_encoded", "work_pref_encoded"]]

# Make prediction
predicted_career = model.predict(X_input)
career = encoder.inverse_transform(predicted_career)

print(career[0])
