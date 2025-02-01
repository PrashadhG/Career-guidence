import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Sample dataset
data = {
    "personality_type": ["Introvert", "Extrovert", "Ambivert", "Introvert", "Extrovert"],
    "interests": ["Tech", "Art", "Business", "Science", "Tech"],
    "work_preferences": ["Remote", "Team", "Flexible", "Solo", "Team"],
    "career": ["Software Engineer", "Graphic Designer", "Business Analyst", "Scientist", "Data Analyst"]
}

df = pd.DataFrame(data)

# Encode categorical data
personality_encoder = LabelEncoder()
interests_encoder = LabelEncoder()
work_pref_encoder = LabelEncoder()
career_encoder = LabelEncoder()

df["personality_encoded"] = personality_encoder.fit_transform(df["personality_type"])
df["interests_encoded"] = interests_encoder.fit_transform(df["interests"])
df["work_pref_encoded"] = work_pref_encoder.fit_transform(df["work_preferences"])
df["career_encoded"] = career_encoder.fit_transform(df["career"])

X = df[["personality_encoded", "interests_encoded", "work_pref_encoded"]]
y = df["career_encoded"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and encoders
with open("career_model.pkl", "wb") as file:
    pickle.dump((model, personality_encoder, interests_encoder, work_pref_encoder, career_encoder), file)

print("Model Trained & Saved Successfully!")
