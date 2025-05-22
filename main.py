from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load data
df = pd.read_csv("destinations_enhanced.csv")
for col in ['type', 'budget', 'season', 'activities', 'State', 'Country']:
    df[col] = df[col].fillna("")

df['combined_features'] = df['type'] + " " + df['budget'] + " " + df['season'] + " " + df['activities'] + " " + df['State'] + " " + df['Country']

# TF-IDF vectorizer
vectorizer = TfidfVectorizer()
feature_matrix = vectorizer.fit_transform(df['combined_features'])

class Input(BaseModel):
    trip_type: Optional[str] = ""
    budget: Optional[str] = ""
    season: Optional[str] = ""
    activities: Optional[List[str]] = []
    state: Optional[str] = ""
    country: Optional[str] = ""

@app.post("/recommend")
def recommend_places(input: Input):
    parts = [input.trip_type, input.budget, input.season] + input.activities
    parts = [p for p in parts if p]
    query = " ".join(parts)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, feature_matrix)[0]

    results = []
    for i, sim in enumerate(similarities):
        boost = 0
        if input.state.lower() == df.iloc[i]['State'].lower(): boost += 0.2
        if input.country.lower() == df.iloc[i]['Country'].lower(): boost += 0.1
        results.append((i, sim + boost))

    top = sorted(results, key=lambda x: x[1], reverse=True)[:10]
    output = []
    for i, score in top:
        row = df.iloc[i]
        output.append({
            "Place": row['Place'],
            "State": row['State'],
            "Country": row['Country'],
            "Type": row['type'],
            "Budget": row['budget'],
            "Season": row['season'],
            "Activities": row['activities'],
            "Score": round(score, 2)
        })
    return output
