from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from functools import lru_cache
import torch

app = FastAPI()

# Load and preprocess the dataset
df = pd.read_csv("destinations_enhanced_global.csv")
for col in ['type', 'budget', 'season', 'activities', 'State', 'Country']:
    df[col] = df[col].fillna("")

df['combined_features'] = (
    df['type'] + " " + df['budget'] + " " + df['season'] + " " +
    df['activities'] + " " + df['State'] + " " + df['Country']
)

# Set device (CPU or GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load sentence embedding model with explicit device
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

@lru_cache()
def get_embeddings():
    return model.encode(df['combined_features'].tolist(), convert_to_tensor=True)

embeddings = get_embeddings()

# Input schema
class Input(BaseModel):
    trip_type: Optional[str] = ""
    budget: Optional[str] = ""
    season: Optional[str] = ""
    activities: Optional[List[str]] = []
    state: Optional[str] = ""
    country: Optional[str] = ""

@app.post("/recommend")
def recommend(input: Input):
    # Ensure at least one field is provided
    if not any([input.trip_type, input.budget, input.season, input.activities, input.state, input.country]):
        raise HTTPException(status_code=400, detail="At least one input field must be provided.")

    # Normalize input
    query_parts = [
        input.trip_type.strip().lower(),
        input.budget.strip().lower(),
        input.season.strip().lower()
    ] + [a.strip().lower() for a in input.activities]

    query = " ".join([p for p in query_parts if p])
    query_embedding = model.encode(query, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, embeddings)[0]

    results = []
    for i, score in enumerate(scores):
        row = df.iloc[i]

        # Apply exact matching for state and country
        if input.state and row['State'].strip().lower() != input.state.strip().lower():
            continue
        if input.country and row['Country'].strip().lower() != input.country.strip().lower():
            continue

        # Match type if provided
        if input.trip_type and row['type'].strip().lower() != input.trip_type.strip().lower():
            continue

        final_score = score.item()

        # Add weight boosts
        if input.trip_type and row['type'].strip().lower() == input.trip_type.strip().lower():
            final_score += 0.2
        if input.season and row['season'].strip().lower() == input.season.strip().lower():
            final_score += 0.1
        if input.budget and row['budget'].strip().lower() == input.budget.strip().lower():
            final_score += 0.1

        results.append({
            "Place": row['Place'],
            "State": row['State'],
            "Country": row['Country'],
            "Type": row['type'],
            "Budget": row['budget'],
            "Season": row['season'],
            "Activities": row['activities'],
            "Latitude": row.get('latitude'),
            "Longitude": row.get('longitude'),
            "Score": round(final_score, 4)
        })

    # Deduplicate results by Place
    unique_places = {}
    for r in results:
        if r["Place"] not in unique_places or r["Score"] > unique_places[r["Place"]]["Score"]:
            unique_places[r["Place"]] = r

    # Return top 10 sorted results
    sorted_results = sorted(unique_places.values(), key=lambda x: x["Score"], reverse=True)
    return sorted_results[:10]