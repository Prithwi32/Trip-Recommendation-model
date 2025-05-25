import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from functools import lru_cache
import torch

# Patch torch for compatibility with Render
if not hasattr(torch, 'get_default_device'):
    torch.get_default_device = lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = FastAPI()

# Load dataset with absolute path
try:
    current_dir = Path(__file__).parent
    csv_path = current_dir / "destinations_enhanced_global.csv"
    df = pd.read_csv(csv_path)
    
    # Preprocess data
    for col in ['type', 'budget', 'season', 'activities', 'State', 'Country']:
        df[col] = df[col].fillna("")

    df['combined_features'] = (
        df['type'] + " " + df['budget'] + " " + df['season'] + " " +
        df['activities'] + " " + df['State'] + " " + df['Country']
    )
except Exception as e:
    raise RuntimeError(f"Failed to load dataset: {str(e)}")

# Initialize model with explicit device handling
device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

@lru_cache(maxsize=1)
def get_embeddings():
    return model.encode(df['combined_features'].tolist(), convert_to_tensor=True)

# Input schema
class Input(BaseModel):
    trip_type: Optional[str] = ""
    budget: Optional[str] = ""
    season: Optional[str] = ""
    activities: Optional[List[str]] = []
    state: Optional[str] = ""
    country: Optional[str] = ""

@app.post("/recommend")
async def recommend(input: Input):
    try:
        # Validate input
        if not any([input.trip_type, input.budget, input.season, input.activities, input.state, input.country]):
            raise HTTPException(status_code=400, detail="At least one input field must be provided.")

        # Generate query
        query_parts = [
            input.trip_type.strip().lower(),
            input.budget.strip().lower(),
            input.season.strip().lower()
        ] + [a.strip().lower() for a in input.activities]
        query = " ".join([p for p in query_parts if p])

        # Get embeddings
        embeddings = get_embeddings()
        query_embedding = model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, embeddings)[0]

        # Process results
        results = []
        for i, score in enumerate(scores):
            row = df.iloc[i]

            # Apply filters
            if input.state and row['State'].strip().lower() != input.state.strip().lower():
                continue
            if input.country and row['Country'].strip().lower() != input.country.strip().lower():
                continue
            if input.trip_type and row['type'].strip().lower() != input.trip_type.strip().lower():
                continue

            # Calculate score with boosts
            final_score = score.item()
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

        # Deduplicate and sort
        unique_places = {}
        for r in results:
            if r["Place"] not in unique_places or r["Score"] > unique_places[r["Place"]]["Score"]:
                unique_places[r["Place"]] = r

        return sorted(unique_places.values(), key=lambda x: x["Score"], reverse=True)[:10]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)