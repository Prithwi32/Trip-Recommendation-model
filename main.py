import os
from pathlib import Path
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from functools import lru_cache
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Patch torch for compatibility
if not hasattr(torch, 'get_default_device'):
    torch.get_default_device = lambda: torch.device('cpu')  # Force CPU on Render

# Initialize FastAPI app with description and version
app = FastAPI(
    title="Trip Recommendation API",
    description="API for recommending travel destinations based on user preferences",
    version="1.0.0"
)

# Optimized dataset loading
def load_dataset():
    try:
        current_dir = Path(__file__).parent
        csv_path = current_dir / "destinations_enhanced_global.csv"
        df = pd.read_csv(csv_path, usecols=['Place', 'type', 'budget', 'season', 'activities', 'State', 'Country', 'latitude', 'longitude'])
        
        for col in ['type', 'budget', 'season', 'activities', 'State', 'Country']:
            df[col] = df[col].fillna("").astype(str)

        df['combined_features'] = (
            df['type'] + " " + df['budget'] + " " + df['season'] + " " +
            df['activities'] + " " + df['State'] + " " + df['Country']
        )
        return df
    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
        raise

# Load dataset at startup
try:
    df = load_dataset()
    logger.info("Dataset loaded successfully")
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    raise

# Lightweight model loading
def load_model():
    try:
        model = SentenceTransformer("paraphrase-MiniLM-L3-v2", device='cpu')
        model.max_seq_length = 128
        for param in model.parameters():
            param.requires_grad = False
        return model
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise

# Load model at startup
try:
    model = load_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

@lru_cache(maxsize=1)
def get_embeddings():
    try:
        # Process in chunks to reduce memory
        embeddings = []
        chunk_size = 100
        for i in range(0, len(df), chunk_size):
            chunk = df['combined_features'].iloc[i:i+chunk_size].tolist()
            embeddings.append(model.encode(chunk, convert_to_tensor=True))
        return torch.cat(embeddings)
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise

class Input(BaseModel):
    trip_type: Optional[str] = ""
    budget: Optional[str] = ""
    season: Optional[str] = ""
    activities: Optional[List[str]] = []
    state: Optional[str] = ""
    country: Optional[str] = ""

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring application status
    """
    try:
        # Verify model and dataset are loaded
        if model is None or df is None:
            raise HTTPException(status_code=503, detail="Service not fully initialized")
        return {
            "status": "healthy",
            "model_loaded": True,
            "dataset_loaded": True,
            "dataset_size": len(df)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/recommend")
async def recommend(input: Input):
    """
    Get travel recommendations based on user preferences
    """
    try:
        # Validate input
        if not any([input.trip_type, input.budget, input.season, input.activities, input.state, input.country]):
            raise HTTPException(400, "At least one input field required")

        # Process query
        query = " ".join([
            input.trip_type.lower().strip(),
            input.budget.lower().strip(),
            input.season.lower().strip(),
            *[a.lower().strip() for a in input.activities]
        ])

        # Get embeddings and calculate similarity
        query_embedding = model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, get_embeddings())[0]

        # Process results efficiently
        results = []
        for idx, score in enumerate(scores.cpu().numpy()):
            row = df.iloc[idx]
            final_score = float(score)
            
            # Apply filters
            if (input.state and input.state.lower() != row['State'].lower()) or \
               (input.country and input.country.lower() != row['Country'].lower()) or \
               (input.trip_type and input.trip_type.lower() != row['type'].lower()):
                continue
                
            # Add boosts
            if input.trip_type and input.trip_type.lower() == row['type'].lower():
                final_score += 0.2
            if input.season and input.season.lower() == row['season'].lower():
                final_score += 0.1
            if input.budget and input.budget.lower() == row['budget'].lower():
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

        # Clean up memory
        del scores
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Deduplicate and return top 10
        unique_results = {r["Place"]: r for r in results}
        return sorted(unique_results.values(), key=lambda x: x["Score"], reverse=True)[:10]

    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=1)