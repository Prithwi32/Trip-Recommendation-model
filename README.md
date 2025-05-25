---
title: Trip Recommendation Model
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "3.9"
app_file: main.py
pinned: false
---

# Trip Recommendation Model

This FastAPI application provides travel recommendations based on user preferences using a sentence transformer model.

## API Endpoints

- `/recommend` - POST endpoint for getting travel recommendations
- `/health` - GET endpoint for health check
- `/docs` - Interactive API documentation

## 🚀 Features

- Semantic search using Sentence Transformers
- Filters for type, budget, season, activities, state, and country
- Cosine similarity-based ranking with intelligent boosting
- FastAPI with auto-generated Swagger UI
- Optimized using caching for embeddings
- Geographical coordinates from OpenStreetMap

## API Usage

Once deployed, you can access the API documentation at:
`https://prithwi-trip-recommendation-model.hf.space/docs`

Example curl request:

```bash
curl -X 'POST' \
  'https://prithwi-trip-recommendation-model.hf.space/recommend' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "trip_type": "beach",
  "budget": "medium",
  "season": "summer",
  "activities": ["swimming", "surfing"],
  "state": "",
  "country": ""
}'
```

## Model Information

Uses the `paraphrase-MiniLM-L3-v2` sentence transformer model for generating travel recommendations based on user preferences.

## Deployment on Hugging Face Spaces

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose:
   - Owner: Your username
   - Space name: trip-recommendation
   - License: MIT
   - SDK: Docker
4. Upload all files from this repository
5. The space will automatically build and deploy

## Environment Variables

No environment variables needed for basic deployment.

## Local Development

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
uvicorn main:app --reload
```

## ▶️ To Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/trip-recommendation-api.git
cd trip-recommendation-api
```

### 2. Create a virtual environment

```bash
python -m venv venv

venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the API

```bash
uvicorn main:app --reload
```

### 5. Test the API:

#### Visit the interactive Swagger documentation at:

```bash
http://127.0.0.1:8000/docs
```
