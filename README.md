# 🌍 Trip Recommendation API

A FastAPI-based semantic recommendation system for travel destinations using Sentence Transformers. Matches user preferences like trip type, budget, season, activities, and location with suitable destinations across India.

---

## 🚀 Features

- Semantic search using Sentence Transformers
- Filters for type, budget, season, activities, state, and country
- Cosine similarity-based ranking with intelligent boosting
- FastAPI with auto-generated Swagger UI
- Optimized using caching for embeddings
- Geographical coordinates from OpenStreetMap

---

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
