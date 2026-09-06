# WhatCanICook-v2

>A lightweight recipe retrieval web app that finds relevant recipes from a dataset using sentence embeddings and FAISS vector search.

## Overview

This project converts recipe text into embeddings (using `sentence-transformers`), stores them as a vector index with FAISS, and exposes a small Flask web UI where users can enter a free-form description (ingredients, cuisine, dish description, time constraints) to retrieve relevant recipes ranked by similarity.

Key components:

- `embeddings/` — scripts to download the dataset and generate sentence embeddings.
- `embeddings/vector_store.py` — FAISS index loader/builder and similarity search helpers.
- `db.py` — prepares a local SQLite database from the CSV and exposes a request-scoped connection.
- `services.py` — application logic that combines vector search results with recipe rows from SQLite.
- `views.py` — Flask routes and templates for the UI.

## Quick start

1. Create and activate a Python virtual environment (Windows PowerShell example):

```powershell
python -m venv venv
& .\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Download the dataset (if not present):

```powershell
py embeddings/download_dataset.py
```

4. Generate embeddings (this may take memory/time):

```powershell
py embeddings/generate_embeddings.py
```

5. Build the FAISS index (or let the app build it on first query):

```powershell
py embeddings/build_faiss_index.py
```

6. Run the web app:

```powershell
py main.py
# or use flask run if you have FLASK_APP configured
```

Then open http://127.0.0.1:5000/ in a browser.

## Usage

Enter a free-form description into the search box. Example queries:

- "chicken, rice, bell peppers, quick 30 minutes"
- "vegetarian pasta with tomato and basil"
- "chocolate dessert with nuts"

Results will show recipe name, score, ingredients and steps.

## Notes & Troubleshooting

- The first app startup may ingest the CSV into a local SQLite DB — this can be slow for large datasets. The code now avoids re-ingesting the DB on every request.
- FAISS index building requires enough memory; if you run into memory issues, consider batching or using a smaller model.

## Files of interest

- `embeddings/generate_embeddings.py` — generate and save embeddings as `.npz`.
- `embeddings/vector_store.py` — load or build FAISS index and run similarity search.
- `templates/tasks/index.html` and `templates/tasks/results.html` — UI templates.

## License

See `LICENSE`.

