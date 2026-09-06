import gc
from pathlib import Path

import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer

base_dir = Path(__file__).resolve().parent
dataset_path = base_dir / "dataset" / "recipes_ingredients.csv"
output_path = base_dir / "receipts_embeddings_400k.npz"

print("Loading model...")
modelo = SentenceTransformer("all-MiniLM-L6-v2")

print("Processing data with polars...")
df = (
    pl.read_csv(
        str(dataset_path),
        columns=["id", "name", "description", "ingredients", "steps", "tags"],
    )
    .fill_null("")
    .with_columns(
        full_text=pl.concat_str(
            [
                pl.lit("Receta: "),
                pl.col("name"),
                pl.lit(". Description: "),
                pl.col("description"),
                pl.lit(". Ing: "),
                pl.col("ingredients"),
                pl.lit(". Steps: "),
                pl.col("steps"),
                pl.lit(". Tags: "),
                pl.col("tags"),
                pl.lit("."),
            ]
        )
    )
)

print("Extracting ids and texts and cleaning RAM...")
ids = df["id"].to_numpy()
texts = df["full_text"].to_list()

# free memory used for df
del df
gc.collect()

print("Generating embeddings...")
# batching to save RAM
embeddings = modelo.encode(
    texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True
)

np.savez_compressed(output_path, ids=ids, embeddings=embeddings)
print(f"READY! Saved {len(ids)} embeddings with their dataset ids to {output_path}")
