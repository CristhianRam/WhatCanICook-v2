import ast
import json
import os
import sqlite3

import polars as pl
from flask import g

DATASET_PATH = "embeddings/dataset/recipes_ingredients.csv"
DATABASE_NAME = "receipts.db"
URI_SQLITE = f"sqlite:///{DATABASE_NAME}"


def clean_list_string(texto):
    if not texto:
        return "[]"
    # Prefer JSON parsing since CSV fields often contain JSON-like lists
    try:
        parsed = json.loads(texto)
        if isinstance(parsed, (list, tuple)):
            return json.dumps(parsed)
    except Exception:
        pass

    # Fallback to ast.literal_eval for Python-like list literals
    try:
        lista_real = ast.literal_eval(texto)
        if isinstance(lista_real, (list, tuple)):
            return json.dumps(lista_real)
    except Exception:
        pass

    return "[]"


def create_db():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.execute("DROP TABLE IF EXISTS recipes")
    conn.execute("""
        CREATE TABLE recipes (
            id INTEGER PRIMARY KEY,
            name TEXT,
            description TEXT,
            ingredients_raw TEXT,
            steps TEXT,
            servings INTEGER,
            serving_size INTEGER
        )
    """)
    conn.close()


def ensure_database_populated():
    """Create and populate the sqlite DB file if it doesn't exist."""
    if os.path.exists(DATABASE_NAME):
        return

    create_db()

    df = pl.read_csv(
        DATASET_PATH,
        columns=[
            "id",
            "name",
            "description",
            "ingredients_raw",
            "steps",
            "servings",
            "serving_size",
        ],
    )

    print("First df row:")
    print(df.head(1))

    if "id" in df.columns:
        df = df.unique(subset=["id"])

    df = df.with_columns(
        pl.col("ingredients_raw").map_elements(
            clean_list_string, return_dtype=pl.String
        ),
        pl.col("steps").map_elements(clean_list_string, return_dtype=pl.String),
    )

    df.write_database(
        table_name="recipes",
        connection=URI_SQLITE,
        if_table_exists="append",
        engine="adbc",
    )


def get_db():
    # Open a sqlite3 connection for the current request context (stored in `g`).
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE_NAME)
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(exc=None):
    from flask import g as flask_g

    db_conn = flask_g.pop("db", None)
    if db_conn is not None:
        db_conn.close()


def init_app(app):
    # Ensure the DB file exists and is populated once at app startup.
    ensure_database_populated()
    app.teardown_appcontext(close_db)
