import ast
import json
import sqlite3

import polars as pl
from flask import g

DATASET_PATH = "embeddings/recipes_ingredients.csv"
DATABASE_NAME = "receipts.db"
URI_SQLITE = f"sqlite:///{DATABASE_NAME}"


def clean_list_string(texto):
    if not texto:
        return "[]"
    try:
        # 1. From "['sal', 'pimienta']" to python list
        lista_real = ast.literal_eval(texto)
        # 2. From list to a valid json ('["sal", "pimienta"]')
        return json.dumps(lista_real)
    except (ValueError, SyntaxError):
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


def initialize_db(g):
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

    g.db = sqlite3.connect(DATABASE_NAME)


def get_db():
    if "db" not in g:
        initialize_db(g)
    return g.db


def close_db(g):
    db = g.pop("db", None)

    if db is not None:
        db.close()


def init_app(app):
    get_db()
    app.teardown_appcontext(close_db)
