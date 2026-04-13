import os
import requests
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / "1_local_setup" / ".env"
load_dotenv(dotenv_path=ENV_PATH)

DBT_HOST = os.getenv("DBT_HOST", "localhost")
DBT_PORT = int(os.getenv("DBT_PORT", "5433"))
DBT_DBNAME = os.getenv("DBT_DBNAME")
DBT_USER = os.getenv("DBT_USER")
DBT_PASSWORD = os.getenv("DBT_PASSWORD")

DB_URL = f"postgresql+psycopg2://{DBT_USER}:{DBT_PASSWORD}@{DBT_HOST}:{DBT_PORT}/{DBT_DBNAME}"

PRODUCTS_URL = "https://fakestoreapi.com/products"


def fetch_products():
    r = requests.get(PRODUCTS_URL, timeout=30)
    r.raise_for_status()
    products = r.json()
    rows = []
    for p in products:
        rating = p.get("rating") or {}
        rows.append(
            {
                "product_id": p.get("id"),
                "title": p.get("title"),
                "description": p.get("description"),
                "category": p.get("category"),
                "price": p.get("price"),
                "image": p.get("image"),
                "rating_rate": rating.get("rate"),
                "rating_count": rating.get("count"),
            }
        )
    df = pd.DataFrame(rows).drop_duplicates(subset=["product_id"]).sort_values("product_id")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["rating_rate"] = pd.to_numeric(df["rating_rate"], errors="coerce")
    df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce")
    return df


def ensure_schema_and_table(conn):
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS bronze;"))
    conn.execute(
        text("""
        CREATE TABLE IF NOT EXISTS bronze.products_raw (
            product_id   INTEGER PRIMARY KEY,
            title        TEXT,
            description  TEXT,
            category     TEXT,
            price        NUMERIC,
            image        TEXT,
            rating_rate  NUMERIC,
            rating_count NUMERIC
        );
    """)
    )


def upsert_products(df: pd.DataFrame):
    engine = create_engine(DB_URL, future=True)
    with engine.begin() as conn:
        ensure_schema_and_table(conn)
        # staging substituída a cada execução
        df.to_sql("_products_stage", con=conn, schema="bronze", if_exists="replace", index=False)
        # upsert idempotente
        conn.execute(
            text("""
            INSERT INTO bronze.products_raw AS t
                (product_id, title, description, category, price, image, rating_rate, rating_count)
            SELECT product_id, title, description, category, price, image, rating_rate, rating_count
            FROM bronze._products_stage s
            ON CONFLICT (product_id) DO UPDATE
            SET title        = EXCLUDED.title,
                description  = EXCLUDED.description,
                category     = EXCLUDED.category,
                price        = EXCLUDED.price,
                image        = EXCLUDED.image,
                rating_rate  = EXCLUDED.rating_rate,
                rating_count = EXCLUDED.rating_count;
        """)
        )
        conn.execute(text("DROP TABLE IF EXISTS bronze._products_stage;"))


def main():
    print(">> Buscando produtos na Fake Store API...")
    df = fetch_products()
    print(f">> {len(df)} produtos obtidos.")
    upsert_products(df)
    print(">> Bronze atualizado: bronze.products_raw")


if __name__ == "__main__":
    main()
