"""Gera (ou atualiza incrementalmente) os artefatos de embeddings.

Uso:
  python build_index.py               # reindexação completa
  python build_index.py --incremental # somente produtos novos
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import create_engine, text

# ── env ───────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
ENV_PATH = ROOT / "1_local_setup" / ".env"
_env = str(ENV_PATH) if ENV_PATH.exists() else find_dotenv()
if not _env:
    raise FileNotFoundError(f".env não encontrado em {ENV_PATH}")
load_dotenv(_env)

DBT_HOST = os.getenv("DBT_HOST", "localhost")
DBT_PORT = int(os.getenv("DBT_PORT", "5433"))
DBT_DBNAME = os.getenv("DBT_DBNAME")
DBT_USER = os.getenv("DBT_USER")
DBT_PASSWORD = os.getenv("DBT_PASSWORD")
DB_URL = (
    f"postgresql+psycopg2://{DBT_USER}:{DBT_PASSWORD}"
    f"@{DBT_HOST}:{DBT_PORT}/{DBT_DBNAME}"
)

ARTIFACTS_DIR = HERE.parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
EMB_PATH = ARTIFACTS_DIR / "embeddings.npy"
META_PATH = ARTIFACTS_DIR / "meta.csv"
MODEL_PATH = ARTIFACTS_DIR / "nn_model.joblib"

_GOLD_TABLE = "public_gold.products_for_embedding"
_META_COLS = ["product_id", "title", "category", "price", "image"]


# ── DB helpers ────────────────────────────────────────────────────────────────
def _load_gold(exclude_ids: list[int] | None = None) -> pd.DataFrame:
    """Carrega produtos da camada gold, opcionalmente excluindo IDs conhecidos."""
    engine = create_engine(DB_URL, future=True)

    with engine.connect() as conn:
        if exclude_ids:
            placeholders = ", ".join(f":id{i}" for i in range(len(exclude_ids)))
            params = {f"id{i}": v for i, v in enumerate(exclude_ids)}
            q = text(
                f"SELECT product_id, title, category, price, image, full_text "
                f"FROM {_GOLD_TABLE} "
                f"WHERE product_id NOT IN ({placeholders}) "
                f"ORDER BY product_id"
            )
            result = conn.execute(q, params)
        else:
            q = text(
                f"SELECT product_id, title, category, price, image, full_text "
                f"FROM {_GOLD_TABLE} ORDER BY product_id"
            )
            result = conn.execute(q)

        return pd.DataFrame(result.fetchall(), columns=list(result.keys()))


# ── embedding helpers ─────────────────────────────────────────────────────────
def _encode(df: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    texts = df["full_text"].fillna("").astype(str).tolist()
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(emb, dtype="float32")


def _build_nn(emb: np.ndarray) -> NearestNeighbors:
    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(emb)
    return nn


def _save(emb: np.ndarray, meta: pd.DataFrame, nn: NearestNeighbors) -> None:
    np.save(EMB_PATH, emb)
    meta[_META_COLS].to_csv(META_PATH, index=False)
    joblib.dump(nn, MODEL_PATH)
    print(f"[OK] {len(meta)} embeddings salvos em {ARTIFACTS_DIR}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Gera embeddings apenas para produtos novos (não presentes em meta.csv).",
    )
    args = parser.parse_args()

    print("Carregando modelo de embeddings...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if args.incremental and META_PATH.exists() and EMB_PATH.exists():
        existing_meta = pd.read_csv(META_PATH)
        known_ids: list[int] = existing_meta["product_id"].tolist()
        print(f"[incremental] {len(known_ids)} produtos já indexados.")

        new_df = _load_gold(exclude_ids=known_ids)
        if new_df.empty:
            print("[incremental] Nenhum produto novo. Índice já está atualizado.")
            return

        print(f"[incremental] Gerando embeddings para {len(new_df)} produtos novos...")
        new_emb = _encode(new_df, model)
        all_emb = np.vstack([np.load(EMB_PATH), new_emb])
        all_meta = pd.concat(
            [existing_meta, new_df[_META_COLS]], ignore_index=True
        )
    else:
        df = _load_gold()
        if df.empty:
            raise RuntimeError(f"{_GOLD_TABLE} está vazia. Execute o dbt antes.")
        print(f"Gerando embeddings para {len(df)} produtos...")
        all_emb = _encode(df, model)
        all_meta = df[_META_COLS].reset_index(drop=True)

    nn = _build_nn(all_emb)
    _save(all_emb, all_meta, nn)


if __name__ == "__main__":
    main()
