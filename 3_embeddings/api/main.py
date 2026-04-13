"""API de recomendação de produtos por embeddings semânticos.

Endpoints:
  GET  /health              — status da API
  GET  /search              — busca semântica por texto (com cache e paginação)
  GET  /similar/{id}        — produtos similares a um id (com paginação)
  POST /reindex             — recarrega artefatos do disco
  GET  /analytics           — análises ad-hoc dos produtos via DuckDB
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import duckdb
import joblib
import numpy as np
import pandas as pd
from cachetools import TTLCache
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ── env ───────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]  # repo root: 3_embeddings/api/main.py → parents[2] = repo root
ENV_PATH = ROOT / "1_local_setup" / ".env"
_env_file = str(ENV_PATH) if ENV_PATH.exists() else find_dotenv()
if _env_file:
    load_dotenv(_env_file)


# ── app ───────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_from_disk()
    yield


app = FastAPI(title="Recomendador por Embeddings", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── global state ──────────────────────────────────────────────────────────────
EMB: Optional[np.ndarray] = None
META: Optional[pd.DataFrame] = None
NN = None
MODEL: Optional[SentenceTransformer] = None

# ── query-embedding cache (TTL = 1 hora, max 256 queries) ────────────────────
_cache: TTLCache = TTLCache(maxsize=256, ttl=3600)
_cache_lock = threading.Lock()


# ── schemas ───────────────────────────────────────────────────────────────────
class ItemOut(BaseModel):
    product_id: int
    title: str
    category: Optional[str] = None
    price: Optional[float] = None
    image: Optional[str] = None
    score: float


class SearchResponse(BaseModel):
    query: str
    k: int
    offset: int
    total: int
    results: list[ItemOut]


class SimilarResponse(BaseModel):
    product_id: int
    k: int
    results: list[ItemOut]


class ReindexResponse(BaseModel):
    status: str
    items: int


class CategoryStat(BaseModel):
    category: Optional[str]
    produtos: int
    preco_medio: Optional[float]


class PriceStat(BaseModel):
    category: Optional[str]
    minimo: Optional[float]
    maximo: Optional[float]
    media: Optional[float]


class AnalyticsResponse(BaseModel):
    total_produtos: int
    categorias: list[CategoryStat]
    preco_por_categoria: list[PriceStat]


# ── helpers ───────────────────────────────────────────────────────────────────
def _artifacts_dir() -> Path:
    """Resolve o diretório de artefatos (env var ou default)."""
    return Path(os.getenv("EMB_ARTIFACTS_DIR", str(ROOT / "3_embeddings" / "artifacts")))


def _load_from_disk() -> None:
    """Lê embeddings, metadados e modelo NN do disco. Limpa o cache."""
    global EMB, META, NN, MODEL

    d = _artifacts_dir()
    emb_path = d / "embeddings.npy"
    meta_path = d / "meta.csv"
    model_path = d / "nn_model.joblib"

    missing = [p for p in (emb_path, meta_path, model_path) if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Artefatos ausentes em {d}: {[p.name for p in missing]}. "
            "Execute build_index.py antes de iniciar a API."
        )

    META = pd.read_csv(meta_path)
    EMB = np.load(emb_path)
    NN = joblib.load(model_path)

    if MODEL is None:
        MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if EMB.shape[0] != len(META):
        raise RuntimeError(
            f"Inconsistência: {EMB.shape[0]} embeddings vs {len(META)} linhas em meta.csv."
        )

    with _cache_lock:
        _cache.clear()

    print(f"[startup] {len(META)} produtos carregados de {d}.")


def _embed_query(query: str) -> np.ndarray:
    """Encode de query com cache TTL por texto."""
    with _cache_lock:
        hit = _cache.get(query)
    if hit is not None:
        return hit

    vec = np.asarray(MODEL.encode([query], normalize_embeddings=True), dtype="float32")

    with _cache_lock:
        _cache[query] = vec

    return vec


def _make_item(row: dict, dist: float) -> ItemOut:
    return ItemOut(
        product_id=int(row["product_id"]),
        title=str(row.get("title", "")),
        category=(row["category"] if pd.notna(row.get("category")) else None),
        price=(float(row["price"]) if pd.notna(row.get("price")) else None),
        image=(str(row["image"]) if pd.notna(row.get("image")) else None),
        score=round(max(0.0, 1.0 - float(dist)), 4),
    )


# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "items": 0 if META is None else len(META)}


@app.get("/search", response_model=SearchResponse)
def search(
    query: str = Query(..., min_length=2, description="Texto de busca"),
    k: int = Query(5, ge=1, le=50, description="Resultados por página"),
    offset: int = Query(0, ge=0, description="Deslocamento para paginação"),
):
    """Busca semântica por texto livre com cache e paginação."""
    if any(x is None for x in (EMB, META, NN, MODEL)):
        raise HTTPException(503, "Artefatos não carregados.")

    total = len(META)
    n_fetch = min(offset + k, total)
    if n_fetch == 0:
        return SearchResponse(query=query, k=k, offset=offset, total=total, results=[])

    q_vec = _embed_query(query)
    distances, indices = NN.kneighbors(q_vec, n_neighbors=n_fetch)

    results = [
        _make_item(META.iloc[idx].to_dict(), dist)
        for idx, dist in zip(indices[0][offset:], distances[0][offset:])
    ]
    return SearchResponse(query=query, k=k, offset=offset, total=total, results=results)


@app.get("/similar/{product_id}", response_model=SimilarResponse)
def similar(
    product_id: int,
    k: int = Query(5, ge=1, le=50, description="Número de similares"),
):
    """Retorna os k produtos mais similares a um produto dado."""
    if any(x is None for x in (EMB, META, NN)):
        raise HTTPException(503, "Artefatos não carregados.")

    positions = META.index[META["product_id"] == product_id].tolist()
    if not positions:
        raise HTTPException(404, f"product_id {product_id} não encontrado.")
    pos = positions[0]

    distances, indices = NN.kneighbors(EMB[pos : pos + 1], n_neighbors=k + 1)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if int(idx) == pos:
            continue
        results.append(_make_item(META.iloc[int(idx)].to_dict(), dist))
        if len(results) >= k:
            break

    return SimilarResponse(product_id=product_id, k=k, results=results)


@app.post("/reindex", response_model=ReindexResponse)
def reindex():
    """Recarrega os artefatos do disco após um novo build_index.py."""
    _load_from_disk()
    return ReindexResponse(status="reloaded", items=len(META))


@app.get("/analytics", response_model=AnalyticsResponse)
def analytics():
    """Análises ad-hoc dos produtos usando DuckDB sobre meta.csv."""
    if META is None:
        raise HTTPException(503, "Artefatos não carregados.")

    meta_path = str(_artifacts_dir() / "meta.csv")
    con = duckdb.connect()

    try:
        total = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{meta_path}')").fetchone()[0]

        cats_df = con.execute(
            f"SELECT category, COUNT(*) AS produtos, ROUND(AVG(price), 2) AS preco_medio "
            f"FROM read_csv_auto('{meta_path}') "
            f"GROUP BY category ORDER BY produtos DESC"
        ).fetchdf()

        price_df = con.execute(
            f"SELECT category, "
            f"ROUND(MIN(price), 2) AS minimo, "
            f"ROUND(MAX(price), 2) AS maximo, "
            f"ROUND(AVG(price), 2) AS media "
            f"FROM read_csv_auto('{meta_path}') "
            f"GROUP BY category ORDER BY media DESC"
        ).fetchdf()
    finally:
        con.close()

    def _to_nullable(val):
        return None if pd.isna(val) else val

    categorias = [
        CategoryStat(
            category=_to_nullable(r["category"]),
            produtos=int(r["produtos"]),
            preco_medio=_to_nullable(r["preco_medio"]),
        )
        for r in cats_df.to_dict(orient="records")
    ]

    preco_por_categoria = [
        PriceStat(
            category=_to_nullable(r["category"]),
            minimo=_to_nullable(r["minimo"]),
            maximo=_to_nullable(r["maximo"]),
            media=_to_nullable(r["media"]),
        )
        for r in price_df.to_dict(orient="records")
    ]

    return AnalyticsResponse(
        total_produtos=total,
        categorias=categorias,
        preco_por_categoria=preco_por_categoria,
    )
