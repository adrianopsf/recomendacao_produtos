from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import os
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer

# -------------------------
# Config e carregamento
# -------------------------
HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]  # sobe até a raiz do repo
ENV_PATH = ROOT / "1_local_setup" / ".env"
env_file = str(ENV_PATH) if ENV_PATH.exists() else find_dotenv()
if env_file:
    load_dotenv(env_file)

# Permite sobrescrever o diretório de artefatos via .env
ARTIFACTS_DIR = Path(os.getenv("EMB_ARTIFACTS_DIR", ROOT / "4_embeddings" / "artifacts"))
EMB_PATH  = ARTIFACTS_DIR / "embeddings.npy"
META_PATH = ARTIFACTS_DIR / "meta.csv"
MODEL_PATH = ARTIFACTS_DIR / "nn_model.joblib"

app = FastAPI(title="Recomendador por Embeddings", version="0.1.0")

# CORS básico para testes locais
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Variáveis globais carregadas na inicialização
EMB = None
META = None
NN = None
MODEL = None  # SentenceTransformer


class ItemOut(BaseModel):
    product_id: int
    title: str
    category: str | None = None
    price: float | None = None
    image: str | None = None
    score: float

class SearchResponse(BaseModel):
    query: str
    k: int
    results: list[ItemOut]

class SimilarResponse(BaseModel):
    product_id: int
    k: int
    results: list[ItemOut]


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


@app.on_event("startup")
def load_artifacts():
    global EMB, META, NN, MODEL
    if not (EMB_PATH.exists() and META_PATH.exists() and MODEL_PATH.exists()):
        raise RuntimeError(
            f"Artefatos não encontrados em {ARTIFACTS_DIR}. "
            "Rode o script de build de índice antes."
        )
    META = pd.read_csv(META_PATH)
    EMB = np.load(EMB_PATH)
    NN = joblib.load(MODEL_PATH)
    MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # Sanidade rápida
    if EMB.shape[0] != len(META):
        raise RuntimeError("Inconsistência: embeddings e meta possuem tamanhos diferentes.")
    # Embeddings devem estar normalizados (o script de build já salvou normalizado);
    # mantemos a função _normalize para robustez em casos futuros.
    print(f"[startup] Artefatos carregados: {len(META)} itens.")


@app.get("/health")
def health():
    return {"status": "ok", "items": 0 if META is None else len(META)}


@app.get("/search", response_model=SearchResponse)
def search(query: str = Query(..., min_length=2), k: int = Query(5, ge=1, le=50)):
    if any(x is None for x in (EMB, META, NN, MODEL)):
        raise HTTPException(status_code=503, detail="Artefatos não carregados.")
    q = MODEL.encode([query], normalize_embeddings=True)  # vetor 1xD normalizado
    distances, indices = NN.kneighbors(q, n_neighbors=k)
    distances = distances[0].tolist()
    indices = indices[0].tolist()

    # similaridade do cosseno = 1 - distância (pq NN usa 'cosine')
    results = []
    for idx, dist in zip(indices, distances):
        row = META.iloc[idx].to_dict()
        results.append(ItemOut(
            product_id=int(row["product_id"]),
            title=str(row.get("title", "")),
            category=(row.get("category") if pd.notna(row.get("category")) else None),
            price=(float(row["price"]) if pd.notna(row.get("price")) else None),
            image=(str(row["image"]) if pd.notna(row.get("image")) else None),
            score=max(0.0, 1.0 - float(dist)),
        ))
    return SearchResponse(query=query, k=k, results=results)


@app.get("/similar/{product_id}", response_model=SimilarResponse)
def similar(product_id: int, k: int = Query(5, ge=1, le=50)):
    if any(x is None for x in (EMB, META, NN)):
        raise HTTPException(status_code=503, detail="Artefatos não carregados.")
    # localizar a posição do product_id em META (mesma ordem de EMB)
    pos = META.index[META["product_id"] == product_id]
    if len(pos) == 0:
        raise HTTPException(status_code=404, detail=f"product_id {product_id} não encontrado.")
    pos = int(pos[0])

    distances, indices = NN.kneighbors(EMB[pos:pos+1], n_neighbors=k+1)
    distances = distances[0].tolist()
    indices = indices[0].tolist()

    results = []
    for idx, dist in zip(indices, distances):
        if idx == pos:  # remove o próprio item
            continue
        row = META.iloc[idx].to_dict()
        results.append(ItemOut(
            product_id=int(row["product_id"]),
            title=str(row.get("title", "")),
            category=(row.get("category") if pd.notna(row.get("category")) else None),
            price=(float(row["price"]) if pd.notna(row.get("price")) else None),
            image=(str(row["image"]) if pd.notna(row.get("image")) else None),
            score=max(0.0, 1.0 - float(dist)),
        ))
        if len(results) >= k:
            break

    return SimilarResponse(product_id=product_id, k=k, results=results)
