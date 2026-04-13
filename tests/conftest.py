"""Fixtures compartilhadas para a suite de testes.

Estratégia:
  - Cria artefatos temporários (embeddings, meta.csv, nn_model.joblib)
    com dados fictícios para evitar dependência de banco ou modelo real.
  - Mocka SentenceTransformer antes de inicializar o TestClient para que
    o startup da API não tente baixar o modelo.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import NearestNeighbors

# Adiciona 3_embeddings/ ao path para que `from api.main import app` funcione
sys.path.insert(0, str(Path(__file__).parent.parent / "3_embeddings"))

EMB_DIM = 384  # dimensão do all-MiniLM-L6-v2

FAKE_PRODUCTS = [
    (1, "Casual T-Shirt", "men's clothing", 9.99, "https://img.example.com/1.jpg"),
    (2, "Mountain Backpack", "bags", 49.99, "https://img.example.com/2.jpg"),
    (3, "Blue Jeans", "men's clothing", 29.99, "https://img.example.com/3.jpg"),
    (4, "Winter Jacket", "men's clothing", 79.99, "https://img.example.com/4.jpg"),
    (5, "Gold Watch", "jewelry", 199.99, "https://img.example.com/5.jpg"),
]


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Cria artefatos temporários com dados fictícios."""
    tmp = tmp_path_factory.mktemp("artifacts")

    meta = pd.DataFrame(FAKE_PRODUCTS, columns=["product_id", "title", "category", "price", "image"])
    meta.to_csv(tmp / "meta.csv", index=False)

    rng = np.random.default_rng(42)
    emb = rng.standard_normal((len(FAKE_PRODUCTS), EMB_DIM)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(tmp / "embeddings.npy", emb)

    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(emb)
    joblib.dump(nn, tmp / "nn_model.joblib")

    return tmp


@pytest.fixture(scope="session")
def client(artifacts_dir: Path):
    """TestClient com artefatos mockados e SentenceTransformer mockado."""
    os.environ["EMB_ARTIFACTS_DIR"] = str(artifacts_dir)

    # Mock do modelo para evitar download e inferência real
    rng = np.random.default_rng(0)
    mock_model = MagicMock()
    mock_model.encode.side_effect = lambda texts, **kw: (
        rng.standard_normal((len(texts), EMB_DIM)).astype("float32")
    )

    # Importa app DEPOIS de setar a env var (ARTIFACTS_DIR é lido no startup)
    from api.main import app  # noqa: PLC0415

    with patch("api.main.SentenceTransformer", return_value=mock_model):
        from fastapi.testclient import TestClient

        with TestClient(app) as c:
            yield c
