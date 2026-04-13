"""Testes da API de recomendação de produtos.

Cobre:
  - GET /health
  - GET /search  (resultados, paginação, parâmetros inválidos)
  - GET /similar/{id}  (resultado, id inexistente)
  - POST /reindex
  - GET /analytics
"""
from __future__ import annotations

import pytest


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["items"] == 5  # 5 produtos fictícios


# ── /search ───────────────────────────────────────────────────────────────────

def test_search_returns_results(client):
    r = client.get("/search", params={"query": "t-shirt", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "t-shirt"
    assert body["k"] == 3
    assert body["offset"] == 0
    assert body["total"] == 5
    assert len(body["results"]) == 3


def test_search_result_schema(client):
    r = client.get("/search", params={"query": "jacket"})
    assert r.status_code == 200
    for item in r.json()["results"]:
        assert "product_id" in item
        assert "title" in item
        assert "score" in item
        assert 0.0 <= item["score"] <= 1.0


def test_search_pagination_offset(client):
    r_all = client.get("/search", params={"query": "clothing", "k": 5, "offset": 0})
    r_paged = client.get("/search", params={"query": "clothing", "k": 2, "offset": 2})

    assert r_all.status_code == 200
    assert r_paged.status_code == 200

    all_ids = [i["product_id"] for i in r_all.json()["results"]]
    paged_ids = [i["product_id"] for i in r_paged.json()["results"]]

    # Os 2 resultados paginados devem corresponder à posição 2 e 3 do resultado completo
    assert paged_ids == all_ids[2:4]


def test_search_query_too_short(client):
    r = client.get("/search", params={"query": "a"})
    assert r.status_code == 422  # Validation error (min_length=2)


def test_search_k_out_of_range(client):
    r = client.get("/search", params={"query": "jacket", "k": 100})
    assert r.status_code == 422  # le=50


def test_search_offset_beyond_total_returns_empty(client):
    r = client.get("/search", params={"query": "anything", "k": 5, "offset": 100})
    assert r.status_code == 200
    assert r.json()["results"] == []


# ── /similar ──────────────────────────────────────────────────────────────────

def test_similar_returns_results(client):
    r = client.get("/similar/1", params={"k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["product_id"] == 1
    assert body["k"] == 3
    assert len(body["results"]) == 3


def test_similar_excludes_self(client):
    r = client.get("/similar/2", params={"k": 4})
    assert r.status_code == 200
    ids = [i["product_id"] for i in r.json()["results"]]
    assert 2 not in ids


def test_similar_not_found(client):
    r = client.get("/similar/9999")
    assert r.status_code == 404


# ── /reindex ──────────────────────────────────────────────────────────────────

def test_reindex(client):
    r = client.post("/reindex")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "reloaded"
    assert body["items"] == 5


# ── /analytics ────────────────────────────────────────────────────────────────

def test_analytics_structure(client):
    r = client.get("/analytics")
    assert r.status_code == 200
    body = r.json()

    assert "total_produtos" in body
    assert body["total_produtos"] == 5

    assert "categorias" in body
    assert len(body["categorias"]) > 0
    for cat in body["categorias"]:
        assert "category" in cat
        assert "produtos" in cat
        assert "preco_medio" in cat

    assert "preco_por_categoria" in body
    for stat in body["preco_por_categoria"]:
        assert "category" in stat
        assert "minimo" in stat
        assert "maximo" in stat
        assert "media" in stat


def test_analytics_category_counts(client):
    r = client.get("/analytics")
    cats = {c["category"]: c["produtos"] for c in r.json()["categorias"]}
    # 3 produtos de men's clothing, 1 bags, 1 jewelry
    assert cats.get("men's clothing") == 3
    assert cats.get("bags") == 1
    assert cats.get("jewelry") == 1


# ── analytics module (DuckDB standalone) ─────────────────────────────────────

def test_analytics_module(artifacts_dir):
    """Testa o módulo 4_analytics/analytics.py diretamente."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from analytics import analytics  # noqa: PLC0415

    meta_path = artifacts_dir / "meta.csv"

    cats = analytics.top_categories(meta_path)
    assert not cats.empty
    assert "category" in cats.columns
    assert "produtos" in cats.columns

    stats = analytics.price_stats(meta_path)
    assert not stats.empty
    assert "media" in stats.columns

    expensive = analytics.most_expensive_per_category(meta_path)
    assert len(expensive) == 3  # 3 categorias distintas

    hist = analytics.price_histogram(meta_path, bins=10)
    assert not hist.empty
