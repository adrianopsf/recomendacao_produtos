# Product Recommendation System

> Sistema de recomendação de produtos baseado em **embeddings semânticos** e busca por similaridade vetorial. Combina uma pipeline de dados (PostgreSQL + dbt) com uma API FastAPI para busca semântica e produtos similares em tempo real.

[![CI](https://github.com/adrianopsf/recomendacao_produtos/actions/workflows/ci.yml/badge.svg)](https://github.com/adrianopsf/recomendacao_produtos/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-latest-blue?style=flat-square&logo=postgresql)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-required-2496ED?style=flat-square&logo=docker)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## Índice

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Pré-requisitos](#pré-requisitos)
- [Instalação e Execução](#instalação-e-execução)
- [Docker Compose (stack completa)](#docker-compose-stack-completa)
- [API Reference](#api-reference)
- [Analytics com DuckDB](#analytics-com-duckdb)
- [Testes](#testes)
- [CI/CD](#cicd)
- [Stack Tecnológica](#stack-tecnológica)

---

## Visão Geral

Este projeto implementa um sistema completo de recomendação de produtos usando NLP e busca vetorial:

1. **Ingestão** — busca produtos via [Fake Store API](https://fakestoreapi.com) e armazena no PostgreSQL (camada Bronze)
2. **Transformação** — dbt constrói as camadas Silver e Gold com dados limpos
3. **Embeddings** — `sentence-transformers` gera vetores semânticos; `NearestNeighbors` monta o índice de busca
4. **API REST** — FastAPI expõe busca semântica, produtos similares, analytics e reindexação
5. **Analytics** — DuckDB realiza consultas ad-hoc sobre os metadados dos produtos

---

## Arquitetura

```
Fake Store API
     │
     ▼
bronze.products_raw        ← fakestore_get_data.py
     │
     ▼ dbt (staging → intermediate → mart)
public_gold.products_for_embedding
     │
     ▼ build_index.py (--incremental disponível)
artifacts/
  ├── embeddings.npy       ← vetores float32 normalizados
  ├── meta.csv             ← metadados dos produtos
  └── nn_model.joblib      ← índice NearestNeighbors (cosine)
     │
     ▼ FastAPI (uvicorn)
  GET /search              ← busca semântica (cache TTL + paginação)
  GET /similar/{id}        ← produtos similares
  POST /reindex            ← recarrega artefatos sem restart
  GET /analytics           ← consultas DuckDB sobre meta.csv
  GET /health
```

---

## Estrutura do Projeto

```
recomendacao_produtos/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD: lint, testes, build Docker
│
├── 1_local_setup/              # Infraestrutura local
│   ├── docker-compose.yml      # PostgreSQL + API em containers
│   ├── pyproject.toml          # Dependências (uv) + config pytest/ruff
│   ├── .env.example            # Template de variáveis de ambiente
│   └── .python-version
│
├── 2_data_warehouse/           # Ingestão e transformação
│   ├── fakestore_get_data.py   # Fetch API → bronze.products_raw
│   └── recomendacao_projetos/  # Projeto dbt (Bronze → Silver → Gold)
│       ├── models/
│       │   ├── staging/        # stg_products (view)
│       │   ├── intermediate/   # products_clean (view)
│       │   └── mart/           # products_for_embedding (table)
│       ├── dbt_project.yml
│       └── profiles.yml
│
├── 3_embeddings/               # Embeddings e API
│   ├── Dockerfile              # Imagem Docker da API
│   ├── build_index.py          # Gera/atualiza artefatos (--incremental)
│   ├── artifacts/              # embeddings.npy, meta.csv, nn_model.joblib
│   └── api/
│       ├── __init__.py
│       └── main.py             # FastAPI: /search /similar /reindex /analytics
│
├── 4_analytics/                # Análises ad-hoc com DuckDB
│   ├── __init__.py
│   └── analytics.py            # top_categories, price_stats, relatório CLI
│
└── tests/                      # Suite de testes (pytest)
    ├── __init__.py
    ├── conftest.py             # Fixtures: artefatos mock + TestClient
    └── test_api.py             # Testes de todos os endpoints + módulo analytics
```

---

## Pré-requisitos

| Ferramenta | Versão mínima | Uso |
|------------|--------------|-----|
| Python     | 3.11+        | Execução dos scripts e API |
| Docker     | 20+          | PostgreSQL e API em container |
| uv         | latest       | Gerenciador de pacotes |
| dbt        | 1.9+         | Transformação bronze → gold |

---

## Instalação e Execução

### 1. Clone o repositório

```bash
git clone https://github.com/adrianopsf/recomendacao_produtos.git
cd recomendacao_produtos
```

### 2. Configure as variáveis de ambiente

```bash
cp .env.example 1_local_setup/.env
# Edite 1_local_setup/.env com suas credenciais
```

### 3. Instale as dependências

```bash
cd 1_local_setup
uv sync --all-groups
```

### 4. Suba o PostgreSQL

```bash
cd 1_local_setup
docker compose up postgres -d
```

### 5. Ingestão de dados (Bronze)

```bash
cd 2_data_warehouse
python fakestore_get_data.py
```

### 6. Transformação dbt (Bronze → Gold)

```bash
cd 2_data_warehouse
dbt run --profiles-dir . --project-dir recomendacao_projetos
```

Cria `public_gold.products_for_embedding`.

### 7. Gere os embeddings

```bash
cd 3_embeddings
python build_index.py

# Para atualizar apenas produtos novos:
python build_index.py --incremental
```

### 8. Suba a API

```bash
cd 3_embeddings
uvicorn api.main:app --reload --port 8001
```

API disponível em: `http://localhost:8001`  
Docs interativos: `http://localhost:8001/docs`

---

## Docker Compose (stack completa)

Sobe PostgreSQL e API juntos com um único comando:

```bash
cd 1_local_setup
docker compose up -d
```

> **Nota:** Execute os passos 5, 6 e 7 antes de subir a API via Docker para garantir que os artefatos existam.

---

## API Reference

### `GET /health`

```json
{ "status": "ok", "items": 20 }
```

### `GET /search`

Busca semântica com **cache TTL** (1h, 256 queries) e **paginação**.

| Param  | Tipo | Default | Descrição |
|--------|------|---------|-----------|
| query  | str  | obrigatório | Texto de busca (mínimo 2 chars) |
| k      | int  | 5       | Resultados por página (máx 50) |
| offset | int  | 0       | Deslocamento para paginação |

```bash
# Primeira página
curl "http://localhost:8001/search?query=casual+t-shirt&k=5"

# Segunda página
curl "http://localhost:8001/search?query=casual+t-shirt&k=5&offset=5"
```

```json
{
  "query": "casual t-shirt",
  "k": 5,
  "offset": 0,
  "total": 20,
  "results": [
    {
      "product_id": 1,
      "title": "Fjallraven Foldsack Backpack",
      "category": "men's clothing",
      "price": 109.95,
      "image": "https://...",
      "score": 0.9214
    }
  ]
}
```

### `GET /similar/{product_id}`

```bash
curl "http://localhost:8001/similar/3?k=5"
```

### `POST /reindex`

Recarrega os artefatos do disco após um novo `build_index.py`, **sem reiniciar a API**.

```bash
curl -X POST "http://localhost:8001/reindex"
```

```json
{ "status": "reloaded", "items": 20 }
```

### `GET /analytics`

Análises ad-hoc via **DuckDB** sobre `meta.csv`.

```bash
curl "http://localhost:8001/analytics"
```

```json
{
  "total_produtos": 20,
  "categorias": [
    { "category": "men's clothing", "produtos": 4, "preco_medio": 63.25 }
  ],
  "preco_por_categoria": [
    { "category": "jewelry", "minimo": 10.99, "maximo": 695.0, "media": 268.4 }
  ]
}
```

---

## Analytics com DuckDB

O módulo `4_analytics/analytics.py` pode ser usado como CLI ou importado:

```bash
# Relatório completo no terminal
python 4_analytics/analytics.py

# Com meta.csv customizado
python 4_analytics/analytics.py --meta /outro/caminho/meta.csv
```

```python
# Como módulo
from analytics.analytics import top_categories, price_stats

df_cats = top_categories()
df_stats = price_stats()
```

---

## Testes

```bash
cd 1_local_setup
uv run pytest ../tests/ -v
```

A suite usa artefatos mock temporários e mocka o `SentenceTransformer`. Não requer banco de dados ou modelo real.

Cobertura:
- `/health`, `/search` (resultados, paginação, validação), `/similar` (resultado, id inexistente), `/reindex`, `/analytics`
- Módulo `4_analytics/analytics.py` via DuckDB

---

## CI/CD

Workflow `.github/workflows/ci.yml` executa em cada push/PR para `main`:

1. **Lint** — `ruff check` + `ruff format --check`
2. **Testes** — `pytest` em Python 3.11 e 3.12
3. **Docker Build** — constrói a imagem e faz smoke test no `/health`

---

## Stack Tecnológica

| Componente | Tecnologia |
|------------|-----------|
| Linguagem | Python 3.11+ |
| API | FastAPI + Uvicorn |
| Banco de dados | PostgreSQL (Docker) |
| ORM / Conexão | SQLAlchemy 2.0 + psycopg2 |
| Transformação | dbt-core + dbt-postgres |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Busca vetorial | scikit-learn NearestNeighbors (cosine) |
| Cache de queries | cachetools TTLCache (in-memory, 1h TTL) |
| Analytics | DuckDB |
| Serialização | NumPy + joblib |
| Contêineres | Docker + Docker Compose |
| Gerenciador | uv |
| Linting | ruff |
| Testes | pytest + FastAPI TestClient |
| CI/CD | GitHub Actions |

---

## Contribuindo

1. Fork o repositório
2. Crie uma branch: `git checkout -b feature/minha-feature`
3. Commit: `git commit -m 'feat: adiciona minha feature'`
4. Push: `git push origin feature/minha-feature`
5. Abra um Pull Request

---

## Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para detalhes.

---

<div align="center">
  <sub>Feito com por <a href="https://github.com/adrianopsf">adrianopsf</a></sub>
</div>
