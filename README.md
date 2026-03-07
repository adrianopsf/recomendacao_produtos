# 🛍️ Product Recommendation System

> Sistema de recomendação de produtos baseado em **embeddings semânticos** e busca por similaridade vetorial. Combina uma pipeline de dados (PostgreSQL + dbt) com uma API FastAPI para busca semântica e produtos similares em tempo real.

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-latest-blue?style=flat-square&logo=postgresql)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-required-2496ED?style=flat-square&logo=docker)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Pré-requisitos](#pré-requisitos)
- [Instalação e Execução](#instalação-e-execução)
- [API Reference](#api-reference)
- [Stack Tecnológica](#stack-tecnológica)
- [Avaliação End-to-End](#avaliação-end-to-end)
- [Problemas Conhecidos e Melhorias](#problemas-conhecidos-e-melhorias)

---

## 🎯 Visão Geral

Este projeto implementa um sistema completo de recomendação de produtos usando NLP e busca vetorial:

1. **Ingestão** de dados de produtos via [Fake Store API](https://fakestoreapi.com)
2. **Armazenamento** em PostgreSQL com camada Bronze/Gold
3. **Geração de embeddings** usando `sentence-transformers` (modelo `all-MiniLM-L6-v2`)
4. **Busca semântica** via NearestNeighbors (cosine similarity)
5. **API REST** com FastAPI expondo endpoints de busca e similaridade

---

## 🏗️ Arquitetura

```
┌───────────────────────────────────────────────────────────────┐
│                     Fluxo de Dados                            │
├──────────────┬───────────────┬──────────────┬─────────────────┤
│  Fake Store  │   PostgreSQL  │  Embeddings  │    FastAPI      │
│     API      │  (Bronze/Gold)│   (Artifacts)│    Service      │
│              │               │              │                 │
│  GET /products──▶bronze.     │  build_index │  /search        │
│  (20 itens)    products_raw  │  .py         │  /similar/{id}  │
│              │               │              │  /health        │
│              │  dbt models   │  embeddings  │                 │
│              │  (gold layer) │  .npy        │                 │
│              │               │  meta.csv    │                 │
│              │               │  nn_model    │                 │
│              │               │  .joblib     │                 │
└──────────────┴───────────────┴──────────────┴─────────────────┘
```

**Camadas de dados:**
- **Bronze** (`bronze.products_raw`): dados brutos da API, sem transformação
- **Gold** (`public_gold.products_for_embedding`): dados tratados, prontos para embeddings
- **Artifacts**: embeddings pré-computados em disco (`.npy`, `.csv`, `.joblib`)

---

## 📁 Estrutura do Projeto

```
recomendacao_produtos/
├── 1_local_setup/              # Infraestrutura local
│   ├── docker-compose.yml      # PostgreSQL via Docker
│   ├── pyproject.toml          # Dependências do projeto (UV)
│   ├── .python-version         # Versão Python recomendada
│   └── .gitignore
│
├── 2_data_warehouse/           # Ingestão e transformação de dados
│   ├── fakestore_get_data.py   # Busca dados da Fake Store API → PostgreSQL
│   ├── recomendacao_projetos/  # Modelos dbt (Bronze → Gold)
│   └── logs/                   # Logs do dbt
│
└── 4_embeddings/               # Sistema de embeddings e API
    ├── build_index.py          # Gera embeddings e índice NN
    ├── artifacts/              # Artefatos gerados (embeddings, model)
    └── api/
        └── main.py             # FastAPI: endpoints de busca
```

---

## ✅ Pré-requisitos

| Ferramenta | Versão mínima | Uso |
|------------|--------------|-----|
| Python     | 3.9+         | Execução dos scripts |
| Docker     | 20+          | PostgreSQL local |
| uv         | latest       | Gerenciador de pacotes |
| dbt        | 1.9+         | Transformação bronze → gold |

---

## 🚀 Instalação e Execução

### 1. Clone o repositório

```bash
git clone https://github.com/adrianopsf/recomendacao_produtos.git
cd recomendacao_produtos
```

### 2. Configure as variáveis de ambiente

```bash
# Em 1_local_setup/, crie o arquivo .env:
cp 1_local_setup/.env.example 1_local_setup/.env
```

Edite o `.env` com suas credenciais:

```env
DBT_USER=postgres
DBT_PASSWORD=sua_senha_segura
DBT_HOST=localhost
DBT_PORT=5433
DBT_DB=dbt_db
```

### 3. Suba o banco de dados

```bash
cd 1_local_setup
docker compose up -d
```

Aguarde o PostgreSQL inicializar (health check automático).

### 4. Instale as dependências

```bash
cd 1_local_setup
uv sync
```

### 5. Ingestão de dados (Bronze)

```bash
cd 2_data_warehouse
python fakestore_get_data.py
```

Isso buscará os 20 produtos da Fake Store API e armazenará em `bronze.products_raw`.

### 6. Transformação dbt (Bronze → Gold)

```bash
cd 2_data_warehouse
dbt run --profiles-dir . --project-dir recomendacao_projetos
```

Cria a view/table `public_gold.products_for_embedding`.

### 7. Gere os embeddings

```bash
cd 4_embeddings
python build_index.py
```

Gera em `artifacts/`:
- `embeddings.npy` — vetores float32 normalizados
- `meta.csv` — metadados dos produtos
- `nn_model.joblib` — índice NearestNeighbors

### 8. Suba a API

```bash
cd 4_embeddings
uvicorn api.main:app --reload --port 8001
```

API disponível em: `http://localhost:8001`
Docs interativos: `http://localhost:8001/docs`

---

## 📡 API Reference

### `GET /health`
Verifica o status da API.

```json
{
  "status": "ok",
  "items": 20
}
```

### `GET /search?query=<texto>&k=<n>`
Busca semântica por texto livre.

**Parâmetros:**
| Param | Tipo | Default | Descrição |
|-------|------|---------|-----------|
| query | str  | obrigatório | Texto de busca |
| k     | int  | 10     | Número de resultados |

**Exemplo:**
```bash
curl "http://localhost:8001/search?query=casual+t-shirt&k=5"
```

```json
{
  "query": "casual t-shirt",
  "results": [
    {
      "product_id": 1,
      "title": "Fjallraven Foldsack Backpack",
      "category": "men's clothing",
      "price": 109.95,
      "image": "https://...",
      "score": 0.92
    }
  ]
}
```

### `GET /similar/{product_id}?k=<n>`
Retorna produtos similares a um produto específico.

```bash
curl "http://localhost:8001/similar/3?k=5"
```

---

## 🧰 Stack Tecnológica

| Componente | Tecnologia |
|------------|-----------|
| Linguagem | Python 3.9+ |
| API | FastAPI + Uvicorn |
| Banco de dados | PostgreSQL (Docker) |
| ORM / Conexão | SQLAlchemy 2.0, psycopg2 |
| Transformação | dbt-core + dbt-postgres |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Busca vetorial | scikit-learn NearestNeighbors (cosine) |
| Serialização | NumPy, joblib |
| Gerenciador | uv |

---

## 🔍 Avaliação End-to-End

> Análise da viabilidade de execução completa do pipeline.

### ✅ O que funciona

| Componente | Status | Observação |
|-----------|--------|-----------|
| Docker PostgreSQL | ✅ | `docker-compose.yml` bem configurado com health check |
| `fakestore_get_data.py` | ✅ | Upsert idempotente, schema bronze correto |
| `build_index.py` | ✅ | Embeddings e índice gerados corretamente |
| `api/main.py` | ✅ | Endpoints `/search`, `/similar`, `/health` funcionais |
| Dependências | ✅ | `pyproject.toml` completo para os módulos existentes |

### ⚠️ Pontos de atenção

| Problema | Severidade | Recomendação |
|---------|-----------|-------------|
| **Modelos dbt não versionados** | 🔴 Alta | Adicionar arquivos `.sql` do dbt ao repositório; a tabela gold não é criada sem eles |
| **FastAPI ausente no `pyproject.toml`** | 🔴 Alta | Adicionar `fastapi>=0.100`, `uvicorn`, `joblib` às dependências |
| **Sem `.env.example`** | 🟡 Média | Criar template de variáveis de ambiente para facilitar onboarding |
| **Sem testes automatizados** | 🟡 Média | Adicionar testes para a API com `pytest` + `httpx` |
| **Step numerado como "4"** | 🟢 Baixa | Renomear para `3_embeddings` para manter consistência sequencial |
| **Sem `__init__.py` em `api/`** | 🟢 Baixa | Pode causar problemas de importação em alguns ambientes |

---

## 🛠️ Problemas Conhecidos e Melhorias

### Correções prioritárias

```bash
# 1. Adicionar FastAPI ao pyproject.toml
# Em 1_local_setup/pyproject.toml, adicionar:
# "fastapi>=0.100.0",
# "uvicorn>=0.24.0",
# "joblib>=1.3.0",

# 2. Adicionar modelos dbt ao repositório
# Criar: 2_data_warehouse/recomendacao_projetos/models/gold/products_for_embedding.sql

# 3. Criar .env.example com as variáveis necessárias
```

### Melhorias futuras

- [ ] Adicionar Docker Compose completo (PostgreSQL + API juntos)
- [ ] Implementar cache para embeddings de queries repetidas
- [ ] Suporte a re-indexação incremental (novos produtos)
- [ ] Adicionar paginação nos resultados da API
- [ ] Integrar DuckDB para análises ad-hoc locais
- [ ] CI/CD com GitHub Actions
- [ ] Dockerfile para a API de embeddings

---

## 🤝 Contribuindo

1. Fork o repositório
2. Crie uma branch: `git checkout -b feature/minha-feature`
3. Commit suas mudanças: `git commit -m 'feat: adiciona minha feature'`
4. Push: `git push origin feature/minha-feature`
5. Abra um Pull Request

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para detalhes.

---

<div align="center">
  <sub>Feito com ❤️ por <a href="https://github.com/adrianopsf">adrianopsf</a></sub>
</div>
