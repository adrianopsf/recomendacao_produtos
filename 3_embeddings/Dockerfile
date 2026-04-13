# ── build context: repo root ──────────────────────────────────────────────────
# docker build -f 3_embeddings/Dockerfile -t recomendacao-api .
FROM python:3.12-slim

WORKDIR /app

# Instala uv para instalação rápida de pacotes
RUN pip install --no-cache-dir uv

# Instala dependências de runtime da API
RUN uv pip install --system \
    "fastapi>=0.100.0" \
    "uvicorn>=0.24.0" \
    "sentence-transformers>=5.0.0" \
    "scikit-learn>=1.6.1" \
    "numpy>=2.0.2" \
    "pandas>=2.3.1" \
    "joblib>=1.3.0" \
    "python-dotenv>=1.1.1" \
    "cachetools>=5.3.0" \
    "duckdb>=1.3.2"

# Copia o código da API
COPY 3_embeddings/ ./3_embeddings/

# Pré-baixa o modelo para evitar latência no primeiro request
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Diretório de artefatos (sobrescrito via volume no docker-compose)
ENV EMB_ARTIFACTS_DIR=/app/3_embeddings/artifacts

# Expõe a porta da API
EXPOSE 8001

# Executa a partir de 3_embeddings/ para que `api.main` seja resolvido
WORKDIR /app/3_embeddings
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
