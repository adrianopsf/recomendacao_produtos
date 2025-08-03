from pathlib import Path
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv, find_dotenv
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import joblib

# --- carregar .env em 1_local_setup/.env ---
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
ENV_PATH = ROOT / "1_local_setup" / ".env"
env_file = str(ENV_PATH) if ENV_PATH.exists() else find_dotenv()
if not env_file:
    raise FileNotFoundError(f".env não encontrado em {ENV_PATH}")
load_dotenv(env_file)

DBT_HOST = os.getenv("DBT_HOST", "localhost")
DBT_PORT = int(os.getenv("DBT_PORT", "5433"))
DBT_DBNAME = os.getenv("DBT_DBNAME")
DBT_USER = os.getenv("DBT_USER")
DBT_PASSWORD = os.getenv("DBT_PASSWORD")
DB_URL = f"postgresql+psycopg2://{DBT_USER}:{DBT_PASSWORD}@{DBT_HOST}:{DBT_PORT}/{DBT_DBNAME}"

ARTIFACTS_DIR = HERE.parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
EMB_PATH = ARTIFACTS_DIR / "embeddings.npy"
META_PATH = ARTIFACTS_DIR / "meta.csv"
MODEL_PATH = ARTIFACTS_DIR / "nn_model.joblib"

def load_gold():
    eng = create_engine(DB_URL, future=True)
    q = """
        select product_id, title, category, price, image, full_text
        from public_gold.products_for_embedding
        order by product_id
    """
    df = pd.read_sql(q, eng)
    if df.empty:
        raise RuntimeError("public_gold.products_for_embedding está vazia.")
    return df

def main():
    df = load_gold()

    texts = df["full_text"].fillna("").astype(str).tolist()
    print("Carregando modelo de embeddings...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Gerando embeddings (normalizados)...")
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")

    # Ajusta NearestNeighbors (busca exata com métrica do cosseno)
    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(emb)

    # Salva artefatos
    np.save(EMB_PATH, emb)
    df[["product_id", "title", "category", "price", "image"]].to_csv(META_PATH, index=False)
    joblib.dump(nn, MODEL_PATH)

    print(f"OK: {len(df)} embeddings salvos em {EMB_PATH}")
    print(f"Metadados salvos em {META_PATH}")
    print(f"Modelo NN salvo em {MODEL_PATH}")

if __name__ == "__main__":
    main()
