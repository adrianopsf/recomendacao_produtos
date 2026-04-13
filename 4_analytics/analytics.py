"""Análises ad-hoc dos produtos usando DuckDB.

Pode ser executado diretamente (gera relatório no terminal) ou importado
como módulo para uso em notebooks e scripts.

Uso direto:
  python analytics.py
  python analytics.py --meta /caminho/para/meta.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

DEFAULT_META = (
    Path(__file__).resolve().parents[1] / "3_embeddings" / "artifacts" / "meta.csv"
)


# ── funções de análise ────────────────────────────────────────────────────────

def top_categories(meta_path: str | Path = DEFAULT_META, limit: int = 10) -> pd.DataFrame:
    """Categorias com mais produtos e preço médio, ordenadas por volume."""
    mp = str(meta_path)
    return duckdb.query(
        f"SELECT category, "
        f"  COUNT(*) AS produtos, "
        f"  ROUND(AVG(price), 2) AS preco_medio "
        f"FROM read_csv_auto('{mp}') "
        f"GROUP BY category "
        f"ORDER BY produtos DESC "
        f"LIMIT {limit}"
    ).df()


def price_stats(meta_path: str | Path = DEFAULT_META) -> pd.DataFrame:
    """Estatísticas de preço (min, max, média, desvio) por categoria."""
    mp = str(meta_path)
    return duckdb.query(
        f"SELECT category, "
        f"  COUNT(*) AS produtos, "
        f"  ROUND(MIN(price), 2)    AS minimo, "
        f"  ROUND(MAX(price), 2)    AS maximo, "
        f"  ROUND(AVG(price), 2)    AS media, "
        f"  ROUND(STDDEV(price), 2) AS desvio "
        f"FROM read_csv_auto('{mp}') "
        f"GROUP BY category "
        f"ORDER BY media DESC"
    ).df()


def most_expensive_per_category(meta_path: str | Path = DEFAULT_META) -> pd.DataFrame:
    """Produto mais caro dentro de cada categoria."""
    mp = str(meta_path)
    return duckdb.query(
        f"WITH ranked AS ("
        f"  SELECT category, title, price, "
        f"         ROW_NUMBER() OVER (PARTITION BY category ORDER BY price DESC) AS rn "
        f"  FROM read_csv_auto('{mp}')"
        f") "
        f"SELECT category, title, price "
        f"FROM ranked WHERE rn = 1 "
        f"ORDER BY price DESC"
    ).df()


def price_histogram(meta_path: str | Path = DEFAULT_META, bins: int = 5) -> pd.DataFrame:
    """Distribuição de preços em faixas."""
    mp = str(meta_path)
    return duckdb.query(
        f"SELECT "
        f"  ROUND(price / {bins}) * {bins} AS faixa_inicio, "
        f"  COUNT(*) AS produtos "
        f"FROM read_csv_auto('{mp}') "
        f"GROUP BY faixa_inicio "
        f"ORDER BY faixa_inicio"
    ).df()


# ── relatório completo ────────────────────────────────────────────────────────

def full_report(meta_path: str | Path = DEFAULT_META) -> None:
    """Imprime relatório completo no terminal."""
    sep = "=" * 60

    print(f"\n{sep}")
    print("  TOP CATEGORIAS")
    print(sep)
    print(top_categories(meta_path).to_string(index=False))

    print(f"\n{sep}")
    print("  ESTATÍSTICAS DE PREÇO POR CATEGORIA")
    print(sep)
    print(price_stats(meta_path).to_string(index=False))

    print(f"\n{sep}")
    print("  PRODUTO MAIS CARO POR CATEGORIA")
    print(sep)
    print(most_expensive_per_category(meta_path).to_string(index=False))

    print(f"\n{sep}")
    print("  DISTRIBUIÇÃO DE PREÇOS (faixas de R$10)")
    print(sep)
    print(price_histogram(meta_path, bins=10).to_string(index=False))
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meta",
        default=str(DEFAULT_META),
        help=f"Caminho para meta.csv (padrão: {DEFAULT_META})",
    )
    args = parser.parse_args()
    full_report(args.meta)
