"""Build data/catalog.csv from the raw Instacart files.

Expects in ./data (Kaggle: Instacart Market Basket Analysis):
  products.csv, aisles.csv, departments.csv, order_products__train.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from grocery_search.features import build_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--min-orders", type=int, default=20)
    args = parser.parse_args()

    d = args.data_dir
    catalog = build_catalog(
        products=pd.read_csv(d / "products.csv"),
        aisles=pd.read_csv(d / "aisles.csv"),
        departments=pd.read_csv(d / "departments.csv"),
        order_products=pd.read_csv(d / "order_products__train.csv"),
        min_orders=args.min_orders,
    )
    out = d / "catalog.csv"
    catalog.to_csv(out, index=False)

    print(f"Wrote {len(catalog):,} products to {out}")
    print(f"Departments: {catalog['department'].nunique()}  Aisles: {catalog['aisle'].nunique()}")
    print(catalog.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
