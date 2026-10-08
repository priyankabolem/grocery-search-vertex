"""Build the product catalog and per-product behavioral features from Instacart data."""
from __future__ import annotations

import pandas as pd

CATALOG_COLUMNS = [
    "product_id", "product_name", "aisle_id", "department_id", "aisle", "department",
    "order_count", "reorder_rate", "avg_cart_position",
]


def build_catalog(
    products: pd.DataFrame,
    aisles: pd.DataFrame,
    departments: pd.DataFrame,
    order_products: pd.DataFrame,
    min_orders: int = 20,
) -> pd.DataFrame:
    """Join product metadata with order behavior and keep products with enough history.

    Returns one row per product with:
      order_count        number of orders containing the product (popularity)
      reorder_rate       share of those orders that were reorders (model target)
      avg_cart_position  mean add-to-cart position (habitual items are added early)
    """
    stats = (
        order_products.groupby("product_id")
        .agg(
            order_count=("order_id", "count"),
            reorder_rate=("reordered", "mean"),
            avg_cart_position=("add_to_cart_order", "mean"),
        )
        .reset_index()
    )

    catalog = (
        products.merge(aisles, on="aisle_id")
        .merge(departments, on="department_id")
        .merge(stats, on="product_id", how="inner")
    )
    catalog = catalog[catalog["order_count"] >= min_orders].copy()

    # Vertex AI Search document IDs must be strings.
    catalog["product_id"] = catalog["product_id"].astype(str)
    return catalog[CATALOG_COLUMNS].sort_values("order_count", ascending=False).reset_index(drop=True)
