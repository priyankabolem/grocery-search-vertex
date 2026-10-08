import pandas as pd
import pytest

from grocery_search.features import CATALOG_COLUMNS, build_catalog


@pytest.fixture
def raw():
    products = pd.DataFrame({
        "product_id": [1, 2, 3],
        "product_name": ["Organic Milk", "Bananas", "Rare Spice"],
        "aisle_id": [10, 20, 30],
        "department_id": [100, 200, 300],
    })
    aisles = pd.DataFrame({"aisle_id": [10, 20, 30], "aisle": ["milk", "fresh fruits", "spices"]})
    departments = pd.DataFrame({
        "department_id": [100, 200, 300],
        "department": ["dairy eggs", "produce", "pantry"],
    })
    # Product 1: 4 orders (3 reorders); product 2: 2 orders (0 reorders); product 3: 1 order.
    order_products = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5, 6, 7],
        "product_id": [1, 1, 1, 1, 2, 2, 3],
        "add_to_cart_order": [1, 1, 2, 2, 3, 5, 9],
        "reordered": [1, 1, 1, 0, 0, 0, 0],
    })
    return products, aisles, departments, order_products


def test_features_are_computed_per_product(raw):
    catalog = build_catalog(*raw, min_orders=1).set_index("product_id")
    assert catalog.loc["1", "order_count"] == 4
    assert catalog.loc["1", "reorder_rate"] == pytest.approx(0.75)
    assert catalog.loc["1", "avg_cart_position"] == pytest.approx(1.5)
    assert catalog.loc["2", "department"] == "produce"


def test_min_orders_filters_low_signal_products(raw):
    catalog = build_catalog(*raw, min_orders=2)
    assert set(catalog["product_id"]) == {"1", "2"}


def test_output_schema_and_string_ids(raw):
    catalog = build_catalog(*raw, min_orders=1)
    assert list(catalog.columns) == CATALOG_COLUMNS
    assert catalog["product_id"].map(type).eq(str).all()


def test_sorted_by_popularity(raw):
    catalog = build_catalog(*raw, min_orders=1)
    assert catalog["order_count"].is_monotonic_decreasing
