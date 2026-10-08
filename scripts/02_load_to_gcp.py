"""Upload data/catalog.csv to Cloud Storage and load it into BigQuery."""
from __future__ import annotations

from google.cloud import bigquery, storage

from grocery_search.config import Settings

F = bigquery.SchemaField
# Explicit schema: Vertex AI Search requires the document ID column (product_id) to be STRING.
SCHEMA = [
    F("product_id", "STRING", mode="REQUIRED"),
    F("product_name", "STRING"),
    F("aisle_id", "INTEGER"),
    F("department_id", "INTEGER"),
    F("aisle", "STRING"),
    F("department", "STRING"),
    F("order_count", "INTEGER"),
    F("reorder_rate", "FLOAT"),
    F("avg_cart_position", "FLOAT"),
]


def main() -> None:
    s = Settings.from_env()
    s.require("bucket")

    blob_path = "data/catalog.csv"
    storage.Client(project=s.project_id).bucket(s.bucket).blob(blob_path).upload_from_filename(blob_path)
    uri = f"gs://{s.bucket}/{blob_path}"
    print(f"Uploaded {uri}")

    bq = bigquery.Client(project=s.project_id)
    bq.create_dataset(f"{s.project_id}.{s.bq_dataset}", exists_ok=True)
    table_id = f"{s.project_id}.{s.bq_dataset}.products"
    job = bq.load_table_from_uri(
        uri,
        table_id,
        job_config=bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1,
            schema=SCHEMA,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        ),
    )
    job.result()
    print(f"Loaded {bq.get_table(table_id).num_rows:,} rows into {table_id}")


if __name__ == "__main__":
    main()
