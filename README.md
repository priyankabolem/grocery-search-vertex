# Grocery Product Search & Re-ranking on Google Cloud

A two-stage product search service for a grocery catalog. **Vertex AI Search** retrieves candidate
products for a text query, a **gradient boosting model served from a Vertex AI endpoint** re-ranks
them by predicted reorder propensity, and a **FastAPI** service on **Cloud Run** blends both signals
into the final ranking.

**Stack:** Python · BigQuery · Cloud Storage · Vertex AI Search · Vertex AI Model Registry & Endpoints ·
scikit-learn · FastAPI · Docker · Cloud Run · GitHub Actions

## Architecture

```
OFFLINE
  Instacart CSVs ─► 01_prepare_data.py ─► catalog.csv ─► Cloud Storage ─► BigQuery (grocery.products)
                                                                              │
                                                     Vertex AI Search data store + search app
  catalog.csv ─► 03_train_and_deploy.py ─► model.joblib ─► Model Registry ─► Vertex AI Endpoint

ONLINE   GET /search?q=organic milk&department=dairy eggs
  FastAPI on Cloud Run
    1. Vertex AI Search    → top-k candidate products (optional department filter)
    2. Vertex AI Endpoint  → reorder-propensity score per candidate (single batched call)
    3. final_score = 0.7 × search_score + 0.3 × model_score → ranked JSON response
```

## Data

[Instacart Market Basket Analysis](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis) (Kaggle, CC0).
Place `products.csv`, `aisles.csv`, `departments.csv` and `order_products__train.csv` in `data/`.

The catalog has one row per product with behavioral features:

| Column | Meaning |
|---|---|
| `order_count` | Number of orders containing the product (popularity) |
| `reorder_rate` | Share of those orders that were reorders — **model target** |
| `avg_cart_position` | Average add-to-cart position (habitual items are added early) |
| `aisle`, `department` | Category names; `department` is a search filter |

Products with fewer than 20 orders are excluded so the statistics are reliable.

## Ranking

| Signal | Source | Definition |
|---|---|---|
| `search_score` | Vertex AI Search | Reciprocal rank: 1.0, 0.5, 0.33, … |
| `model_score` | Vertex AI endpoint | Predicted reorder propensity |
| `final_score` | API | `0.7 × search_score + 0.3 × model_score` (weights configurable) |

Example: retrieval order A, B, C with model scores 0.40, 0.85, 0.60 → final scores 0.82, 0.61, 0.41.
Relevance dominates, and the model lifts frequently re-bought items when relevance is close.

**Graceful degradation:** if no endpoint is configured or a prediction call fails, the API logs the
error and returns results in retrieval order (`"reranked": false`) instead of failing the request.

## Project structure

```
grocery_search/
  config.py        settings from environment variables, feature list
  features.py      catalog + feature engineering (pure pandas, unit tested)
  ranking.py       score blending and re-ranking (no cloud dependencies, unit tested)
  api.py           FastAPI app: /health, /search, OpenAPI docs at /docs
scripts/
  01_prepare_data.py     build data/catalog.csv
  02_load_to_gcp.py      upload to Cloud Storage, load BigQuery with an explicit schema
  03_train_and_deploy.py train + evaluate, register in Model Registry, deploy endpoint
  04_cleanup.py          undeploy and delete the endpoint
tests/                   pytest unit tests for features and ranking
.github/workflows/ci.yml lint + tests on every push
Dockerfile               non-root image for Cloud Run (also runs on GKE)
Makefile                 shortcuts for every step
```

## Setup

Requires Python 3.11 or 3.12, the [gcloud CLI](https://cloud.google.com/sdk/docs/install), and a
Google Cloud project with billing enabled.

```bash
# Python environment
python3.11 -m venv .venv && source .venv/bin/activate
make install                       # dependencies + editable install of grocery_search
make test                          # unit tests run locally, no cloud access needed

# Google Cloud
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable aiplatform.googleapis.com discoveryengine.googleapis.com \
  bigquery.googleapis.com storage.googleapis.com run.googleapis.com \
  cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud storage buckets create gs://YOUR_BUCKET --location=us-central1

# Configuration
cp .env.example .env               # set PROJECT_ID and BUCKET
set -a; source .env; set +a
```

## Run

| Step | Command | Result |
|---|---|---|
| 1. Build catalog | `make prepare` | `data/catalog.csv` |
| 2. Load to GCP | `make load` | BigQuery table `grocery.products` |
| 3. Search app | Console (below) | `SEARCH_ENGINE_ID` |
| 4. Train + deploy | `make train` | `artifacts/metrics.json`, `ENDPOINT_ID` |
| 5. Serve locally | `make serve` | http://127.0.0.1:8000/docs |
| 6. Deploy | `make deploy` | Public Cloud Run URL |
| 7. Clean up | `make cleanup` | Endpoint deleted (stops hourly billing) |

**Step 3 – Vertex AI Search (console):** AI Applications → **Create app** → **Search** (location
`global`) → create a data store from **BigQuery**, table `grocery.products`, **structured data**,
ID field `product_id`. In the schema, mark `product_name` searchable and `department` filterable.
Wait for indexing to finish, then copy the app ID into `SEARCH_ENGINE_ID` in `.env`.

**Step 4 options:** `python scripts/03_train_and_deploy.py --skip-deploy` trains and evaluates
locally (reports MAE against a mean-prediction baseline and R²) without creating cloud resources.

**Step 6 permissions:** the Cloud Run service account needs `roles/aiplatform.user` and
`roles/discoveryengine.user`.

### Example response (shape; values illustrative)

```bash
curl "http://127.0.0.1:8000/search?q=organic%20milk&department=dairy%20eggs&k=5"
```

```json
{
  "query": "organic milk",
  "department": "dairy eggs",
  "reranked": true,
  "latency_ms": 212.4,
  "results": [
    {"product_id": "27845", "product_name": "Organic Whole Milk", "department": "dairy eggs",
     "aisle": "milk", "search_score": 1.0, "model_score": 0.81, "final_score": 0.943}
  ]
}
```

## Design decisions

- **Retrieve, then re-rank.** Retrieval is cheap and recall-oriented across the whole catalog; the
  model runs only on the top-k candidates, in one batched prediction call.
- **Explicit BigQuery schema.** Vertex AI Search requires a STRING document ID, which schema
  autodetection would infer as INTEGER.
- **Pinned scikit-learn 1.3.x** to match Vertex AI's prebuilt `sklearn-cpu.1-3` serving container.
- **Cloud-free core logic.** Feature engineering and ranking are pure functions, so they are unit
  tested in CI without credentials.
- **Cloud Run** for low idle cost (scales to zero); the same image runs on GKE.

## Limitations and next steps

- The model scores products, not users or queries. Next: learning-to-rank on query–click/purchase
  data with user features, and personalization through Vertex AI Search for commerce user events.
- Blend weights are hand-set. Tune them against offline NDCG@k / MRR and validate with an online A/B test.
- Latency: cache popular queries, precompute product-level scores in BigQuery, keep warm replicas.
- Monitoring: Cloud Monitoring alerts on p95/p99 latency and error rate; Vertex AI Model Monitoring for drift.

## License

MIT
