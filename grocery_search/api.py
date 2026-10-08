"""FastAPI service: Vertex AI Search retrieval + Vertex AI endpoint re-ranking."""
from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from google.cloud import aiplatform
from google.cloud import discoveryengine_v1 as discoveryengine
from pydantic import BaseModel

from grocery_search.config import FEATURES, Settings
from grocery_search.ranking import rerank

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("grocery_search.api")

settings = Settings.from_env()
settings.require("search_engine_id")

search_client = discoveryengine.SearchServiceClient()
SERVING_CONFIG = (
    f"projects/{settings.project_id}/locations/global/collections/default_collection/"
    f"engines/{settings.search_engine_id}/servingConfigs/default_search"
)

endpoint: aiplatform.Endpoint | None = None
if settings.endpoint_id:
    aiplatform.init(project=settings.project_id, location=settings.region)
    endpoint = aiplatform.Endpoint(settings.endpoint_id)

app = FastAPI(
    title="Grocery Product Search",
    description="Vertex AI Search retrieval with reorder-propensity re-ranking.",
    version="0.1.0",
)


class Product(BaseModel):
    product_id: str
    product_name: str | None = None
    department: str | None = None
    aisle: str | None = None
    search_score: float
    model_score: float | None = None
    final_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    department: str | None
    reranked: bool
    latency_ms: float
    results: list[Product]


def retrieve(query: str, k: int, department: str | None) -> list[dict[str, Any]]:
    """Fetch the top-k candidate products from Vertex AI Search."""
    request = discoveryengine.SearchRequest(
        serving_config=SERVING_CONFIG,
        query=query,
        page_size=k,
        filter=f'department: ANY("{department}")' if department else "",
    )
    docs = []
    for result in search_client.search(request).results:
        doc = dict(result.document.struct_data)
        # Prefer the catalog's product_id; the data store may assign its own document IDs.
        doc["product_id"] = str(doc.get("product_id", result.document.id))
        docs.append(doc)
    return docs


def predict_scores(docs: list[dict[str, Any]]) -> list[float] | None:
    """Score all candidates in one batched call; return None if the model is unavailable."""
    if endpoint is None or not docs:
        return None
    instances = [[float(doc.get(f) or 0) for f in FEATURES] for doc in docs]
    try:
        return [float(p) for p in endpoint.predict(instances=instances).predictions]
    except Exception:  # serve search-only results rather than failing the request
        logger.exception("Model endpoint call failed; returning retrieval order")
        return None


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "reranking_enabled": endpoint is not None}


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, max_length=200, description="Search text"),
    k: int = Query(10, ge=1, le=50, description="Number of results"),
    department: str | None = Query(None, description="Optional department filter, e.g. 'dairy eggs'"),
) -> SearchResponse:
    start = time.perf_counter()
    try:
        docs = retrieve(q, k, department)
    except Exception as exc:
        logger.exception("Vertex AI Search request failed")
        raise HTTPException(status_code=502, detail="Search backend unavailable") from exc

    scores = predict_scores(docs)
    ranked = rerank(docs, scores, settings.search_weight, settings.model_weight)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info("query=%r department=%r results=%d reranked=%s latency_ms=%.1f",
                q, department, len(ranked), scores is not None, latency_ms)

    return SearchResponse(
        query=q,
        department=department,
        reranked=scores is not None,
        latency_ms=latency_ms,
        results=[Product(**{k2: v for k2, v in d.items() if k2 in Product.model_fields}) for d in ranked],
    )
