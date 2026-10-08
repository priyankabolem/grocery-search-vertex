"""Runtime configuration, read once from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass

# Model input features, in the order the model was trained on.
FEATURES: list[str] = ["order_count", "avg_cart_position", "aisle_id", "department_id"]
TARGET: str = "reorder_rate"


@dataclass(frozen=True)
class Settings:
    project_id: str
    region: str = "us-central1"
    bucket: str | None = None
    bq_dataset: str = "grocery"
    search_engine_id: str | None = None
    endpoint_id: str | None = None
    search_weight: float = 0.7
    model_weight: float = 0.3

    @classmethod
    def from_env(cls) -> Settings:
        project_id = os.environ.get("PROJECT_ID")
        if not project_id:
            raise RuntimeError("PROJECT_ID is not set. Copy .env.example to .env and load it.")
        return cls(
            project_id=project_id,
            region=os.environ.get("REGION", "us-central1"),
            bucket=os.environ.get("BUCKET") or None,
            bq_dataset=os.environ.get("BQ_DATASET", "grocery"),
            search_engine_id=os.environ.get("SEARCH_ENGINE_ID") or None,
            endpoint_id=os.environ.get("ENDPOINT_ID") or None,
            search_weight=float(os.environ.get("SEARCH_WEIGHT", 0.7)),
            model_weight=float(os.environ.get("MODEL_WEIGHT", 0.3)),
        )

    def require(self, *names: str) -> None:
        """Fail fast with a clear message when a script needs a setting that is missing."""
        missing = [n for n in names if not getattr(self, n)]
        if missing:
            env_names = ", ".join(n.upper() for n in missing)
            raise RuntimeError(f"Missing required setting(s): {env_names}. Set them in .env.")
