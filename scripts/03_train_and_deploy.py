"""Train the reorder-propensity model, register it in Vertex AI, and deploy it to an endpoint.

Use --skip-deploy to train and evaluate locally without creating cloud resources.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from grocery_search.config import FEATURES, TARGET, Settings

ARTIFACTS = Path("artifacts")
# Vertex AI prebuilt scikit-learn serving container; must match the local scikit-learn minor version.
SERVING_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"


def train(df: pd.DataFrame) -> tuple[GradientBoostingRegressor, dict[str, float]]:
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    baseline = np.full_like(y_te, y_tr.mean(), dtype=float)
    metrics = {
        "test_mae": round(mean_absolute_error(y_te, pred), 4),
        "baseline_mae": round(mean_absolute_error(y_te, baseline), 4),
        "test_r2": round(r2_score(y_te, pred), 4),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
    }
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-deploy", action="store_true", help="Train and evaluate only")
    args = parser.parse_args()

    model, metrics = train(pd.read_csv("data/catalog.csv"))
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS / "model.joblib")  # filename required by the prebuilt container
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Metrics:", json.dumps(metrics))

    if args.skip_deploy:
        return

    from google.cloud import aiplatform, storage

    s = Settings.from_env()
    s.require("bucket")
    storage.Client(project=s.project_id).bucket(s.bucket).blob("model/model.joblib") \
        .upload_from_filename(str(ARTIFACTS / "model.joblib"))

    aiplatform.init(project=s.project_id, location=s.region)
    registered = aiplatform.Model.upload(
        display_name="grocery-reorder-model",
        artifact_uri=f"gs://{s.bucket}/model/",
        serving_container_image_uri=SERVING_IMAGE,
        labels={"project": "grocery-search"},
    )
    print(f"Registered model: {registered.resource_name}")

    print("Deploying to endpoint (typically 10-20 minutes)...")
    endpoint = registered.deploy(machine_type="n1-standard-2", min_replica_count=1, max_replica_count=1)
    print(f"ENDPOINT_ID={endpoint.name}")

    sample = pd.read_csv("data/catalog.csv")[FEATURES].head(3).values.tolist()
    print("Smoke test predictions:", endpoint.predict(instances=sample).predictions)
    print("Endpoint bills hourly while deployed; run scripts/04_cleanup.py when finished.")


if __name__ == "__main__":
    main()
