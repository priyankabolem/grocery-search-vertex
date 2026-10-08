"""Undeploy and delete the Vertex AI endpoint so it stops billing.

The registered model, BigQuery table and Cloud Storage files are kept (negligible cost),
so the endpoint can be recreated later with 03_train_and_deploy.py.
"""
from __future__ import annotations

from google.cloud import aiplatform

from grocery_search.config import Settings


def main() -> None:
    s = Settings.from_env()
    s.require("endpoint_id")
    aiplatform.init(project=s.project_id, location=s.region)

    endpoint = aiplatform.Endpoint(s.endpoint_id)
    endpoint.undeploy_all()
    endpoint.delete()
    print(f"Endpoint {s.endpoint_id} undeployed and deleted. Clear ENDPOINT_ID in .env.")


if __name__ == "__main__":
    main()
