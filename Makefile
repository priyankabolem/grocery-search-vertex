.PHONY: install test lint prepare load train serve deploy cleanup

install:            ## Install runtime + dev dependencies and the package (editable)
	pip install -r requirements-dev.txt
	pip install -e . --no-deps

test:               ## Run unit tests
	pytest -q

lint:               ## Lint and check import order
	ruff check .

prepare:            ## Build data/catalog.csv from raw Instacart files
	python scripts/01_prepare_data.py

load:               ## Upload catalog to Cloud Storage and BigQuery
	python scripts/02_load_to_gcp.py

train:              ## Train, register and deploy the model to a Vertex AI endpoint
	python scripts/03_train_and_deploy.py

serve:              ## Run the API locally with auto-reload
	uvicorn grocery_search.api:app --reload

deploy:             ## Build and deploy the API to Cloud Run
	gcloud run deploy grocery-search --source . --region $(REGION) --allow-unauthenticated \
	  --set-env-vars PROJECT_ID=$(PROJECT_ID),REGION=$(REGION),SEARCH_ENGINE_ID=$(SEARCH_ENGINE_ID),ENDPOINT_ID=$(ENDPOINT_ID)

cleanup:            ## Undeploy and delete the Vertex AI endpoint
	python scripts/04_cleanup.py
