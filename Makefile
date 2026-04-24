# =============================================================================
# Project Cascade Defect — automation entrypoints
# =============================================================================
# All Azure operations target the subscription set in .env (AZURE_SUBSCRIPTION_ID).
# Run `make help` for a list of available targets.
# =============================================================================

SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

# ── Load .env if present (no-op otherwise) ───────────────────────────────────
ifneq (,$(wildcard .env))
	include .env
	export
endif

# ── Defaults (override via .env or CLI: `make infra-up LOCATION=...`) ────────
PROJECT_NAME      ?= cascade
ENV_NAME          ?= dev
LOCATION          ?= westeurope
RESOURCE_GROUP    ?= $(PROJECT_NAME)-$(ENV_NAME)-rg
DEPLOYMENT_NAME   ?= $(PROJECT_NAME)-$(ENV_NAME)-$(shell date +%Y%m%d-%H%M%S)
PARAM_FILE        ?= infra/main.parameters.json
TEMPLATE_FILE     ?= infra/main.bicep

.DEFAULT_GOAL := help

# =============================================================================
# Help
# =============================================================================
help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Local dev
# =============================================================================
sync: ## Install Python deps via uv.
	uv sync

test: ## Run pytest suite.
	uv run pytest tests/ -v

lint: ## Lint with ruff.
	uv run ruff check .
	uv run ruff format --check .

fmt: ## Auto-format with ruff.
	uv run ruff format .
	uv run ruff check --fix .

# =============================================================================
# Azure infrastructure (Bicep)
# =============================================================================
az-login: ## Set the active subscription from .env.
	@if [ -z "$$AZURE_SUBSCRIPTION_ID" ]; then echo "AZURE_SUBSCRIPTION_ID not set"; exit 1; fi
	az account set --subscription "$$AZURE_SUBSCRIPTION_ID"
	az account show --query "{name:name,id:id,tenantId:tenantId}" -o table

infra-rg: az-login ## Create the resource group if missing.
	az group create -n $(RESOURCE_GROUP) -l $(LOCATION) -o table

infra-plan: infra-rg ## Preview infrastructure changes (what-if).
	az deployment group what-if \
	  --resource-group $(RESOURCE_GROUP) \
	  --template-file $(TEMPLATE_FILE) \
	  --parameters @$(PARAM_FILE)

infra-up: infra-rg ## Deploy / update Azure infrastructure.
	az deployment group create \
	  --name $(DEPLOYMENT_NAME) \
	  --resource-group $(RESOURCE_GROUP) \
	  --template-file $(TEMPLATE_FILE) \
	  --parameters @$(PARAM_FILE) \
	  -o table

infra-show: az-login ## Show outputs from the most recent deployment.
	az deployment group list \
	  --resource-group $(RESOURCE_GROUP) \
	  --query "[0].properties.outputs" -o json

infra-down: az-login ## Delete the resource group (irreversible).
	@read -p "Delete resource group $(RESOURCE_GROUP)? [y/N] " ans; \
	  if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
	    az group delete -n $(RESOURCE_GROUP) --yes --no-wait; \
	    echo "Deletion started (no-wait)."; \
	  else echo "Aborted."; fi

# =============================================================================
# Container images (ACR remote build — no local Docker required)
# =============================================================================
IMAGE_TAG ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "dev")

images-build: az-login ## Build all images in ACR (base + 4 layer images).
	az acr build --registry $(ACR_NAME) --image cascade-base:$(IMAGE_TAG)   --image cascade-base:latest   --file docker/base.Dockerfile   .
	az acr build --registry $(ACR_NAME) --image cascade-layer1:$(IMAGE_TAG) --image cascade-layer1:latest --file docker/layer1.Dockerfile --build-arg REGISTRY=$(ACR_LOGIN_SERVER) --build-arg BASE_TAG=$(IMAGE_TAG) .
	az acr build --registry $(ACR_NAME) --image cascade-layer2:$(IMAGE_TAG) --image cascade-layer2:latest --file docker/layer2.Dockerfile --build-arg REGISTRY=$(ACR_LOGIN_SERVER) --build-arg BASE_TAG=$(IMAGE_TAG) .
	az acr build --registry $(ACR_NAME) --image cascade-layer3:$(IMAGE_TAG) --image cascade-layer3:latest --file docker/layer3.Dockerfile --build-arg REGISTRY=$(ACR_LOGIN_SERVER) --build-arg BASE_TAG=$(IMAGE_TAG) .
	az acr build --registry $(ACR_NAME) --image cascade-router:$(IMAGE_TAG) --image cascade-router:latest --file docker/router.Dockerfile --build-arg REGISTRY=$(ACR_LOGIN_SERVER) --build-arg BASE_TAG=$(IMAGE_TAG) .

images-push: images-build ## Alias for images-build (ACR builds push automatically).
	@echo "ACR build pushes images automatically — nothing more to do."

apps-deploy: az-login ## Deploy the four Container Apps (router + 3 layers).
	@if [ -z "$$AOAI_API_KEY" ]; then echo "AOAI_API_KEY missing in .env"; exit 1; fi
	az deployment group create \
	  --name "cascade-apps-$$(date +%Y%m%d-%H%M%S)" \
	  --resource-group $(RESOURCE_GROUP) \
	  --template-file infra/apps.bicep \
	  --parameters \
	    environmentName=$(ACA_ENVIRONMENT) \
	    acrName=$(ACR_NAME) \
	    storageAccountName=$(BLOB_ACCOUNT) \
	    aoaiEndpoint=$(AOAI_ENDPOINT) \
	    aoaiApiKey=$(AOAI_API_KEY) \
	    aoaiDeployment=$(AOAI_DEPLOYMENT) \
	    imageTag=$(IMAGE_TAG) \
	  -o table

apps-show: az-login ## Print router URL.
	@az containerapp show -n cascade-router -g $(RESOURCE_GROUP) --query "properties.configuration.ingress.fqdn" -o tsv

# =============================================================================
# Data
# =============================================================================
data-fetch: ## Download NEU dataset from Kaggle (Phase D).
	uv run python -m cascade_defect.data.ingest

data-upload: ## Upload local data/raw to Blob (Phase D).
	uv run python -m cascade_defect.data.upload

data-split: ## Generate train/val/test splits (Phase D).
	uv run python -m cascade_defect.data.split

.PHONY: help sync test lint fmt az-login infra-rg infra-plan infra-up infra-show infra-down images-build images-push apps-deploy apps-show data-fetch data-upload data-split
