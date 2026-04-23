# Skill: Azure Container Apps (ACA) — GPU Deployment in West Europe

## Context
This project deploys ML inference containers on **Azure Container Apps** in the **West Europe** region
using **NVIDIA T4 GPUs** (A100s are *not* available for ACA Consumption workload profiles in West Europe).

---

## Key CLI Patterns

### 1. Create an ACA Environment with GPU Workload Profile
```bash
az containerapp env create \
  --name cascade-defect-env \
  --resource-group cascade-defect-rg \
  --location westeurope \
  --enable-workload-profiles \
  --workload-profile-name gpu-t4 \
  --workload-profile-type Consumption-GPU-NC8as-T4 \
  --min-count 0 \
  --max-count 10
```

### 2. Deploy a Container App (Layer 2 — YOLOv8) with T4 GPU
```bash
az containerapp create \
  --name layer2-yolo \
  --resource-group cascade-defect-rg \
  --environment cascade-defect-env \
  --workload-profile-name gpu-t4 \
  --image <acr-name>.azurecr.io/layer2-yolo:latest \
  --registry-server <acr-name>.azurecr.io \
  --min-replicas 0 \
  --max-replicas 5 \
  --cpu 8 \
  --memory 56Gi \
  --env-vars \
      AZURE_STORAGE_ACCOUNT_NAME=secretref:storage-account-name \
      AZURE_OPENAI_ENDPOINT=secretref:openai-endpoint
```

### 3. KEDA Scale Rule — Azure Service Bus Queue
```bash
az containerapp update \
  --name layer2-yolo \
  --resource-group cascade-defect-rg \
  --scale-rule-name servicebus-keda \
  --scale-rule-type azure-servicebus \
  --scale-rule-metadata \
      queueName=defect-queue \
      messageCount=1 \
  --scale-rule-auth \
      trigger=connection \
      secretRef=servicebus-connection-string
```

### 4. Build & Push Image to Azure Container Registry
```bash
az acr build \
  --registry <acr-name> \
  --image layer2-yolo:latest \
  --file docker/layer2.Dockerfile \
  .
```

---

## Important Constraints

| Constraint | Detail |
|-----------|--------|
| **Region** | Always use `westeurope` |
| **GPU SKU** | `Consumption-GPU-NC8as-T4` — T4 only (A100 not supported in ACA West Europe) |
| **Quota** | Default GPU quota is **0**. Open Azure Portal support ticket for "Managed Environment Consumption T4 Gpus" — allow 24–48 h |
| **Scale-to-zero** | Set `--min-replicas 0`; cold-start penalty is 30–90 s |
| **Image pull** | Always use ACR with Managed Identity auth, not admin credentials |

---

## Resource Naming Convention
```
Resource Group : cascade-defect-rg
ACA Environment: cascade-defect-env
ACR            : cascadedefectacr  (no hyphens — ACR names must be alphanumeric)
Layer 1 App    : layer1-autoencoder
Layer 2 App    : layer2-yolo
Service Bus NS : cascade-defect-sb
Service Bus Q  : defect-queue
ADLS Account   : cascadedefectadls
AML Workspace  : cascade-defect-aml
```

---

## Azure OpenAI — West Europe Fallback
West Europe may face GPT-4o quota limits. If so, deploy the Azure OpenAI resource in
`swedencentral` (nearest region with reliable GPT-4o quota) and reference the endpoint
via an environment variable `AZURE_OPENAI_ENDPOINT`.

```bash
az cognitiveservices account create \
  --name cascade-openai \
  --resource-group cascade-defect-rg \
  --kind OpenAI \
  --sku S0 \
  --location swedencentral

az cognitiveservices account deployment create \
  --name cascade-openai \
  --resource-group cascade-defect-rg \
  --deployment-name gpt-4o \
  --model-name gpt-4o \
  --model-version "2024-08-06" \
  --model-format OpenAI \
  --sku-capacity 10 \
  --sku-name GlobalStandard
```
