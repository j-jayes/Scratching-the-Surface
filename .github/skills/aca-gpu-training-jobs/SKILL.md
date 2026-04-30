---
name: aca-gpu-training-jobs
description: 'Deploy GPU training jobs (YOLO/PyTorch) on Azure Container Apps Jobs. Use when: setting up cascade-yolo-train-metal or similar GPU jobs; debugging ACA job failures (FileNotFoundError on data, CUDA not available, OSError 28 No space left on device, DataLoader Bus error, ACR build "unexpected dockerfile format"); mounting Azure File shares or EmptyDir into ACA jobs; building CUDA torch overlay images; uploading datasets via PowerShell to storage accounts with special-char keys.'
---

# Azure Container Apps GPU Training Jobs

Hard-won fixes for running long training jobs (YOLOv8 on Consumption-GPU-NC8as-T4) on ACA Jobs in this workspace. All issues observed during Phase J.4 of the cascade-defect refit.

## When to Use

- Adding/modifying a training job in [infra/jobs.bicep](infra/jobs.bicep)
- Debugging a failed `cascade-yolo-train-metal` (or sibling) execution
- Building a new ACR image that needs CUDA torch
- Uploading a dataset to the `yolo-data` Azure File share

## The Six Fixes (apply in this order)

### 1. Mount data via Azure File share, not blob — and skip in-job data prep

ACA Jobs cannot mount blob containers directly; only Azure File / EmptyDir / Secret. Pre-stage the prepared dataset on a file share rather than running data-prep inside the job.

```bicep
volumeMounts: [
  { mountPath: '/work/yolo-data', volumeName: 'yolo-data' }
]
volumes: [
  { name: 'yolo-data', storageType: 'AzureFile', storageName: 'yolo-data' }
]
```

The `storageName` must first be registered with the managed env:

```powershell
az containerapp env storage set --name cascade-dev-aca-env --resource-group cascade-dev-rg `
  --storage-name yolo-data --azure-file-account-name cascadedev6ya7a3px `
  --azure-file-account-key $env:AZURE_STORAGE_KEY --azure-file-share-name yolo-data `
  --access-mode ReadWrite
```

Use a portable `data.yaml` with `path: /work/yolo-data` (not local paths).

### 2. Always upload via PowerShell `$env:` — never cmd `%VAR%`

Storage account keys contain `=`, `/`, `+`. cmd.exe `%VAR%` expansion mangles these and gives `Authentication failure`.

```powershell
$env:AZURE_STORAGE_KEY = (az storage account keys list ... --query "[0].value" -o tsv)
az storage file upload-batch --account-name cascadedev6ya7a3px --auth-mode key `
  --destination yolo-data --source $env:TEMP\yolo-upload
```

### 3. Mount EmptyDir at `/dev/shm` — the 64 MiB cap is fatal for PyTorch

ACA's default `/dev/shm` is 64 MiB and **not configurable** via API. Any `multiprocessing.Pool`, `ThreadPool`, or PyTorch `DataLoader(num_workers>0)` will eventually crash with `OSError: [Errno 28] No space left on device` in `SemLock.__init__`, often surfacing as `DataLoader worker (pid N) is killed by signal: Bus error`.

The fix: mount an `EmptyDir` volume at `/dev/shm`. ACA gives EmptyDir ~1 GiB which is plenty.

```bicep
volumeMounts: [
  { mountPath: '/work/yolo-data', volumeName: 'yolo-data' }
  { mountPath: '/dev/shm', volumeName: 'dshm' }
]
volumes: [
  { name: 'yolo-data', storageType: 'AzureFile', storageName: 'yolo-data' }
  { name: 'dshm', storageType: 'EmptyDir' }
]
```

Monkey-patching `ultralytics.utils.NUM_THREADS` or replacing `multiprocessing.pool.ThreadPool` with a `concurrent.futures` shim does **not** fully work — Pool itself, and PyTorch's DataLoader workers, still call `SemLock`. Mount the volume; don't try to dodge the problem in Python.

### 4. Build a CUDA-torch overlay image, parameterize separately

`cascade-base:latest` ships CPU-only torch (`torch-2.x+cpu`). On gpu-t4 it crashes with `Invalid CUDA 'device=0' requested ... torch.cuda.is_available(): False`. Build a sibling image:

```dockerfile
# docker/base.phasej-gpu.Dockerfile
FROM cascadedevacr6ya7a3.azurecr.io/cascade-base:latest
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
RUN /usr/local/bin/uv pip install --python /app/.venv/bin/python --reinstall --index-url https://download.pytorch.org/whl/cu121 torch torchvision
RUN /app/.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.version.cuda)"
```

Add a separate bicep param so CPU jobs (AE / PatchCore) keep using the small image:

```bicep
param trainerImageName string = 'cascade-base'           // CPU
param yoloTrainerImageName string = 'cascade-base-gpu'   // CUDA 12.1
```

### 5. ACR `az acr build` is picky about Dockerfile format

The ACR build scanner rejects with `unexpected dockerfile format` if you use:

- **Multi-line `RUN \\` continuations** — collapse to single-line `RUN`
- **`ARG` before `FROM ${ARG}`** — inline the registry/tag

Also: `.dockerignore` parsing is unreliable. Stage the build context manually:

```powershell
$ctx = "$env:TEMP\cascade-acr-context"
Remove-Item -Recurse -Force $ctx -ErrorAction Ignore
New-Item -ItemType Directory -Path $ctx | Out-Null
Copy-Item pyproject.toml, uv.lock $ctx
Copy-Item -Recurse src $ctx\src
Copy-Item -Recurse docker $ctx\docker
cd $ctx
az acr build --registry cascadedevacr6ya7a3 --image cascade-base-gpu:latest --file docker/base.phasej-gpu.Dockerfile .
```

Building FROM a cached ACR image (`cascade-base:latest`) avoids Docker Hub rate-limits on the public `python:3.11-slim` base.

### 6. Trigger + verify pattern

```powershell
$exec = az containerapp job start --name cascade-yolo-train-metal --resource-group cascade-dev-rg --query "name" -o tsv
Write-Host "exec=$exec"
Start-Sleep 360
az containerapp job execution show --name cascade-yolo-train-metal --resource-group cascade-dev-rg `
  --job-execution-name $exec --query "{state:properties.status}" -o tsv
az containerapp job logs show --name cascade-yolo-train-metal --resource-group cascade-dev-rg `
  --container yolo-train --execution $exec --tail 60 2>&1 |
  Select-String -NotMatch '^WARNING|At line|CategoryInfo|FullyQualified' |
  Select-Object -Last 30
```

Healthy startup signs in logs:
- `torch X.Y.Z+cu121 CUDA:0 (Tesla T4 ...)` — CUDA wheel installed
- `Class counts before: {...}` — data mount works
- `1/50 ... 460/460 ... 8.x it/s` — training proceeds past `cache_labels` (means /dev/shm fix worked)

## Anti-patterns observed in this repo

- Doing dataset prep inside the GPU job — slow, wastes GPU minutes, and depends on raw data being mounted. Prep locally, push artefacts to the share.
- Adding `--shm-size` flags or trying to remount `/dev/shm` with `mount -o size=...` — neither is permitted in ACA. Only EmptyDir works.
- Patching Python's `multiprocessing` to "avoid" SemLock — Pool/SimpleQueue still hits `/dev/shm` even with `num_workers=0`. Fix the mount, leave the code alone.
- Reusing `trainerImageName` for both CPU and GPU jobs — rebuilding the CPU base with CUDA torch bloats it from ~1 GB to ~6 GB and slows AE/PatchCore cold starts.

## File map

- [infra/jobs.bicep](infra/jobs.bicep) — job defs, volumes, mounts, image params
- [docker/base.phasej.Dockerfile](docker/base.phasej.Dockerfile) — CPU base overlay
- [docker/base.phasej-gpu.Dockerfile](docker/base.phasej-gpu.Dockerfile) — CUDA torch overlay
- [src/cascade_defect/layer2_yolo/train_metal.py](src/cascade_defect/layer2_yolo/train_metal.py) — YOLO trainer entrypoint
- [models/yolo_metal/dataset/data.azure.yaml](models/yolo_metal/dataset/data.azure.yaml) — portable dataset YAML for the share
