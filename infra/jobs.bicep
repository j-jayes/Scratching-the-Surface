// =============================================================================
// ACA Jobs — one-shot training runs for the Phase J metal-surface refit.
// =============================================================================
// Three jobs, all triggered manually::
//
//   az containerapp job start --name cascade-ae-train-metal      -g cascade-dev-rg
//   az containerapp job start --name cascade-patchcore-train-metal -g cascade-dev-rg
//   az containerapp job start --name cascade-yolo-train-metal    -g cascade-dev-rg
//
// They share the same managed env + UAMI as the always-on apps and all
// persist artefacts to the ``models`` blob container under the same
// identity used by the inference workloads.
// =============================================================================

targetScope = 'resourceGroup'

@description('Existing ACA managed environment.')
param environmentName string

@description('Existing ACR name (e.g. cascadedevacr6ya7a3).')
param acrName string

@description('ACR login server (e.g. cascadedevacr6ya7a3.azurecr.io).')
param acrLoginServer string

@description('Image tag (typically the git short SHA).')
param imageTag string = 'latest'

@description('Training image — typically the cascade-base or a dedicated trainer image.')
param trainerImageName string = 'cascade-base'

@description('YOLO trainer image — needs CUDA torch wheels for the gpu-t4 profile.')
param yoloTrainerImageName string = 'cascade-base-gpu'

@description('Storage account name for blob read/write.')
param storageAccountName string

@description('Whether the YOLO trainer should run on the GPU workload profile.')
param yoloUseGpu bool = true

@description('Maximum job runtime in seconds (per replica). Default 4 h.')
param replicaTimeoutSeconds int = 14400

@description('MLflow tracking URI (typically the AML workspace URI). Empty disables MLflow.')
param mlflowTrackingUri string = ''

@description('Whether the YOLO trainer should include KSDD2 defectives as a 5th class (collapses Track C into B).')
param yoloIncludeKsdd2 bool = false

@description('Z-score threshold the AE trainer should bake into calibration.json.')
param aeZThreshold string = '3.0'

param location string = resourceGroup().location

var commonTags = {
  project: 'cascade-defect'
  managedBy: 'bicep'
  workload: 'training'
  phase: 'J'
}

resource env 'Microsoft.App/managedEnvironments@2024-03-01' existing = {
  name: environmentName
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: acrName
}

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' existing = {
  name: storageAccountName
}

// Reuse the existing UAMI from apps.bicep so all training reads/writes happen
// under the same identity that the inference apps use.
resource uami 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' existing = {
  name: 'cascade-apps-uami'
}

// Training jobs need WRITE access to Blob (the inference UAMI is Reader only).
resource blobWriter 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, uami.id, 'StorageBlobDataContributor', 'training')
  scope: storage
  properties: {
    // Storage Blob Data Contributor
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')
    principalId: uami.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

var registries = [
  {
    server: acrLoginServer
    identity: uami.id
  }
]

var commonEnv = [
  { name: 'BLOB_ACCOUNT', value: storageAccountName }
  { name: 'BLOB_CONTAINER_RAW', value: 'raw' }
  { name: 'BLOB_CONTAINER_MODELS', value: 'models' }
  { name: 'AZURE_CLIENT_ID', value: uami.properties.clientId }
  { name: 'PYTHONUNBUFFERED', value: '1' }
  { name: 'MLFLOW_TRACKING_URI', value: mlflowTrackingUri }
]

// ─── AE trainer (CPU is fine — model is tiny) ────────────────────────────────
resource aeTrainJob 'Microsoft.App/jobs@2024-03-01' = {
  name: 'cascade-ae-train-metal'
  location: location
  tags: commonTags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${uami.id}': {} }
  }
  properties: {
    environmentId: env.id
    workloadProfileName: 'Consumption'
    configuration: {
      triggerType: 'Manual'
      replicaTimeout: 7200
      replicaRetryLimit: 0
      manualTriggerConfig: {
        replicaCompletionCount: 1
        parallelism: 1
      }
      registries: registries
    }
    template: {
      containers: [
        {
          name: 'ae-train'
          image: '${acrLoginServer}/${trainerImageName}:${imageTag}'
          command: [
            'python'
            '-u'
            '-m'
            'cascade_defect.layer1_autoencoder.train_metal'
          ]
          args: [
            '--data-dir'
            '/work/data/splits_metal'
            '--output-dir'
            '/work/models/autoencoder_metal'
            '--epochs'
            '15'
            '--image-size'
            '256'
            '--z-threshold'
            aeZThreshold
          ]
          resources: { cpu: json('2'), memory: '4Gi' }
          env: commonEnv
        }
      ]
    }
  }
  dependsOn: [
    blobWriter
  ]
}

// ─── PatchCore-lite trainer (CPU; ResNet18 features, no decoder) ────────────
resource patchcoreTrainJob 'Microsoft.App/jobs@2024-03-01' = {
  name: 'cascade-patchcore-train-metal'
  location: location
  tags: commonTags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${uami.id}': {} }
  }
  properties: {
    environmentId: env.id
    workloadProfileName: 'Consumption'
    configuration: {
      triggerType: 'Manual'
      replicaTimeout: 7200
      replicaRetryLimit: 0
      manualTriggerConfig: {
        replicaCompletionCount: 1
        parallelism: 1
      }
      registries: registries
    }
    template: {
      containers: [
        {
          name: 'patchcore-train'
          image: '${acrLoginServer}/${trainerImageName}:${imageTag}'
          command: [
            'python'
            '-u'
            '-m'
            'cascade_defect.layer1_autoencoder.train_patchcore'
          ]
          args: [
            '--data-dir'
            '/work/data/splits_metal'
            '--output-dir'
            '/work/models/patchcore_metal'
            '--image-size'
            '224'
            '--bank-fraction'
            '0.10'
          ]
          resources: { cpu: json('4'), memory: '8Gi' }
          env: commonEnv
        }
      ]
    }
  }
  dependsOn: [
    blobWriter
  ]
}

// ─── YOLO trainer (GPU profile preferred; falls back to CPU) ────────────────
resource yoloTrainJob 'Microsoft.App/jobs@2024-03-01' = {
  name: 'cascade-yolo-train-metal'
  location: location
  tags: commonTags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${uami.id}': {} }
  }
  properties: {
    environmentId: env.id
    workloadProfileName: yoloUseGpu ? 'gpu-t4' : 'Consumption'
    configuration: {
      triggerType: 'Manual'
      replicaTimeout: replicaTimeoutSeconds
      replicaRetryLimit: 0
      manualTriggerConfig: {
        replicaCompletionCount: 1
        parallelism: 1
      }
      registries: registries
    }
    template: {
      containers: [
        {
          name: 'yolo-train'
          image: '${acrLoginServer}/${yoloTrainerImageName}:${imageTag}'
          // Dataset (~500 MB of images + labels + portable data.yaml + yolov8n.pt)
          // is pre-staged on the ``yolo-data`` Azure File share mounted at
          // ``/work/yolo-data``. The trainer rebuilds the oversampled manifest
          // in-place and writes ``best.pt`` + ``summary.json`` back to the share.
          command: [
            'python'
            '-u'
            '-m'
            'cascade_defect.layer2_yolo.train_metal'
          ]
          args: [
            '--dataset-dir'
            '/work/yolo-data'
            '--output-dir'
            '/work/yolo-data/runs'
            '--base-weights'
            '/work/yolo-data/yolov8n.pt'
            '--epochs'
            '50'
            '--image-size'
            '640'
            '--batch-size'
            yoloUseGpu ? '16' : '4'
            '--device'
            yoloUseGpu ? '0' : ''
          ]
          // GPU profile gives 8 vCPU + 56 GiB; CPU profile we keep modest.
          resources: yoloUseGpu ? { cpu: json('8'), memory: '56Gi' } : { cpu: json('2'), memory: '4Gi' }
          env: commonEnv
          volumeMounts: [
            {
              mountPath: '/work/yolo-data'
              volumeName: 'yolo-data'
            }
            {
              mountPath: '/dev/shm'
              volumeName: 'dshm'
            }
          ]
        }
      ]
      volumes: [
        {
          name: 'yolo-data'
          storageType: 'AzureFile'
          storageName: 'yolo-data'
        }
        {
          name: 'dshm'
          storageType: 'EmptyDir'
        }
      ]
    }
  }
  dependsOn: [
    blobWriter
  ]
}

output aeJobName string = aeTrainJob.name
output patchcoreJobName string = patchcoreTrainJob.name
output yoloJobName string = yoloTrainJob.name
