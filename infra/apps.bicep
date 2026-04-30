// Cascade Container Apps — Layer 1, Layer 2, Layer 3, and Router.
// Deployed AFTER the base infrastructure exists and images are pushed to ACR.
//
// Layers 1/2 run on the Consumption profile (CPU). They scale 0→N on HTTP traffic.
// Layer 3 also runs CPU (calls Azure OpenAI for inference).
// The router is the only externally-ingressed app.

targetScope = 'resourceGroup'

@description('Name of the existing ACA managed environment.')
param environmentName string

@description('Name of the existing Azure Container Registry.')
param acrName string

@description('ACR login server (e.g. myreg.azurecr.io). Saves a runtime lookup.')
param acrLoginServer string

@description('Tag of the images in ACR (e.g. git short SHA).')
param imageTag string = 'latest'

@description('Storage account name for managed-identity blob access.')
param storageAccountName string

@description('Azure OpenAI endpoint URL.')
param aoaiEndpoint string

@description('Azure OpenAI deployment name.')
@secure()
param aoaiApiKey string

param aoaiDeployment string = 'oracle'
param location string = resourceGroup().location

@description('Layer 1 autoencoder MSE threshold above which a frame escalates (legacy AE scorer; ignored when SCORER=patchcore).')
param mseThreshold string = '0.0098'

@description('PatchCore z-score threshold for the Severstal domain (Phase L calibrated knee).')
param zThresholdSeverstal string = '-0.5'

@description('PatchCore z-score threshold for the KSDD2 domain (Phase L calibrated knee).')
param zThresholdKsdd2 string = '1.0'

@description('Daily USD cap on Azure OpenAI spend for the live demo (Layer 3).')
param aoaiDailyUsdCap string = '5.0'

@description('Per-IP rate limit applied to /predict on the public router.')
param routerRateLimit string = '10/minute;200/day'

var commonTags = {
  project: 'cascade-defect'
  managedBy: 'bicep'
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

// Shared user-assigned identity so each app can pull from ACR + read Blob.
resource uami 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'cascade-apps-uami'
  location: location
  tags: commonTags
}

// Allow the UAMI to pull from ACR.
resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, uami.id, 'AcrPull')
  scope: acr
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
    principalId: uami.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Allow the UAMI to read blobs (model artifacts).
resource blobReader 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, uami.id, 'StorageBlobDataReader')
  scope: storage
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1')
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

// ─── Layer 1 ─────────────────────────────────────────────────────────────────
resource layer1 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'cascade-layer1'
  location: location
  tags: commonTags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${uami.id}': {} }
  }
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      registries: registries
      ingress: {
        external: false
        targetPort: 8000
        transport: 'http'
      }
    }
    template: {
      containers: [
        {
          name: 'layer1'
          image: '${acrLoginServer}/cascade-layer1:${imageTag}'
          resources: { cpu: json('0.5'), memory: '1Gi' }
          env: [
            { name: 'SCORER', value: 'patchcore' }
            { name: 'PATCHCORE_DIR', value: '/app/models/patchcore_metal' }
            { name: 'MSE_THRESHOLD', value: mseThreshold }
            { name: 'Z_THRESHOLD', value: '3.0' }
            { name: 'Z_THRESHOLD_SEVERSTAL', value: zThresholdSeverstal }
            { name: 'Z_THRESHOLD_KSDD2', value: zThresholdKsdd2 }
            { name: 'DEFAULT_DOMAIN', value: 'severstal' }
            { name: 'IMAGE_SIZE', value: '224' }
          ]
        }
      ]
      scale: { minReplicas: 0, maxReplicas: 3 }
    }
  }
}

// ─── Layer 2 ─────────────────────────────────────────────────────────────────
resource layer2 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'cascade-layer2'
  location: location
  tags: commonTags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${uami.id}': {} }
  }
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      registries: registries
      ingress: {
        external: false
        targetPort: 8000
        transport: 'http'
      }
    }
    template: {
      containers: [
        {
          name: 'layer2'
          image: '${acrLoginServer}/cascade-layer2:${imageTag}'
          resources: { cpu: json('1'), memory: '2Gi' }
        }
      ]
      scale: { minReplicas: 0, maxReplicas: 3 }
    }
  }
}

// ─── Layer 3 (Oracle) ────────────────────────────────────────────────────────
resource layer3 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'cascade-layer3'
  location: location
  tags: commonTags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${uami.id}': {} }
  }
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      registries: registries
      ingress: {
        external: false
        targetPort: 8000
        transport: 'http'
      }
      secrets: [
        {
          name: 'aoai-api-key'
          value: aoaiApiKey
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'layer3'
          image: '${acrLoginServer}/cascade-layer3:${imageTag}'
          resources: { cpu: json('0.5'), memory: '1Gi' }
          env: [
            { name: 'AOAI_ENDPOINT', value: aoaiEndpoint }
            { name: 'AOAI_DEPLOYMENT', value: aoaiDeployment }
            { name: 'AOAI_API_VERSION', value: '2024-10-21' }
            { name: 'AOAI_API_KEY', secretRef: 'aoai-api-key' }
            { name: 'AOAI_DAILY_USD_CAP', value: aoaiDailyUsdCap }
            { name: 'USE_ORACLE_CACHE', value: '1' }
            { name: 'DEFAULT_DOMAIN', value: 'severstal' }
          ]
        }
      ]
      scale: { minReplicas: 0, maxReplicas: 2 }
    }
  }
}

// ─── Router (only external ingress) ──────────────────────────────────────────
resource router 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'cascade-router'
  location: location
  tags: commonTags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${uami.id}': {} }
  }
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      registries: registries
      ingress: {
        external: true
        targetPort: 8000
        transport: 'http'
        allowInsecure: false
      }
    }
    template: {
      containers: [
        {
          name: 'router'
          image: '${acrLoginServer}/cascade-router:${imageTag}'
          resources: { cpu: json('0.25'), memory: '0.5Gi' }
          env: [
            { name: 'LAYER1_URL', value: 'https://${layer1.properties.configuration.ingress.fqdn}' }
            { name: 'LAYER2_URL', value: 'https://${layer2.properties.configuration.ingress.fqdn}' }
            { name: 'LAYER3_URL', value: 'https://${layer3.properties.configuration.ingress.fqdn}' }
            { name: 'L2_CONF_ESCALATE_BELOW', value: '0.7' }
            { name: 'STATIC_DIR', value: '/app/static' }
            { name: 'MAX_UPLOAD_BYTES', value: '2097152' }
            { name: 'MAX_IMAGE_DIM', value: '2048' }
            { name: 'RATE_LIMIT_PREDICT', value: routerRateLimit }
          ]
        }
      ]
      scale: { minReplicas: 0, maxReplicas: 5 }
    }
  }
}

output routerFqdn string = router.properties.configuration.ingress.fqdn
output layer1Fqdn string = layer1.properties.configuration.ingress.fqdn
output layer2Fqdn string = layer2.properties.configuration.ingress.fqdn
output layer3Fqdn string = layer3.properties.configuration.ingress.fqdn
