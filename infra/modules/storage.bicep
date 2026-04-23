// Blob storage account with the four containers used by the cascade pipeline.
// Hierarchical namespace is OFF — flat blob is sufficient and cheaper.

@minLength(3)
@maxLength(24)
param storageAccountName string
param location string
param tags object = {}

var containerNames = [
  'raw'
  'processed'
  'models'
  'splits'
]

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageAccountName
  location: location
  tags: tags
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    allowBlobPublicAccess: false
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    publicNetworkAccess: 'Enabled' // dev only; tighten later
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storage
  name: 'default'
  properties: {
    deleteRetentionPolicy: {
      enabled: true
      days: 7
    }
  }
}

resource containers 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = [for name in containerNames: {
  parent: blobService
  name: name
  properties: {
    publicAccess: 'None'
  }
}]

output accountName string = storage.name
output blobEndpoint string = storage.properties.primaryEndpoints.blob
output containerNames array = containerNames
