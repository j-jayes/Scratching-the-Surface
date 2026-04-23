// Azure Container Registry (Basic) for cascade inference images.

@minLength(5)
@maxLength(50)
param acrName string
param location string
param tags object = {}

resource acr 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' = {
  name: acrName
  location: location
  tags: tags
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: false // managed identity will pull
    publicNetworkAccess: 'Enabled'
    anonymousPullEnabled: false
  }
}

output name string = acr.name
output loginServer string = acr.properties.loginServer
output id string = acr.id
