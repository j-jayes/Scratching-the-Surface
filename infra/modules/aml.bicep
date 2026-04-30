// =============================================================================
// Azure Machine Learning workspace + dependencies.
// =============================================================================
// Used for **MLflow tracking + Model Registry only** in this project — compute
// runs on ACA Jobs (we have no modern GPU VM quota in this subscription).
//
// Dependencies (all required by AML):
//   * Storage account — REUSED from main.bicep via parameter
//   * Key Vault       — created here
//   * Application Insights + Log Analytics — created here (LAW reused)
// =============================================================================

@description('AML workspace name.')
param workspaceName string

@description('Region. Should match the project region.')
param location string

@description('Existing storage account ID to reuse for AML datastore.')
param storageAccountId string

@description('Existing Log Analytics workspace ID for App Insights backing.')
param logAnalyticsWorkspaceId string

@description('Common tags.')
param tags object = {}

var keyVaultName = take('${workspaceName}-kv-${uniqueString(resourceGroup().id, workspaceName)}', 24)
var appInsightsName = '${workspaceName}-ai'

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  tags: tags
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    enablePurgeProtection: true
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalyticsWorkspaceId
  }
}

resource workspace 'Microsoft.MachineLearningServices/workspaces@2024-04-01' = {
  name: workspaceName
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    friendlyName: workspaceName
    description: 'MLflow tracking + model registry for cascade-defect (compute runs on ACA Jobs).'
    storageAccount: storageAccountId
    keyVault: keyVault.id
    applicationInsights: appInsights.id
    publicNetworkAccess: 'Enabled'
    hbiWorkspace: false
    v1LegacyMode: false
  }
}

output workspaceName string = workspace.name
output workspaceId string = workspace.id
output discoveryUrl string = workspace.properties.discoveryUrl
output mlflowTrackingUri string = 'azureml://${location}.api.azureml.ms/mlflow/v1.0/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.MachineLearningServices/workspaces/${workspace.name}'
