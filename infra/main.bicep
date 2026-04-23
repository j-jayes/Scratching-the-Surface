// =============================================================================
// Project Cascade Defect — Root infrastructure deployment (resource-group scope)
// =============================================================================
// Provisions the shared platform: Storage, ACR, Service Bus, Container Apps env
// (with serverless T4 GPU workload profile), Log Analytics, and a $/mo budget.
// Azure OpenAI is *not* deployed here — we reuse the existing nexer resource.
// Container Apps themselves are deployed in `apps.bicep` after images exist.
// =============================================================================

targetScope = 'resourceGroup'

@minLength(3)
@maxLength(11)
@description('Short project name used as a prefix for all resources. Lowercase alphanumerics only.')
param projectName string = 'cascade'

@allowed([
  'dev'
  'prod'
])
param environmentName string = 'dev'

@description('Azure region. West Europe is the only one verified to support Consumption-GPU-NC8as-T4.')
param location string = 'westeurope'

@description('Monthly budget cap in USD for cost-management alerting.')
param monthlyBudgetUsd int = 50

@description('Email address(es) to notify on budget threshold breaches.')
param budgetAlertEmails array = []

@description('Whether to add the serverless T4 GPU workload profile. Set false to disable if quota fails.')
param enableGpuWorkloadProfile bool = true

// ─── Naming ──────────────────────────────────────────────────────────────────
var resourceToken = uniqueString(subscription().subscriptionId, resourceGroup().id, projectName, environmentName)
var namePrefix = '${projectName}-${environmentName}'
var storageAccountName = toLower('${projectName}${environmentName}${take(resourceToken, 8)}')
var acrName = toLower('${projectName}${environmentName}acr${take(resourceToken, 6)}')
var serviceBusNamespaceName = '${namePrefix}-sb-${take(resourceToken, 6)}'
var logAnalyticsName = '${namePrefix}-logs'
var acaEnvironmentName = '${namePrefix}-aca-env'

// Tags applied to every resource for cost reporting + cleanup.
var commonTags = {
  project: 'cascade-defect'
  environment: environmentName
  managedBy: 'bicep'
}

// ─── Modules ─────────────────────────────────────────────────────────────────
module storage 'modules/storage.bicep' = {
  name: 'storage-deploy'
  params: {
    storageAccountName: storageAccountName
    location: location
    tags: commonTags
  }
}

module acr 'modules/acr.bicep' = {
  name: 'acr-deploy'
  params: {
    acrName: acrName
    location: location
    tags: commonTags
  }
}

module servicebus 'modules/servicebus.bicep' = {
  name: 'servicebus-deploy'
  params: {
    namespaceName: serviceBusNamespaceName
    location: location
    tags: commonTags
  }
}

module logs 'modules/log-analytics.bicep' = {
  name: 'logs-deploy'
  params: {
    workspaceName: logAnalyticsName
    location: location
    tags: commonTags
  }
}

module acaEnv 'modules/aca-env.bicep' = {
  name: 'aca-env-deploy'
  params: {
    environmentName: acaEnvironmentName
    location: location
    logAnalyticsWorkspaceCustomerId: logs.outputs.customerId
    logAnalyticsWorkspacePrimaryKey: logs.outputs.primarySharedKey
    enableGpuWorkloadProfile: enableGpuWorkloadProfile
    tags: commonTags
  }
}

module budget 'modules/budget.bicep' = if (!empty(budgetAlertEmails)) {
  name: 'budget-deploy'
  params: {
    budgetName: '${namePrefix}-budget'
    monthlyBudgetUsd: monthlyBudgetUsd
    contactEmails: budgetAlertEmails
  }
}

// ─── Outputs (consumed by Makefile to populate .env) ─────────────────────────
output storageAccountName string = storage.outputs.accountName
output storageBlobEndpoint string = storage.outputs.blobEndpoint
output acrLoginServer string = acr.outputs.loginServer
output acrName string = acr.outputs.name
output serviceBusNamespace string = servicebus.outputs.namespaceName
output serviceBusQueueName string = servicebus.outputs.queueName
output acaEnvironmentName string = acaEnv.outputs.name
output acaEnvironmentId string = acaEnv.outputs.id
output logAnalyticsWorkspaceId string = logs.outputs.id
