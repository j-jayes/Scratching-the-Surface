// Azure OpenAI resource + GPT-4o (vision) deployment.
// Deployed only when the parent template sets `deployOpenAi = true`.

@minLength(2)
@maxLength(64)
param accountName string
param location string
param tags object = {}

@description('Model deployment name used by the cascade Layer 3 (the Oracle).')
param oracleDeploymentName string = 'oracle'

@description('Tokens-per-minute capacity (1 unit = 1k TPM).')
@minValue(1)
@maxValue(50)
param oracleCapacity int = 10

resource aoai 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: accountName
  location: location
  tags: tags
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  properties: {
    customSubDomainName: accountName
    publicNetworkAccess: 'Enabled'
    disableLocalAuth: false // allow key auth for dev; prefer AAD where possible
  }
}

resource oracle 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: aoai
  name: oracleDeploymentName
  sku: {
    name: 'Standard'
    capacity: oracleCapacity
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4.1-mini'
      version: '2025-04-14'
    }
    raiPolicyName: 'Microsoft.DefaultV2'
  }
}

output endpoint string = aoai.properties.endpoint
output accountName string = aoai.name
output deploymentName string = oracle.name
output modelName string = 'gpt-4.1-mini'
