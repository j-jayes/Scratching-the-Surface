// Service Bus namespace + single queue used as a KEDA scaler trigger
// (wired in the quality phase; provisioned now so the env is ready).

@minLength(6)
@maxLength(50)
param namespaceName string
param location string
param queueName string = 'frame-ingest'
param tags object = {}

resource sb 'Microsoft.ServiceBus/namespaces@2022-10-01-preview' = {
  name: namespaceName
  location: location
  tags: tags
  sku: {
    name: 'Basic'
    tier: 'Basic'
  }
}

resource queue 'Microsoft.ServiceBus/namespaces/queues@2022-10-01-preview' = {
  parent: sb
  name: queueName
  properties: {
    maxDeliveryCount: 5
    lockDuration: 'PT30S'
    enablePartitioning: false
  }
}

output namespaceName string = sb.name
output queueName string = queue.name
output namespaceId string = sb.id
