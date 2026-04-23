// Container Apps managed environment with:
//   * Default Consumption profile (CPU, scale-to-zero)
//   * Optional Consumption-GPU-NC8as-T4 profile (serverless T4)
// Apps are deployed in a separate template once images exist.

@minLength(4)
@maxLength(50)
param environmentName string
param location string
param logAnalyticsWorkspaceCustomerId string

@secure()
param logAnalyticsWorkspacePrimaryKey string

param enableGpuWorkloadProfile bool = true
param tags object = {}

var baseProfiles = [
  {
    name: 'Consumption'
    workloadProfileType: 'Consumption'
  }
]

var gpuProfile = [
  {
    name: 'gpu-t4'
    workloadProfileType: 'Consumption-GPU-NC8as-T4'
  }
]

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: environmentName
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalyticsWorkspaceCustomerId
        sharedKey: logAnalyticsWorkspacePrimaryKey
      }
    }
    workloadProfiles: enableGpuWorkloadProfile ? concat(baseProfiles, gpuProfile) : baseProfiles
    zoneRedundant: false
  }
}

output id string = env.id
output name string = env.name
output defaultDomain string = env.properties.defaultDomain
