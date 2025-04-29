// This file contains types for the database layer.

interface AppInfo {
    name: string,
    description: string,
    created: string,
    modified: string,
    owner: string,
    deploymentLink?: string
}