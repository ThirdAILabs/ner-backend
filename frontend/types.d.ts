//DB Layer Types
interface AppInfo {
    name: string,
    description: string,
    created: string,
    modified: string,
    owner: string,
    deploymentLink?: string
}

//Nav Types
interface BreadCrumb {
    href: string,
    label: string,
    icon?: any
}

interface Tabs {
    tabs: string[],
}