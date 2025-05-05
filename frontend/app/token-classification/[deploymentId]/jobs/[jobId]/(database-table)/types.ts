export interface ObjectDatabaseRecord {
  taggedTokens: [string, string][];
  tokenContexts?: { left: string; right: string }[];
  sourceObject: string;
  groups: string[];
}

export interface ClassifiedTokenDatabaseRecord {
  token: string;
  tag: string;
  sourceObject: string;
  groups: string[];
  context?: {
    left: string;
    right: string;
  };
  start?: number;
  end?: number;
}

export interface DatabaseTableProps {
  loadMoreObjectRecords: () => Promise<ObjectDatabaseRecord[]>;
  loadMoreClassifiedTokenRecords: () => Promise<ClassifiedTokenDatabaseRecord[]>;
  groups: string[];
  tags: string[];
}

export type ViewMode = 'object' | 'classified-token'; 