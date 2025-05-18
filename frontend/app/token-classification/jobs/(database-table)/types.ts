import { Tag } from '@/components/AnalyticsDashboard';

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
  groups: string[];
  tags: Tag[];
}

export interface TableContentProps {
  viewMode: ViewMode;
  objectRecords: ObjectDatabaseRecord[];
  tokenRecords: ClassifiedTokenDatabaseRecord[];
  groupFilters: Record<string, boolean>;
  tagFilters: Record<string, boolean>;
  isLoadingObjectRecords: boolean;
  isLoadingTokenRecords: boolean;
  tags: Tag[];
  hasMoreTokens?: boolean;
  hasMoreObjects?: boolean;
  onLoadMore?: () => void;
  showFilterContent: boolean;
}

export type ViewMode = 'object' | 'classified-token';
