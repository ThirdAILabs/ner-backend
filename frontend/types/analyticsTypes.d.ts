import { FeedbackMetadata } from '@/components/feedback/useFeedbackState';

interface Tag {
  type: string;
  count: number;
}

interface ObjectDatabaseRecord {
  taggedTokens: [string, string][];
  tokenContexts?: { left: string; right: string }[];
  sourceObject: string;
  groups: string[];
}

interface ClassifiedTokenDatabaseRecord {
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

interface DatabaseTableProps {
  groups: string[];
  tags: Tag[];
  uploadId?: string;
  addFeedback: (tokens: FeedbackMetadata, objectTokens: string[], objectTags: string[]) => void;
  initialSelectedTag?: string | null;
}

interface TableContentProps {
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
  pathMap?: Record<string, string>;
  addFeedback: (tokens: FeedbackMetadata, objectTokens: string[], objectTags: string[]) => void;
}

type ViewMode = 'object' | 'classified-token';
