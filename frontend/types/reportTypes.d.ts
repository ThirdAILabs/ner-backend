interface Model {
  Id: string;
  Name: string;
  Status: string;
  BaseModelId?: string;
  Tags?: string[];
}

interface Group {
  Id: string;
  Name: string;
  Query: string;
  Objects?: string[];
}

interface TaskStatusCategory {
  TotalTasks: number;
  TotalSize: number;
  CompletedSize: number;
}

interface Report {
  Id: string;
  Model: Model;
  SourceS3Bucket: string;
  SourceS3Prefix?: string;
  IsUpload?: boolean;
  CreationTime: string;
  Tags?: string[];
  CustomTags?: { [key: string]: string };
  FileCount: number;
  SucceededFileCount: number;
  FailedFileCount: number;
  Groups?: Group[];
  ShardDataTaskStatus?: string;
  InferenceTaskStatuses?: { [key: string]: TaskStatusCategory };
  Errors?: string[];
  ReportName: string;
  TagCounts: { [key: string]: number };
  TotalInferenceTimeSeconds: number;
  ShardDataTimeSeconds: number;
}

interface ReportWithStatus extends Report {
  isLoadingStatus?: boolean;
  detailedStatus?: {
    ShardDataTaskStatus?: string;
    InferenceTaskStatuses?: { [key: string]: TaskStatusCategory };
  };
}

interface Entity {
  Object: string;
  Start: number;
  End: number;
  Label: string;
  Text: string;
  LContext?: string;
  RContext?: string;
}

interface ObjectPreview {
  object: string;
  tokens: string[];
  tags: string[];
}

interface CreateReportRequest {
  ModelId: string;
  UploadId?: string;
  S3Endpoint?: string;
  S3Region?: string;
  SourceS3Bucket?: string;
  SourceS3Prefix?: string;
  Tags: string[];
  CustomTags?: { [key: string]: string };
  Groups?: { [key: string]: string };
  report_name: string;
}

interface InferenceMetrics {
  Completed: number;
  Failed: number;
  InProgress: number;
  DataProcessedMB: number;
  TokensProcessed: number;
}

interface ThroughputMetrics {
  ModelID: string;
  ReportID?: string;
  ThroughputMBPerHour: number;
}

interface ChatResponse {
  InputText: string;
  Reply: string;
  TagMap: Record<string, string>;
}
