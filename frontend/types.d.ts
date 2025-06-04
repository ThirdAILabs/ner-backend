interface Model {
  Id: string;
  Name: string;
  Status: string;
}

interface Group {
  Id: string;
  Name: string;
  Query: string;
  Objects?: string[];
}

interface TaskStatus {
  TotalTasks: number;
  TotalSize: number;
}

interface Report {
  Id: string;
  Model: Model;
  SourceS3Bucket: string;
  SourceS3Prefix: string;
  IsUpload?: boolean;
  CreationTime: string;
  Tags?: string[];
  CustomTags?: {
    [key: string]: string;
  };
  FileCount: number;
  SucceededFileCount: number;
  FailedFileCount: number;
  Groups?: Group[];
  ShardDataTaskStatus?: string;
  InferenceTaskStatuses?: { [key: string]: TaskStatusCategory };
  Errors?: string[];
  ReportName: string;
  TotalInferenceTimeSeconds: number;
  ShardDataTimeSeconds: number;
  TagCounts: { [key: string]: number };
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

interface CreateReportRequest {
  ModelId: string;
  UploadId?: string;
  SourceS3Bucket: string;
  SourceS3Prefix?: string;
  Groups: Record<string, string>;
}
