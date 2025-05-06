interface Model {
    Id: string;
    Name: string;
    Type: string;
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
    CreationTime: string;
    Tags?: { [key: string]: number };
    CustomTags?: {
        [key: string]: {
            Pattern: string;
            Count: number;
        }
    };
    Groups?: Group[];
    ShardDataTaskStatus?: string;
    InferenceTaskStatuses?: { [key: string]: TaskStatusCategory };
    Errors?: string[];
    report_name: string;
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