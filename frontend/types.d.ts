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
    SourceS3Prefix?: string;
    CreationTime: string;
    Groups: Group[];
    ShardDataTaskStatus: string;
    InferenceTaskStatuses: {
        COMPLETED: TaskStatus;
        RUNNING: TaskStatus;
        QUEUED: TaskStatus;
        FAILED: TaskStatus;
    };
}

interface Entity {
    Object: string;
    Start: number;
    End: number;
    Label: string;
    Text: string;
}

interface CreateReportRequest {
    ModelId: string;
    UploadId?: string;
    SourceS3Bucket: string;
    SourceS3Prefix?: string;
    Groups: Record<string, string>;
}