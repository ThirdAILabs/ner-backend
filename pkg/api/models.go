package api

import (
	"encoding/json"
	"ner-backend/internal/core/types"
	"ner-backend/internal/licensing"
	"time"

	"github.com/google/uuid"
)

type Model struct {
	Id           uuid.UUID
	BaseModelId  *uuid.UUID
	Name         string
	Status       string
	CreationTime time.Time
	Finetunable  bool

	Tags []string `json:"Tags,omitempty"`
}

type Group struct {
	Id    uuid.UUID
	Name  string
	Query string

	Objects []string `json:"Objects,omitempty"`
}

type TaskStatusCategory struct {
	TotalTasks    int
	TotalSize     int
	CompletedSize int
}

type Report struct {
	Id uuid.UUID

	Model         Model
	ReportName    string
	StorageType   string
	StorageParams json.RawMessage

	Stopped            bool
	CreationTime       time.Time
	TotalFileCount     int
	SucceededFileCount int
	FailedFileCount    int

	Tags       []string          `json:"Tags,omitempty"`
	CustomTags map[string]string `json:"CustomTags,omitempty"`
	TagCounts  map[string]uint64 `json:"TagCounts,omitempty"`

	Groups []Group

	ShardDataTaskStatus       string                        `json:"ShardDataTaskStatus,omitempty"`
	InferenceTaskStatuses     map[string]TaskStatusCategory `json:"InferenceTaskStatuses,omitempty"`
	TotalInferenceTimeSeconds float64                       `json:"TotalInferenceTimeSeconds,omitempty"`
	ShardDataTimeSeconds      float64                       `json:"ShardDataTimeSeconds,omitempty"`

	Errors []string `json:"Errors,omitempty"`
}

type Entity struct {
	Object   string
	Start    int
	End      int
	Label    string
	Text     string
	LContext string
	RContext string
}

type CreateReportRequest struct {
	ModelId uuid.UUID

	ReportName    string
	StorageType   string
	StorageParams json.RawMessage

	Tags       []string
	CustomTags map[string]string

	Groups map[string]string
}

type CreateReportResponse struct {
	ReportId uuid.UUID
}

type Sample struct {
	Tokens []string
	Labels []string
}

type FinetuneRequest struct {
	Name                string          `json:"Name"`
	TaskPrompt          *string         `json:"TaskPrompt"`
	GenerateData        bool            `json:"GenerateData"`
	VerifyGeneratedData bool            `json:"VerifyGeneratedData" default:"true"`
	Tags                []types.TagInfo `json:"Tags"`
	Samples             []Sample        `json:"Samples,omitempty"`
}

type FinetuneResponse struct {
	ModelId uuid.UUID
}

type SearchResponse struct {
	Objects []string
}

type UploadResponse struct {
	Id uuid.UUID
}

type ObjectPreviewResponse struct {
	Object string   `json:"object"`
	Tokens []string `json:"tokens"`
	Tags   []string `json:"tags"`
}

type InferenceMetricsResponse struct {
	Completed       int64   `json:"Completed"`
	Failed          int64   `json:"Failed"`
	InProgress      int64   `json:"InProgress"`
	DataProcessedMB float64 `json:"DataProcessedMB"`
	TokensProcessed int64   `json:"TokensProcessed"`
}

type ThroughputResponse struct {
	ModelID             uuid.UUID `json:"ModelId"`
	ReportID            uuid.UUID `json:"ReportId"`
	ThroughputMBPerHour float64   `json:"ThroughputMBPerHour"`
}

type GetLicenseResponse struct {
	LicenseInfo  licensing.LicenseInfo
	LicenseError string
}

type GetEnterpriseInfoResponse struct {
	IsEnterpriseMode bool `json:"IsEnterpriseMode"`
}

type ValidateGroupDefinitionRequest struct {
	GroupQuery string
}

type ValidateS3BucketRequest struct {
	S3Endpoint     string
	Region         string
	SourceS3Bucket string
	SourceS3Prefix string
}

type FileNameToPath struct {
	Mapping map[string]string
}

type FeedbackRequest struct {
	Tokens []string
	Labels []string
}

type FeedbackResponse struct {
	Id     uuid.UUID
	Tokens []string
	Labels []string
}
