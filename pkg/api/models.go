package api

import (
	"ner-backend/internal/licensing"
	"time"

	"github.com/google/uuid"
)

type Model struct {
	Id          uuid.UUID
	BaseModelId *uuid.UUID
	Name        string
	Status      string

	Tags []string `json:"Tags,omitempty"`
}

type Group struct {
	Id    uuid.UUID
	Name  string
	Query string

	Objects []string `json:"Objects,omitempty"`
}

type TaskStatusCategory struct {
	TotalTasks int
	TotalSize  int
}

type Report struct {
	Id uuid.UUID

	Model          Model
	ReportName     string
	SourceS3Bucket string
	SourceS3Prefix string
	IsUpload       bool

	Stopped            bool
	CreationTime       time.Time
	TotalFileCount     int `json:"FileCount,omitempty"`
	CompletedFileCount int `json:"CompletedFileCount,omitempty"`

	Tags       []string          `json:"Tags,omitempty"`
	CustomTags map[string]string `json:"CustomTags,omitempty"`
	TagCounts  map[string]uint64 `json:"TagCounts,omitempty"`

	Groups []Group

	ShardDataTaskStatus   string                        `json:"ShardDataTaskStatus,omitempty"`
	InferenceTaskStatuses map[string]TaskStatusCategory `json:"InferenceTaskStatuses,omitempty"`

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

	ReportName     string `json:"report_name"`
	UploadId       uuid.UUID
	S3Endpoint     string
	S3Region       string
	SourceS3Bucket string
	SourceS3Prefix string

	Tags       []string
	CustomTags map[string]string

	Groups map[string]string
}

type CreateReportResponse struct {
	ReportId uuid.UUID
}

type TagInfo struct {
	Name        string
	Description string
	Examples    []string
}

type Sample struct {
	Tokens []string
	Labels []string
}

type FinetuneRequest struct {
	Name string

	TaskPrompt string
	Tags       []TagInfo
	Samples    []Sample
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
	LicenseType  licensing.LicenseType
	LicenseInfo  licensing.LicenseInfo
	LicenseError string
}
