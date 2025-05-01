package api

import (
	"time"

	"github.com/google/uuid"
)

type Model struct {
	Id          uuid.UUID
	BaseModelId *uuid.UUID
	Name        string
	Type        string
	Status      string
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

	Model Model

	SourceS3Bucket string
	SourceS3Prefix string

	CreationTime time.Time

	Groups []Group

	ShardDataTaskStatus   string                        `json:"ShardDataTaskStatus,omitempty"`
	InferenceTaskStatuses map[string]TaskStatusCategory `json:"InferenceTaskStatuses,omitempty"`
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

	UploadId       uuid.UUID
	SourceS3Bucket string
	SourceS3Prefix string

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

type Tags struct {
	TagId uuid.UUID
	Tag   string

	ModelId uuid.UUID
}
