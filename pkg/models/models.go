package models

import (
	"database/sql"
	"time"
	// Needed if you add custom marshaling/unmarshaling
)

type ModelStatus string
type JobStatus string

const (
	StatusPending  ModelStatus = "PENDING"
	StatusTraining ModelStatus = "TRAINING"
	StatusTrained  ModelStatus = "TRAINED"
	StatusFailed   ModelStatus = "FAILED"
	JobPending     JobStatus   = "PENDING"
	JobRunning     JobStatus   = "RUNNING"
	JobCompleted   JobStatus   = "COMPLETED"
	JobFailed      JobStatus   = "FAILED"
)

type Model struct {
	ModelID           string         `json:"model_id" db:"model_id"`
	Status            ModelStatus    `json:"status" db:"status"`
	SourceDataPath    sql.NullString `json:"source_data_path,omitempty" db:"source_data_path"` // Use omitempty for optional fields
	ModelArtifactPath sql.NullString `json:"model_artifact_path,omitempty" db:"model_artifact_path"`
	CreationTime      time.Time      `json:"creation_time" db:"creation_time"`
	CompletionTime    sql.NullTime   `json:"completion_time,omitempty" db:"completion_time"`
}

type InferenceJob struct {
	JobID          string         `json:"job_id" db:"job_id"`
	ModelID        string         `json:"model_id" db:"model_id"`
	SourceS3Bucket string         `json:"source_s3_bucket" db:"source_s3_bucket"`
	SourceS3Prefix sql.NullString `json:"source_s3_prefix,omitempty" db:"source_s3_prefix"`
	DestS3Bucket   string         `json:"dest_s3_bucket" db:"dest_s3_bucket"`
	ResultS3Prefix sql.NullString `json:"result_s3_prefix,omitempty" db:"result_s3_prefix"`
	Status         JobStatus      `json:"status" db:"status"`
	CreationTime   time.Time      `json:"creation_time" db:"creation_time"`
	CompletionTime sql.NullTime   `json:"completion_time,omitempty" db:"completion_time"`
	TaskCount      int            `json:"task_count,omitempty"` // Not in DB, added for API response
}

// --- Task Payload Structs ---

// TrainTaskPayload defines the message sent to the training queue
type TrainTaskPayload struct {
	ModelID          string `json:"model_id"`
	SourceS3PathTags string `json:"source_s3_path_tags"` // Path to training data in S3/MinIO
}

// InferenceTaskPayload defines the message sent to the inference queue
type InferenceTaskPayload struct {
	JobID             string `json:"job_id"`
	ModelID           string `json:"model_id"`
	ModelArtifactPath string `json:"model_artifact_path"` // S3 Path to the trained model artifact
	SourceS3Bucket    string `json:"source_s3_bucket"`
	SourceS3Key       string `json:"source_s3_key"`  // Specific file key to process
	DestS3Bucket      string `json:"dest_s3_bucket"` // Bucket to upload results to
}

// --- API Request/Response Structs ---

type TrainRequest struct {
	SourceS3Path string `json:"source_s3_path" validate:"required,url"`
}

type TrainSubmitResponse struct {
	Message string `json:"message"`
	ModelID string `json:"model_id"`
}

type InferenceRequest struct {
	ModelID        string `json:"model_id" validate:"required"`
	SourceS3Bucket string `json:"source_s3_bucket" validate:"required"`
	SourceS3Prefix string `json:"source_s3_prefix"` // Optional prefix
	DestS3Bucket   string `json:"dest_s3_bucket" validate:"required"`
}

type InferenceSubmitResponse struct {
	Message   string `json:"message"`
	JobID     string `json:"job_id"`
	TaskCount int    `json:"task_count"`
}

// Note: For status responses, we are directly using the Model and InferenceJob structs defined above.
// You could create specific response structs if you want to tailor the output further.
