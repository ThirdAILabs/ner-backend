package models

import (
	"github.com/google/uuid"
	// Needed if you add custom marshaling/unmarshaling
)

// --- Task Payload Structs ---

// TrainTaskPayload defines the message sent to the training queue
type TrainTaskPayload struct {
	ModelId          uuid.UUID
	SourceS3PathTags string // Path to training data in S3/MinIO
}

// InferenceTaskPayload defines the message sent to the inference queue
type InferenceTaskPayload struct {
	JobId             uuid.UUID
	ModelId           uuid.UUID
	ModelArtifactPath string // S3 Path to the trained model artifact
	SourceS3Bucket    string
	SourceS3Key       string // Specific file key to process
	DestS3Bucket      string // Bucket to upload results to
}

type InferenceRequest struct {
	ModelId        uuid.UUID `validate:"required"`
	SourceS3Bucket string    `validate:"required"`
	SourceS3Prefix string
	DestS3Bucket   string `validate:"required"`

	Groups map[string]string
}

type InferenceResponse struct {
	JobId uuid.UUID
}
