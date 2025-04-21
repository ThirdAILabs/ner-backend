package models

import (
	"github.com/google/uuid"
	// Needed if you add custom marshaling/unmarshaling
)

// --- Task Payload Structs ---

type TrainTaskPayload struct {
	ModelId          uuid.UUID
	SourceS3PathTags string // Path to training data in S3/MinIO
}

type ShardDataPayload struct {
	ReportId uuid.UUID
}

type InferenceTaskPayload struct {
	ReportId uuid.UUID
	TaskId   int
}
