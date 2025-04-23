package messaging

import (
	"context"
	"time"

	"github.com/google/uuid"
)

const (
	TrainingQueue   = "training_queue"
	InferenceQueue  = "inference_queue"
	ShardDataQueue  = "shard_data_queue"
	RetryDelay      = 5 * time.Second
	MaxConnectRetry = 5
)

type Task interface {
	Type() string

	Payload() []byte

	Ack() error

	Nack() error

	Reject() error
}

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

type Publisher interface {
	PublishTrainTask(ctx context.Context, payload TrainTaskPayload) error

	PublishShardDataTask(ctx context.Context, payload ShardDataPayload) error

	PublishInferenceTask(ctx context.Context, payload InferenceTaskPayload) error

	Close()
}

type Reciever interface {
	Tasks() <-chan Task

	Close()
}
