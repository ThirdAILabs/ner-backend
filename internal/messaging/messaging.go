package messaging

import (
	"context"
	"ner-backend/pkg/api"
	"time"

	"github.com/google/uuid"
)

const (
	FinetuneQueue   = "finetune_queue"
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

type FinetuneTaskPayload struct {
	ModelId     uuid.UUID
	BaseModelId uuid.UUID

	TaskPrompt string
	Tags       []api.TagInfo
	Samples    []api.Sample

	GenerateData       bool
	NumValuesPerTag    int
	RecordsToGenerate  int
	RecordsPerTemplate int
	TestSplit          float32
}

type ShardDataPayload struct {
	ReportId uuid.UUID
}

type InferenceTaskPayload struct {
	ReportId uuid.UUID
	TaskId   int
}

type Publisher interface {
	PublishFinetuneTask(ctx context.Context, payload FinetuneTaskPayload) error

	PublishShardDataTask(ctx context.Context, payload ShardDataPayload) error

	PublishInferenceTask(ctx context.Context, payload InferenceTaskPayload) error

	Close()
}

type Reciever interface {
	Tasks() <-chan Task

	Close()
}
