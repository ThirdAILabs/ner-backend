package messaging

import (
	"context"
	"ner-backend/pkg/models"
	"time"
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

type Publisher interface {
	PublishTrainTask(ctx context.Context, payload models.TrainTaskPayload) error

	PublishShardDataTask(ctx context.Context, payload models.ShardDataPayload) error

	PublishInferenceTask(ctx context.Context, payload models.InferenceTaskPayload) error

	Close()
}

type Reciever interface {
	Tasks() <-chan Task

	Close()
}
