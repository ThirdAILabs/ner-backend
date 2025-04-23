package messaging

import (
	"context"
	"encoding/json"
	"ner-backend/pkg/models"
)

type inMemoryTask struct {
	queue   string
	payload []byte
}

func (t *inMemoryTask) Type() string {
	return t.queue
}

func (t *inMemoryTask) Payload() []byte {
	return t.payload
}

func (t *inMemoryTask) Ack() error {
	return nil
}

func (t *inMemoryTask) Nack() error {
	return nil
}

func (t *inMemoryTask) Reject() error {
	return nil
}

type InMemoryQueue struct {
	tasks chan Task
}

func NewInMemoryQueue() *InMemoryQueue {
	return &InMemoryQueue{
		tasks: make(chan Task, 100),
	}
}

func (q *InMemoryQueue) publishTaskInternal(queue string, payload interface{}) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	q.tasks <- &inMemoryTask{queue: queue, payload: data}

	return nil
}

func (q *InMemoryQueue) PublishTrainTask(ctx context.Context, payload models.TrainTaskPayload) error {
	return q.publishTaskInternal(TrainingQueue, payload)
}

func (q *InMemoryQueue) PublishShardDataTask(ctx context.Context, payload models.ShardDataPayload) error {
	return q.publishTaskInternal(ShardDataQueue, payload)
}

func (q *InMemoryQueue) PublishInferenceTask(ctx context.Context, payload models.InferenceTaskPayload) error {
	return q.publishTaskInternal(InferenceQueue, payload)
}

func (q *InMemoryQueue) Tasks() <-chan Task {
	return q.tasks
}

func (q *InMemoryQueue) Close() {
	if q.tasks != nil {
		close(q.tasks)
		q.tasks = nil
	}
}
