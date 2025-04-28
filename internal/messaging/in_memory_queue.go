package messaging

import (
	"context"
	"encoding/json"
	"sync"
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
	tasks      chan Task
	destructor sync.Once
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

func (q *InMemoryQueue) PublishFinetuneTask(ctx context.Context, payload FinetuneTaskPayload) error {
	return q.publishTaskInternal(FinetuneQueue, payload)
}

func (q *InMemoryQueue) PublishShardDataTask(ctx context.Context, payload ShardDataPayload) error {
	return q.publishTaskInternal(ShardDataQueue, payload)
}

func (q *InMemoryQueue) PublishInferenceTask(ctx context.Context, payload InferenceTaskPayload) error {
	return q.publishTaskInternal(InferenceQueue, payload)
}

func (q *InMemoryQueue) Tasks() <-chan Task {
	return q.tasks
}

func (q *InMemoryQueue) Close() {
	q.destructor.Do(func() {
		close(q.tasks)
	})
}
