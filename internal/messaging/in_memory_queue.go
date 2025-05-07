package messaging

import (
	"context"
	"encoding/json"
	"fmt"
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
		tasks: make(chan Task, 1000),
	}
}

var ErrQueueFull = fmt.Errorf("queue is full")

func (q *InMemoryQueue) publishTaskInternal(queue string, payload interface{}) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	// This is to prevent a deadlock case if the shard data task in the worker tries to 
	// publish a task to the same queue and the queue is full. The deadlock occurs because 
	// only the worker can pull from the queue, but it is blocked waiting for the task to
	// be published.
	select {
	case q.tasks <- &inMemoryTask{queue: queue, payload: data}:
		return nil
	default:
		return ErrQueueFull
	}
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
