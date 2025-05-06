package integrationtests

import (
	"context"
	"encoding/json"
	"ner-backend/internal/messaging"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRabbitMQ(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	publisher, receiver := setupRabbitMQContainer(t, ctx)

	// Test publishing and receiving a FinetuneTask
	t.Run("Publish and Receive FinetuneTask", func(t *testing.T) {
		payload := messaging.FinetuneTaskPayload{ModelId: uuid.New()}
		err := publisher.PublishFinetuneTask(ctx, payload)
		require.NoError(t, err)

		select {
		case task := <-receiver.Tasks():
			assert.Equal(t, messaging.FinetuneQueue, task.Type())

			var receivedPayload messaging.FinetuneTaskPayload
			err := json.Unmarshal(task.Payload(), &receivedPayload)
			require.NoError(t, err)
			assert.Equal(t, payload, receivedPayload)

			err = task.Ack()
			require.NoError(t, err)
		case <-time.After(4 * time.Second):
			t.Fatal("Timed out waiting for task")
		}
	})

	// Test publishing and receiving a ShardDataTask
	t.Run("Publish and Receive ShardDataTask", func(t *testing.T) {
		payload := messaging.ShardDataPayload{ReportId: uuid.New()}
		err := publisher.PublishShardDataTask(ctx, payload)
		require.NoError(t, err)

		select {
		case task := <-receiver.Tasks():
			assert.Equal(t, messaging.ShardDataQueue, task.Type())

			var receivedPayload messaging.ShardDataPayload
			err := json.Unmarshal(task.Payload(), &receivedPayload)
			require.NoError(t, err)
			assert.Equal(t, payload, receivedPayload)

			err = task.Ack()
			require.NoError(t, err)
		case <-time.After(4 * time.Second):
			t.Fatal("Timed out waiting for task")
		}
	})

	// Test publishing and receiving an InferenceTask
	t.Run("Publish and Receive InferenceTask", func(t *testing.T) {
		payload := messaging.InferenceTaskPayload{ReportId: uuid.New(), TaskId: 10}
		err := publisher.PublishInferenceTask(ctx, payload)
		require.NoError(t, err)

		select {
		case task := <-receiver.Tasks():
			assert.Equal(t, messaging.InferenceQueue, task.Type())

			var receivedPayload messaging.InferenceTaskPayload
			err := json.Unmarshal(task.Payload(), &receivedPayload)
			require.NoError(t, err)
			assert.Equal(t, payload, receivedPayload)

			err = task.Ack()
			require.NoError(t, err)
		case <-time.After(4 * time.Second):
			t.Fatal("Timed out waiting for task")
		}
	})
}
