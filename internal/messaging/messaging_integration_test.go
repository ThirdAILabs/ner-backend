//go:build integration
// +build integration

// The build tag 'integration' allows separating integration tests from unit tests.
// Run unit tests with: go test ./...
// Run integration tests with: go test -tags=integration ./...

package messaging

import (
	"context"
	"encoding/json"
	"log"
	"ner_backend/pkg/models" // Adjust import path if needed
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/modules/rabbitmq"
)

// TestPublishConsumeInferenceTask tests the full cycle for an inference task
func TestPublishConsumeInferenceTask(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode.")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute) // Timeout for the whole test
	defer cancel()

	log.Println("Setting up RabbitMQ container...")
	// Start RabbitMQ container
	rabbitmqContainer, err := rabbitmq.RunContainer(ctx,
		testcontainers.WithImage("rabbitmq:3.11-management"),
		// rabbitmq.WithAdminUsername("user"), // Default is guest/guest
		// rabbitmq.WithAdminPassword("password"),
	)
	require.NoError(t, err, "Failed to start RabbitMQ container")
	// Clean up the container after the test function returns
	defer func() {
		log.Println("Terminating RabbitMQ container...")
		if err := rabbitmqContainer.Terminate(context.Background()); err != nil {
			log.Printf("Warning: failed to terminate RabbitMQ container: %v", err)
		}
	}()

	// Get connection string for the test container
	// Note: Use AMQP port 5672
	// CORRECT:
	connStr, err := rabbitmqContainer.AmqpURL(ctx)             // Gets amqp://guest:guest@host:port
	require.NoError(t, err, "Failed to get RabbitMQ AMQP URL") // Moved require check here too
	log.Printf("RabbitMQ container ready at: %s", connStr)

	// --- Test Setup ---
	var wg sync.WaitGroup
	processedSignal := make(chan bool, 1) // Channel to signal message processing

	// --- Publisher Setup ---
	publisher, err := NewTaskPublisher(connStr)
	require.NoError(t, err, "Failed to create task publisher")
	defer publisher.Close()

	// --- Worker Setup ---
	// Create a simple worker instance for the test
	wg.Add(1)
	go func() {
		defer wg.Done()
		log.Println("[Test Worker] Starting...")
		conn, err := connectToRabbitMQ(connStr) // Use test helper func
		if err != nil {
			log.Printf("[Test Worker] Failed to connect: %v", err)
			processedSignal <- false // Signal failure
			return
		}
		defer conn.Close()

		channel, err := conn.Channel()
		if err != nil {
			log.Printf("[Test Worker] Failed to open channel: %v", err)
			processedSignal <- false
			return
		}
		defer channel.Close()

		// Declare queue (publisher also declares, but good practice)
		_, err = channel.QueueDeclare(InferenceQueue, true, false, false, false, nil)
		if err != nil {
			log.Printf("[Test Worker] Failed to declare queue %s: %v", InferenceQueue, err)
			processedSignal <- false
			return
		}

		// Consume messages
		msgs, err := channel.Consume(
			InferenceQueue,  // queue
			"test-consumer", // consumer tag
			false,           // auto-ack = false
			false,           // exclusive
			false,           // no-local
			false,           // no-wait
			nil,             // args
		)
		if err != nil {
			log.Printf("[Test Worker] Failed to start consuming: %v", err)
			processedSignal <- false
			return
		}
		log.Println("[Test Worker] Waiting for messages...")

		// Wait for one message relevant to this test
		select {
		case d, ok := <-msgs:
			if !ok {
				log.Println("[Test Worker] Delivery channel closed unexpectedly.")
				processedSignal <- false
				return
			}
			log.Printf("[Test Worker] Received a message: %s", string(d.Body))

			// Basic check: Can we unmarshal it?
			var payload models.InferenceTaskPayload
			err := json.Unmarshal(d.Body, &payload)
			if err != nil {
				log.Printf("[Test Worker] Failed to unmarshal payload: %v", err)
				d.Nack(false, false) // Don't requeue bad message
				processedSignal <- false
				return
			}

			// TODO: Add more specific checks if needed (e.g., check JobID)
			log.Printf("[Test Worker] Successfully processed message for JobID: %s", payload.JobID)
			d.Ack(false)            // Acknowledge the message
			processedSignal <- true // Signal success

		case <-ctx.Done(): // Handle test timeout
			log.Println("[Test Worker] Test context cancelled.")
			processedSignal <- false
		}
	}()

	// --- Test Execution ---
	// Prepare a test payload
	testPayload := models.InferenceTaskPayload{
		JobID:             "test-job-123",
		ModelID:           "test-model-abc",
		ModelArtifactPath: "s3://models/test-model-abc/model.bin",
		SourceS3Bucket:    "test-input-bucket",
		SourceS3Key:       "test/input/file.txt",
		DestS3Bucket:      "test-output-bucket",
	}

	// Publish the message
	log.Println("Publishing test message...")
	err = publisher.PublishInferenceTask(ctx, testPayload)
	require.NoError(t, err, "Failed to publish inference task")
	log.Println("Test message published.")

	// --- Verification ---
	// Wait for the worker to process the message or timeout
	log.Println("Waiting for worker signal...")
	select {
	case success := <-processedSignal:
		assert.True(t, success, "Worker did not signal successful processing")
		if success {
			log.Println("Worker successfully processed the message.")
		} else {
			log.Println("Worker signaled failure or timed out.")
		}
	case <-ctx.Done():
		t.Fatal("Test timed out waiting for worker to process message")
	}

	// Wait for the worker goroutine to finish its cleanup
	log.Println("Waiting for worker goroutine to exit...")
	wg.Wait()
	log.Println("Test finished.")
}
