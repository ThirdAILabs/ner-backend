package messaging

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"ner-backend/internal/config" // Adjust import path
	"ner-backend/internal/core"
	"ner-backend/internal/database"
	"ner-backend/internal/s3"
	"ner-backend/pkg/models"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	amqp "github.com/rabbitmq/amqp091-go"
	"gorm.io/gorm"
)

type Worker struct {
	DB        *gorm.DB
	S3Client  *s3.Client
	Config    *config.Config
	WaitGroup *sync.WaitGroup
	Publisher *TaskPublisher
}

func (worker *Worker) StartThreads(rabbitMQURL string) error {
	_, err := connectToRabbitMQ(rabbitMQURL)
	if err != nil {
		return err
	}

	numWorkers := worker.Config.WorkerConcurrency
	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU() // Default to number of CPUs
		log.Printf("Worker concurrency not specified or invalid, defaulting to %d", numWorkers)
	}

	queuesToConsume := strings.Split(worker.Config.QueueNames, ",")
	log.Printf("Starting %d worker goroutines, consuming from queues: %v", numWorkers, queuesToConsume)

	worker.WaitGroup.Add(numWorkers) // Add count for all potential worker goroutines

	for i := 0; i < numWorkers; i++ {
		go worker.runWorkerInstance(i, rabbitMQURL, queuesToConsume)
	}

	log.Printf("%d worker instances initiated.", numWorkers)
	return nil
}

func connectToRabbitMQ(url string) (*amqp.Connection, error) {
	var conn *amqp.Connection
	var err error
	for i := 0; i < MaxConnectRetry; i++ {
		conn, err = amqp.Dial(url)
		if err == nil {
			log.Println("Worker connected to RabbitMQ.")
			return conn, nil
		}
		log.Printf("Worker failed to connect (attempt %d/%d): %v. Retrying in %v", i+1, MaxConnectRetry, err, RetryDelay)
		time.Sleep(RetryDelay)
	}
	return nil, fmt.Errorf("worker failed to connect after %d attempts: %w", MaxConnectRetry, err)
}

func (worker *Worker) runWorkerInstance(id int, rabbitMQURL string, queues []string) {
	defer worker.WaitGroup.Done()
	log.Printf("[Worker %d] Starting...", id)

	var conn *amqp.Connection
	var channel *amqp.Channel
	var msgs <-chan amqp.Delivery
	var err error

	connect := func() bool {
		conn, err = connectToRabbitMQ(rabbitMQURL)
		if err != nil {
			log.Printf("[Worker %d] Failed to connect, will retry: %v", id, err)
			return false
		}
		channel, err = conn.Channel()
		if err != nil {
			log.Printf("[Worker %d] Failed to open channel: %v", id, err)
			conn.Close()
			return false
		}

		// Set QoS to process one message at a time per worker instance
		// Adjust prefetch count if needed for batching or higher throughput per worker
		err = channel.Qos(1, 0, false)
		if err != nil {
			log.Printf("[Worker %d] Failed to set QoS: %v", id, err)
			channel.Close()
			conn.Close()
			return false
		}

		// Declare queues this worker will consume from
		// We merge deliveries from multiple queues into one channel 'msgs'
		deliveries := make(chan amqp.Delivery) // Channel to merge deliveries
		msgs = deliveries                      // Assign to the variable workers will read from

		for _, qName := range queues {
			qName = strings.TrimSpace(qName)
			if qName == "" {
				continue
			}

			// Declare queue (idempotent)
			_, err = channel.QueueDeclare(qName, true, false, false, false, nil)
			if err != nil {
				log.Printf("[Worker %d] Failed to declare queue %s: %v", id, qName, err)
				channel.Close()
				conn.Close()
				return false
			}

			// Start consuming from this queue
			individualMsgs, err := channel.Consume(
				qName,                                  // queue
				fmt.Sprintf("worker-%d-%s", id, qName), // consumer tag
				false,                                  // auto-ack (set to false for manual ack)
				false,                                  // exclusive
				false,                                  // no-local
				false,                                  // no-wait
				nil,                                    // args
			)
			if err != nil {
				log.Printf("[Worker %d] Failed to start consuming from %s: %v", id, qName, err)
				channel.Close()
				conn.Close()
				return false
			}
			log.Printf("[Worker %d] Consuming from queue %s", id, qName)

			// Goroutine to forward messages from this queue's channel to the merged channel
			go func(deliveryChan <-chan amqp.Delivery) {
				for d := range deliveryChan {
					deliveries <- d
				}
				log.Printf("[Worker %d] Consumption channel for a queue closed.", id)
				// If one consumer stops, maybe we should signal the main loop to reconnect?
			}(individualMsgs)
		}
		log.Printf("[Worker %d] Established channel and consumers.", id)
		return true
	}

	for { // Main worker loop
		if channel == nil || conn == nil || conn.IsClosed() || channel.IsClosed() {
			log.Printf("[Worker %d] Connection/Channel lost. Attempting to reconnect...", id)
			time.Sleep(RetryDelay) // Wait before retrying connection
			if !connect() {
				continue // Retry connection attempt
			}
		}

		select {
		case d, ok := <-msgs:
			if !ok {
				log.Printf("[Worker %d] Delivery channel closed. Attempting reconnect...", id)
				// Channel closed, likely connection issue, trigger reconnect logic
				if channel != nil {
					channel.Close()
				}
				if conn != nil {
					conn.Close()
				}
				channel = nil
				conn = nil
				continue // Loop back to reconnect
			}
			log.Printf("[Worker %d] Received message from queue: %s", id, d.RoutingKey) // d.RoutingKey is the queue name here
			worker.processMessage(d)                                                    // Process the message

			// Add a timeout or heartbeat check if needed
			// case <-time.After(30 * time.Second):
			//    log.Printf("[Worker %d] No message received for 30s", id)
		}
	}
	// log.Printf("[Worker %d] Exiting.", id) // This part is unlikely to be reached in the current loop structure
}

func (worker *Worker) processMessage(d amqp.Delivery) {
	ctx := context.Background() // Create a new context for each task
	var err error
	processed := false // Flag to track if processing was attempted

	// Determine task type based on the queue it came from
	switch d.RoutingKey { // Using RoutingKey which equals Queue Name here
	case TrainingQueue:
		var payload models.TrainTaskPayload
		if err = json.Unmarshal(d.Body, &payload); err != nil {
			log.Printf("Error unmarshalling training task: %v. Body: %s", err, string(d.Body))
			// Reject message (non-requeueable) as it's malformed
			d.Reject(false) // false = don't requeue
			return
		}
		processed = true
		err = worker.handleTrainTask(ctx, payload)

	case InferenceQueue:
		var payload models.InferenceTaskPayload
		if err = json.Unmarshal(d.Body, &payload); err != nil {
			log.Printf("Error unmarshalling inference task: %v. Body: %s", err, string(d.Body))
			d.Reject(false)
			return
		}
		processed = true
		err = worker.handleInferenceTask(ctx, payload)

	case GenerateInferenceTasksQueue:
		var payload models.GenerateInferenceTasksPayload
		if err = json.Unmarshal(d.Body, &payload); err != nil {
			log.Printf("Error unmarshalling generate tasks task: %v. Body: %s", err, string(d.Body))
			d.Reject(false) // Discard malformed message
			return
		}
		processed = true
		err = worker.handleGenerateInferenceTasksTask(ctx, payload) // Call new handler

	default:
		log.Printf("Received message from unknown queue: %s. Discarding.", d.RoutingKey)
		d.Reject(false) // Discard unknown message
		return
	}

	// Handle ACK/NACK based on processing result
	if err != nil {
		log.Printf("Error processing task (Queue: %s): %v", d.RoutingKey, err)
		// Basic retry: NACK and requeue if possible (RabbitMQ handles redelivery)
		// Be careful with infinite loops if task always fails!
		// Consider dead-letter queues in RabbitMQ for persistent failures.
		// d.Nack(false, true) multiple=false, requeue=true (for transient errors)
		// If error is permanent (e.g., bad input), use d.Reject(false) or d.Nack(false, false)
		d.Nack(false, false)
	} else if processed {
		log.Printf("Successfully processed task (Queue: %s). Acknowledging.", d.RoutingKey)
		d.Ack(false) // Acknowledge successful processing
	} else {
		// Should not happen if routing is correct, but safety check
		log.Printf("Message from queue %s was not processed. Discarding.", d.RoutingKey)
		d.Reject(false)
	}
}

// --- Task Handlers ---

func (worker *Worker) handleTrainTask(ctx context.Context, payload models.TrainTaskPayload) error {
	log.Printf("Handling training task for model %s", payload.ModelId)
	tempDir, err := os.MkdirTemp("", fmt.Sprintf("train-%s-*", payload.ModelId))
	if err != nil {
		return fmt.Errorf("failed to create temp dir: %w", err)
	}
	defer os.RemoveAll(tempDir) // Clean up temp dir

	dataDir := filepath.Join(tempDir, "data")
	modelDir := filepath.Join(tempDir, "model")
	localModelSavePath := filepath.Join(modelDir, "model.bin") // Use .bin or actual format

	if err := database.UpdateModelStatus(ctx, worker.DB, payload.ModelId, database.ModelTraining); err != nil {
		// Log error but continue training attempt? Or fail here?
		log.Printf("Warning: Failed to update model %s status to TRAINING: %v", payload.ModelId, err)
		// return fmt.Errorf("failed to set status to training: %w", err) // Option: fail fast
	}

	// Download Data
	log.Println("Downloading training data...")
	_, err = worker.S3Client.DownloadTrainingData(ctx, payload.SourceS3PathTags, dataDir)
	if err != nil {
		database.UpdateModelStatus(ctx, worker.DB, payload.ModelId, database.ModelFailed)
		return fmt.Errorf("failed to download training data: %w", err)
	}
	log.Println("Training data downloaded.")

	// Run Training (Placeholder)
	log.Println("Starting training...")
	err = core.TrainModel(dataDir, localModelSavePath)
	if err != nil {
		database.UpdateModelStatus(ctx, worker.DB, payload.ModelId, database.ModelFailed)
		return fmt.Errorf("training failed: %w", err)
	}
	log.Println("Training complete.")

	// Upload Artifact
	log.Println("Uploading model artifact...")
	_, err = worker.S3Client.UploadModelArtifact(ctx, localModelSavePath, payload.ModelId, filepath.Base(localModelSavePath))
	if err != nil {
		database.UpdateModelStatus(ctx, worker.DB, payload.ModelId, database.ModelFailed)
		return fmt.Errorf("failed to upload model artifact: %w", err)
	}
	log.Println("Model artifact uploaded.")

	// Final DB Update on Success
	err = database.UpdateModelStatus(ctx, worker.DB, payload.ModelId, database.ModelTrained)
	if err != nil {
		// Log error, but task succeeded overall
		log.Printf("Warning: Failed to update model %s status to TRAINED: %v", payload.ModelId, err)
		// Return success anyway, as the core work is done
	}

	return nil // Success
}

func (worker *Worker) handleInferenceTask(ctx context.Context, payload models.InferenceTaskPayload) error {

	// TODO: download s3 keys from bucket, download model, run inference of model on files, upload results to s3

	return nil // Success
}

func (worker *Worker) handleGenerateInferenceTasksTask(ctx context.Context, payload models.GenerateInferenceTasksPayload) error {
	log.Printf("Handling generate tasks for job %s, bucket %s, prefix %s", payload.JobId, payload.SourceS3Bucket, payload.SourceS3Prefix)

	// Optional: Update job status to GENERATING
	// err := deps.DB.UpdateInferenceJobStatus(ctx, payload.JobID, models.JobGeneratingTasks)
	// if err != nil { log.Printf("Warning: Failed to update job %s status to GENERATING: %v", payload.JobID, err) }

	s3Client := worker.S3Client // Get underlying *s3.Client

	targetBytes := payload.ChunkTargetBytes
	if targetBytes <= 0 {
		targetBytes = 10 * 1024 * 1024 * 1024 // Default 10GB if not set or invalid
		log.Printf("Using default chunk target size: %d bytes for job %s", targetBytes, payload.JobId)
	}

	var totalTasksQueued int = 0 // Counter managed by the callback

	// Define the callback function to process each chunk
	processChunkCallback := func(ctx context.Context, chunkKeys []string, chunkSize int64) error {
		// This function now contains the logic specific to creating and publishing the task
		currentTaskIndex := totalTasksQueued + 1 // For logging clarity
		log.Printf("Handler: Processing chunk %d for job %s (Size: %d bytes, Keys: %d)", currentTaskIndex, payload.JobId, chunkSize, len(chunkKeys))

		inferencePayload := models.InferenceTaskPayload{
			JobId:             payload.JobId,
			ModelId:           payload.ModelId,
			ModelArtifactPath: payload.ModelArtifactPath,
			SourceS3Bucket:    payload.SourceS3Bucket,
			SourceS3Keys:      chunkKeys,
			DestS3Bucket:      payload.DestS3Bucket,
		}

		if err := worker.Publisher.PublishInferenceTask(ctx, inferencePayload); err != nil {
			log.Printf("ERROR Handler: Failed to publish inference chunk %d for job %s: %v", currentTaskIndex, payload.JobId, err)
			// Return the error so the helper function knows processing failed
			return fmt.Errorf("failed to publish inference chunk %d: %w", currentTaskIndex, err)
		}

		totalTasksQueued++                                                                                    // Increment count *after* successful publishing
		log.Printf("Handler: Successfully published chunk %d for job %s.", currentTaskIndex-1, payload.JobId) // Log index starts from 1
		return nil
	}

	// Call the helper function
	processedChunks, err := s3Client.ListAndChunkS3Objects(
		ctx,
		payload.SourceS3Bucket,
		payload.SourceS3Prefix,
		targetBytes,
		payload.JobId, // Pass JobID for logging context in helper
		processChunkCallback,
	)

	// Check for errors from the helper (S3 listing or callback failure)
	if err != nil {
		log.Printf("ERROR Handler: Failed during S3 processing/chunk publishing for job %s: %v", payload.JobId, err)
		// Attempt to mark the job as failed
		_ = database.UpdateGenerateInferenceTasksTaskStatus(ctx, worker.DB, payload.JobId, database.JobFailed)
		// Return the error from the handler
		return fmt.Errorf("failed during task generation for job %s: %w", payload.JobId, err)
	}

	// Double-check if the counter matches (it should if no errors occurred)
	if processedChunks != totalTasksQueued {
		log.Printf("Warning Handler: Mismatch between processed chunks (%d) and tasks queued (%d) for job %s. This might indicate an issue.", processedChunks, totalTasksQueued, payload.JobId)
	}

	log.Printf("Handler: Finished generating %d inference task chunks for job %s.", totalTasksQueued, payload.JobId)

	// Update job status based on whether any tasks were generated
	finalStatus := database.JobRunning
	if totalTasksQueued == 0 {
		log.Printf("Handler: No inference tasks were generated for job %s (no files found or matched prefix?). Setting status to COMPLETED.", payload.JobId)
		finalStatus = database.JobCompleted
	} else {
		log.Printf("Handler: Setting job %s status to RUNNING.", payload.JobId)
	}

	dbErr := database.UpdateGenerateInferenceTasksTaskStatus(ctx, worker.DB, payload.JobId, finalStatus)
	if dbErr != nil {
		// Log error, but main task generation succeeded (or completed with 0 tasks)
		log.Printf("Warning Handler: Failed to update job %s final status to %s: %v", payload.JobId, finalStatus, dbErr)
		// Do not return this as a handler error, as the core task succeeded.
	}

	return nil // Success for the generator task handler
}
