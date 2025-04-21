package messaging

import (
	"context"
	"encoding/json"
	"fmt"
	"log" // Adjust import path
	"log/slog"
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

	"github.com/google/uuid"
	amqp "github.com/rabbitmq/amqp091-go"
	"gorm.io/gorm"
)

type WorkerConfig struct {
	WorkerConcurrency int
	QueueNames        string
}

type Worker struct {
	DB        *gorm.DB
	S3Client  *s3.Client
	Config    *WorkerConfig
	WaitGroup *sync.WaitGroup
	Publisher *TaskPublisher

	Processor *core.InferenceJobProcessor
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

	case ShardDataQueue:
		var payload models.ShardDataPayload
		if err = json.Unmarshal(d.Body, &payload); err != nil {
			log.Printf("Error unmarshalling shard data task: %v. Body: %s", err, string(d.Body))
			d.Reject(false) // Discard malformed message
			return
		}
		processed = true
		err = worker.handleShardDataTask(ctx, payload) // Call new handler

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
	reportId := payload.ReportId

	var task database.InferenceTask
	if err := worker.DB.WithContext(ctx).Preload("Report").Preload("Report.Model").Preload("Report.Groups").First(&task, "report_id = ? AND task_id = ?", reportId, payload.TaskId).Error; err != nil {
		slog.Error("error fetching inference task", "report_id", reportId, "task_id", payload.TaskId, "error", err)
		return fmt.Errorf("error getting inference task: %w", err)
	}

	if err := database.UpdateInferenceTaskStatus(ctx, worker.DB, reportId, payload.TaskId, database.JobRunning); err != nil {
		slog.Error("error updating inference task status to running", "report_id", reportId, "task_id", payload.TaskId, "error", err)
	}

	s3Objects := strings.Split(task.SourceS3Keys, ";")

	groupToQuery := map[uuid.UUID]string{}
	for _, group := range task.Report.Groups {
		groupToQuery[group.Id] = group.Query
	}

	workerErr := worker.Processor.RunInferenceTask(task.ReportId, task.Report.Model.Id, task.Report.Model.Type, groupToQuery, task.SourceS3Bucket, s3Objects)
	if workerErr != nil {
		slog.Error("error running inference task", "report_id", reportId, "task_id", payload.TaskId, "error", workerErr)
		if err := database.UpdateInferenceTaskStatus(ctx, worker.DB, reportId, payload.TaskId, database.JobFailed); err != nil {
			slog.Error("error updating inference task status to failed", "report_id", reportId, "task_id", payload.TaskId, "error", err)
		}
		return fmt.Errorf("error running inference task: %w", workerErr)
	}

	if err := database.UpdateInferenceTaskStatus(ctx, worker.DB, reportId, payload.TaskId, database.JobCompleted); err != nil {
		slog.Error("error updating inference task status to complete", "report_id", reportId, "task_id", payload.TaskId, "error", err)
		return fmt.Errorf("error updating inference task status to complete: %w", err)
	}

	return nil
}

func (worker *Worker) handleShardDataTask(ctx context.Context, payload models.ShardDataPayload) error {
	reportId := payload.ReportId

	var task database.ShardDataTask
	if err := worker.DB.WithContext(ctx).Preload("Report").First(&task, "report_id = ?", reportId).Error; err != nil {
		slog.Error("error fetching shard data task", "report_id", reportId, "error", err)
		return fmt.Errorf("error getting shard data task: %w", err)
	}

	slog.Info("Handling generate tasks", "jobId", task.ReportId, "sourceBucket", task.Report.SourceS3Bucket, "sourcePrefix", task.Report.SourceS3Prefix)

	targetBytes := task.ChunkTargetBytes
	if targetBytes <= 0 {
		targetBytes = 10 * 1024 * 1024 * 1024 // Default 10GB if not set or invalid
		slog.Info("Using default chunk target size", "targetBytes", targetBytes, "jobId", reportId)
	}

	var taskIndex int = 0

	// Define the callback function to process each chunk
	processChunkCallback := func(ctx context.Context, chunkKeys []string, chunkSize int64) error {
		taskIndex++ // Increment count *after* successful publishing
		slog.Info("Handler: Processing chunk", "chunkIndex", taskIndex, "jobId", reportId, "chunkSize", chunkSize, "keyCount", len(chunkKeys))

		task := database.InferenceTask{
			ReportId:     reportId,
			TaskId:       taskIndex,
			Status:       database.JobQueued,
			CreationTime: time.Now().UTC(),

			SourceS3Bucket: task.Report.SourceS3Bucket,
			SourceS3Keys:   strings.Join(chunkKeys, ";"),
			TotalSize:      chunkSize,
		}

		inferencePayload := models.InferenceTaskPayload{
			ReportId: task.ReportId, TaskId: task.TaskId,
		}

		if err := worker.Publisher.PublishInferenceTask(ctx, inferencePayload); err != nil {
			// Use slog.Error for failures
			slog.Error("Handler: Failed to publish inference chunk", "report_id", reportId, "task_id", taskIndex, "error", err)
			// Return the error so the helper function knows processing failed
			return fmt.Errorf("failed to publish inference chunk %d: %w", taskIndex, err)
		}

		if err := worker.DB.WithContext(ctx).Create(&task).Error; err != nil {
			slog.Error("error saving inference task to db", "report_id", task.ReportId, "task_id", task.TaskId, "error", err)
			return fmt.Errorf("error saving inference task to db: %w", err)
		}

		return nil
	}

	processedChunks, err := worker.S3Client.ListAndChunkS3Objects(
		ctx,
		task.Report.SourceS3Bucket,
		task.Report.SourceS3Prefix.String,
		targetBytes,
		reportId,
		processChunkCallback,
	)

	// Check for errors from the helper (S3 listing or callback failure)
	if err != nil {
		// Use slog.Error
		slog.Error("Failed during S3 processing/chunk publishing", "report_id", reportId, "error", err)
		_ = database.UpdateShardDataTaskStatus(ctx, worker.DB, reportId, database.JobFailed)
		return fmt.Errorf("failed during task generation for job %s: %w", reportId, err)
	}

	// Double-check if the counter matches (it should if no errors occurred)
	if processedChunks != taskIndex {
		// Use slog.Warn for potential issues/mismatches
		slog.Warn("Mismatch between processed chunks and tasks queued", "processedChunks", processedChunks, "n_tasks", taskIndex, "report_id", reportId)
	}

	slog.Info("Finished generating inference task chunks", "n_tasks", taskIndex, "report_id", reportId)

	dbErr := database.UpdateShardDataTaskStatus(ctx, worker.DB, reportId, database.JobCompleted)
	if dbErr != nil {
		// Use slog.Error
		slog.Error("Failed to update job final status", "report_id", reportId, "status", database.JobCompleted, "error", dbErr)
	}

	return nil
}
