package messaging

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"ner_backend/internal/config" // Adjust import path
	"ner_backend/internal/core"
	"ner_backend/internal/database"
	"ner_backend/internal/s3"
	"ner_backend/pkg/models"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	amqp "github.com/rabbitmq/amqp091-go"
)

type WorkerDependencies struct {
	DB        *database.Queries
	S3Client  *s3.Client
	Config    *config.Config
	WaitGroup *sync.WaitGroup
}

func StartWorkers(rabbitMQURL string, deps WorkerDependencies) error {
	_, err := connectToRabbitMQ(rabbitMQURL)
	if err != nil {
		return err
	}

	numWorkers := deps.Config.WorkerConcurrency
	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU() // Default to number of CPUs
		log.Printf("Worker concurrency not specified or invalid, defaulting to %d", numWorkers)
	}

	queuesToConsume := strings.Split(deps.Config.QueueNames, ",")
	log.Printf("Starting %d worker goroutines, consuming from queues: %v", numWorkers, queuesToConsume)

	deps.WaitGroup.Add(numWorkers) // Add count for all potential worker goroutines

	for i := 0; i < numWorkers; i++ {
		go runWorkerInstance(i, rabbitMQURL, deps, queuesToConsume)
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

func runWorkerInstance(id int, rabbitMQURL string, deps WorkerDependencies, queues []string) {
	defer deps.WaitGroup.Done()
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
			processMessage(d, deps)                                                     // Process the message

			// Add a timeout or heartbeat check if needed
			// case <-time.After(30 * time.Second):
			//    log.Printf("[Worker %d] No message received for 30s", id)
		}
	}
	// log.Printf("[Worker %d] Exiting.", id) // This part is unlikely to be reached in the current loop structure
}

func processMessage(d amqp.Delivery, deps WorkerDependencies) {
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
		err = handleTrainTask(ctx, payload, deps)

	case InferenceQueue:
		var payload models.InferenceTaskPayload
		if err = json.Unmarshal(d.Body, &payload); err != nil {
			log.Printf("Error unmarshalling inference task: %v. Body: %s", err, string(d.Body))
			d.Reject(false)
			return
		}
		processed = true
		err = handleInferenceTask(ctx, payload, deps)

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

func handleTrainTask(ctx context.Context, payload models.TrainTaskPayload, deps WorkerDependencies) error {
	log.Printf("Handling training task for model %s", payload.ModelID)
	tempDir, err := os.MkdirTemp("", fmt.Sprintf("train-%s-*", payload.ModelID))
	if err != nil {
		return fmt.Errorf("failed to create temp dir: %w", err)
	}
	defer os.RemoveAll(tempDir) // Clean up temp dir

	dataDir := filepath.Join(tempDir, "data")
	modelDir := filepath.Join(tempDir, "model")
	localModelSavePath := filepath.Join(modelDir, "model.bin") // Use .bin or actual format

	// Update DB status immediately (moved from Celery task)
	err = deps.DB.UpdateModelStatus(ctx, payload.ModelID, models.StatusTraining, "")
	if err != nil {
		// Log error but continue training attempt? Or fail here?
		log.Printf("Warning: Failed to update model %s status to TRAINING: %v", payload.ModelID, err)
		// return fmt.Errorf("failed to set status to training: %w", err) // Option: fail fast
	}

	// Download Data
	log.Println("Downloading training data...")
	_, err = deps.S3Client.DownloadTrainingData(ctx, payload.SourceS3PathTags, dataDir)
	if err != nil {
		deps.DB.UpdateModelStatus(ctx, payload.ModelID, models.StatusFailed, "") // Update DB on failure
		return fmt.Errorf("failed to download training data: %w", err)
	}
	log.Println("Training data downloaded.")

	// Run Training (Placeholder)
	log.Println("Starting training...")
	err = core.TrainModel(dataDir, localModelSavePath)
	if err != nil {
		deps.DB.UpdateModelStatus(ctx, payload.ModelID, models.StatusFailed, "")
		return fmt.Errorf("training failed: %w", err)
	}
	log.Println("Training complete.")

	// Upload Artifact
	log.Println("Uploading model artifact...")
	artifactS3Path, err := deps.S3Client.UploadModelArtifact(ctx, localModelSavePath, payload.ModelID, filepath.Base(localModelSavePath))
	if err != nil {
		deps.DB.UpdateModelStatus(ctx, payload.ModelID, models.StatusFailed, "") // Update DB on failure
		return fmt.Errorf("failed to upload model artifact: %w", err)
	}
	log.Println("Model artifact uploaded.")

	// Final DB Update on Success
	err = deps.DB.UpdateModelStatus(ctx, payload.ModelID, models.StatusTrained, artifactS3Path)
	if err != nil {
		// Log error, but task succeeded overall
		log.Printf("Warning: Failed to update model %s status to TRAINED: %v", payload.ModelID, err)
		// Return success anyway, as the core work is done
	}

	return nil // Success
}

func handleInferenceTask(ctx context.Context, payload models.InferenceTaskPayload, deps WorkerDependencies) error {
	log.Printf("Handling inference task for job %s, file %s", payload.JobID, payload.SourceS3Key)
	tempDir, err := os.MkdirTemp("", fmt.Sprintf("infer-%s-*", payload.JobID))
	if err != nil {
		return fmt.Errorf("failed to create temp dir: %w", err)
	}
	defer os.RemoveAll(tempDir)

	localInputPath := filepath.Join(tempDir, filepath.Base(payload.SourceS3Key))
	localResultPath := filepath.Join(tempDir, "results.json")
	modelDir := filepath.Join(tempDir, "model")

	// Download Model
	log.Println("Downloading model artifact...")
	localModelPath, err := deps.S3Client.DownloadModelArtifact(ctx, payload.ModelID, modelDir, "model.bin") // Assuming .bin from training
	if err != nil {
		return fmt.Errorf("failed to download model %s: %w", payload.ModelID, err)
	}
	log.Println("Model artifact downloaded.")

	// Download Source File
	log.Println("Downloading source file...")
	err = deps.S3Client.DownloadFile(ctx, payload.SourceS3Bucket, payload.SourceS3Key, localInputPath)
	if err != nil {
		return fmt.Errorf("failed to download source file s3://%s/%s: %w", payload.SourceS3Bucket, payload.SourceS3Key, err)
	}
	log.Println("Source file downloaded.")

	// Run Inference (Placeholder)
	log.Println("Running inference...")
	results, err := core.RunInference(localModelPath, localInputPath)
	if err != nil {
		return fmt.Errorf("inference failed: %w", err)
	}
	log.Println("Inference complete.")

	// Save results locally
	resultBytes, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal inference results: %w", err)
	}
	err = os.WriteFile(localResultPath, resultBytes, 0644)
	if err != nil {
		return fmt.Errorf("failed to save results locally: %w", err)
	}

	// Upload Results
	log.Println("Uploading results...")
	resultFilename := strings.ReplaceAll(payload.SourceS3Key, "/", "_") + ".json"
	resultS3Key := fmt.Sprintf("results/%s/%s", payload.JobID, resultFilename)
	_, err = deps.S3Client.UploadFile(ctx, localResultPath, payload.DestS3Bucket, resultS3Key)
	if err != nil {
		return fmt.Errorf("failed to upload results: %w", err)
	}
	log.Println("Results uploaded.")

	// Optional: Update individual task status in DB if tracking is implemented

	return nil // Success
}
