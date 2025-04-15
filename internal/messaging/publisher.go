package messaging

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"ner_backend/pkg/models" // Adjust import path

	amqp "github.com/rabbitmq/amqp091-go"
)

const (
	TrainingQueue   = "training_queue"
	InferenceQueue  = "inference_queue"
	RetryDelay      = 5 * time.Second
	MaxConnectRetry = 5
)

type TaskPublisher struct {
	conn    *amqp.Connection
	channel *amqp.Channel
	url     string
}

func NewTaskPublisher(rabbitMQURL string) (*TaskPublisher, error) {
	p := &TaskPublisher{url: rabbitMQURL}
	if err := p.connect(); err != nil {
		return nil, err
	}
	return p, nil
}

func (p *TaskPublisher) connect() error {
	var err error
	for i := 0; i < MaxConnectRetry; i++ {
		p.conn, err = amqp.Dial(p.url)
		if err == nil {
			log.Println("RabbitMQ connection established.")
			p.channel, err = p.conn.Channel()
			if err != nil {
				p.conn.Close() // Close connection if channel fails
				return fmt.Errorf("failed to open RabbitMQ channel: %w", err)
			}

			// Declare queues to ensure they exist
			_, err = p.channel.QueueDeclare(TrainingQueue, true, false, false, false, nil) // Durable
			if err != nil {
				p.channel.Close()
				p.conn.Close()
				return fmt.Errorf("failed to declare queue %s: %w", TrainingQueue, err)
			}
			_, err = p.channel.QueueDeclare(InferenceQueue, true, false, false, false, nil) // Durable
			if err != nil {
				p.channel.Close()
				p.conn.Close()
				return fmt.Errorf("failed to declare queue %s: %w", InferenceQueue, err)
			}

			log.Println("RabbitMQ channel opened and queues declared.")

			// Handle reconnects in background
			go p.handleReconnect()

			return nil
		}
		log.Printf("Failed to connect to RabbitMQ (attempt %d/%d): %v. Retrying in %v...", i+1, MaxConnectRetry, err, RetryDelay)
		time.Sleep(RetryDelay)
	}
	return fmt.Errorf("failed to connect to RabbitMQ after %d attempts: %w", MaxConnectRetry, err)
}

func (p *TaskPublisher) handleReconnect() {
	notifyClose := make(chan *amqp.Error)
	p.conn.NotifyClose(notifyClose)

	err := <-notifyClose // Block until connection closes
	log.Printf("RabbitMQ connection closed: %v. Attempting to reconnect...", err)
	p.channel = nil // Mark channel as invalid
	p.conn = nil    // Mark connection as invalid
	// Loop indefinitely trying to reconnect
	for {
		if p.connect() == nil {
			log.Println("Successfully reconnected to RabbitMQ.")
			return // Exit goroutine on successful reconnect
		}
		time.Sleep(RetryDelay * 2) // Wait longer between reconnect cycles
	}
}

func (p *TaskPublisher) ensureChannel() error {
	if p.channel == nil {
		log.Println("RabbitMQ channel is nil, attempting to reconnect...")
		if err := p.connect(); err != nil {
			return fmt.Errorf("cannot publish: failed to reconnect: %w", err)
		}
	}
	// Optional: Check if channel is open, though publish errors usually handle this
	// if p.channel.IsClosed() { ... }
	return nil
}

// PublishTrainTask sends a training task to the queue
func (p *TaskPublisher) PublishTrainTask(ctx context.Context, payload models.TrainTaskPayload) error {
	if err := p.ensureChannel(); err != nil {
		return err
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal train task payload: %w", err)
	}

	err = p.channel.PublishWithContext(ctx,
		"",            // exchange (default)
		TrainingQueue, // routing key (queue name)
		false,         // mandatory
		false,         // immediate
		amqp.Publishing{
			ContentType:  "application/json",
			DeliveryMode: amqp.Persistent, // Make message persistent
			Body:         body,
		})
	if err != nil {
		// Handle potential connection errors here too, maybe trigger reconnect
		log.Printf("Failed to publish train task, may need reconnect: %v", err)
		return fmt.Errorf("failed to publish train task: %w", err)
	}
	log.Printf("Published training task for model %s", payload.ModelID)
	return nil
}

// PublishInferenceTask sends an inference task to the queue
func (p *TaskPublisher) PublishInferenceTask(ctx context.Context, payload models.InferenceTaskPayload) error {
	if err := p.ensureChannel(); err != nil {
		return err
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal inference task payload: %w", err)
	}

	err = p.channel.PublishWithContext(ctx,
		"",             // exchange (default)
		InferenceQueue, // routing key (queue name)
		false,          // mandatory
		false,          // immediate
		amqp.Publishing{
			ContentType:  "application/json",
			DeliveryMode: amqp.Persistent, // Make message persistent
			Body:         body,
		})
	if err != nil {
		log.Printf("Failed to publish inference task, may need reconnect: %v", err)
		return fmt.Errorf("failed to publish inference task: %w", err)
	}
	// Avoid logging every single inference task publication if high volume
	// log.Printf("Published inference task for job %s, key %s", payload.JobID, payload.SourceS3Key)
	return nil
}

func (p *TaskPublisher) Close() {
	if p.channel != nil {
		p.channel.Close()
	}
	if p.conn != nil {
		p.conn.Close()
	}
	log.Println("RabbitMQ publisher connection closed.")
}

// --- Define Payload Structs ---
// (Could also be in pkg/models)

type TrainTaskPayload struct {
	ModelID          string `json:"model_id"`
	SourceS3PathTags string `json:"source_s3_path_tags"`
}

type InferenceTaskPayload struct {
	JobID             string `json:"job_id"`
	ModelID           string `json:"model_id"`
	ModelArtifactPath string `json:"model_artifact_path"` // Pass from API
	SourceS3Bucket    string `json:"source_s3_bucket"`
	SourceS3Key       string `json:"source_s3_key"`
	DestS3Bucket      string `json:"dest_s3_bucket"`
}
