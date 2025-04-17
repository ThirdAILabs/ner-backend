package messaging

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"log/slog"
	"time"

	"ner-backend/pkg/models" // Adjust import path

	amqp "github.com/rabbitmq/amqp091-go"
)

const (
	TrainingQueue   = "training_queue"
	InferenceQueue  = "inference_queue"
	ShardDataQueue  = "shard_data_queue"
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
			_, err = p.channel.QueueDeclare(ShardDataQueue, true, false, false, false, nil) // Durable
			if err != nil {
				p.channel.Close()
				p.conn.Close()
				return fmt.Errorf("failed to declare queue %s: %w", ShardDataQueue, err)
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

func (p *TaskPublisher) publishTaskInternal(ctx context.Context, queueName string, payload interface{}, taskType string) error {
	if err := p.ensureChannel(); err != nil {
		return err
	}

	body, err := json.Marshal(payload)
	if err != nil {
		slog.Error("Failed to marshal payload",
			slog.String("task_type", taskType),
			slog.Any("error", err),
		)
		return fmt.Errorf("failed to marshal %s payload: %w", taskType, err)
	}

	err = p.channel.PublishWithContext(ctx,
		"",        // exchange (default)
		queueName, // routing key (queue name)
		false,     // mandatory
		false,     // immediate
		amqp.Publishing{
			ContentType:  "application/json",
			DeliveryMode: amqp.Persistent, // Make message persistent
			Body:         body,
		})

	if err != nil {
		slog.Error("Failed to publish task, potential connection issue",
			slog.String("task_type", taskType),
			slog.String("queue", queueName),
			slog.Any("error", err),
		)
		return fmt.Errorf("failed to publish %s: %w", taskType, err)
	}

	return nil
}

func (p *TaskPublisher) PublishTrainTask(ctx context.Context, payload models.TrainTaskPayload) error {
	taskType := "train_task"
	err := p.publishTaskInternal(ctx, TrainingQueue, payload, taskType)
	if err != nil {
		return err
	}
	return nil
}

func (p *TaskPublisher) PublishShardDataTask(ctx context.Context, payload models.ShardDataPayload) error {
	taskType := "shard_data_task"
	err := p.publishTaskInternal(ctx, ShardDataQueue, payload, taskType)
	if err != nil {
		return err
	}
	return nil
}

func (p *TaskPublisher) PublishInferenceTask(ctx context.Context, payload models.InferenceTaskPayload) error {
	taskType := "inference_task"
	err := p.publishTaskInternal(ctx, InferenceQueue, payload, taskType)
	if err != nil {
		return err
	}
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
