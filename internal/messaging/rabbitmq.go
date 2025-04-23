package messaging

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"ner-backend/pkg/models"
	"sync"
	"time"

	amqp "github.com/rabbitmq/amqp091-go"
)

func connectToRabbitMQ(url string) (*amqp.Connection, error) {
	var conn *amqp.Connection
	var err error
	for i := 0; i < MaxConnectRetry; i++ {
		conn, err = amqp.Dial(url)
		if err == nil {
			slog.Info("connected to rabbitmq")
			return conn, nil
		}
		slog.Warn("failed to connect to rabbitmq", "attempt", i+1, "max_attempts", MaxConnectRetry, "error", err)
		time.Sleep(RetryDelay)
	}
	slog.Error("failed to connect to rabbitmq", "attempts", MaxConnectRetry, "error", err)
	return nil, fmt.Errorf("worker failed to connect after %d attempts: %w", MaxConnectRetry, err)
}

type RabbitMQPublisher struct {
	connLock   sync.RWMutex
	conn       *amqp.Connection
	channel    *amqp.Channel
	url        string
	destructor sync.Once
}

func NewRabbitMQPublisher(rabbitMQURL string) (*RabbitMQPublisher, error) {
	p := &RabbitMQPublisher{url: rabbitMQURL}
	if err := p.connect(); err != nil {
		return nil, err
	}
	return p, nil
}

func (p *RabbitMQPublisher) connect() error {
	var err error
	p.conn, err = connectToRabbitMQ(p.url)
	if err != nil {
		return err
	}

	p.channel, err = p.conn.Channel()
	if err != nil {
		p.conn.Close() // Close connection if channel fails
		slog.Error("failed to open rabbitmq channel", "error", err)
		return fmt.Errorf("failed to open rabbitmq channel: %w", err)
	}

	queues := []string{TrainingQueue, InferenceQueue, ShardDataQueue}
	for _, queue := range queues {
		_, err := p.channel.QueueDeclare(queue, true, false, false, false, nil)
		if err != nil {
			p.conn.Close()
			return fmt.Errorf("failed to declare rabbitmq queue %s: %w", queue, err)
		}
	}

	slog.Info("rabbitmq channel opened and queues declared")

	// Handle reconnects in background
	go p.handleReconnect()

	return nil
}

func (p *RabbitMQPublisher) handleReconnect() {
	notifyClose := make(chan *amqp.Error)
	p.channel.NotifyClose(notifyClose)

	err, ok := <-notifyClose
	if !ok { // channel is just closed on graceful close
		slog.Info("rabbitmq connection closed", "error", err)
		return
	}

	slog.Warn("rabbit connection closed, attempting to reconnect", "error", err)

	p.connLock.Lock() // This is to ensure that the connection is not used while we are reconnecting
	defer p.connLock.Unlock()

	p.channel = nil
	p.conn = nil
	for {
		if p.connect() == nil {
			slog.Info("successfully reconnected to rabbitmq.")
			return
		}
		time.Sleep(RetryDelay * 10)
	}
}

func (p *RabbitMQPublisher) publishTaskInternal(ctx context.Context, queueName string, payload interface{}) error {
	p.connLock.RLock()
	defer p.connLock.RUnlock()

	if p.channel == nil || p.channel.IsClosed() {
		return fmt.Errorf("rabbitmq connection is closed")
	}

	body, err := json.Marshal(payload)
	if err != nil {
		slog.Error("Failed to marshal payload", "queue", queueName, "error", err)
		return fmt.Errorf("failed to marshal %s payload: %w", queueName, err)
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
		slog.Error("Failed to publish task, potential connection issue", "queue", queueName, "error", err)
		return fmt.Errorf("failed to publish %s: %w", queueName, err)
	}

	return nil
}

func (p *RabbitMQPublisher) PublishTrainTask(ctx context.Context, payload models.TrainTaskPayload) error {
	return p.publishTaskInternal(ctx, TrainingQueue, payload)
}

func (p *RabbitMQPublisher) PublishShardDataTask(ctx context.Context, payload models.ShardDataPayload) error {
	return p.publishTaskInternal(ctx, ShardDataQueue, payload)
}

func (p *RabbitMQPublisher) PublishInferenceTask(ctx context.Context, payload models.InferenceTaskPayload) error {
	return p.publishTaskInternal(ctx, InferenceQueue, payload)
}

func (p *RabbitMQPublisher) Close() {
	p.destructor.Do(func() {
		if err := p.conn.Close(); err != nil {
			slog.Error("error closing rabbitmq connection", "error", err)
		}
	})
}

type RabbitMQTask struct {
	d amqp.Delivery
}

func (t *RabbitMQTask) Type() string {
	return t.d.RoutingKey
}

func (t *RabbitMQTask) Payload() []byte {
	return t.d.Body
}

func (t *RabbitMQTask) Ack() error {
	return t.d.Ack(false)
}

func (t *RabbitMQTask) Nack() error {
	// Basic retry: NACK and requeue if possible (RabbitMQ handles redelivery)
	// Be careful with infinite loops if task always fails!
	// Consider dead-letter queues in RabbitMQ for persistent failures.
	// d.Nack(false, true) multiple=false, requeue=true (for transient errors)
	// If error is permanent (e.g., bad input), use d.Reject(false) or d.Nack(false, false)
	return t.d.Nack(false, false)
}

func (t *RabbitMQTask) Reject() error {
	return t.d.Reject(false)
}

type RabbitMQReceiver struct {
	tasks chan Task
	url   string
	stop  chan struct{}
}

func NewRabbitMQReceiver(rabbitMQURL string) (*RabbitMQReceiver, error) {
	c := &RabbitMQReceiver{
		tasks: make(chan Task),
		url:   rabbitMQURL,
	}

	if err := c.receiveTasks(); err != nil {
		return nil, err
	}
	return c, nil
}

func (c *RabbitMQReceiver) consume(msgs <-chan amqp.Delivery) {
	for d := range msgs {
		task := &RabbitMQTask{d: d}
		c.tasks <- task
	}
}

func (c *RabbitMQReceiver) receiveTasks() error {
	conn, err := connectToRabbitMQ(c.url)
	if err != nil {
		return err
	}
	channel, err := conn.Channel()
	if err != nil {
		slog.Error("failed to open rabbitmq channel", "error", err)
		conn.Close()
		return fmt.Errorf("failed to open rabbitmq channel: %w", err)
	}

	// Set QoS to process one message at a time per worker instance
	// Adjust prefetch count if needed for batching or higher throughput per worker
	err = channel.Qos(1, 0, false)
	if err != nil {
		slog.Error("failed to set channel qos", "error", err)
		conn.Close()
		return fmt.Errorf("failed to set channel qos: %w", err)
	}

	queues := []string{TrainingQueue, InferenceQueue, ShardDataQueue}

	for _, queue := range queues {
		msgs, err := channel.Consume(queue, "", false, false, false, false, nil)
		if err != nil {
			slog.Error("failed to consume from rabbitmq queue", "queue", queue, "error", err)
			conn.Close()
			return fmt.Errorf("failed to consume from rabbitmq queue %s: %w", queue, err)
		}

		go c.consume(msgs)
	}

	go c.handleReconnect(conn, channel)

	return nil
}

func (c *RabbitMQReceiver) handleReconnect(conn *amqp.Connection, channel *amqp.Channel) {
	notifyClose := make(chan *amqp.Error)
	channel.NotifyClose(notifyClose)

	select {
	case err, ok := <-notifyClose:
		if !ok { // channel is just closed on graceful close
			slog.Info("rabbitmq connection closed", "error", err)
			return
		}

		slog.Warn("rabbit connection closed, attempting to reconnect", "error", err)

		for {
			if c.receiveTasks() == nil {
				slog.Info("successfully restarted rabbitmq consumer")
				return
			}
			time.Sleep(RetryDelay * 10)
		}
	case <-c.stop:
		slog.Info("stopping rabbitmq consumer")
		if err := conn.Close(); err != nil {
			slog.Error("error closing rabbitmp conn", "error", err)
		}
		return
	}
}

func (c *RabbitMQReceiver) Tasks() <-chan Task {
	return c.tasks
}

func (c *RabbitMQReceiver) Close() {
	close(c.stop)
}
