package main

import (
	"log"
	"ner-backend/internal/config" // Adjust import path
	"ner-backend/internal/database"
	"ner-backend/internal/messaging"
	"ner-backend/internal/s3"
	"os"
	"os/signal"
	"sync"
	"syscall"
)

func main() {
	log.Println("Starting Worker Process...")

	cfg, err := config.LoadConfig()
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	db, err := database.NewDatabase(cfg.DatabaseURL)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}

	// Initialize S3 Client
	s3Client, err := s3.NewS3Client(cfg)
	if err != nil {
		log.Fatalf("Worker: Failed to create S3 client: %v", err)
	}

	// WaitGroup to wait for worker goroutines to finish
	var wg sync.WaitGroup

	publisher, err := messaging.NewTaskPublisher(cfg.RabbitMQURL)
	if err != nil {
		log.Fatalf("Failed to connect to RabbitMQ: %v", err)
	}
	defer publisher.Close()

	// Worker Dependencies
	worker := messaging.Worker{
		DB:        db,
		S3Client:  s3Client,
		Config:    cfg,
		WaitGroup: &wg,
		Publisher: publisher,
	}

	// Start worker consumers

	if err := worker.StartThreads(cfg.RabbitMQURL); err != nil {
		log.Fatalf("Worker: Failed to start message consumers: %v", err)
	}

	log.Println("Worker started. Waiting for tasks. Press Ctrl+C to exit.")

	// Wait for termination signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutdown signal received, waiting for workers to finish...")

	// TODO: Implement graceful shutdown for workers if needed
	// e.g., signal workers to stop consuming new messages and wait for current ones
	// For now, just wait for WaitGroup (which might not finish if workers loop forever)
	// wg.Wait() // This might block indefinitely if workers don't exit cleanly

	log.Println("Worker process stopped.")
}
