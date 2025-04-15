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

	// Initialize Database Pool
	dbPool, err := database.NewConnectionPool(cfg.DatabaseURL)
	if err != nil {
		log.Fatalf("Worker: Failed to connect to database: %v", err)
	}
	defer dbPool.Close()
	dbQueries := database.NewQueries(dbPool)

	// Initialize S3 Client
	s3Client, err := s3.NewS3Client(cfg)
	if err != nil {
		log.Fatalf("Worker: Failed to create S3 client: %v", err)
	}

	// WaitGroup to wait for worker goroutines to finish
	var wg sync.WaitGroup

	// Worker Dependencies
	deps := messaging.WorkerDependencies{
		DB:        dbQueries,
		S3Client:  s3Client,
		Config:    cfg,
		WaitGroup: &wg,
	}

	// Start worker consumers
	err = messaging.StartWorkers(cfg.RabbitMQURL, deps)
	if err != nil {
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
