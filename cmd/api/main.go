package main

import (
	"context"
	"log"
	"ner-backend/cmd" // Adjust import path
	"ner-backend/internal/api"
	"ner-backend/internal/database"
	"ner-backend/internal/messaging"
	"ner-backend/internal/s3"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/caarlos0/env/v11"
	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
)

type APIConfig struct {
	DatabaseURL       string `env:"DATABASE_URL,notEmpty,required"`
	RabbitMQURL       string `env:"RABBITMQ_URL,notEmpty,required"`
	S3EndpointURL     string `env:"S3_ENDPOINT_URL,notEmpty,required"`
	S3AccessKeyID     string `env:"AWS_ACCESS_KEY_ID,notEmpty,required"`
	S3SecretAccessKey string `env:"AWS_SECRET_ACCESS_KEY,notEmpty,required"`
	S3Region          string `env:"AWS_REGION,notEmpty,required"`
	ModelBucketName   string `env:"MODEL_BUCKET_NAME" envDefault:"models"`
	QueueNames        string `env:"QUEUE_NAMES" envDefault:"inference_queue,training_queue,shard_data_queue"`
	WorkerConcurrency int    `env:"CONCURRENCY" envDefault:"1"`
	APIPort           string `env:"API_PORT" envDefault:"8001"`
	ChunkTargetBytes  int64  `env:"S3_CHUNK_TARGET_BYTES" envDefault:"10737418240"`
}

func main() {
	log.Println("Starting API Server...")

	cmd.LoadEnvFile()

	var cfg APIConfig
	if err := env.Parse(&cfg); err != nil {
		log.Fatalf("error parsing config: %v", err)
	}

	db, err := database.NewDatabase(cfg.DatabaseURL)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}

	// Initialize S3 Client
	s3Cfg := s3.Config{
		S3EndpointURL:     cfg.S3EndpointURL,
		S3AccessKeyID:     cfg.S3AccessKeyID,
		S3SecretAccessKey: cfg.S3SecretAccessKey,
		S3Region:          cfg.S3Region,
		ModelBucketName:   cfg.ModelBucketName,
	}
	s3Client, err := s3.NewS3Client(&s3Cfg)
	if err != nil {
		log.Fatalf("Failed to create S3 client: %v", err)
	}

	// Initialize RabbitMQ Publisher
	publisher, err := messaging.NewRabbitMQPublisher(cfg.RabbitMQURL)
	if err != nil {
		log.Fatalf("Failed to connect to RabbitMQ: %v", err)
	}
	defer publisher.Close()

	// --- Chi Router Setup ---
	r := chi.NewRouter()

	// Middleware
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Logger)                    // Log requests
	r.Use(middleware.Recoverer)                 // Recover from panics
	r.Use(middleware.Timeout(60 * time.Second)) // Set request timeout

	// API Handlers (dependency injection)
	apiHandler := api.NewBackendService(db, publisher, s3Client, cfg.ChunkTargetBytes)

	apiHandler.AddRoutes(r)

	// --- Start HTTP Server ---
	server := &http.Server{
		Addr:    ":" + cfg.APIPort,
		Handler: r,
	}

	// Goroutine for graceful shutdown
	go func() {
		quit := make(chan os.Signal, 1)
		signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
		<-quit
		log.Println("Shutting down server...")

		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		if err := server.Shutdown(ctx); err != nil {
			log.Fatalf("Server forced to shutdown: %v", err)
		}
	}()

	log.Printf("API server listening on port %s", cfg.APIPort)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Could not listen on %s: %v\n", cfg.APIPort, err)
	}

	log.Println("Server stopped.")
}
