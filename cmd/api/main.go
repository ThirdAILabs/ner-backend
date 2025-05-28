package main

import (
	"context"
	"log"
	"log/slog"
	"ner-backend/cmd"
	"ner-backend/internal/api"
	"ner-backend/internal/database"
	"ner-backend/internal/messaging"
	"ner-backend/internal/storage"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/caarlos0/env/v11"
	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/go-chi/cors"
)

type APIConfig struct {
	DatabaseURL       string `env:"DATABASE_URL,notEmpty,required"`
	RabbitMQURL       string `env:"RABBITMQ_URL,notEmpty,required"`
	S3EndpointURL     string `env:"S3_ENDPOINT_URL,notEmpty,required"`
	S3AccessKeyID     string `env:"INTERNAL_AWS_ACCESS_KEY_ID,notEmpty,required"`
	S3SecretAccessKey string `env:"INTERNAL_AWS_SECRET_ACCESS_KEY,notEmpty,required"`
	ModelBucketName   string `env:"MODEL_BUCKET_NAME" envDefault:"ner-models"`
	QueueNames        string `env:"QUEUE_NAMES" envDefault:"inference_queue,training_queue,shard_data_queue"`
	WorkerConcurrency int    `env:"CONCURRENCY" envDefault:"1"`
	APIPort           string `env:"API_PORT" envDefault:"8001"`
	ChunkTargetBytes  int64  `env:"S3_CHUNK_TARGET_BYTES" envDefault:"10737418240"`
	HostModelDir      string `env:"HOST_MODEL_DIR" envDefault:"/app/models"`
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

	s3Cfg := storage.S3ProviderConfig{
		S3EndpointURL:     cfg.S3EndpointURL,
		S3AccessKeyID:     cfg.S3AccessKeyID,
		S3SecretAccessKey: cfg.S3SecretAccessKey,
	}
	s3Client, err := storage.NewS3Provider(s3Cfg)
	if err != nil {
		log.Fatalf("Worker: Failed to create S3 client: %v", err)
	}

	if err := s3Client.CreateBucket(context.Background(), cfg.ModelBucketName); err != nil {
		slog.Error("error creating model bucket", "error", err)
		panic("failed to create model bucket")
	}

	cmd.InitializePresidioModel(db)

	if err := cmd.InitializeCnnNerExtractor(context.Background(), db, s3Client, cfg.ModelBucketName, "advanced", cfg.HostModelDir); err != nil {
		log.Fatalf("Failed to init & upload CNN NER model: %v", err)
	}

	if err := cmd.InitializeTransformerModel(context.Background(), db, s3Client, cfg.ModelBucketName, "ultra", cfg.HostModelDir); err != nil {
		log.Fatalf("Failed to init & upload Transformer model: %v", err)
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
	r.Use(cors.Handler(cors.Options{
		AllowedOrigins:   []string{"*"},                                       // Allow all origins (TODO: make this an env var)
		AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}, // Allow all HTTP methods
		AllowedHeaders:   []string{"*"},                                       // Allow all headers
		ExposedHeaders:   []string{"*"},                                       // Expose all headers
		AllowCredentials: true,                                                // Allow cookies/auth headers
		MaxAge:           300,                                                 // Cache preflight response for 5 minutes
	}))
	r.Use(middleware.RequestID)
	r.Use(middleware.Logger)                    // Log requests
	r.Use(middleware.Recoverer)                 // Recover from panics
	r.Use(middleware.Timeout(60 * time.Second)) // Set request timeout

	apiHandler := api.NewBackendService(db, s3Client, publisher, cfg.ChunkTargetBytes)

	// Your existing API routes should be prefixed with /api to avoid conflicts
	r.Route("/api/v1", func(r chi.Router) {
		apiHandler.AddRoutes(r)
	})

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
