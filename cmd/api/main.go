package main

import (
	"context"
	"log"
	"ner_backend/internal/config" // Adjust import path
	"ner_backend/internal/database"
	"ner_backend/internal/messaging"
	"ner_backend/internal/s3"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/go-chi/render"
)

func main() {
	log.Println("Starting API Server...")

	cfg, err := config.LoadConfig()
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	// Initialize Database Pool
	dbPool, err := database.NewConnectionPool(cfg.DatabaseURL)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}
	defer dbPool.Close()
	dbQueries := database.NewQueries(dbPool)

	// Initialize S3 Client
	s3Client, err := s3.NewS3Client(cfg)
	if err != nil {
		log.Fatalf("Failed to create S3 client: %v", err)
	}

	// Initialize RabbitMQ Publisher
	publisher, err := messaging.NewTaskPublisher(cfg.RabbitMQURL)
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
	r.Use(render.SetContentType(render.ContentTypeJSON))

	// API Handlers (dependency injection)
	apiHandler := NewAPIHandler(dbQueries, publisher, s3Client)

	// Routes
	r.Get("/health", apiHandler.HealthCheck)
	r.Route("/models", func(r chi.Router) {
		r.Post("/", apiHandler.SubmitTrainingJob)
		r.Get("/{modelID}", apiHandler.GetModelStatus)
	})
	r.Route("/inference", func(r chi.Router) {
		r.Post("/", apiHandler.SubmitInferenceJob)
		r.Get("/{jobID}", apiHandler.GetInferenceJobStatus)
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
