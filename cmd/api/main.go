package main

import (
	"context"
	"log"
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
	ort "github.com/yalue/onnxruntime_go"
)

type APIConfig struct {
	DatabaseURL       string `env:"DATABASE_URL,notEmpty,required"`
	RabbitMQURL       string `env:"RABBITMQ_URL,notEmpty,required"`
	S3EndpointURL     string `env:"S3_ENDPOINT_URL"`
	S3Region          string `env:"S3_REGION"`
	S3AccessKeyID     string `env:"INTERNAL_AWS_ACCESS_KEY_ID"`
	S3SecretAccessKey string `env:"INTERNAL_AWS_SECRET_ACCESS_KEY"`
	BucketName        string `env:"BUCKET_NAME,notEmpty,required"`
	LicenseKey        string `env:"LICENSE_KEY" envDefault:""`
	APIPort           string `env:"API_PORT" envDefault:"8001"`
	ChunkTargetBytes  int64  `env:"S3_CHUNK_TARGET_BYTES" envDefault:"10737418240"`
	EnterpriseMode    bool   `env:"ENTERPRISE_MODE" envDefault:"false"`
	HostModelDir      string `env:"HOST_MODEL_DIR" envDefault:"/app/models"`
	OnnxRuntimeDylib  string `env:"ONNX_RUNTIME_DYLIB"`
}

func main() {
	log.Println("Starting API Server...")

	cmd.LoadEnvFile()

	var cfg APIConfig
	if err := env.Parse(&cfg); err != nil {
		log.Fatalf("error parsing config: %v", err)
	}

	if cfg.OnnxRuntimeDylib == "" {
		log.Fatalf("ONNX_RUNTIME_DYLIB must be set")
	}
	ort.SetSharedLibraryPath(cfg.OnnxRuntimeDylib)
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatalf("could not init ONNX Runtime: %v", err)
	}
	defer func() {
		if err := ort.DestroyEnvironment(); err != nil {
			log.Fatalf("error destroying onnx env: %v", err)
		}
	}()

	db, err := database.NewDatabase(cfg.DatabaseURL)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}

	s3Cfg := storage.S3ClientConfig{
		Endpoint:        cfg.S3EndpointURL,
		Region:          cfg.S3Region,
		AccessKeyID:     cfg.S3AccessKeyID,
		SecretAccessKey: cfg.S3SecretAccessKey,
	}
	s3ObjectStore, err := storage.NewS3ObjectStore(cfg.BucketName, s3Cfg)
	if err != nil {
		log.Fatalf("Failed to create S3 object store: %v", err)
	}

	if err != nil {
		log.Fatalf("Worker: Failed to create S3 client: %v", err)
	}

	if err := cmd.InitializeOnnxCnnModel(context.Background(), db, s3ObjectStore, cmd.ModelBucketName, "basic", cfg.HostModelDir); err != nil {
		log.Fatalf("failed to init ONNX model: %v", err)
	}

	if err := cmd.RemoveExcludedTagsFromAllModels(db); err != nil {
		log.Fatalf("Failed to remove excluded tags from all models: %v", err)
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

	licensing := cmd.CreateLicenseVerifier(db, cfg.LicenseKey)
	licenseInfo, err := licensing.VerifyLicense(context.Background())
	if err != nil {
		log.Fatalf("License verification failed - Info: %v, Error: %v", licenseInfo, err)
	}

	apiHandler := api.NewBackendService(db, s3ObjectStore, cmd.UploadBucketName, publisher, cfg.ChunkTargetBytes, licensing, cfg.EnterpriseMode)

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
