package main

import (
	"context"
	"fmt"
	"log"
	"log/slog"
	"path/filepath"

	"ner-backend/cmd"
	"ner-backend/internal/api"
	"ner-backend/internal/core"
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
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
)

type Config struct {
	Root          string `env:"ROOT" envDefault:"./pocket-shield"`
	Port          int    `env:"PORT" envDefault:"3001"`
	License       string `env:"LICENSE_KEY" envDefault:""`
	BoltModelPath string `env:"MODEL_PATH" envDefault:""`
}

const (
	chunkTargetBytes = 200 * 1024 * 1024 // 200MB
)

func createDatabase(root string) *gorm.DB {
	path := filepath.Join(root, "db", "pocket-shield.db")
	if err := os.MkdirAll(filepath.Dir(path), os.ModePerm); err != nil {
		log.Fatalf("Failed to create database directory: %v", err)
	}

	db, err := gorm.Open(sqlite.Open(path), &gorm.Config{})
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}

	if err := database.GetMigrator(db).Migrate(); err != nil {
		log.Fatalf("Failed to migrate database: %v", err)
	}

	return db
}

func createQueue(db *gorm.DB) *messaging.InMemoryQueue {
	var shardTasks []database.ShardDataTask
	if err := db.Where("status = ?", database.JobQueued).Find(&shardTasks).Error; err != nil {
		log.Fatalf("Failed to fetch tasks from database: %v", err)
	}

	var inferenceTasks []database.InferenceTask
	if err := db.Where("status = ?", database.JobQueued).Find(&inferenceTasks).Error; err != nil {
		log.Fatalf("Failed to fetch tasks from database: %v", err)
	}

	queue := messaging.NewInMemoryQueue()

	for _, task := range shardTasks {
		if err := queue.PublishShardDataTask(context.Background(), messaging.ShardDataPayload{
			ReportId: task.ReportId,
		}); err != nil {
			log.Fatalf("Failed to publish shard task: %v", err)
		}
	}

	for _, task := range inferenceTasks {
		if err := queue.PublishInferenceTask(context.Background(), messaging.InferenceTaskPayload{
			ReportId: task.ReportId,
			TaskId:   task.TaskId,
		}); err != nil {
			log.Fatalf("Failed to publish inference task: %v", err)
		}
	}

	return queue
}

func createServer(db *gorm.DB, storage storage.Provider, queue messaging.Publisher, port int) *http.Server {
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

	apiHandler := api.NewBackendService(db, storage, queue, chunkTargetBytes)

	r.Route("/api/v1", func(r chi.Router) {
		apiHandler.AddRoutes(r)
	})

	return &http.Server{
		Addr:    fmt.Sprintf(":%d", port),
		Handler: r,
	}
}

func main() {
	const modelBucket = "models"

	var cfg Config
	if err := env.Parse(&cfg); err != nil {
		log.Fatalf("error parsing config: %v", err)
	}

	slog.Info("starting backend", "root", cfg.Root, "port", cfg.Port)

	db := createDatabase(cfg.Root)

	storage, err := storage.NewLocalProvider(filepath.Join(cfg.Root, "storage"))
	if err != nil {
		log.Fatalf("Worker: Failed to create storage client: %v", err)
	}

	if cfg.BoltModelPath != "" {
		if err := cmd.InitializeBoltModel(db, storage, modelBucket, "basic", cfg.BoltModelPath); err != nil {
			log.Fatalf("Failed to initialize basic model: %v", err)
		}
	} else {
		cmd.InitializePresidioModel(db)
	}

	queue := createQueue(db)

	licensing := cmd.CreateLicenseVerifier(db, cfg.License)

	worker := core.NewTaskProcessor(db, storage, queue, queue, licensing, filepath.Join(cfg.Root, "models"), modelBucket)

	server := createServer(db, storage, queue, cfg.Port)

	slog.Info("starting worker")
	go worker.Start()

	// Goroutine for graceful shutdown
	go func() {
		quit := make(chan os.Signal, 1)
		signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
		<-quit
		slog.Info("shutting down server")

		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		if err := server.Shutdown(ctx); err != nil {
			log.Fatalf("Server forced to shutdown: %v", err)
		}

		slog.Info("shutting down worker")
		worker.Stop()
	}()

	slog.Info("server started", "port", cfg.Port)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Could not listen on %d: %v\n", cfg.Port, err)
	}

	slog.Info("server stopped")
}
