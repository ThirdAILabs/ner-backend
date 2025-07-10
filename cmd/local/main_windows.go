//go:build windows

package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"log/slog"
	"path/filepath"

	"ner-backend/cmd"
	"ner-backend/internal/api"
	"ner-backend/internal/core"
	"ner-backend/internal/core/python"
	"ner-backend/internal/database"
	"ner-backend/internal/database/versions/migration_6"
	"ner-backend/internal/licensing"
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
	Root             string `env:"ROOT" envDefault:"./pocket-shield"`
	Port             int    `env:"PORT" envDefault:"3001"`
	License          string `env:"LICENSE_KEY" envDefault:""`
	EnterpriseMode   bool   `env:"ENTERPRISE_MODE" envDefault:"false"`
	ModelDir         string `env:"MODEL_DIR" envDefault:""`
	ModelType        string `env:"MODEL_TYPE"`
	UploadBucket     string `env:"UPLOAD_BUCKET" envDefault:"uploads"`
	AppDataDir       string `env:"APP_DATA_DIR" envDefault:"./pocket-shield"`
	OnnxRuntimeDylib string `env:"ONNX_RUNTIME_DYLIB"`
	EnablePython     bool   `env:"ENABLE_PYTHON" envDefault:"false"`
}

const (
	chunkTargetBytes = 200 * 1024 * 1024 // 200MB
)

func createDatabase(root string) *gorm.DB {
	err := migration_6.SetDefaultStorageProvider(string(storage.LocalType))
	if err != nil {
		log.Fatalf("Failed to set default storage provider: %v", err)
	}

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

func createServer(db *gorm.DB, storage storage.ObjectStore, queue messaging.Publisher, port int, modelDir string, modelType core.ModelType, uploadBucket string, licensing licensing.LicenseVerifier, enterpriseMode bool) *http.Server {
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

	apiHandler := api.NewBackendService(db, storage, uploadBucket, queue, chunkTargetBytes, licensing, enterpriseMode)

	loaders := core.NewModelLoaders()

	nerModel, err := loaders[modelType](modelDir)
	if err != nil {
		log.Fatalf("could not load NER model: %v", err)
	}

	chatHandler := api.NewChatService(db, nerModel)

	r.Route("/api/v1", func(r chi.Router) {
		apiHandler.AddRoutes(r)
		chatHandler.AddRoutes(r)
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

	// Skip ONNX runtime initialization on Windows
	if cfg.OnnxRuntimeDylib != "" {
		log.Println("Warning: ONNX runtime is not supported on Windows, skipping initialization")
	}

	log.SetFlags(log.LstdFlags | log.Lshortfile)
	if err := os.MkdirAll(cfg.Root, os.ModePerm); err != nil {
		log.Fatalf("error creating directory for log file: %v", err)
	}

	f, err := os.OpenFile(filepath.Join(cfg.Root, "backend.log"), os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		log.Fatalf("error opening log file: %v", err)
	}
	defer f.Close()

	log.SetOutput(io.MultiWriter(f, os.Stderr))

	if cfg.EnablePython {
		python.EnablePythonPlugin("python", "plugin/plugin-python/plugin.py")
	}

	slog.Info("starting backend", "root", cfg.Root, "port", cfg.Port, "app_data_dir", cfg.AppDataDir, "model_dir", cfg.ModelDir, "model_type", cfg.ModelType)

	localBaseDir := filepath.Join(cfg.Root, "storage")

	db := createDatabase(cfg.AppDataDir)

	if err := db.
		Model(&database.ShardDataTask{}).
		Where("status IN ?", []string{database.JobQueued, database.JobRunning}).
		Update("status", database.JobAborted).
		Error; err != nil {
		log.Fatalf("failed to abort stale shard tasks: %v", err)
	}

	if err := db.
		Model(&database.InferenceTask{}).
		Where("status IN ?", []string{database.JobQueued, database.JobRunning}).
		Update("status", database.JobAborted).
		Error; err != nil {
		log.Fatalf("failed to abort stale inference tasks: %v", err)
	}

	queue := createQueue(db)

	if cfg.License != "" {
		log.Printf("License found: %s", cfg.License)
	} else {
		log.Printf("No license found")
	}

	enterpriseMode := cfg.EnterpriseMode

	cmd.InitializePresidioModel(db)

	localstorage, err := storage.NewLocalObjectStore(localBaseDir)
	if err != nil {
		log.Fatalf("Failed to create local storage: %v", err)
	}

	if err := localstorage.CreateBucket(context.Background(), cfg.UploadBucket); err != nil {
		log.Fatalf("error creating upload bucket: %v", err)
	}

	if err := localstorage.CreateBucket(context.Background(), modelBucket); err != nil {
		log.Fatalf("error creating model bucket: %v", err)
	}

	if cfg.ModelType == "" {
		log.Fatalf("MODEL_TYPE must be set")
	}

	modelType := core.ParseModelType(cfg.ModelType)

	licensing := cmd.CreateLicenseVerifier(db, cfg.License)

	server := createServer(db, localstorage, queue, cfg.Port, cfg.ModelDir, modelType, cfg.UploadBucket, licensing, enterpriseMode)

	// Start the server
	go func() {
		log.Printf("Server starting on port %d", cfg.Port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("listen: %s\n", err)
		}
	}()

	// Wait for interrupt signal to gracefully shutdown the server
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("Shutting down server...")

	// The context is used to inform the server it has 5 seconds to finish
	// the request it is currently handling
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := server.Shutdown(ctx); err != nil {
		log.Fatal("Server forced to shutdown:", err)
	}

	log.Println("Server exiting")
}
