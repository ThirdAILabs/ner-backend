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
	"strings"
	"syscall"
	"time"

	"github.com/caarlos0/env/v11"
	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/go-chi/cors"
	ort "github.com/yalue/onnxruntime_go"
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
	err := migration_6.SetDefaultStorageProvider(string(storage.UploadType))
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

	slog.Info("Stating backend", "root", cfg.Root, "port", cfg.Port, "enterprise_mode", cfg.EnterpriseMode, "model_dir", cfg.ModelDir, "model_type", cfg.ModelType, "upload_bucket", cfg.UploadBucket, "app_data_dir", cfg.AppDataDir, "enable_python", cfg.EnablePython)
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

	objectStore, err := storage.NewLocalObjectStore(localBaseDir)
	if err != nil {
		log.Fatalf("Worker: Failed to create storage client: %v", err)
	}

	if cfg.ModelDir != "" {
		switch core.ParseModelType(cfg.ModelType) {
		case core.BoltUdt:
			if err := cmd.InitializeBoltUdtModel(context.Background(), db, objectStore, modelBucket, "basic", cfg.ModelDir); err != nil {
				log.Fatalf("Failed to init & upload bolt model: %v", err)
			}
		case core.PythonCnn:
			if err := cmd.InitializePythonCnnModel(context.Background(), db, objectStore, modelBucket, "basic", cfg.ModelDir); err != nil {
				log.Fatalf("Failed to init & upload python CNN model: %v", err)
			}
		case core.OnnxCnn:
			if err := cmd.InitializeOnnxCnnModel(context.Background(), db, objectStore, modelBucket, "basic", cfg.ModelDir); err != nil {
				log.Fatalf("failed to init ONNX model: %v", err)
			}
		default:
			log.Fatalf("Invalid model type: %s. Must be either 'bolt_udt' or 'python_cnn'", cfg.ModelType)
		}
	} else {
		cmd.InitializePresidioModel(db)
	}

	if err := cmd.RemoveExcludedTagsFromAllModels(db); err != nil {
		log.Fatalf("Failed to remove excluded tags from all models: %v", err)
	}

	queue := createQueue(db)

	licensing := cmd.CreateLicenseVerifier(db, cfg.License)

	worker := core.NewTaskProcessor(db, objectStore, queue, queue, licensing, filepath.Join(cfg.Root, "models"), modelBucket, cfg.UploadBucket, core.NewModelLoaders())

	var basicModel database.Model
	if err := db.Where("name = ?", "basic").First(&basicModel).Error; err != nil {
		log.Fatalf("could not lookup basic model: %v", err)
	}

	basicModelDir := filepath.Join(cfg.Root, "models", basicModel.Id.String())
	if err := objectStore.DownloadDir(context.Background(), filepath.Join(modelBucket, basicModel.Id.String()), basicModelDir, true); err != nil {
		log.Fatalf("failed to download model: %v", err)
	}
	server := createServer(db, objectStore, queue, cfg.Port, basicModelDir, core.ParseModelType(cfg.ModelType), cfg.UploadBucket, licensing, cfg.EnterpriseMode)

	// Read OpenAI API key from file and set as environment variable for worker
	if apiKeyData, err := os.ReadFile("api-key.txt"); err == nil {
		apiKey := strings.TrimSpace(string(apiKeyData))
		if apiKey != "" {
			os.Setenv("OPENAI_API_KEY", apiKey)
			slog.Info("loaded OpenAI API key for worker")
		}
	}

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
