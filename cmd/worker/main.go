package main

import (
	"log" // Adjust import path
	"ner-backend/cmd"
	"ner-backend/internal/core"
	"ner-backend/internal/core/python"
	"ner-backend/internal/database"
	"ner-backend/internal/messaging"
	"ner-backend/internal/storage"
	"os"
	"os/signal"
	"syscall"

	"github.com/caarlos0/env/v11"
	ort "github.com/yalue/onnxruntime_go"
)

type WorkerConfig struct {
	DatabaseURL                 string `env:"DATABASE_URL,notEmpty,required"`
	RabbitMQURL                 string `env:"RABBITMQ_URL,notEmpty,required"`
	S3EndpointURL               string `env:"S3_ENDPOINT_URL"`
	S3Region                    string `env:"S3_REGION"`
	S3AccessKeyID               string `env:"INTERNAL_AWS_ACCESS_KEY_ID"`
	S3SecretAccessKey           string `env:"INTERNAL_AWS_SECRET_ACCESS_KEY"`
	BucketName                  string `env:"BUCKET_NAME,notEmpty,required"`
	LicenseKey                  string `env:"LICENSE_KEY" envDefault:""`
	PythonExecutablePath        string `env:"PYTHON_EXECUTABLE_PATH" envDefault:"python"`
	PythonModelPluginScriptPath string `env:"PYTHON_MODEL_PLUGIN_SCRIPT_PATH" envDefault:"plugin/plugin-python/plugin.py"`
	OnnxRuntimeDylib            string `env:"ONNX_RUNTIME_DYLIB"`
}

func main() {
	log.Println("Starting Worker Process...")

	cmd.LoadEnvFile()

	var cfg WorkerConfig
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

	s3ObjectStoreCfg := storage.S3ClientConfig{
		Endpoint:        cfg.S3EndpointURL,
		Region:          cfg.S3Region,
		AccessKeyID:     cfg.S3AccessKeyID,
		SecretAccessKey: cfg.S3SecretAccessKey,
	}
	s3ObjectStore, err := storage.NewS3ObjectStore(cfg.BucketName, s3ObjectStoreCfg)
	if err != nil {
		log.Fatalf("Worker: Failed to create S3 client: %v", err)
	}

	publisher, err := messaging.NewRabbitMQPublisher(cfg.RabbitMQURL)
	if err != nil {
		log.Fatalf("Failed to connect to RabbitMQ: %v", err)
	}
	defer publisher.Close()

	receiver, err := messaging.NewRabbitMQReceiver(cfg.RabbitMQURL)
	if err != nil {
		log.Fatalf("Failed to connect to RabbitMQ: %v", err)
	}

	licensing := cmd.CreateLicenseVerifier(db, cfg.LicenseKey)

	python.EnablePythonPlugin(cfg.PythonExecutablePath, cfg.PythonModelPluginScriptPath)

	loaders := core.NewModelLoaders()

	worker := core.NewTaskProcessor(db, s3ObjectStore, publisher, receiver, licensing, "./tmp_models_TODO", cmd.ModelBucketName, cmd.UploadBucketName, loaders)

	go worker.Start()

	log.Println("Worker started. Waiting for tasks. Press Ctrl+C to exit.")

	// Wait for termination signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutdown signal received, waiting for workers to finish...")

	worker.Stop()

	// TODO: Implement graceful shutdown for workers if needed
	// e.g., signal workers to stop consuming new messages and wait for current ones
	// For now, just wait for WaitGroup (which might not finish if workers loop forever)
	// wg.Wait() // This might block indefinitely if workers don't exit cleanly

	log.Println("Worker process stopped.")
}
