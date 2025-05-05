package main

import (
	"fmt"
	"log" // Adjust import path
	"log/slog"
	"ner-backend/cmd"
	"ner-backend/internal/core"
	"ner-backend/internal/database"
	"ner-backend/internal/licensing"
	"ner-backend/internal/messaging"
	"ner-backend/internal/s3"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/caarlos0/env/v11"
)

type WorkerConfig struct {
	DatabaseURL                 string `env:"DATABASE_URL,notEmpty,required"`
	RabbitMQURL                 string `env:"RABBITMQ_URL,notEmpty,required"`
	S3EndpointURL               string `env:"S3_ENDPOINT_URL,notEmpty,required"`
	S3AccessKeyID               string `env:"AWS_ACCESS_KEY_ID,notEmpty,required"`
	S3SecretAccessKey           string `env:"AWS_SECRET_ACCESS_KEY,notEmpty,required"`
	S3Region                    string `env:"AWS_REGION,notEmpty,required"`
	ModelBucketName             string `env:"MODEL_BUCKET_NAME" envDefault:"models"`
	QueueNames                  string `env:"QUEUE_NAMES" envDefault:"inference_queue,training_queue,shard_data_queue"`
	WorkerConcurrency           int    `env:"CONCURRENCY" envDefault:"1"`
	LicenseKey                  string `env:"LICENSE_KEY" envDefault:""`
	PythonExecutablePath        string `env:"PYTHON_EXECUTABLE_PATH" envDefault:"python"`
	PythonModelPluginScriptPath string `env:"PYTHON_MODEL_PLUGIN_SCRIPT_PATH" envDefault:"plugin/plugin-python/plugin.py"`
}

func main() {
	log.Println("Starting Worker Process...")

	cmd.LoadEnvFile()

	var cfg WorkerConfig
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

	var licenseVerifier licensing.LicenseVerifier
	if strings.HasPrefix(cfg.LicenseKey, "local:") {
		var err error
		licenseVerifier, err = licensing.NewFileLicenseVerifier([]byte(licensing.FileLicensePublicKey), strings.TrimSpace(strings.TrimPrefix(cfg.LicenseKey, "local:")))
		if err != nil {
			log.Fatalf("Failed to create file license verifier: %v", err)
		}
	} else if cfg.LicenseKey != "" {
		licenseVerifier, err = licensing.NewKeygenLicenseVerifier(cfg.LicenseKey)
		if err != nil {
			log.Fatalf("Failed to create license verifier: %v", err)
		}
	} else {
		slog.Warn(fmt.Sprintf("License key not provided. Using free license verifier with %.2fGB total file size limit", float64(licensing.DefaultFreeLicenseMaxBytes)/(1024*1024*1024)))
		licenseVerifier = licensing.NewFreeLicenseVerifier(db, licensing.DefaultFreeLicenseMaxBytes)
	}

	worker := core.NewTaskProcessor(db, s3Client, publisher, receiver, licenseVerifier, "./tmp_models_TODO", cfg.ModelBucketName)

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
