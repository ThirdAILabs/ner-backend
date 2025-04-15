package config

import (
	"log"
	"os"
	"strconv"

	"github.com/joho/godotenv"
)

type Config struct {
	DatabaseURL       string
	RabbitMQURL       string
	S3EndpointURL     string
	S3AccessKeyID     string
	S3SecretAccessKey string
	S3Region          string
	S3UseSSL          bool
	ModelBucketName   string
	QueueNames        string // Comma-separated
	WorkerConcurrency int
	APIPort           string
}

func LoadConfig() (*Config, error) {
	// Load .env file if it exists (useful for local development)
	err := godotenv.Load()
	if err != nil {
		log.Println("No .env file found or error loading, continuing with environment variables")
	}

	concurrencyStr := getEnv("CONCURRENCY", "1")
	concurrency, err := strconv.Atoi(concurrencyStr)
	if err != nil {
		log.Printf("Invalid CONCURRENCY value '%s', using default 1", concurrencyStr)
		concurrency = 4
	}

	useSSLStr := getEnv("S3_USE_SSL", "false")
	useSSL := useSSLStr == "true" || useSSLStr == "1"

	cfg := &Config{
		DatabaseURL:       getEnv("DATABASE_URL", "postgresql://user:password@localhost:5432/ner_backend?sslmode=disable"),
		RabbitMQURL:       getEnv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/"),
		S3EndpointURL:     getEnv("S3_ENDPOINT_URL", ""),
		S3AccessKeyID:     getEnv("AWS_ACCESS_KEY_ID", ""),
		S3SecretAccessKey: getEnv("AWS_SECRET_ACCESS_KEY", ""),
		S3Region:          getEnv("AWS_REGION", "us-east-1"),
		S3UseSSL:          useSSL,
		ModelBucketName:   getEnv("MODEL_BUCKET_NAME", "trained-models"),
		QueueNames:        getEnv("QUEUE_NAMES", "inference_queue,training_queue"),
		WorkerConcurrency: concurrency,
		APIPort:           getEnv("API_PORT", "8001"),
	}

	if cfg.S3EndpointURL != "" && (cfg.S3AccessKeyID == "" || cfg.S3SecretAccessKey == "") {
		log.Println("Warning: S3_ENDPOINT_URL is set, but AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY are missing.")
	}

	return cfg, nil
}

func getEnv(key, fallback string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return fallback
}
