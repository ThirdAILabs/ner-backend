package database

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

func NewConnectionPool(databaseURL string) (*pgxpool.Pool, error) {
	log.Println("Connecting to database...")
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		return nil, fmt.Errorf("unable to parse database URL: %w", err)
	}

	config.MaxConns = 10                 // Example: max 10 connections
	config.MinConns = 2                  // Example: keep 2 idle connections
	config.MaxConnIdleTime = time.Minute // Example: close connections idle for > 1 min

	pool, err := pgxpool.NewWithConfig(context.Background(), config)
	if err != nil {
		return nil, fmt.Errorf("unable to create connection pool: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	err = pool.Ping(ctx)
	if err != nil {
		pool.Close() // Close pool if ping fails
		return nil, fmt.Errorf("unable to ping database: %w", err)
	}

	log.Println("Database connection pool established.")
	return pool, nil
}

// InitializeSchema creates database tables if they don't exist.
func InitializeSchema(pool *pgxpool.Pool) error {
	schemaSQL := `
    CREATE TABLE IF NOT EXISTS models (
        model_id VARCHAR(255) PRIMARY KEY,
        status VARCHAR(50),
        source_data_path TEXT,
        model_artifact_path TEXT,
        creation_time TIMESTAMPTZ DEFAULT NOW(),
        completion_time TIMESTAMPTZ
    );

    CREATE INDEX IF NOT EXISTS idx_models_status ON models (status);

    CREATE TABLE IF NOT EXISTS inference_jobs (
        job_id VARCHAR(255) PRIMARY KEY,
        model_id VARCHAR(255) REFERENCES models(model_id),
        source_s3_bucket TEXT,
        source_s3_prefix TEXT,
        dest_s3_bucket TEXT,
        result_s3_prefix TEXT,
        status VARCHAR(50),
        creation_time TIMESTAMPTZ DEFAULT NOW(),
        completion_time TIMESTAMPTZ
    );

    CREATE INDEX IF NOT EXISTS idx_inference_jobs_model_id ON inference_jobs (model_id);
    CREATE INDEX IF NOT EXISTS idx_inference_jobs_status ON inference_jobs (status);
    `
	log.Println("Initializing database schema...")
	_, err := pool.Exec(context.Background(), schemaSQL)
	if err != nil {
		return fmt.Errorf("failed to initialize schema: %w", err)
	}
	log.Println("Database schema initialized/verified.")
	return nil
}
