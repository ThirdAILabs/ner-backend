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
