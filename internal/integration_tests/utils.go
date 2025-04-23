package integrationtests

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/modules/minio"
	"github.com/testcontainers/testcontainers-go/modules/postgres"
	"github.com/testcontainers/testcontainers-go/wait"
)

const (
	minioUsername = "admin"
	minioPassword = "password"
)

// func setupRabbitMQContainer(t *testing.T, ctx context.Context) string {
// 	// Start RabbitMQ container
// 	rabbitmqContainer, err := rabbitmq.RunContainer(ctx,
// 		testcontainers.WithImage("rabbitmq:3.11-management"),
// 	)
// 	require.NoError(t, err, "Failed to start RabbitMQ container")

// 	t.Cleanup(func() {
// 		err := rabbitmqContainer.Terminate(context.Background())
// 		require.NoError(t, err, "Failed to terminate RabbitMQ container")
// 	})

// 	// Get connection string for the test container
// 	connStr, err := rabbitmqContainer.AmqpURL(ctx)
// 	require.NoError(t, err, "Failed to get RabbitMQ AMQP URL")

// 	return connStr
// }

func setupMinioContainer(t *testing.T, ctx context.Context) string {
	minioContainer, err := minio.Run(
		ctx,
		"minio/minio:RELEASE.2024-01-16T16-07-38Z",
		minio.WithUsername(minioUsername),
		minio.WithPassword(minioPassword),
	)
	require.NoError(t, err, "Failed to start MinIO container")

	t.Cleanup(func() {
		err := minioContainer.Terminate(context.Background())
		require.NoError(t, err, "Failed to terminate MinIO container")
	})

	connStr, err := minioContainer.ConnectionString(ctx)
	require.NoError(t, err, "Failed to get MinIO connection string")

	return "http://" + connStr
}

func setupPostgresContainer(t *testing.T, ctx context.Context) string {
	dbName, dbUser, dbPassword := "test_db", "test_user", "test_password"

	postgresContainer, err := postgres.Run(ctx,
		"postgres:16-alpine",
		postgres.WithDatabase(dbName),
		postgres.WithUsername(dbUser),
		postgres.WithPassword(dbPassword),
		testcontainers.WithWaitStrategy(
			wait.ForLog("database system is ready to accept connections").
				WithOccurrence(2).
				WithStartupTimeout(5*time.Second)),
	)
	require.NoError(t, err, "Failed to start PostgreSQL container")

	t.Cleanup(func() {
		err := postgresContainer.Terminate(context.Background())
		require.NoError(t, err, "Failed to terminate PostgreSQL container")
	})

	connStr, err := postgresContainer.ConnectionString(ctx)
	require.NoError(t, err, "Failed to get PostgreSQL connection string")

	return connStr
}

func httpRequest(api http.Handler, method, endpoint string, payload any, dest any) error {
	var body io.Reader
	if payload != nil {
		requestBody, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		body = bytes.NewReader(requestBody)
	}

	req := httptest.NewRequest(method, endpoint, body)
	req.Header.Set("Content-Type", "application/json")

	rr := httptest.NewRecorder()
	api.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		return fmt.Errorf("expected status code 200, got %d: %v", rr.Code, rr.Body.String())
	}

	if dest != nil {
		if err := json.Unmarshal(rr.Body.Bytes(), dest); err != nil {
			return fmt.Errorf("failed to unmarshal response: %w", err)
		}
	}

	return nil
}
