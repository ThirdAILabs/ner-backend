package integrationtests

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"ner-backend/internal/core"
	"ner-backend/internal/core/types"
	"ner-backend/internal/database"
	"ner-backend/internal/s3"
	"ner-backend/pkg/api"
	"net/http"
	"net/http/httptest"
	"os"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/modules/minio"
	"github.com/testcontainers/testcontainers-go/modules/postgres"
	"github.com/testcontainers/testcontainers-go/wait"
	"gorm.io/gorm"
)

type regexModel struct {
	patterns map[string]regexp.Regexp
}

func (m *regexModel) Predict(text string) ([]types.Entity, error) {
	var entities []types.Entity
	for label, pattern := range m.patterns {
		matches := pattern.FindAllStringSubmatchIndex(text, -1)
		for _, match := range matches {
			if len(match) > 0 {
				entities = append(entities, types.Entity{
					Label:    label,
					Text:     text[match[0]:match[1]],
					Start:    match[0],
					End:      match[1],
					LContext: text[max(0, match[0]-20):match[0]],
					RContext: text[match[1]:min(len(text), match[1]+20)],
				})
			}
		}
	}
	return entities, nil
}

func (m *regexModel) Finetune(taskPrompt string, tags []api.TagInfo, samples []api.Sample) error {
	for _, tag := range tags {
		pattern, err := regexp.Compile(tag.Name)
		if err != nil {
			return fmt.Errorf("error compiling regex pattern: %w", err)
		}
		m.patterns[tag.Name] = *pattern
	}

	return nil
}

func (m *regexModel) Save(path string) error {
	data := make(map[string]string)
	for label, pattern := range m.patterns {
		data[label] = pattern.String()
	}
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("error saving model: %w", err)
	}
	defer file.Close()

	if err := json.NewEncoder(file).Encode(data); err != nil {
		return fmt.Errorf("error encoding model data: %w", err)
	}

	return nil
}

func (m *regexModel) Release() {}

func loadRegexModel(path string) (core.Model, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var patterns map[string]string

	if err := json.NewDecoder(file).Decode(&patterns); err != nil {
		return nil, err
	}

	compiledPatterns := make(map[string]regexp.Regexp)
	for label, pattern := range patterns {
		compiledPattern, err := regexp.Compile(pattern)
		if err != nil {
			return nil, err
		}
		compiledPatterns[label] = *compiledPattern
	}

	return &regexModel{patterns: compiledPatterns}, nil
}

const (
	modelBucket = "test-model-bucket"
)

func createModel(t *testing.T, s3Client *s3.Client, db *gorm.DB, modelBucket string) uuid.UUID {
	modelData := `{"phone": "\\d{3}-\\d{3}-\\d{4}", "email": "\\w+@email\\.com"}`

	require.NoError(t, s3Client.CreateBucket(context.Background(), modelBucket))

	modelId := uuid.New()
	_, err := s3Client.UploadObject(context.Background(), modelBucket, modelId.String()+"/model.bin", strings.NewReader(modelData))
	require.NoError(t, err)

	model := database.Model{
		Id:           modelId,
		Name:         "test-model",
		Type:         "regex",
		Status:       database.ModelTrained,
		CreationTime: time.Now().UTC(),
		Tags:         []database.ModelTag{{ModelId: modelId, Tag: "phone"}, {ModelId: modelId, Tag: "email"}},
	}

	require.NoError(t, db.Create(&model).Error)

	core.RegisterModelLoader("regex", loadRegexModel)

	return modelId
}

func createDB(t *testing.T) *gorm.DB {
	uri := setupPostgresContainer(t, context.Background())
	db, err := database.NewDatabase(uri)
	require.NoError(t, err)

	return db
}

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

const (
	minioUsername = "admin"
	minioPassword = "password"
)

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

type DummyLicenseVerifier struct{}

func (d *DummyLicenseVerifier) VerifyLicense(context.Context) error {
	return nil
}
