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
	"ner-backend/internal/licensing"
	"ner-backend/internal/messaging"
	"ner-backend/internal/storage"
	"ner-backend/pkg/api"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/modules/minio"
	"github.com/testcontainers/testcontainers-go/modules/postgres"
	"github.com/testcontainers/testcontainers-go/modules/rabbitmq"
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
					LContext: strings.ToValidUTF8(text[max(0, match[0]-20):match[0]], ""),
					RContext: strings.ToValidUTF8(text[match[1]:min(len(text), match[1]+20)], ""),
				})
			}
		}
	}
	return entities, nil
}

func (m *regexModel) FinetuneAndSave(taskPrompt string, tags []types.TagInfo, samples []api.Sample, savePath string) error {
	for _, tag := range tags {
		pattern, err := regexp.Compile(tag.Name)
		if err != nil {
			return fmt.Errorf("error compiling regex pattern: %w", err)
		}
		m.patterns[tag.Name] = *pattern
	}

	if err := m.Save(savePath); err != nil {
		return fmt.Errorf("error saving model: %w", err)
	}

	return nil
}

func (m *regexModel) Save(modelDir string) error {
	data := make(map[string]string)
	for label, pattern := range m.patterns {
		data[label] = pattern.String()
	}
	file, err := os.Create(filepath.Join(modelDir, "model.json"))
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

func loadRegexModel(modelDir string) (core.Model, error) {
	file, err := os.Open(filepath.Join(modelDir, "model.json"))
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

func createModel(t *testing.T, storage storage.ObjectStore, db *gorm.DB, modelBucket string) (string, core.ModelLoader, uuid.UUID) {
	modelData := `{"phone": "\\d{3}-\\d{3}-\\d{4}", "email": "\\w+@email\\.com"}`

	modelId := uuid.New()
	err := storage.PutObject(context.Background(), filepath.Join(modelBucket, modelId.String(), "model.json"), strings.NewReader(modelData))
	require.NoError(t, err)

	model := database.Model{
		Id:           modelId,
		Name:         "test-model",
		Type:         "regex",
		Status:       database.ModelTrained,
		CreationTime: time.Now().UTC(),
		Tags:         []database.ModelTag{{ModelId: modelId, Tag: "phone"}, {ModelId: modelId, Tag: "email"}, {ModelId: modelId, Tag: "xyz"}},
	}

	require.NoError(t, db.Create(&model).Error)

	return "regex", loadRegexModel, modelId
}

func createDB(t *testing.T) *gorm.DB {
	uri := setupPostgresContainer(t, context.Background())
	db, err := database.NewDatabase(uri)
	require.NoError(t, err)

	return db
}

func setupRabbitMQContainer(t *testing.T, ctx context.Context) (messaging.Publisher, messaging.Reciever) {
	rabbitmqContainer, err := rabbitmq.Run(ctx, "rabbitmq:3.11-management-alpine")

	require.NoError(t, err, "Failed to start RabbitMQ container")

	t.Cleanup(func() {
		err := rabbitmqContainer.Terminate(context.Background())
		require.NoError(t, err, "Failed to terminate RabbitMQ container")
	})

	connStr, err := rabbitmqContainer.AmqpURL(ctx)
	require.NoError(t, err, "Failed to get RabbitMQ AMQP URL")

	publisher, err := messaging.NewRabbitMQPublisher(connStr)
	require.NoError(t, err, "Failed to create RabbitMQ publisher")

	t.Cleanup(func() {
		publisher.Close()
	})

	reciever, err := messaging.NewRabbitMQReceiver(connStr)
	require.NoError(t, err, "Failed to create RabbitMQ reciever")

	t.Cleanup(func() {
		reciever.Close()
	})

	return publisher, reciever
}

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

func (d *DummyLicenseVerifier) VerifyLicense(context.Context) (licensing.LicenseInfo, error) {
	return licensing.LicenseInfo{}, nil
}
