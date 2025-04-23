package integrationtests

import (
	"context"
	"encoding/json"
	"fmt"
	backend "ner-backend/internal/api"
	"ner-backend/internal/core"
	"ner-backend/internal/core/types"
	"ner-backend/internal/database"
	"ner-backend/internal/messaging"
	"ner-backend/internal/s3"
	"ner-backend/pkg/api"
	"net/http"
	"os"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gorm.io/driver/sqlite"
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
					Label: label,
					Text:  text[match[0]:match[1]],
					Start: match[0],
					End:   match[1],
				})
			}
		}
	}
	return entities, nil
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

func createDB(t *testing.T) *gorm.DB {
	db, err := gorm.Open(sqlite.Open("file::memory:"), &gorm.Config{})
	require.NoError(t, err)

	require.NoError(t, database.GetMigrator(db).Migrate())

	return db
}

const (
	modelBucket = "models"
	dataBucket  = "test-data"
)

func createModel(t *testing.T, s3Client *s3.Client, db *gorm.DB) uuid.UUID {
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
	}

	require.NoError(t, db.Create(&model).Error)

	core.RegisterModelLoader("regex", loadRegexModel)

	return modelId
}

func createData(t *testing.T, s3Client *s3.Client) {
	require.NoError(t, s3Client.CreateBucket(context.Background(), dataBucket))

	for i := 0; i < 10; i++ {
		phonePath := fmt.Sprintf("phone-%d.txt", i)
		phoneData := fmt.Sprintf("this file contains a phone number %d%d%d-123-4567", i, i, i)

		_, err := s3Client.UploadObject(context.Background(), dataBucket, phonePath, strings.NewReader(phoneData))
		require.NoError(t, err)

		emailPath := fmt.Sprintf("email-%d.txt", i)
		emailData := fmt.Sprintf("this file contains a email address id-%d@email.com", i)

		_, err = s3Client.UploadObject(context.Background(), dataBucket, emailPath, strings.NewReader(emailData))
		require.NoError(t, err)
	}
}

func createReport(t *testing.T, router http.Handler, req api.CreateReportRequest) uuid.UUID {
	var res api.CreateReportResponse
	err := httpRequest(router, "POST", "/reports", req, &res)
	require.NoError(t, err)
	return res.ReportId
}

func reportIsComplete(report api.Report) bool {
	return report.ShardDataTaskStatus == database.JobCompleted &&
		report.InferenceTaskStatuses[database.JobQueued].TotalTasks == 0 &&
		report.InferenceTaskStatuses[database.JobRunning].TotalTasks == 0
}

func waitForReport(t *testing.T, router http.Handler, jobId uuid.UUID) api.Report {
	var report api.Report

	for i := 0; i < 20; i++ {
		time.Sleep(500 * time.Millisecond)
		err := httpRequest(router, "GET", fmt.Sprintf("/reports/%s", jobId), nil, &report)
		require.NoError(t, err)

		if reportIsComplete(report) {
			return report
		}
	}

	t.Fatal("timeout reached before report completed")
	return report
}

func getReportGroup(t *testing.T, router http.Handler, jobId, groupId uuid.UUID) api.Group {
	var res api.Group
	err := httpRequest(router, "GET", fmt.Sprintf("/reports/%s/groups/%s", jobId, groupId), nil, &res)
	require.NoError(t, err)
	return res
}

func getReportEntities(t *testing.T, router http.Handler, jobId uuid.UUID) []api.Entity {
	var res []api.Entity
	err := httpRequest(router, "GET", fmt.Sprintf("/reports/%s/entities", jobId), nil, &res)
	require.NoError(t, err)
	return res
}

func TestInferenceWorklow(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	minioUrl := setupMinioContainer(t, ctx)

	s3, err := s3.NewS3Client(&s3.Config{
		S3EndpointURL:     minioUrl,
		S3AccessKeyID:     minioUsername,
		S3SecretAccessKey: minioPassword,
	})
	require.NoError(t, err)

	db := createDB(t)

	queue := messaging.NewInMemoryQueue()

	backend := backend.NewBackendService(db, queue, 120)
	router := chi.NewRouter()
	backend.AddRoutes(router)

	worker := core.NewTaskProcessor(db, s3, queue, queue, t.TempDir(), modelBucket)

	go worker.Start()
	defer worker.Stop()

	modelId := createModel(t, s3, db)

	createData(t, s3)

	reportId := createReport(t, router, api.CreateReportRequest{
		ModelId:        modelId,
		SourceS3Bucket: dataBucket,

		Groups: map[string]string{
			"phone": `COUNT(phone) > 0`,
			"email": `COUNT(email) > 0`,
		},
	})

	report := waitForReport(t, router, reportId)

	assert.Equal(t, modelId, report.Model.Id)
	assert.Equal(t, dataBucket, report.SourceS3Bucket)
	assert.Equal(t, 10, report.InferenceTaskStatuses[database.JobCompleted].TotalTasks)
	assert.Equal(t, 2, len(report.Groups))

	entities := getReportEntities(t, router, reportId)
	assert.Equal(t, 20, len(entities))

	for _, group := range report.Groups {
		group := getReportGroup(t, router, reportId, group.Id)
		assert.Equal(t, fmt.Sprintf("COUNT(%s) > 0", group.Name), group.Query)
		assert.Equal(t, 10, len(group.Objects))
		for _, obj := range group.Objects {
			assert.Contains(t, obj, group.Name)
		}
	}
}
