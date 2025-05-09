package integrationtests

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"mime/multipart"
	backend "ner-backend/internal/api"
	"ner-backend/internal/core"
	"ner-backend/internal/database"
	"ner-backend/internal/storage"
	"ner-backend/pkg/api"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	dataBucket = "test-data"
)

func createData(t *testing.T, storage storage.Provider) {
	require.NoError(t, storage.CreateBucket(context.Background(), dataBucket))

	for i := 0; i < 10; i++ {
		phonePath := fmt.Sprintf("phone-%d.txt", i)
		phoneData := fmt.Sprintf("this file contains a phone number %d%d%d-123-4567", i, i, i)

		err := storage.PutObject(context.Background(), dataBucket, phonePath, strings.NewReader(phoneData))
		require.NoError(t, err)

		emailPath := fmt.Sprintf("email-%d.txt", i)
		emailData := fmt.Sprintf("this file contains a email address id-%d@email.com", i)

		err = storage.PutObject(context.Background(), dataBucket, emailPath, strings.NewReader(emailData))
		require.NoError(t, err)
	}

	err := storage.PutObject(context.Background(), dataBucket, "custom-token.txt", strings.NewReader("this is a custom token a1b2c3"))
	require.NoError(t, err)
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
	for i := 0; i < 20; i++ {
		var report api.Report
		time.Sleep(500 * time.Millisecond)
		err := httpRequest(router, "GET", fmt.Sprintf("/reports/%s", jobId), nil, &report)
		require.NoError(t, err)

		if reportIsComplete(report) {
			return report
		}
	}

	t.Fatal("timeout reached before report completed")
	return api.Report{}
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

func TestInferenceWorkflowOnBucket(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	minioUrl := setupMinioContainer(t, ctx)

	s3, err := storage.NewS3Provider(storage.S3ProviderConfig{
		S3EndpointURL:     minioUrl,
		S3AccessKeyID:     minioUsername,
		S3SecretAccessKey: minioPassword,
	})
	require.NoError(t, err)

	os.Setenv("AWS_ACCESS_KEY_ID", minioUsername)
	os.Setenv("AWS_SECRET_ACCESS_KEY", minioPassword)

	db := createDB(t)

	publisher, reciever := setupRabbitMQContainer(t, ctx)

	backend := backend.NewBackendService(db, s3, publisher, 120)
	router := chi.NewRouter()
	backend.AddRoutes(router)

	worker := core.NewTaskProcessor(db, s3, publisher, reciever, &DummyLicenseVerifier{}, t.TempDir(), modelBucket)

	go worker.Start()
	defer worker.Stop()

	modelId := createModel(t, s3, db, modelBucket)

	createData(t, s3)

	reportId := createReport(t, router, api.CreateReportRequest{
		ReportName:     "test-report",
		ModelId:        modelId,
		S3Endpoint:     minioUrl,
		SourceS3Bucket: dataBucket,
		Tags:           []string{"phone", "email"},
		CustomTags:     map[string]string{"custom-token": `(\w\d){3}`},
		Groups: map[string]string{
			"phone": `COUNT(phone) > 0`,
			"email": `COUNT(email) > 0`,
		},
	})

	report := waitForReport(t, router, reportId)

	assert.Equal(t, modelId, report.Model.Id)
	assert.Equal(t, dataBucket, report.SourceS3Bucket)
	assert.Equal(t, 11, report.InferenceTaskStatuses[database.JobCompleted].TotalTasks)
	assert.Equal(t, 2, len(report.Groups))

	entities := getReportEntities(t, router, reportId)
	assert.Equal(t, 21, len(entities))

	for _, group := range report.Groups {
		group := getReportGroup(t, router, reportId, group.Id)
		assert.Equal(t, fmt.Sprintf("COUNT(%s) > 0", group.Name), group.Query)
		assert.Equal(t, 10, len(group.Objects))
		for _, obj := range group.Objects {
			assert.Contains(t, obj, group.Name)
		}
	}
}

func createUpload(t *testing.T, router http.Handler) uuid.UUID {
	buf := new(bytes.Buffer)
	writer := multipart.NewWriter(buf)

	f1, err := writer.CreateFormFile("files", "file1.txt")
	require.NoError(t, err)
	_, err = f1.Write([]byte("this is a test file with a phone number 123-456-7890"))
	require.NoError(t, err)

	f2, err := writer.CreateFormFile("files", "file2.txt")
	require.NoError(t, err)
	_, err = f2.Write([]byte("this is a test file with an email address abc@email.com"))
	require.NoError(t, err)

	require.NoError(t, writer.Close())

	req := httptest.NewRequest(http.MethodPost, "/uploads", buf)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	rr := httptest.NewRecorder()
	router.ServeHTTP(rr, req)

	require.Equal(t, http.StatusOK, rr.Code)

	var res api.UploadResponse
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &res))

	return res.Id
}

func TestInferenceWorkflowOnUpload(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	minioUrl := setupMinioContainer(t, ctx)

	s3, err := storage.NewS3Provider(storage.S3ProviderConfig{
		S3EndpointURL:     minioUrl,
		S3AccessKeyID:     minioUsername,
		S3SecretAccessKey: minioPassword,
	})
	require.NoError(t, err)

	db := createDB(t)

	publisher, reciever := setupRabbitMQContainer(t, ctx)

	backend := backend.NewBackendService(db, s3, publisher, 120)
	router := chi.NewRouter()
	backend.AddRoutes(router)

	worker := core.NewTaskProcessor(db, s3, publisher, reciever, &DummyLicenseVerifier{}, t.TempDir(), modelBucket)

	go worker.Start()
	defer worker.Stop()

	modelId := createModel(t, s3, db, modelBucket)

	uploadId := createUpload(t, router)

	reportId := createReport(t, router, api.CreateReportRequest{
		ReportName: "test-report",
		ModelId:    modelId,
		UploadId:   uploadId,
		Tags:       []string{"phone", "email"},
	})

	report := waitForReport(t, router, reportId)

	assert.Equal(t, modelId, report.Model.Id)
	assert.Equal(t, "uploads", report.SourceS3Bucket)
	assert.Equal(t, uploadId.String(), report.SourceS3Prefix)

	entities := getReportEntities(t, router, reportId)
	assert.Equal(t, 2, len(entities))
}

func TestInferenceWorkflowForCNN(t *testing.T) {
	os.Setenv("HOST_MODEL_DIR", "/share/ner/model")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	minioUrl := setupMinioContainer(t, ctx)

	s3, err := storage.NewS3Provider(storage.S3ProviderConfig{
		S3EndpointURL:     minioUrl,
		S3AccessKeyID:     minioUsername,
		S3SecretAccessKey: minioPassword,
	})
	require.NoError(t, err)

	db := createDB(t)

	publisher, reciever := setupRabbitMQContainer(t, ctx)

	backend := backend.NewBackendService(db, s3, publisher, 120)
	router := chi.NewRouter()
	backend.AddRoutes(router)

	worker := core.NewTaskProcessor(db, s3, publisher, reciever, &DummyLicenseVerifier{}, t.TempDir(), modelBucket)

	go worker.Start()
	defer worker.Stop()

	var model database.Model

	db_err := db.Where("name = ?", "advance").First(&model).Error
	require.NoError(t, db_err)

	uploadId := createUpload(t, router)

	reportId := createReport(t, router, api.CreateReportRequest{
		ReportName: "test-report-cnn",
		ModelId:    model.Id,
		UploadId:   uploadId,
		Tags: []string{"ADDRESS", "CARD_NUMBER", "COMPANY", "CREDIT_SCORE", "DATE",
			"EMAIL", "ETHNICITY", "GENDER", "ID_NUMBER", "LICENSE_PLATE",
			"LOCATION", "NAME", "PHONENUMBER", "SERVICE_CODE",
			"SEXUAL_ORIENTATION", "SSN", "URL", "VIN", "O"},
	})

	report := waitForReport(t, router, reportId)

	assert.Equal(t, model.Id, report.Model.Id)
	assert.Equal(t, "uploads", report.SourceS3Bucket)
	assert.Equal(t, uploadId.String(), report.SourceS3Prefix)

	entities := getReportEntities(t, router, reportId)
	fmt.Println(entities)
}
