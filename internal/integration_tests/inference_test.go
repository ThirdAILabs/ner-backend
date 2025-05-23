package integrationtests

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"mime/multipart"
	"ner-backend/cmd"
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

	"gorm.io/gorm"

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

func waitForReport(t *testing.T, router http.Handler, jobId uuid.UUID, timeoutSeconds int) api.Report {
	for i := 0; i < timeoutSeconds; i++ {
		var report api.Report
		time.Sleep(1 * time.Second)
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

	modelName, modelLoader, modelId := createModel(t, s3, db, modelBucket)

	worker := core.NewTaskProcessor(db, s3, publisher, reciever, &DummyLicenseVerifier{}, t.TempDir(), modelBucket, map[string]core.ModelLoader{
		modelName: modelLoader,
	})

	go worker.Start()
	defer worker.Stop()

	createData(t, s3)

	reportId := createReport(t, router, api.CreateReportRequest{
		ReportName:     "test-report",
		ModelId:        modelId,
		S3Endpoint:     minioUrl,
		SourceS3Bucket: dataBucket,
		Tags:           []string{"phone", "email"},
		CustomTags:     map[string]string{"custom_token": `(\w\d){3}`},
		Groups: map[string]string{
			"phone": `COUNT(phone) > 0`,
			"email": `COUNT(email) > 0`,
		},
	})

	report := waitForReport(t, router, reportId, 10)

	assert.Equal(t, modelId, report.Model.Id)
	assert.Equal(t, dataBucket, report.SourceS3Bucket)
	assert.Equal(t, 11, report.InferenceTaskStatuses[database.JobCompleted].TotalTasks)
	assert.Equal(t, 2, len(report.Groups))
	assert.Greater(t, report.TotalInferenceTimeSeconds, 0.0)
	assert.Greater(t, report.ShardDataTimeSeconds, 0.0)

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

	modelName, modelLoader, modelId := createModel(t, s3, db, modelBucket)

	worker := core.NewTaskProcessor(db, s3, publisher, reciever, &DummyLicenseVerifier{}, t.TempDir(), modelBucket, map[string]core.ModelLoader{
		modelName: modelLoader,
	})

	go worker.Start()
	defer worker.Stop()

	uploadId := createUpload(t, router)

	reportId := createReport(t, router, api.CreateReportRequest{
		ReportName: "test-report",
		ModelId:    modelId,
		UploadId:   uploadId,
		Tags:       []string{"phone", "email"},
	})

	report := waitForReport(t, router, reportId, 10)

	assert.Equal(t, modelId, report.Model.Id)
	assert.Equal(t, "uploads", report.SourceS3Bucket)
	assert.Equal(t, uploadId.String(), report.SourceS3Prefix)

	entities := getReportEntities(t, router, reportId)
	assert.Equal(t, 2, len(entities))
}

func TestInferenceWorkflowForModels(t *testing.T) {
	if os.Getenv("PYTHON_EXECUTABLE_PATH") == "" || os.Getenv("PYTHON_MODEL_PLUGIN_SCRIPT_PATH") == "" {
		t.Fatalf("PYTHON_EXECUTABLE_PATH and PYTHON_MODEL_PLUGIN_SCRIPT_PATH must be set")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	minioURL := setupMinioContainer(t, ctx)

	s3, err := storage.NewS3Provider(storage.S3ProviderConfig{
		S3EndpointURL:     minioURL,
		S3AccessKeyID:     minioUsername,
		S3SecretAccessKey: minioPassword,
	})
	require.NoError(t, err)
	require.NoError(t, s3.CreateBucket(ctx, modelBucket))

	db := createDB(t)

	publisher, receiver := setupRabbitMQContainer(t, ctx)

	backendSvc := backend.NewBackendService(db, s3, publisher, 120)
	router := chi.NewRouter()
	backendSvc.AddRoutes(router)

	tempDir := t.TempDir()
	worker := core.NewTaskProcessor(db, s3, publisher, receiver, &DummyLicenseVerifier{}, tempDir, modelBucket, core.NewModelLoaders(os.Getenv("PYTHON_EXECUTABLE_PATH"), os.Getenv("PYTHON_MODEL_PLUGIN_SCRIPT_PATH")))
	go worker.Start()
	t.Cleanup(worker.Stop)

	models := []struct {
		label      string
		tag        string
		initFn     func(context.Context, *gorm.DB, *storage.S3Provider, string) error
		expectedDB string
	}{
		{
			label: "CNN Model",
			tag:   "cnn",
			initFn: func(c context.Context, db *gorm.DB, s3 *storage.S3Provider, bucket string) error {
				return cmd.InitializeCnnNerExtractor(c, db, s3, bucket, "advanced", os.Getenv("HOST_MODEL_DIR"))
			},
			expectedDB: "advanced",
		},
		{
			label: "Transformer Model",
			tag:   "transformer",
			initFn: func(c context.Context, db *gorm.DB, s3 *storage.S3Provider, bucket string) error {
				return cmd.InitializeTransformerModel(c, db, s3, bucket, "ultra", os.Getenv("HOST_MODEL_DIR"))
			},
			expectedDB: "ultra",
		},
	}

	for _, m := range models {
		m := m // capture range variable
		t.Run(m.label, func(t *testing.T) {
			require.NoError(t, m.initFn(ctx, db, s3, modelBucket))

			var model database.Model
			require.NoError(t, db.Where("name = ?", m.expectedDB).First(&model).Error)

			uploadID := createUpload(t, router)
			reportID := createReport(t, router, api.CreateReportRequest{
				ReportName: fmt.Sprintf("test-report-%s", m.tag),
				ModelId:    model.Id,
				UploadId:   uploadID,
				Tags: []string{"ADDRESS", "CARD_NUMBER", "COMPANY", "CREDIT_SCORE", "DATE",
					"EMAIL", "ETHNICITY", "GENDER", "ID_NUMBER", "LICENSE_PLATE",
					"LOCATION", "NAME", "PHONENUMBER", "SERVICE_CODE", "SEXUAL_ORIENTATION",
					"SSN", "URL", "VIN", "O"},
			})

			report := waitForReport(t, router, reportID, 180)

			assert.Equal(t, model.Id, report.Model.Id)
			assert.Equal(t, "uploads", report.SourceS3Bucket)
			assert.Equal(t, uploadID.String(), report.SourceS3Prefix)

			entities := getReportEntities(t, router, reportID)
			assert.Greater(t, len(entities), 0)
		})
	}
}
