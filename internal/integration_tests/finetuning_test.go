package integrationtests

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"testing"
	"time"

	"ner-backend/cmd"
	backendpkg "ner-backend/internal/api"
	"ner-backend/internal/core"
	"ner-backend/internal/core/python"
	"ner-backend/internal/database"
	"ner-backend/internal/messaging"
	"ner-backend/internal/storage"
	"ner-backend/pkg/api"

	"github.com/go-chi/chi/v5"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	ort "github.com/yalue/onnxruntime_go"
	"gorm.io/gorm"
)

func setupCommon(t *testing.T) (
	ctx context.Context,
	cancel func(),
	s3 *storage.S3Provider,
	db *gorm.DB,
	pub messaging.Publisher,
	sub messaging.Reciever,
	router *chi.Mux,
) {
	ctx, cancel = context.WithTimeout(context.Background(), 5*time.Minute)

	minioURL := setupMinioContainer(t, ctx)
	s3Prov, err := storage.NewS3Provider(storage.S3ProviderConfig{
		S3EndpointURL:     minioURL,
		S3AccessKeyID:     minioUsername,
		S3SecretAccessKey: minioPassword,
	})
	require.NoError(t, err)

	db = createDB(t)
	pub, sub = setupRabbitMQContainer(t, ctx)

	backendSvc := backendpkg.NewBackendService(db, s3Prov, pub, 120, &DummyLicenseVerifier{})
	r := chi.NewRouter()
	backendSvc.AddRoutes(r)

	return ctx, cancel, s3Prov, db, pub, sub, r
}

func startWorker(
	t *testing.T,
	db *gorm.DB,
	s3 *storage.S3Provider,
	pub messaging.Publisher,
	sub messaging.Reciever,
	bucket string,
	loaders map[core.ModelType]core.ModelLoader,
) (stop func()) {
	worker := core.NewTaskProcessor(db, s3, pub, sub, &DummyLicenseVerifier{}, t.TempDir(), bucket, loaders)
	go worker.Start()
	return worker.Stop
}

func finetune(
	t *testing.T,
	router *chi.Mux,
	baseModelID string,
	req api.FinetuneRequest,
	attempts int,
	delay time.Duration,
) (resp api.FinetuneResponse, m api.Model) {
	require.NoError(t, httpRequest(router, "POST", fmt.Sprintf("/models/%s/finetune", baseModelID), req, &resp))

	for i := 0; i < attempts; i++ {
		time.Sleep(delay)
		require.NoError(t, httpRequest(router, "GET", fmt.Sprintf("/models/%s", resp.ModelId), nil, &m))
		if m.Status == database.ModelTrained || m.Status == database.ModelFailed {
			break
		}
	}
	return
}

func TestFinetuning(t *testing.T) {
	_, cancel, s3, db, pub, sub, router := setupCommon(t)
	defer cancel()

	baseName, baseLoader, baseID := createModel(t, s3, db, modelBucket)

	stop := startWorker(t, db, s3, pub, sub, modelBucket, map[core.ModelType]core.ModelLoader{
		core.ParseModelType(baseName): baseLoader,
	})
	defer stop()

	sample := api.FeedbackRequest{
		Tokens: []string{"foo"},
		Labels: []string{"xyz"},
	}
	require.NoError(t, httpRequest(
		router,
		"POST",
		fmt.Sprintf("/models/%s/feedback", baseID.String()),
		sample,
		nil,
	))

	tp := "Finetuning test"

	// No in-body samples; just do a normal finetune request
	ftReq := api.FinetuneRequest{
		Name:       "finetuned-model",
		TaskPrompt: &tp,
		Tags:       []api.TagInfo{{Name: "URL", Description: "uniform resource locators (URLs), which are references to web resources or pages on the internet", Examples: []string{"https://example.com", "http://thirdai.com"}}},
	}
	_, model := finetune(t, router, baseID.String(), ftReq, 10, 100*time.Millisecond)

	assert.Equal(t, database.ModelTrained, model.Status)
	assert.Equal(t, "finetuned-model", model.Name)
	require.NotNil(t, model.BaseModelId)
	assert.Equal(t, baseID, *model.BaseModelId)

	stream, err := s3.GetObjectStream(modelBucket, fmt.Sprintf("%s/model.json", model.Id))
	require.NoError(t, err)
	var data map[string]string
	require.NoError(t, json.NewDecoder(stream).Decode(&data))
	assert.Contains(t, data, "URL")
}

func finetuningTestHelper(t *testing.T, modelInit func(ctx context.Context, db *gorm.DB, s3p storage.Provider, bucket, name, hostModelDir string) error) {
	var (
		modelName    = "test-model"
		pythonExec   = os.Getenv("PYTHON_EXECUTABLE_PATH")
		pluginScript = os.Getenv("PYTHON_MODEL_PLUGIN_SCRIPT_PATH")
		hostModelDir = os.Getenv("HOST_MODEL_DIR")
	)
	if pythonExec == "" || pluginScript == "" || hostModelDir == "" {
		t.Fatalf("PYTHON_EXECUTABLE_PATH, PYTHON_MODEL_PLUGIN_SCRIPT_PATH, and HOST_MODEL_DIR env vars must be set")
	}

	ctx, cancel, s3, db, pub, sub, router := setupCommon(t)
	defer cancel()

	require.NoError(t, s3.CreateBucket(context.Background(), modelBucket))

	os.Setenv("AWS_ACCESS_KEY_ID", minioUsername)
	os.Setenv("AWS_SECRET_ACCESS_KEY", minioPassword)

	python.EnablePythonPlugin(pythonExec, pluginScript)

	stop := startWorker(t, db, s3, pub, sub, modelBucket, core.NewModelLoaders())
	defer stop()

	require.NoError(t, modelInit(ctx, db, s3, modelBucket, modelName, hostModelDir))

	var base database.Model
	require.NoError(t, db.Where("name = ?", modelName).First(&base).Error)

	samples := []api.Sample{
		{
			Tokens: []string{"I", "started", "working", "at", "ThirdAI", "in", "2022"},
			Labels: []string{"O", "O", "O", "O", "COMPANY", "O", "DATE"},
		},
		{
			Tokens: []string{"I", "started", "working", "at", "ThirdAI", "in", "2022"},
			Labels: []string{"O", "O", "O", "O", "COMPANY", "O", "DATE"},
		},
	}
	for _, sample := range samples {
		feedbackReq := api.FeedbackRequest(sample)
		require.NoError(t, httpRequest(
			router,
			"POST",
			fmt.Sprintf("/models/%s/feedback", base.Id.String()),
			feedbackReq,
			nil,
		))
	}

	var saved []api.FeedbackRequest
	require.NoError(t, httpRequest(
		router,
		"GET",
		fmt.Sprintf("/models/%s/feedback", base.Id.String()),
		nil,
		&saved,
	))
	require.Len(t, saved, 2)
	assert.Equal(t, samples[0].Tokens, saved[0].Tokens)
	assert.Equal(t, samples[0].Labels, saved[0].Labels)
	assert.Equal(t, samples[1].Tokens, saved[1].Tokens)
	assert.Equal(t, samples[1].Labels, saved[1].Labels)

	tp := fmt.Sprintf("%s finetune test", modelName)

	ftReq := api.FinetuneRequest{
		Name:       fmt.Sprintf("finetuned-%s", modelName),
		TaskPrompt: &tp,
		Tags:       []api.TagInfo{{Name: "xyz"}},
	}

	_, model := finetune(
		t, router,
		base.Id.String(),
		ftReq,
		50, 5*time.Second,
	)
	assert.Equal(t, database.ModelTrained, model.Status)
	assert.Equal(t, fmt.Sprintf("finetuned-%s", modelName), model.Name)
	require.NotNil(t, model.BaseModelId)
	assert.Equal(t, base.Id, *model.BaseModelId)
}

func TestFinetuningUDT(t *testing.T) {
	finetuningTestHelper(t, cmd.InitializeBoltUdtModel)
}

func TestFinetuningCNN(t *testing.T) {
	finetuningTestHelper(t, cmd.InitializePythonCnnModel)
}

func TestFinetuningTransformer(t *testing.T) {
	finetuningTestHelper(t, cmd.InitializePythonTransformerModel)
}

func TestFinetuningWithGenerateData(t *testing.T) {
	_, cancel, s3, db, pub, sub, router := setupCommon(t)
	defer cancel()

	baseName, baseLoader, baseID := createModel(t, s3, db, modelBucket)

	stop := startWorker(t, db, s3, pub, sub, modelBucket, map[core.ModelType]core.ModelLoader{
		core.ParseModelType(baseName): baseLoader,
	})
	defer stop()

	sample := api.FeedbackRequest{
		Tokens: []string{"foo"},
		Labels: []string{"xyz"},
	}
	require.NoError(t, httpRequest(
		router,
		"POST",
		fmt.Sprintf("/models/%s/feedback", baseID.String()),
		sample,
		nil,
	))

	tp := "Finetuning with data generation"
	ftReq := api.FinetuneRequest{
		Name:         "finetuned-with-gen",
		TaskPrompt:   &tp,
		Tags:         []api.TagInfo{{Name: "xyz"}},
		GenerateData: true,
	}

	_, model := finetune(t, router, baseID.String(), ftReq, 10, 100*time.Millisecond)

	assert.Equal(t, database.ModelTraining, model.Status)
	assert.Equal(t, "finetuned-with-gen", model.Name)
	require.NotNil(t, model.BaseModelId)
	assert.Equal(t, baseID, *model.BaseModelId)
}

func TestFinetuningOnnxModel(t *testing.T) {
	var (
		modelName    = "onnx_cnn"
		onnxDylib    = os.Getenv("ONNX_RUNTIME_DYLIB_PATH")
		pythonExec   = os.Getenv("PYTHON_EXECUTABLE_PATH")
		pluginScript = os.Getenv("PYTHON_MODEL_PLUGIN_SCRIPT_PATH")
		hostModelDir = os.Getenv("HOST_MODEL_DIR")
	)

	if onnxDylib == "" || pythonExec == "" || pluginScript == "" || hostModelDir == "" {
		t.Fatalf("ONNX_RUNTIME_DYLIB_PATH, PYTHON_EXECUTABLE_PATH, PYTHON_MODEL_PLUGIN_SCRIPT_PATH, and HOST_MODEL_DIR env vars must be set")
	}

	ctx, cancel, s3, db, pub, sub, router := setupCommon(t)
	defer cancel()

	require.NoError(t, s3.CreateBucket(context.Background(), modelBucket))

	os.Setenv("AWS_ACCESS_KEY_ID", minioUsername)
	os.Setenv("AWS_SECRET_ACCESS_KEY", minioPassword)

	python.EnablePythonPlugin(
		os.Getenv("PYTHON_EXECUTABLE_PATH"),
		os.Getenv("PYTHON_MODEL_PLUGIN_SCRIPT_PATH"),
	)

	stop := startWorker(t, db, s3, pub, sub, modelBucket, core.NewModelLoaders())
	defer stop()

	ort.SetSharedLibraryPath(onnxDylib)
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatalf("could not init ONNX Runtime: %v", err)
	}
	defer func() {
		if err := ort.DestroyEnvironment(); err != nil {
			log.Fatalf("error destroying onnx env: %v", err)
		}
	}()

	require.NoError(t, cmd.InitializeOnnxCnnModel(ctx, db, s3, modelBucket, modelName, hostModelDir))

	var base database.Model
	require.NoError(t, db.First(&base, "name = ?", modelName).Error)

	tags := []string{
		"ADDRESS", "CARD_NUMBER", "COMPANY", "CREDIT_SCORE", "DATE",
		"EMAIL", "ETHNICITY", "GENDER", "ID_NUMBER", "LICENSE_PLATE",
		"LOCATION", "NAME", "O", "PHONENUMBER", "SERVICE_CODE",
		"SEXUAL_ORIENTATION", "SSN", "URL", "VIN",
	}

	tagInfos := make([]api.TagInfo, len(tags))
	for i, tag := range tags {
		tagInfos[i] = api.TagInfo{Name: tag}
	}

	tp := fmt.Sprintf("%s finetune test", modelName)

	ftReq := api.FinetuneRequest{
		Name:       fmt.Sprintf("finetuned-%s", modelName),
		TaskPrompt: &tp,
		Tags:       tagInfos,
		Samples: []api.Sample{
			{
				Tokens: []string{"I", "started", "working", "at", "ThirdAI", "in", "2022"},
				Labels: []string{"O", "O", "O", "O", "COMPANY", "O", "DATE"},
			},
			{
				Tokens: []string{"I", "started", "working", "at", "ThirdAI", "in", "2022"},
				Labels: []string{"O", "O", "O", "O", "COMPANY", "O", "DATE"},
			},
		},
	}

	_, model := finetune(t, router, base.Id.String(), ftReq, 50, 5*time.Second)

	assert.Equal(t, database.ModelTrained, model.Status)

	uploadId := createUpload(t, router)

	reportId := createReport(t, router, api.CreateReportRequest{
		ReportName: "test-report",
		ModelId:    model.Id,
		UploadId:   uploadId,
		Tags:       tags,
	})

	report := waitForReport(t, router, reportId, 10)

	assert.Equal(t, model.Id, report.Model.Id)

	totalTags := uint64(0)
	for _, cnt := range report.TagCounts {
		totalTags += cnt
	}

	assert.Greater(t, totalTags, uint64(0))
}
