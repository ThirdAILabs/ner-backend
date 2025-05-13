package integrationtests

import (
	"context"
	"encoding/json"
	"fmt"
	"ner-backend/cmd"
	backend "ner-backend/internal/api"
	"ner-backend/internal/core"
	"ner-backend/internal/database"
	"ner-backend/internal/storage"
	"ner-backend/pkg/api"
	"os"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFinetuning(t *testing.T) {
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

	baseModelId := createModel(t, s3, db, modelBucket)

	var res api.FinetuneResponse
	err = httpRequest(router, "POST", fmt.Sprintf("/models/%s/finetune", baseModelId.String()), api.FinetuneRequest{
		Name: "finetuned-model", TaskPrompt: "finetuning test", Tags: []api.TagInfo{{Name: "xyz"}},
	}, &res)
	require.NoError(t, err)

	var model api.Model
	for i := 0; i < 10; i++ {
		time.Sleep(100 * time.Millisecond)
		err := httpRequest(router, "GET", fmt.Sprintf("/models/%s", res.ModelId), nil, &model)
		require.NoError(t, err)

		if model.Status == database.ModelTrained || model.Status == database.ModelFailed {
			break
		}
	}

	assert.Equal(t, database.ModelTrained, model.Status)
	assert.Equal(t, "finetuned-model", model.Name)
	assert.NotEqual(t, nil, model.BaseModelId)
	assert.Equal(t, baseModelId, *model.BaseModelId)

	stream, err := s3.GetObjectStream(modelBucket, fmt.Sprintf("%s/model.json", model.Id))
	require.NoError(t, err)

	var modelData map[string]string
	if err := json.NewDecoder(stream).Decode(&modelData); err != nil {
		t.Fatalf("failed to decode model data: %v", err)
	}
	assert.Equal(t, 3, len(modelData))
	// The finetuning method for the test model just adds the new tag names as regex patterns
	assert.Contains(t, modelData, "xyz")
}

func TestFinetuning_CNNModel(t *testing.T) {
	os.Setenv("PYTHON_EXECUTABLE_PATH", "/opt/conda/envs/pii-presidio-3.10/bin/python3")
	os.Setenv("PYTHON_PLUGIN_PATH", "/home/ubuntu/pratik/ner-backend/plugin/plugin-python/plugin.py")
	os.Setenv("HOST_MODEL_DIR", "/home/ubuntu/shubh/ner/misc/ner-models")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	minioUrl := setupMinioContainer(t, ctx)
	s3, err := storage.NewS3Provider(storage.S3ProviderConfig{
		S3EndpointURL:     minioUrl,
		S3AccessKeyID:     minioUsername,
		S3SecretAccessKey: minioPassword,
	})
	require.NoError(t, err)
	err = s3.CreateBucket(context.Background(), modelBucket)
	require.NoError(t, err)

	os.Setenv("AWS_ACCESS_KEY_ID", minioUsername)
	os.Setenv("AWS_SECRET_ACCESS_KEY", minioPassword)

	db := createDB(t)

	publisher, receiver := setupRabbitMQContainer(t, ctx)

	backendSvc := backend.NewBackendService(db, s3, publisher, 120)
	router := chi.NewRouter()
	backendSvc.AddRoutes(router)

	worker := core.NewTaskProcessor(db, s3, publisher, receiver, &DummyLicenseVerifier{}, t.TempDir(), modelBucket)
	go worker.Start()
	defer worker.Stop()

	require.NoError(t, cmd.InitializeCnnNerExtractor(ctx, db, s3, modelBucket))

	var baseModel database.Model
	require.NoError(t, db.Where("name = ?", "advance").First(&baseModel).Error)

	var ftResp api.FinetuneResponse
	err = httpRequest(
		router, "POST",
		fmt.Sprintf("/models/%s/finetune", baseModel.Id),
		api.FinetuneRequest{
			Name:       "finetuned-cnn",
			TaskPrompt: "CNN finetune test",
			Tags:       []api.TagInfo{{Name: "xyz"}},
		},
		&ftResp,
	)
	require.NoError(t, err)

	var m api.Model
	for i := 0; i < 50; i++ {
		time.Sleep(5 * time.Second)
		err = httpRequest(router, "GET", fmt.Sprintf("/models/%s", ftResp.ModelId), nil, &m)
		require.NoError(t, err)
		if m.Status == database.ModelTraining {
			t.Logf("Model is still training, waiting...")
			continue
		}
		if m.Status == database.ModelTrained || m.Status == database.ModelFailed {
			break
		}
	}

	assert.Equal(t, database.ModelTrained, m.Status, "CNN model should be trained")
	assert.Equal(t, "finetuned-cnn", m.Name)
	assert.NotNil(t, m.BaseModelId)
	assert.Equal(t, baseModel.Id, *m.BaseModelId)
}
