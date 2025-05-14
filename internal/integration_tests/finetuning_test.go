package integrationtests

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"testing"
	"time"

	"ner-backend/cmd"
	backendpkg "ner-backend/internal/api"
	"ner-backend/internal/core"
	"ner-backend/internal/database"
	"ner-backend/internal/messaging"
	"ner-backend/internal/storage"
	"ner-backend/pkg/api"

	"github.com/go-chi/chi/v5"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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

	backendSvc := backendpkg.NewBackendService(db, s3Prov, pub, 120)
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
	loaders map[string]core.ModelLoader,
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

	stop := startWorker(t, db, s3, pub, sub, modelBucket, map[string]core.ModelLoader{
		baseName: baseLoader,
	})
	defer stop()

	ftReq := api.FinetuneRequest{
		Name:       "finetuned-model",
		TaskPrompt: "finetuning test",
		Tags:       []api.TagInfo{{Name: "xyz"}},
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
	assert.Contains(t, data, "xyz")
}

func TestFinetuning_CNNModel(t *testing.T) {
	if os.Getenv("PYTHON_EXECUTABLE_PATH") == "" || os.Getenv("PYTHON_MODEL_PLUGIN_SCRIPT_PATH") == "" {
		t.Fatalf("PYTHON_EXECUTABLE_PATH and PYTHON_MODEL_PLUGIN_SCRIPT_PATH must be set")
	}

	ctx, cancel, s3, db, pub, sub, router := setupCommon(t)
	defer cancel()

	require.NoError(t, s3.CreateBucket(context.Background(), modelBucket))

	os.Setenv("AWS_ACCESS_KEY_ID", minioUsername)
	os.Setenv("AWS_SECRET_ACCESS_KEY", minioPassword)

	stop := startWorker(t, db, s3, pub, sub, modelBucket,
		core.NewModelLoaders(
			os.Getenv("PYTHON_EXECUTABLE_PATH"),
			os.Getenv("PYTHON_MODEL_PLUGIN_SCRIPT_PATH"),
		),
	)
	defer stop()

	require.NoError(t, cmd.InitializeCnnNerExtractor(ctx, db, s3, modelBucket, os.Getenv("HOST_MODEL_DIR")))
	var base database.Model
	require.NoError(t, db.Where("name = ?", "advanced").First(&base).Error)

	ftReq := api.FinetuneRequest{
		Name:       "finetuned-cnn",
		TaskPrompt: "CNN finetune test",
		Tags:       []api.TagInfo{{Name: "xyz"}},
	}
	_, model := finetune(
		t, router,
		base.Id.String(),
		ftReq,
		50, 5*time.Second,
	)

	assert.Equal(t, database.ModelTrained, model.Status)
	assert.Equal(t, "finetuned-cnn", model.Name)
	require.NotNil(t, model.BaseModelId)
	assert.Equal(t, base.Id, *model.BaseModelId)
}
