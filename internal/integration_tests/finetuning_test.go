package integrationtests

import (
	"context"
	"encoding/json"
	"fmt"
	backend "ner-backend/internal/api"
	"ner-backend/internal/core"
	"ner-backend/internal/database"
	"ner-backend/internal/messaging"
	"ner-backend/internal/s3"
	"ner-backend/pkg/api"
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

	s3, err := s3.NewS3Client(&s3.Config{
		S3EndpointURL:     minioUrl,
		S3AccessKeyID:     minioUsername,
		S3SecretAccessKey: minioPassword,
	})
	require.NoError(t, err)

	db := createDB(t)

	queue := messaging.NewInMemoryQueue()

	backend := backend.NewBackendService(db, s3, queue, 120)
	router := chi.NewRouter()
	backend.AddRoutes(router)

	worker := core.NewTaskProcessor(db, s3, s3, queue, queue, t.TempDir(), modelBucket)

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

	stream := s3.DownloadFileStream(modelBucket, fmt.Sprintf("%s/model.bin", model.Id))

	var modelData map[string]string
	if err := json.NewDecoder(stream).Decode(&modelData); err != nil {
		t.Fatalf("failed to decode model data: %v", err)
	}
	assert.Equal(t, 3, len(modelData))
	// The finetuning method for the test model just adds the new tag names as regex patterns
	assert.Contains(t, modelData, "xyz")
}
