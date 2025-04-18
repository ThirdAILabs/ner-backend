package core

import (
	"context"
	"fmt"
	"log/slog"
	"ner-backend/internal/core/types"
	"ner-backend/internal/database"
	"ner-backend/internal/s3"
	"os"
	"path/filepath"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

type InferenceJobProcessor struct {
	db       *gorm.DB
	s3client *s3.Client

	localModelDir     string
	modelArtifactPath string
}

func (proc *InferenceJobProcessor) RunInferenceTask(
	jobId uuid.UUID,
	modelId uuid.UUID,
	modelType string,
	groupToQuery map[uuid.UUID]string,
	bucket string,
	objects []string,
) {
	parser := NewDefaultParser()

	model, err := proc.loadModel(modelId, modelType)
	if err != nil {
	}

	groupToFilter := make(map[uuid.UUID]Filter)
	for groupId, query := range groupToQuery {
		filter, err := ParseQuery(query)
		if err != nil {

		}
		groupToFilter[groupId] = filter
	}

	for _, object := range objects {
		entities, groups, err := proc.processObject(jobId, parser, model, groupToFilter, bucket, object)
		if err != nil {
			slog.Error("error processing object", "object", object, "error", err)
			continue
		}

		if err := proc.db.CreateInBatches(&entities, 100).Error; err != nil {
			slog.Error("error saving entities to database", "object", object, "error", err)
			continue
		}

		if err := proc.db.CreateInBatches(groups, 100); err != nil {
			slog.Error("error saving groups to database", "object", object, "error", err)
			continue
		}
	}
}

func (proc *InferenceJobProcessor) loadModel(modelId uuid.UUID, modelType string) (Model, error) {
	localPath := filepath.Join(proc.localModelDir, modelId.String(), "model.bin")

	// Check if the model file exists locally
	if _, err := os.Stat(localPath); os.IsNotExist(err) {
		slog.Info("model not found locally, downloading from S3", "modelId", modelId)

		modelObjectKey := filepath.Join(modelId.String(), "model.bin")

		if err := proc.s3client.DownloadFile(context.TODO(), proc.modelArtifactPath, modelObjectKey, localPath); err != nil {
			return nil, fmt.Errorf("failed to download model from S3: %w", err)
		}
	}

	// Load the model from the local path
	model, err := LoadModel(localPath, modelType)
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}

	return model, nil
}

func (proc *InferenceJobProcessor) processObject(
	jobId uuid.UUID,
	parser Parser,
	model Model,
	groupFilter map[uuid.UUID]Filter,
	bucket string,
	object string) (
	[]database.ObjectEntity, []database.ObjectGroup, error) {

	chunks := parser.Parse(object, proc.s3client.DownloadFileStream(bucket, object))

	labelToEntities := make(map[string][]types.Entity)

	for chunk := range chunks {
		if chunk.Error != nil {
			return nil, nil, fmt.Errorf("error parsing document: %w", chunk.Error)
		}

		chunkEntities, err := model.Predict(chunk.Text)
		if err != nil {
			return nil, nil, fmt.Errorf("error running model inference: %w", err)
		}

		for _, entity := range chunkEntities {
			entity.Start += chunk.Offset
			entity.End += chunk.Offset
			labelToEntities[entity.Label] = append(labelToEntities[entity.Label], entity)
		}
	}

	groups := make([]database.ObjectGroup, 0)
	for groupId, filter := range groupFilter {
		if filter.Matches(labelToEntities) {
			groups = append(groups, database.ObjectGroup{
				InferenceJobId: jobId,
				GroupId:        groupId,
				Object:         object,
			})
		}
	}

	allEntities := make([]database.ObjectEntity, 0)
	for _, entities := range labelToEntities {
		for _, entity := range entities {
			allEntities = append(allEntities, database.ObjectEntity{
				InferenceJobId: jobId,
				Label:          entity.Label,
				Text:           entity.Text,
				Start:          entity.Start,
				End:            entity.End,
				Object:         object,
			})
		}
	}

	return allEntities, groups, nil
}
