package database

import (
	"context"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

func UpdateModelStatus(ctx context.Context, txn *gorm.DB, modelId uuid.UUID, status string) error {
	if err := txn.WithContext(ctx).Model(&Model{Id: modelId}).Update("status", status).Error; err != nil {
		return err
	}
	return nil
}

func UpdateInferenceTaskStatus(ctx context.Context, txn *gorm.DB, jobId uuid.UUID, status string) error {
	if err := txn.WithContext(ctx).Model(&InferenceTask{Id: jobId}).Update("status", status).Error; err != nil {
		return err
	}
	return nil
}

func UpdateGenerateInferenceTasksTaskStatus(ctx context.Context, txn *gorm.DB, jobId uuid.UUID, status string) error {
	if err := txn.WithContext(ctx).Model(&GenerateInferenceTasksTask{Id: jobId}).Update("status", status).Error; err != nil {
		return err
	}
	return nil
}

func GetModelByID(ctx context.Context, db *gorm.DB, modelId uuid.UUID) (*Model, error) {
	var model Model
	if err := db.WithContext(ctx).Where("id = ?", modelId).First(&model).Error; err != nil {
		return nil, err
	}
	return &model, nil
}
