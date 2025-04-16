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

func UpdateInferenceJobStatus(ctx context.Context, txn *gorm.DB, jobId uuid.UUID, status string) error {
	if err := txn.WithContext(ctx).Model(&InferenceJob{Id: jobId}).Update("status", status).Error; err != nil {
		return err
	}
	return nil
}
