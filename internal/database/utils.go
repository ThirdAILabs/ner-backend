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

func UpdateInferenceTaskStatus(ctx context.Context, txn *gorm.DB, reportId uuid.UUID, taskId int, status string) error {
	if err := txn.WithContext(ctx).Model(&InferenceTask{ReportId: reportId, TaskId: taskId}).Update("status", status).Error; err != nil {
		return err
	}
	return nil
}

func UpdateShardDataTaskStatus(ctx context.Context, txn *gorm.DB, reportId uuid.UUID, status string) error {
	if err := txn.WithContext(ctx).Model(&ShardDataTask{ReportId: reportId}).Update("status", status).Error; err != nil {
		return err
	}
	return nil
}
