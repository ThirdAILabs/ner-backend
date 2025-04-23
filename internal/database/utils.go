package database

import (
	"context"
	"log/slog"
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

func UpdateModelStatus(ctx context.Context, txn *gorm.DB, modelId uuid.UUID, status string) error {
	updates := map[string]any{"status": status}
	if status == JobCompleted || status == JobFailed {
		updates["completion_time"] = time.Now().UTC()
	}

	if err := txn.WithContext(ctx).Model(&Model{Id: modelId}).Updates(updates).Error; err != nil {
		slog.Error("error updating model status", "model_id", modelId, "status", status, "error", err)
		return err
	}
	return nil
}

func UpdateInferenceTaskStatus(ctx context.Context, txn *gorm.DB, reportId uuid.UUID, taskId int, status string) error {
	updates := map[string]any{"status": status}
	if status == JobCompleted || status == JobFailed {
		updates["completion_time"] = time.Now().UTC()
	}

	if err := txn.WithContext(ctx).Model(&InferenceTask{ReportId: reportId, TaskId: taskId}).Updates(updates).Error; err != nil {
		slog.Error("error updating inference task status", "report_id", reportId, "task_id", taskId, "status", status, "error", err)
		return err
	}
	return nil
}

func UpdateShardDataTaskStatus(ctx context.Context, txn *gorm.DB, reportId uuid.UUID, status string) error {
	updates := map[string]any{"status": status}
	if status == JobCompleted || status == JobFailed {
		updates["completion_time"] = time.Now().UTC()
	}

	if err := txn.WithContext(ctx).Model(&ShardDataTask{ReportId: reportId}).Updates(updates).Error; err != nil {
		slog.Error("error updating shard data task status", "report_id", reportId, "status", status, "error", err)
		return err
	}
	return nil
}
