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

func SaveReportError(ctx context.Context, txn *gorm.DB, reportId uuid.UUID, errorMessage string) {
	reportError := ReportError{
		ReportId:  reportId,
		ErrorId:   uuid.New(),
		Error:     errorMessage,
		Timestamp: time.Now().UTC(),
	}

	if err := txn.WithContext(ctx).Create(&reportError).Error; err != nil {
		slog.Error("error saving report error", "report_id", reportId, "error", err)
	}
}

func SetModelTags(ctx context.Context, db *gorm.DB, modelId uuid.UUID, tags []string) error {
	modelTags := make([]ModelTag, len(tags))
	for i, tag := range tags {
		modelTags[i] = ModelTag{
			ModelId: modelId,
			Tag:     tag,
		}
	}

	if err := db.WithContext(ctx).
		Model(&Model{Id: modelId}).
		Association("Tags").
		Replace(&modelTags); err != nil {
		slog.Error("failed to attach tags to model", "model_id", modelId, "error", err)
		return err
	}

	return nil
}
