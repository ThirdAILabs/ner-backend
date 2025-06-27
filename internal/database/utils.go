package database

import (
	"context"
	"encoding/json"
	"fmt"
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
	newTags := make([]ModelTag, len(tags))
	for i, t := range tags {
		newTags[i] = ModelTag{ModelId: modelId, Tag: t}
	}

	if err := db.WithContext(ctx).
		Where("model_id = ?", modelId).
		Delete(&ModelTag{}).
		Error; err != nil {
		return fmt.Errorf("could not clear old tags: %w", err)
	}

	if len(newTags) > 0 {
		if err := db.WithContext(ctx).
			Create(&newTags).
			Error; err != nil {
			return fmt.Errorf("could not add new tags: %w", err)
		}
	}
	return nil
}

func SaveFeedbackSample(ctx context.Context, db *gorm.DB, modelId uuid.UUID, tokens []string, labels []string) error {
	bTokens, err := json.Marshal(tokens)
	if err != nil {
		return fmt.Errorf("could not marshal tokens: %w", err)
	}
	bLabels, err := json.Marshal(labels)
	if err != nil {
		return fmt.Errorf("could not marshal labels: %w", err)
	}

	fs := FeedbackSample{
		ID:      uuid.New(),
		ModelId: modelId,
		Tokens:  bTokens,
		Labels:  bLabels,
	}
	if err := db.WithContext(ctx).Create(&fs).Error; err != nil {
		return fmt.Errorf("failed to save feedback sample: %w", err)
	}
	return nil
}

func GetFeedbackSamples(ctx context.Context, db *gorm.DB, modelId uuid.UUID) ([]uuid.UUID, [][]string, [][]string, error) {
	var rows []FeedbackSample
	if err := db.WithContext(ctx).Where("model_id = ?", modelId).Find(&rows).Error; err != nil {
		return nil, nil, nil, fmt.Errorf("could not query feedback samples: %w", err)
	}

	allIds := make([]uuid.UUID, 0, len(rows))
	allTokens := make([][]string, 0, len(rows))
	allLabels := make([][]string, 0, len(rows))
	for _, r := range rows {
		var toks []string
		var labs []string
		if err := json.Unmarshal(r.Tokens, &toks); err != nil {
			return nil, nil, nil, fmt.Errorf("invalid tokens JSON: %w", err)
		}
		if err := json.Unmarshal(r.Labels, &labs); err != nil {
			return nil, nil, nil, fmt.Errorf("invalid labels JSON: %w", err)
		}
		allIds = append(allIds, r.ID)
		allTokens = append(allTokens, toks)
		allLabels = append(allLabels, labs)
	}
	return allIds, allTokens, allLabels, nil
}
