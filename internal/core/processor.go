package core

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"ner-backend/internal/database"
	"ner-backend/internal/messaging"
	"ner-backend/internal/s3"
	"ner-backend/pkg/models"
	"strings"
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

type TaskProcessor struct {
	db        *gorm.DB
	inference *InferenceJobProcessor
	s3Client  *s3.Client
	publisher messaging.Publisher
	reciever  messaging.Reciever
}

func NewTaskProcessor(db *gorm.DB, s3client *s3.Client, publisher messaging.Publisher, reciever messaging.Reciever, localModelDir string, modelArtifactPath string) *TaskProcessor {
	inference := NewInferenceJobProcessor(db, s3client, localModelDir, modelArtifactPath)
	return &TaskProcessor{
		db:        db,
		inference: inference,
		s3Client:  s3client,
		publisher: publisher,
		reciever:  reciever,
	}
}

func (proc *TaskProcessor) Start() {
	slog.Info("starting task processor")

	for task := range proc.reciever.Tasks() {
		proc.ProcessTask(task)
	}
}

func (proc *TaskProcessor) Stop() {
	slog.Info("stopping task processor")

	proc.publisher.Close()
	proc.reciever.Close()

}

func (proc *TaskProcessor) ProcessTask(task messaging.Task) {
	ctx := context.Background()

	var err error
	switch task.Type() {

	case messaging.InferenceQueue:
		var payload models.InferenceTaskPayload
		if err = json.Unmarshal(task.Payload(), &payload); err != nil {
			slog.Error("error unmarshalling inference task", "error", err)
			task.Reject() // Discard malformed message
			return
		}
		err = proc.processInferenceTask(ctx, payload)

	case messaging.ShardDataQueue:
		var payload models.ShardDataPayload
		if err = json.Unmarshal(task.Payload(), &payload); err != nil {
			slog.Error("error unmarshalling shard data task", "error", err)
			task.Reject() // Discard malformed message
			return
		}
		err = proc.processShardDataTask(ctx, payload) // Call new handler

	case messaging.TrainingQueue:
		var payload models.TrainTaskPayload
		if err = json.Unmarshal(task.Payload(), &payload); err != nil {
			slog.Error("error unmarshalling training task", "error", err)
			// Reject message (non-requeueable) as it's malformed
			task.Reject()
			return
		}
		err = proc.processTrainTask(ctx, payload)

	default:
		slog.Error("received unknown task type", "queue", task.Type())
		task.Reject() // reject unknown message type
		return
	}

	if err != nil {
		slog.Error("error processing task", "queue", task.Type(), "error", err)
		task.Nack()
	} else {
		slog.Info("successfully processed task", "queue", task.Type())
		task.Ack()
	}
}

func (proc *TaskProcessor) processInferenceTask(ctx context.Context, payload models.InferenceTaskPayload) error {
	reportId := payload.ReportId

	var task database.InferenceTask
	if err := proc.db.Preload("Report").Preload("Report.Model").Preload("Report.Groups").First(&task, "report_id = ? AND task_id = ?", reportId, payload.TaskId).Error; err != nil {
		slog.Error("error fetching inference task", "report_id", reportId, "task_id", payload.TaskId, "error", err)
		return fmt.Errorf("error getting inference task: %w", err)
	}

	proc.updateInferenceTaskStatus(ctx, reportId, payload.TaskId, database.JobRunning)

	s3Objects := strings.Split(task.SourceS3Keys, ";")

	groupToQuery := map[uuid.UUID]string{}
	for _, group := range task.Report.Groups {
		groupToQuery[group.Id] = group.Query
	}

	workerErr := proc.inference.RunInferenceTask(task.ReportId, task.Report.Model.Id, task.Report.Model.Type, groupToQuery, task.SourceS3Bucket, s3Objects)
	if workerErr != nil {
		slog.Error("error running inference task", "report_id", reportId, "task_id", payload.TaskId, "error", workerErr)
		proc.updateInferenceTaskStatus(ctx, reportId, payload.TaskId, database.JobFailed)
		return fmt.Errorf("error running inference task: %w", workerErr)
	}

	if err := proc.updateInferenceTaskStatus(ctx, reportId, payload.TaskId, database.JobCompleted); err != nil {
		return fmt.Errorf("error updating inference task status to complete: %w", err)
	}

	return nil
}

func (proc *TaskProcessor) processShardDataTask(ctx context.Context, payload models.ShardDataPayload) error {
	reportId := payload.ReportId

	var task database.ShardDataTask
	if err := proc.db.Preload("Report").First(&task, "report_id = ?", reportId).Error; err != nil {
		slog.Error("error fetching shard data task", "report_id", reportId, "error", err)
		return fmt.Errorf("error getting shard data task: %w", err)
	}

	slog.Info("Handling generate tasks", "jobId", task.ReportId, "sourceBucket", task.Report.SourceS3Bucket, "sourcePrefix", task.Report.SourceS3Prefix)

	targetBytes := task.ChunkTargetBytes
	if targetBytes <= 0 {
		targetBytes = 10 * 1024 * 1024 * 1024 // Default 10GB if not set or invalid
		slog.Info("Using default chunk target size", "targetBytes", targetBytes, "jobId", reportId)
	}

	var taskIndex int = 0

	// Define the callback function to process each chunk
	processChunkCallback := func(ctx context.Context, chunkKeys []string, chunkSize int64) error {
		taskIndex++ // Increment count *after* successful publishing
		slog.Info("Handler: Processing chunk", "chunkIndex", taskIndex, "jobId", reportId, "chunkSize", chunkSize, "keyCount", len(chunkKeys))

		task := database.InferenceTask{
			ReportId:     reportId,
			TaskId:       taskIndex,
			Status:       database.JobQueued,
			CreationTime: time.Now().UTC(),

			SourceS3Bucket: task.Report.SourceS3Bucket,
			SourceS3Keys:   strings.Join(chunkKeys, ";"),
			TotalSize:      chunkSize,
		}

		inferencePayload := models.InferenceTaskPayload{
			ReportId: task.ReportId, TaskId: task.TaskId,
		}

		if err := proc.publisher.PublishInferenceTask(ctx, inferencePayload); err != nil {
			// Use slog.Error for failures
			slog.Error("Handler: Failed to publish inference chunk", "report_id", reportId, "task_id", taskIndex, "error", err)
			// Return the error so the helper function knows processing failed
			return fmt.Errorf("failed to publish inference chunk %d: %w", taskIndex, err)
		}

		if err := proc.db.WithContext(ctx).Create(&task).Error; err != nil {
			slog.Error("error saving inference task to db", "report_id", task.ReportId, "task_id", task.TaskId, "error", err)
			return fmt.Errorf("error saving inference task to db: %w", err)
		}

		return nil
	}

	processedChunks, err := proc.s3Client.ListAndChunkS3Objects(
		ctx,
		task.Report.SourceS3Bucket,
		task.Report.SourceS3Prefix.String,
		targetBytes,
		reportId,
		processChunkCallback,
	)

	if err != nil {
		slog.Error("Failed during S3 processing/chunk publishing", "report_id", reportId, "error", err)
		proc.updateShardDataTaskStatus(ctx, reportId, database.JobFailed)
		return fmt.Errorf("failed during task generation for job %s: %w", reportId, err)
	}

	if processedChunks != taskIndex {
		slog.Warn("Mismatch between processed chunks and tasks queued", "processedChunks", processedChunks, "n_tasks", taskIndex, "report_id", reportId)
	}

	slog.Info("Finished generating inference task chunks", "n_tasks", taskIndex, "report_id", reportId)

	if err := proc.updateShardDataTaskStatus(ctx, reportId, database.JobCompleted); err != nil {
		return fmt.Errorf("failed to update job final status: %w", err)
	}

	return nil
}

func (proc *TaskProcessor) processTrainTask(ctx context.Context, payload models.TrainTaskPayload) error {
	slog.Error("train tasks are not implemented yet", "payload", payload)
	return nil
}

func (proc *TaskProcessor) updateInferenceTaskStatus(ctx context.Context, reportId uuid.UUID, taskId int, status string) error {
	if err := proc.db.WithContext(ctx).Model(&database.InferenceTask{ReportId: reportId, TaskId: taskId}).Update("status", status).Error; err != nil {
		slog.Error("error updating inference task status", "report_id", reportId, "task_id", taskId, "status", status, "error", err)
		return err
	}
	return nil
}

func (proc *TaskProcessor) updateShardDataTaskStatus(ctx context.Context, reportId uuid.UUID, status string) error {
	if err := proc.db.WithContext(ctx).Model(&database.ShardDataTask{ReportId: reportId}).Update("status", status).Error; err != nil {
		slog.Error("error updating shard data task status", "report_id", reportId, "status", status, "error", err)
		return err
	}
	return nil
}
