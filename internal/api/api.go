package api

import (
	"database/sql"
	"errors"
	"fmt"
	"log" // Adjust import path
	"log/slog"
	"ner-backend/internal/database"
	"ner-backend/internal/messaging"
	"ner-backend/internal/s3"
	"ner-backend/pkg/models"
	"net/http"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"gorm.io/gorm"
)

type BackendService struct {
	db        *gorm.DB
	publisher *messaging.TaskPublisher
	s3Client  *s3.Client
}

func NewBackendService(db *gorm.DB, pub *messaging.TaskPublisher, s3c *s3.Client) *BackendService {
	return &BackendService{db: db, publisher: pub, s3Client: s3c}
}

func (s *BackendService) AddRoutes(r chi.Router) {
	r.Get("/health", RestHandler(func(r *http.Request) (any, error) { return nil, nil }))
	r.Route("/models", func(r chi.Router) {
		r.Post("/", RestHandler(s.SubmitTrainingJob))
		r.Get("/{model_id}", RestHandler(s.GetModel))
	})
	r.Route("/inference", func(r chi.Router) {
		r.Post("/", RestHandler(s.SubmitInferenceJob))
		r.Get("/{job_id}", RestHandler(s.GetInferenceJob))
	})
}

func (s *BackendService) SubmitTrainingJob(r *http.Request) (any, error) {
	var req models.TrainRequest
	req, err := ParseRequest[models.TrainRequest](r)
	if err != nil {
		return nil, err
	}

	if req.SourceS3Path == "" || !strings.HasPrefix(req.SourceS3Path, "s3://") {
		return nil, CodedErrorf(http.StatusBadRequest, "invalid source_s3_path, source_s3_path is required and must start wiht s3://")
	}

	ctx := r.Context() // Use request context

	model := &database.Model{
		Id:           uuid.New(),
		Name:         req.ModelName,
		Type:         "TODO",
		Status:       database.ModelQueued,
		CreationTime: time.Now(),
	}

	if err := s.db.WithContext(ctx).Create(&model).Error; err != nil {
		slog.Error("error creating model", "error", err)
		return nil, CodedErrorf(http.StatusInternalServerError, "failed to create model entry")
	}

	payload := models.TrainTaskPayload{
		ModelId:          model.Id,
		SourceS3PathTags: req.SourceS3Path,
	}

	// Publish task in background? For now, do it directly.
	if err := s.publisher.PublishTrainTask(ctx, payload); err != nil {
		// Attempt to rollback DB state or mark as failed? Difficult without transactions.
		// For now, log error and return failure to client.
		slog.Error("error publishing training task", "model_id", model.Id, "error", err)
		return nil, CodedErrorf(http.StatusInternalServerError, "failed to queue training task")
	}

	slog.Info("Submitted training job for model", "model_id", model.Id)
	return models.TrainSubmitResponse{Message: "Training job submitted", ModelId: model.Id}, nil
}

func (s *BackendService) GetModel(r *http.Request) (any, error) {
	modelId, err := URLParamUUID(r, "model_id")
	if err != nil {
		return nil, err
	}

	ctx := r.Context()

	var model database.Model
	if err := s.db.WithContext(ctx).Find(&model, "id = ?", modelId).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, CodedErrorf(http.StatusNotFound, "model not found")
		}
		slog.Error("error getting model", "error", err)
		return nil, CodedErrorf(http.StatusInternalServerError, "error retrieving model record")
	}

	return model, nil
}

func (s *BackendService) SubmitInferenceJob(r *http.Request) (any, error) {
	var req models.InferenceRequest

	req, err := ParseRequest[models.InferenceRequest](r)
	if err != nil {
		return nil, err
	}

	if req.SourceS3Bucket == "" || req.DestS3Bucket == "" {
		return nil, CodedErrorf(http.StatusUnprocessableEntity, "missing required fields: model_id, source_s3_bucket, dest_s3_bucket")
	}

	ctx := r.Context()

	var model database.Model
	if err := s.db.WithContext(ctx).Find(&model, "id = ?", req.ModelId).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, CodedErrorf(http.StatusNotFound, "model not found")
		}
		slog.Error("error getting model", "error", err)
		return nil, CodedErrorf(http.StatusInternalServerError, "error retrieving model record")
	}

	if model.Status != database.ModelTrained {
		return nil, CodedErrorf(http.StatusUnprocessableEntity, "model is not ready: model has status: %s", model.Status)
	}
	if model.ModelArtifactPath == "" {
		return nil, CodedErrorf(http.StatusInternalServerError, "model artifact path is missing")
	}

	job := database.InferenceJob{
		Id:             uuid.New(),
		ModelId:        model.Id,
		SourceS3Bucket: req.SourceS3Bucket,
		SourceS3Prefix: sql.NullString{String: req.SourceS3Prefix, Valid: req.SourceS3Prefix != ""},
		DestS3Bucket:   req.DestS3Bucket,
		Status:         database.JobQueued,
		CreationTime:   time.Now(),
	}

	if err := s.db.WithContext(ctx).Create(&job).Error; err != nil {
		slog.Error("error creating inference job", "error", err)
		return nil, CodedErrorf(http.StatusInternalServerError, "failed to create inference job entry")
	}

	// 3. List files (Consider doing this asynchronously for large buckets)
	filesToProcess, listErr := s.s3Client.ListFiles(ctx, req.SourceS3Bucket, req.SourceS3Prefix)
	if listErr != nil {
		slog.Error("error listing files in s3 bucket", "bucket", req.SourceS3Bucket, "prefix", req.SourceS3Prefix, "error", listErr)
		database.UpdateInferenceJobStatus(ctx, s.db, job.Id, database.JobFailed)
		return nil, CodedErrorf(http.StatusInternalServerError, "failed to list files in source S3 bucket: %v", listErr)
	}

	taskCount := len(filesToProcess)
	if taskCount == 0 {
		database.UpdateInferenceJobStatus(ctx, s.db, job.Id, database.JobCompleted)
		log.Printf("No files found for inference job %s", job.Id)
		return models.InferenceSubmitResponse{
			Message:   "No files found to process in specified location",
			JobId:     job.Id,
			TaskCount: 0,
		}, nil
	}

	// 4. Submit tasks (Can do this in a goroutine?)
	log.Printf("Queueing %d inference tasks for job %s...", taskCount, job.Id)
	publishErrors := false
	for _, s3Key := range filesToProcess {
		payload := models.InferenceTaskPayload{
			JobId:             job.Id,
			ModelId:           req.ModelId,
			ModelArtifactPath: model.ModelArtifactPath,
			SourceS3Bucket:    req.SourceS3Bucket,
			SourceS3Key:       s3Key,
			DestS3Bucket:      req.DestS3Bucket,
		}
		// Use context from request, maybe add timeout?
		if err := s.publisher.PublishInferenceTask(ctx, payload); err != nil {
			log.Printf("ERROR: Failed to publish inference task for key %s, job %s: %v", s3Key, job.Id, err)
			publishErrors = true
			// Decide whether to stop or continue queueing other tasks
			// break // Option: stop queueing on first error
		}
	}

	// 5. Update job status
	finalStatus := database.JobRunning
	if publishErrors {
		// Decide final status if some tasks failed to publish
		// Could mark as FAILED, or RUNNING_WITH_ERRORS if we had such a state
		log.Printf("WARNING: Some tasks failed to publish for job %s", job.Id)
		// Keeping status as RUNNING for now, assuming some tasks might have queued
	}

	if err := database.UpdateInferenceJobStatus(ctx, s.db, job.Id, finalStatus); err != nil {
		// Log error, but tasks might already be running
		log.Printf("ERROR: Failed to update job %s status to %s after queueing: %v", job.Id, finalStatus, err)
	}

	log.Printf("Submitted inference job %s with %d tasks (Publish errors: %v)", job.Id, taskCount, publishErrors)

	return models.InferenceSubmitResponse{
		Message:   fmt.Sprintf("Inference job submitted with %d tasks", taskCount),
		JobId:     job.Id,
		TaskCount: taskCount,
	}, nil
}

func (s *BackendService) GetInferenceJob(r *http.Request) (any, error) {
	jobId, err := URLParamUUID(r, "job_id")
	if err != nil {
		return nil, err
	}

	ctx := r.Context()

	var job database.InferenceJob
	if err := s.db.WithContext(ctx).Find(&job, "id = ?", jobId).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, CodedErrorf(http.StatusNotFound, "inference job not found")
		}
		return nil, CodedErrorf(http.StatusInternalServerError, "error retrieving inference job record")
	}

	return job, nil
}
