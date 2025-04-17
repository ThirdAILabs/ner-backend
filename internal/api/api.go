package api

import (
	"errors"
	"log" // Adjust import path
	"log/slog"
	"ner-backend/internal/database"
	"ner-backend/internal/messaging"
	"ner-backend/internal/s3"
	"ner-backend/pkg/models"
	"net/http"
	"os"
	"strconv"
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
		return nil, CodedErrorf(http.StatusBadRequest, "invalid source_s3_path, source_s3_path is required and must start with s3://")
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
	modelId, err := URLParamUUID(r, "model_id") // Assuming URLParamUUID is defined
	if err != nil {
		return nil, CodedErrorf(http.StatusBadRequest, "invalid model_id format")
	}

	ctx := r.Context()

	model, err := database.GetModelByID(ctx, s.db, modelId) // Use the new helper
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, CodedErrorf(http.StatusNotFound, "model not found")
		}
		slog.Error("error getting model from database", "error", err, "model_id", modelId)
		return nil, CodedErrorf(http.StatusInternalServerError, "error retrieving model record")
	}

	return model, nil // Return the pointer returned by the helper
}

func (s *BackendService) SubmitInferenceJob(r *http.Request) (any, error) {
	var req models.InferenceRequest
	req, err := ParseRequest[models.InferenceRequest](r)
	if err != nil {
		return nil, err
	}

	// Basic validation
	if req.SourceS3Bucket == "" || req.DestS3Bucket == "" {
		return nil, CodedErrorf(http.StatusBadRequest, "model_id, source_s3_bucket, and dest_s3_bucket are required")
	}

	ctx := r.Context()

	// 1. Check model status (Same as before)
	model, err := database.GetModelByID(ctx, s.db, req.ModelId)
	if err != nil { /* handle */
		return nil, CodedErrorf(http.StatusInternalServerError, "error retrieving model record")
	}
	if model == nil { /* handle */
		return nil, err
	}
	if model.Status != database.ModelTrained { /* handle */
		return nil, CodedErrorf(http.StatusBadRequest, "model_id, source_s3_bucket, and dest_s3_bucket are required")
	}

	// 2. Create inference job entry with status QUEUED
	jobId := uuid.New()

	// 3. Publish ONE GenerateInferenceTasks message
	log.Printf("Queueing GenerateInferenceTasks task for job %s...", jobId)

	// Get chunk size from config (or use a default)
	// Assuming config is accessible here, otherwise add it to APIHandler dependencies
	// For simplicity, let's read env var directly here, though config struct is better
	chunkTargetBytesStr := os.Getenv("S3_CHUNK_TARGET_BYTES")
	chunkTargetBytes, convErr := strconv.ParseInt(chunkTargetBytesStr, 10, 64)
	if convErr != nil || chunkTargetBytes <= 0 {
		log.Printf("Warning: Invalid or missing S3_CHUNK_TARGET_BYTES env var ('%s'), using default 10GB", chunkTargetBytesStr)
		chunkTargetBytes = 10 * 1024 * 1024 * 1024 // Default 10GB
	}

	genPayload := models.GenerateInferenceTasksPayload{
		JobId:             jobId,
		ModelId:           req.ModelId,
		ModelArtifactPath: model.ModelArtifactPath,
		SourceS3Bucket:    req.SourceS3Bucket,
		SourceS3Prefix:    req.SourceS3Prefix,
		DestS3Bucket:      req.DestS3Bucket,
		ChunkTargetBytes:  chunkTargetBytes,
	}

	err = s.publisher.PublishGenerateInferenceTasksTask(ctx, genPayload)
	if err != nil {
		// If publishing fails, should we mark the job as failed?
		log.Printf("CRITICAL: Failed to publish GenerateInferenceTasks task for job %s: %v", jobId, err)
		_ = database.UpdateGenerateInferenceTasksTaskStatus(ctx, s.db, jobId, database.JobFailed) // Mark job failed
		return nil, err
	}

	log.Printf("Submitted job generation task for inference job %s", jobId)
	return models.InferenceSubmitResponse{
		Message: "Inference job accepted for processing",
		JobId:   jobId,
	}, nil
}

func (s *BackendService) GetInferenceJob(r *http.Request) (any, error) {
	jobId, err := URLParamUUID(r, "job_id")
	if err != nil {
		return nil, err
	}

	ctx := r.Context()

	var job database.GenerateInferenceTasksTask
	if err := s.db.WithContext(ctx).Find(&job, "id = ?", jobId).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, CodedErrorf(http.StatusNotFound, "inference job not found")
		}
		return nil, CodedErrorf(http.StatusInternalServerError, "error retrieving inference job record")
	}

	return job, nil
}
