package api

import (
	"database/sql"
	"errors"
	"log" // Adjust import path
	"log/slog"
	"ner-backend/internal/core"
	"ner-backend/internal/database"
	"ner-backend/internal/messaging"
	"ner-backend/internal/s3"
	"ner-backend/pkg/models"
	"net/http"
	"strconv"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
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
		r.Get("/", RestHandler(s.ListModels))
		r.Get("/{model_id}", RestHandler(s.GetModel))
	})
	r.Route("/inference", func(r chi.Router) {
		r.Get("/", RestHandler(s.ListInferenceJobs))
		r.Post("/", RestHandler(s.SubmitInferenceJob))
		r.Get("/{job_id}", RestHandler(s.GetInferenceJob))
		r.Get("/{job_id}/groups/{group_id}", RestHandler(s.GetInferenceJobGroup))
		r.Get("/{job_id}/entities", RestHandler(s.GetInferenceJobEntities))
	})
}

func (s *BackendService) ListModels(r *http.Request) (any, error) {
	ctx := r.Context()
	var models []database.Model
	if err := s.db.WithContext(ctx).Find(&models).Error; err != nil {
		slog.Error("error getting models", "error", err)
		return nil, CodedErrorf(http.StatusInternalServerError, "error retrieving model records")
	}
	return models, nil
}

func (s *BackendService) GetModel(r *http.Request) (any, error) {
	modelId, err := URLParamUUID(r, "model_id")
	if err != nil {
		return nil, err
	}

	ctx := r.Context()

	var model database.Model
	if err := s.db.WithContext(ctx).First(&model, "id = ?", modelId).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, CodedErrorf(http.StatusNotFound, "model not found")
		}
		slog.Error("error getting model", "model_id", modelId, "error", err)
		return nil, CodedErrorf(http.StatusInternalServerError, "error retrieving model record")
	}

	return model, nil
}

func (s *BackendService) ListInferenceJobs(r *http.Request) (any, error) {
	ctx := r.Context()
	var jobs []database.InferenceJob
	if err := s.db.WithContext(ctx).Preload("Groups").Find(&jobs).Error; err != nil {
		slog.Error("error getting inference jobs", "error", err)
		return nil, CodedErrorf(http.StatusInternalServerError, "error retrieving inference job records")
	}
	return jobs, nil
}

func (s *BackendService) SubmitInferenceJob(r *http.Request) (any, error) {
	req, err := ParseRequest[models.InferenceRequest](r)
	if err != nil {
		return nil, err
	}

	if req.SourceS3Bucket == "" || req.DestS3Bucket == "" {
		return nil, CodedErrorf(http.StatusUnprocessableEntity, "missing required fields: model_id, source_s3_bucket, dest_s3_bucket")
	}

	job := database.InferenceJob{
		Id:             uuid.New(),
		ModelId:        req.ModelId,
		SourceS3Bucket: req.SourceS3Bucket,
		SourceS3Prefix: sql.NullString{String: req.SourceS3Prefix, Valid: req.SourceS3Prefix != ""},
		DestS3Bucket:   req.DestS3Bucket,
		Status:         database.JobQueued,
		CreationTime:   time.Now(),
	}

	for name, query := range req.Groups {
		if _, err := core.ParseQuery(query); err != nil {
			return nil, CodedErrorf(http.StatusUnprocessableEntity, "invalid query '%s' for group '%s': %v", query, name, err)
		}

		job.Groups = append(job.Groups, database.Group{
			Id:             uuid.New(),
			Name:           name,
			InferenceJobId: job.Id,
			Query:          query,
		})
	}

	ctx := r.Context()

	var model database.Model
	
	if err := s.db.WithContext(ctx).Transaction(func(txn *gorm.DB) error {
		if err := txn.First(&model, "id = ?", req.ModelId).Error; err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return CodedErrorf(http.StatusNotFound, "model not found")
			}
			slog.Error("error getting model", "error", err)
			return CodedErrorf(http.StatusInternalServerError, "error retrieving model record")
		}

		if model.Status != database.ModelTrained {
			return CodedErrorf(http.StatusUnprocessableEntity, "model is not ready: model has status: %s", model.Status)
		}
		if model.ModelArtifactPath == "" {
			return CodedErrorf(http.StatusInternalServerError, "model artifact path is missing")
		}

		if err := s.db.WithContext(ctx).Create(&job).Error; err != nil {
			slog.Error("error creating inference job", "error", err)
			return CodedErrorf(http.StatusInternalServerError, "failed to create inference job entry")
		}
		return nil
	}); err != nil {
		return nil, err
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
		return models.InferenceResponse{JobId: job.Id}, nil
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

	return models.InferenceResponse{JobId: job.Id}, nil
}

func (s *BackendService) GetInferenceJob(r *http.Request) (any, error) {
	jobId, err := URLParamUUID(r, "job_id")
	if err != nil {
		return nil, err
	}

	ctx := r.Context()

	var job database.InferenceJob
	if err := s.db.WithContext(ctx).First(&job, "id = ?", jobId).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, CodedErrorf(http.StatusNotFound, "inference job not found")
		}
		slog.Error("error getting inference job", "job_id", jobId, "error", err)
		return nil, CodedErrorf(http.StatusInternalServerError, "error retrieving inference job record")
	}

	return job, nil
}

func (s *BackendService) GetInferenceJobGroup(r *http.Request) (any, error) {
	jobId, err := URLParamUUID(r, "job_id")
	if err != nil {
		return nil, err
	}

	groupId, err := URLParamUUID(r, "group_id")
	if err != nil {
		return nil, err
	}

	ctx := r.Context()

	var group database.Group
	if err := s.db.WithContext(ctx).Preload("Objects").First(&group, "inference_job_id = ? AND id = ?", jobId, groupId).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, CodedErrorf(http.StatusNotFound, "inference job group not found")
		}
		slog.Error("error getting inference job group", "job_id", jobId, "group_id", groupId, "error", err)
		return nil, CodedErrorf(http.StatusInternalServerError, "error retrieving inference job group record")
	}

	return group, nil
}

func (s *BackendService) GetInferenceJobEntities(r *http.Request) (any, error) {
	jobId, err := URLParamUUID(r, "job_id")
	if err != nil {
		return nil, err
	}

	params := r.URL.Query()

	var offset, limit int = 0, 100
	if offsetStr := params.Get("offset"); offsetStr != "" {
		if offset, err = strconv.Atoi(offsetStr); err != nil {
			return nil, CodedErrorf(http.StatusBadRequest, "invalid offset value")
		}
	}
	if limitStr := params.Get("limit"); limitStr != "" {
		if limit, err = strconv.Atoi(limitStr); err != nil {
			if limit > 200 {
				return nil, CodedErrorf(http.StatusBadRequest, "limit value must be <= 200")
			}
			return nil, CodedErrorf(http.StatusBadRequest, "invalid limit value")
		}
	}

	ctx := r.Context()

	query := s.db.WithContext(ctx).Offset(offset).Limit(limit).Order(clause.OrderByColumn{
		Column: clause.Column{Table: clause.CurrentTable, Name: clause.PrimaryKey},
	})
	if object := params.Get("object"); object != "" {
		query = query.Where("object = ?", object)
	}

	var entities []database.ObjectEntity
	if err := query.Find(&entities, "inference_job_id = ?", jobId).Error; err != nil {
		slog.Error("error getting job entities", "job_id", jobId, "error", err)
		return nil, CodedErrorf(http.StatusInternalServerError, "error retrieving inference job entities")
	}

	return entities, nil
}
