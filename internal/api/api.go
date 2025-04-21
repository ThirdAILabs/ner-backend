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
	db               *gorm.DB
	publisher        *messaging.TaskPublisher
	s3Client         *s3.Client
	chunkTargetBytes int64
}

func NewBackendService(db *gorm.DB, pub *messaging.TaskPublisher, s3c *s3.Client, chunkTargetBytes int64) *BackendService {
	return &BackendService{db: db, publisher: pub, s3Client: s3c, chunkTargetBytes: chunkTargetBytes}
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

	model, err := database.GetModelByID(ctx, s.db, modelId)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, CodedErrorf(http.StatusNotFound, "model not found")
		}
		slog.Error("error getting model from database", "error", err, "model_id", modelId)
		return nil, CodedErrorf(http.StatusInternalServerError, "error retrieving model record")
	}

	return model, nil
}

func (s *BackendService) ListInferenceJobs(r *http.Request) (any, error) {
	ctx := r.Context()
	var jobs []database.ShardDataTask
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

	// Basic validation
	if req.SourceS3Bucket == "" || req.DestS3Bucket == "" {
		return nil, CodedErrorf(http.StatusUnprocessableEntity, "missing required fields: model_id, source_s3_bucket, dest_s3_bucket")
	}

	job := database.ShardDataTask{
		Id:               uuid.New(),
		ModelId:          req.ModelId,
		SourceS3Bucket:   req.SourceS3Bucket,
		SourceS3Prefix:   sql.NullString{String: req.SourceS3Prefix, Valid: req.SourceS3Prefix != ""},
		DestS3Bucket:     req.DestS3Bucket,
		ChunkTargetBytes: s.chunkTargetBytes,
		CreationTime:     time.Now(),
		Status:           database.JobQueued,
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

	log.Printf("Queueing ShardData task for job %s...", job.Id)

	shardDataPayload := models.ShardDataPayload{
		JobId:             job.Id,
		ModelId:           req.ModelId,
		ModelArtifactPath: model.ModelArtifactPath,
		SourceS3Bucket:    req.SourceS3Bucket,
		SourceS3Prefix:    req.SourceS3Prefix,
		DestS3Bucket:      req.DestS3Bucket,
		ChunkTargetBytes:  s.chunkTargetBytes,
	}

	err = s.publisher.PublishShardDataTask(ctx, shardDataPayload)
	if err != nil {
		log.Printf("CRITICAL: Failed to publish ShardData task for job %s: %v", job.Id, err)
		_ = database.UpdateShardDataTaskStatus(ctx, s.db, job.Id, database.JobFailed)
		return nil, err
	}

	finalStatus := database.JobRunning

	if err := database.UpdateShardDataTaskStatus(ctx, s.db, job.Id, finalStatus); err != nil {
		log.Printf("ERROR: Failed to update job %s status to %s after queueing: %v", job.Id, finalStatus, err)
	}

	log.Printf("Submitted inference job %s.", job.Id)
	return models.InferenceResponse{JobId: job.Id}, nil
}

func (s *BackendService) GetInferenceJob(r *http.Request) (any, error) {
	jobId, err := URLParamUUID(r, "job_id")
	if err != nil {
		return nil, err
	}

	ctx := r.Context()

	// TODO: get statuses of all inference tasks associated with shard data task
	var job database.ShardDataTask
	if err := s.db.WithContext(ctx).Find(&job, "id = ?", jobId).Error; err != nil {
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
		if offset < 0 {
			return nil, CodedErrorf(http.StatusBadRequest, "offset value must be >= 0")
		}
	}
	if limitStr := params.Get("limit"); limitStr != "" {
		if limit, err = strconv.Atoi(limitStr); err != nil {
			if limit > 200 || limit < 0 {
				return nil, CodedErrorf(http.StatusBadRequest, "limit value must be >= 0 and <= 200")
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
