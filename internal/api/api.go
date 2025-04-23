package api

import (
	"database/sql"
	"errors" // Adjust import path
	"log/slog"
	"ner-backend/internal/core"
	"ner-backend/internal/database"
	"ner-backend/internal/messaging"
	"ner-backend/pkg/api"
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
	publisher        messaging.Publisher
	chunkTargetBytes int64
}

func NewBackendService(db *gorm.DB, pub messaging.Publisher, chunkTargetBytes int64) *BackendService {
	return &BackendService{db: db, publisher: pub, chunkTargetBytes: chunkTargetBytes}
}

func (s *BackendService) AddRoutes(r chi.Router) {
	r.Get("/health", RestHandler(func(r *http.Request) (any, error) { return nil, nil }))
	r.Route("/models", func(r chi.Router) {
		r.Get("/", RestHandler(s.ListModels))
		r.Get("/{model_id}", RestHandler(s.GetModel))
	})
	r.Route("/reports", func(r chi.Router) {
		r.Get("/", RestHandler(s.ListReports))
		r.Post("/", RestHandler(s.CreateReport))
		r.Get("/{report_id}", RestHandler(s.GetReport))
		r.Get("/{report_id}/groups/{group_id}", RestHandler(s.GetReportGroup))
		r.Get("/{report_id}/entities", RestHandler(s.GetReportEntities))
	})
}

func (s *BackendService) ListModels(r *http.Request) (any, error) {
	ctx := r.Context()
	var models []database.Model
	if err := s.db.WithContext(ctx).Find(&models).Error; err != nil {
		slog.Error("error getting models", "error", err)
		return nil, CodedErrorf(http.StatusInternalServerError, "error retrieving model records")
	}

	return convertModels(models), nil
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
		slog.Error("error getting model from database", "error", err, "model_id", modelId)
		return nil, CodedErrorf(http.StatusInternalServerError, "error retrieving model record")
	}

	return convertModel(model), nil
}

func (s *BackendService) ListReports(r *http.Request) (any, error) {
	ctx := r.Context()
	var reports []database.Report
	if err := s.db.WithContext(ctx).Preload("Model").Preload("Groups").Find(&reports).Error; err != nil {
		slog.Error("error getting inference jobs", "error", err)
		return nil, CodedErrorf(http.StatusInternalServerError, "error retrieving inference job records")
	}

	return convertReports(reports), nil
}

func (s *BackendService) CreateReport(r *http.Request) (any, error) {
	req, err := ParseRequest[api.CreateReportRequest](r)
	if err != nil {
		return nil, err
	}

	if req.ModelId == uuid.Nil || req.SourceS3Bucket == "" {
		return nil, CodedErrorf(http.StatusUnprocessableEntity, "the following fields are required: ModelId, SourceS3Bucket")
	}

	report := database.Report{
		Id:             uuid.New(),
		ModelId:        req.ModelId,
		SourceS3Bucket: req.SourceS3Bucket,
		SourceS3Prefix: sql.NullString{String: req.SourceS3Prefix, Valid: req.SourceS3Prefix != ""},
		CreationTime:   time.Now().UTC(),
	}

	for name, query := range req.Groups {
		if _, err := core.ParseQuery(query); err != nil {
			return nil, CodedErrorf(http.StatusUnprocessableEntity, "invalid query '%s' for group '%s': %v", query, name, err)
		}

		report.Groups = append(report.Groups, database.Group{
			Id:       uuid.New(),
			Name:     name,
			ReportId: report.Id,
			Query:    query,
		})
	}

	task := database.ShardDataTask{
		ReportId:         report.Id,
		Status:           database.JobQueued,
		CreationTime:     time.Now().UTC(),
		ChunkTargetBytes: s.chunkTargetBytes,
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

		if err := txn.WithContext(ctx).Create(&report).Error; err != nil {
			slog.Error("error creating report entry", "error", err)
			return CodedErrorf(http.StatusInternalServerError, "failed to create report entry")
		}

		err = s.publisher.PublishShardDataTask(ctx, messaging.ShardDataPayload{ReportId: report.Id})
		if err != nil {
			slog.Error("error queueing shard data task", "error", err)
			_ = database.UpdateShardDataTaskStatus(ctx, txn, report.Id, database.JobFailed)
			return CodedErrorf(http.StatusInternalServerError, "failed to queue shard data task")
		}

		if err := txn.WithContext(ctx).Create(&task).Error; err != nil {
			slog.Error("error creating shard data task", "error", err)
			return CodedErrorf(http.StatusInternalServerError, "failed to create shard data task")
		}

		return nil
	}); err != nil {
		return nil, err
	}

	slog.Info("created report", "report_id", report.Id)
	return api.CreateReportResponse{ReportId: report.Id}, nil
}

func (s *BackendService) GetReport(r *http.Request) (any, error) {
	reportId, err := URLParamUUID(r, "report_id")
	if err != nil {
		return nil, err
	}

	ctx := r.Context()

	var report database.Report
	if err := s.db.WithContext(ctx).Preload("Model").Preload("Groups").Preload("ShardDataTask").Find(&report, "id = ?", reportId).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, CodedErrorf(http.StatusNotFound, "inference job not found")
		}
		slog.Error("error getting inference job", "report_id", reportId, "error", err)
		return nil, CodedErrorf(http.StatusInternalServerError, "error retrieving report data")
	}

	type taskStatusCategory struct {
		Status string
		Count  int
		Total  int
	}

	var statusCategories []taskStatusCategory

	if err := s.db.WithContext(ctx).Model(&database.InferenceTask{}).
		Where("report_id = ?", reportId).
		Select("status, COUNT(*) as count, sum(total_size) as total").
		Group("status").
		Find(&statusCategories).Error; err != nil {
		slog.Error("error getting inference job task statuses", "report_id", reportId, "error", err)
		return nil, CodedErrorf(http.StatusInternalServerError, "error retrieving inference task statuses")
	}

	apiReport := convertReport(report)
	apiReport.InferenceTaskStatuses = make(map[string]api.TaskStatusCategory)
	for _, category := range statusCategories {
		apiReport.InferenceTaskStatuses[category.Status] = api.TaskStatusCategory{
			TotalTasks: category.Count,
			TotalSize:  category.Total,
		}
	}

	return apiReport, nil
}

func (s *BackendService) GetReportGroup(r *http.Request) (any, error) {
	reportId, err := URLParamUUID(r, "report_id")
	if err != nil {
		return nil, err
	}

	groupId, err := URLParamUUID(r, "group_id")
	if err != nil {
		return nil, err
	}

	ctx := r.Context()

	var group database.Group
	if err := s.db.WithContext(ctx).Preload("Objects").First(&group, "report_id = ? AND id = ?", reportId, groupId).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, CodedErrorf(http.StatusNotFound, "inference job group not found")
		}
		slog.Error("error getting inference job group", "report_id", reportId, "group_id", groupId, "error", err)
		return nil, CodedErrorf(http.StatusInternalServerError, "error retrieving inference job group record")
	}

	return convertGroup(group), nil
}

func (s *BackendService) GetReportEntities(r *http.Request) (any, error) {
	reportId, err := URLParamUUID(r, "report_id")
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
	if err := query.Find(&entities, "report_id = ?", reportId).Error; err != nil {
		slog.Error("error getting job entities", "report_id", reportId, "error", err)
		return nil, CodedErrorf(http.StatusInternalServerError, "error retrieving inference job entities")
	}

	return convertEntities(entities), nil
}
