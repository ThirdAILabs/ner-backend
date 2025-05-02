package api

import (
	"context"
	"database/sql"
	"errors"
	"io"
	"log/slog"
	"mime"
	"mime/multipart"
	"ner-backend/internal/core"
	"ner-backend/internal/database"
	"ner-backend/internal/messaging"
	"ner-backend/internal/s3"
	"slices"

	"ner-backend/pkg/api"
	"net/http"
	"path/filepath"
	"strconv"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

type BackendService struct {
	db               *gorm.DB
	s3               *s3.Client
	publisher        messaging.Publisher
	chunkTargetBytes int64
}

const (
	uploadBucket = "uploads"
)

func NewBackendService(db *gorm.DB, s3 *s3.Client, pub messaging.Publisher, chunkTargetBytes int64) *BackendService {
	if err := s3.CreateBucket(context.Background(), uploadBucket); err != nil {
		slog.Error("error creating upload bucket", "error", err)
		panic("failed to create upload bucket")
	}

	return &BackendService{db: db, s3: s3, publisher: pub, chunkTargetBytes: chunkTargetBytes}
}

func (s *BackendService) AddRoutes(r chi.Router) {
	r.Get("/health", RestHandler(func(r *http.Request) (any, error) { return nil, nil }))
	r.Route("/models", func(r chi.Router) {
		r.Get("/", RestHandler(s.ListModels))
		r.Get("/{model_id}", RestHandler(s.GetModel))
		r.Post("/{model_id}/finetune", RestHandler(s.FinetuneModel))
	})
	r.Route("/reports", func(r chi.Router) {
		r.Get("/", RestHandler(s.ListReports))
		r.Post("/", RestHandler(s.CreateReport))
		r.Get("/{report_id}", RestHandler(s.GetReport))
		r.Get("/{report_id}/groups/{group_id}", RestHandler(s.GetReportGroup))
		r.Get("/{report_id}/entities", RestHandler(s.GetReportEntities))
		r.Get("/{report_id}/search", RestHandler(s.ReportSearch))
	})

	r.Route("/uploads", func(r chi.Router) {
		r.Post("/", RestHandler(s.UploadFiles))
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
	if err := s.db.WithContext(ctx).Preload("Tags").First(&model, "id = ?", modelId).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, CodedErrorf(http.StatusNotFound, "model not found")
		}
		slog.Error("error getting model from database", "error", err, "model_id", modelId)
		return nil, CodedErrorf(http.StatusInternalServerError, "error retrieving model record")
	}

	return convertModel(model), nil
}

func (s *BackendService) FinetuneModel(r *http.Request) (any, error) {
	modelId, err := URLParamUUID(r, "model_id")
	if err != nil {
		return nil, err
	}
	req, err := ParseRequest[api.FinetuneRequest](r)
	if err != nil {
		return nil, err
	}

	if req.Name == "" {
		return nil, CodedErrorf(http.StatusUnprocessableEntity, "the following fields are required: Name")
	}

	ctx := r.Context()

	var model database.Model

	if err := s.db.WithContext(ctx).Transaction(func(txn *gorm.DB) error {
		var baseModel database.Model
		if err := txn.First(&baseModel, "id = ?", modelId).Error; err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return CodedErrorf(http.StatusNotFound, "base model not found")
			}
			slog.Error("error getting base model", "error", err)
			return CodedErrorf(http.StatusInternalServerError, "error retrieving base model record")
		}

		if baseModel.Status != database.ModelTrained {
			return CodedErrorf(http.StatusUnprocessableEntity, "base model is not ready for finetuning: model has status: %s", baseModel.Status)
		}

		model = database.Model{
			Id:           uuid.New(),
			BaseModelId:  uuid.NullUUID{UUID: baseModel.Id, Valid: true},
			Name:         req.Name,
			Type:         baseModel.Type,
			Status:       database.ModelQueued,
			CreationTime: time.Now().UTC(),
		}

		if err := txn.WithContext(ctx).Create(&model).Error; err != nil {
			slog.Error("error creating model entry", "error", err)
			return CodedErrorf(http.StatusInternalServerError, "failed to create model entry")
		}

		return nil
	}); err != nil {
		return nil, err
	}

	if err := s.publisher.PublishFinetuneTask(ctx, messaging.FinetuneTaskPayload{
		ModelId:     model.Id,
		BaseModelId: model.BaseModelId.UUID,
		TaskPrompt:  req.TaskPrompt,
		Tags:        req.Tags,
		Samples:     req.Samples,
	}); err != nil {
		slog.Error("error queueing finetune task", "error", err)
		_ = database.UpdateModelStatus(ctx, s.db, model.Id, database.ModelFailed)
		return nil, CodedErrorf(http.StatusInternalServerError, "failed to queue finetune task")
	}

	slog.Info("created model", "model_id", model.Id, "base_model_id", model.BaseModelId, "name", model.Name)

	return api.FinetuneResponse{ModelId: model.Id}, nil
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

	if req.ModelId == uuid.Nil {
		return nil, CodedErrorf(http.StatusUnprocessableEntity, "the following fields are required: ModelId")
	}

	sourceS3Bucket, s3Prefix := req.SourceS3Bucket, req.SourceS3Prefix

	if req.UploadId != uuid.Nil {
		sourceS3Bucket = uploadBucket
		s3Prefix = req.UploadId.String()
	}

	if sourceS3Bucket == "" {
		return nil, CodedErrorf(http.StatusUnprocessableEntity, "the following fields are required: SourceS3Bucket or UploadId")
	}

	report := database.Report{
		Id:             uuid.New(),
		ModelId:        req.ModelId,
		SourceS3Bucket: sourceS3Bucket,
		SourceS3Prefix: sql.NullString{String: s3Prefix, Valid: s3Prefix != ""},
		CreationTime:   time.Now().UTC(),
		TagCount:       []byte{},
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
		if err := txn.Preload("Tags").First(&model, "id = ?", req.ModelId).Error; err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return CodedErrorf(http.StatusNotFound, "model not found")
			}
			slog.Error("error getting model", "error", err)
			return CodedErrorf(http.StatusInternalServerError, "error retrieving model record")
		}

		if model.Status != database.ModelTrained {
			return CodedErrorf(http.StatusUnprocessableEntity, "model is not ready: model has status: %s", model.Status)
		}

		modelTags := make([]string, 0)
		for _, tag := range model.Tags {
			modelTags = append(modelTags, tag.Name)
		}

		erroneousTags := make([]string, 0)
		for _, tag := range req.Tags {
			if !slices.Contains(modelTags, tag) {
				erroneousTags = append(erroneousTags, tag)
			}
		}

		if len(erroneousTags) > 0 {
			return CodedErrorf(http.StatusUnprocessableEntity, "model does not have the following tags: %s", erroneousTags)
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

func (s *BackendService) ReportSearch(r *http.Request) (any, error) {
	reportId, err := URLParamUUID(r, "report_id")
	if err != nil {
		return nil, err
	}

	queryStr := r.URL.Query().Get("query")
	if queryStr == "" {
		return nil, CodedErrorf(http.StatusBadRequest, "query parameter is required")
	}

	filter, err := core.ToSql(s.db, queryStr)
	if err != nil {
	}

	ctx := r.Context()

	var objects []string
	if err := s.db.WithContext(ctx).Model(&database.ObjectEntity{}).Distinct("object").Where("report_id = ?", reportId).Where(filter).Find(&objects).Error; err != nil {
		slog.Error("error getting job entities", "report_id", reportId, "error", err)
		return nil, CodedErrorf(http.StatusInternalServerError, "error retrieving inference job entities")
	}

	return api.SearchResponse{Objects: objects}, nil
}

func getMultipartBoundary(r *http.Request) (string, error) {
	contentType := r.Header.Get("Content-Type")
	if contentType == "" {
		return "", CodedErrorf(http.StatusBadRequest, "missing 'Content-Type' header")
	}
	mediaType, params, err := mime.ParseMediaType(contentType)
	if err != nil {
		return "", CodedErrorf(http.StatusBadRequest, "error parsing media type in request: %w", err)
	}
	if mediaType != "multipart/form-data" {
		return "", CodedErrorf(http.StatusBadRequest, "expected media type to be 'multipart/form-data'")
	}

	boundary, ok := params["boundary"]
	if !ok {
		return "", CodedErrorf(http.StatusBadRequest, "missing 'boundary' parameter in 'Content-Type' header")
	}

	return boundary, nil
}

func (s *BackendService) UploadFiles(r *http.Request) (any, error) {
	boundary, err := getMultipartBoundary(r)
	if err != nil {
		return nil, err
	}

	uploadId := uuid.New()

	reader := multipart.NewReader(r.Body, boundary)

	var filenames []string

	for {
		part, err := reader.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, CodedErrorf(http.StatusBadRequest, "error parsing multipart request: %v", err)
		}
		defer part.Close()

		if part.FormName() == "files" {
			if part.FileName() == "" {
				return nil, CodedErrorf(http.StatusUnprocessableEntity, "invalid filename detected in upload files: filename cannot be empty")
			}

			filenames = append(filenames, part.FileName())

			newFilepath := filepath.Join(uploadId.String(), part.FileName())

			if _, err := s.s3.UploadObject(r.Context(), uploadBucket, newFilepath, part); err != nil {
				slog.Error("error uploading file to S3", "error", err)
				return nil, CodedErrorf(http.StatusInternalServerError, "error saving file")
			}
		}
	}

	slog.Info("created upload record", "upload_id", uploadId, "filenames", filenames)

	return api.UploadResponse{Id: uploadId}, nil
}
