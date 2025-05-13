package api

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"mime"
	"mime/multipart"
	"ner-backend/internal/core"
	"ner-backend/internal/database"
	"ner-backend/internal/messaging"
	"ner-backend/internal/storage"
	"regexp"

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
	storage          storage.Provider
	publisher        messaging.Publisher
	chunkTargetBytes int64
}

const (
	uploadBucket = "uploads"
	ErrCodeDB    = 1001 // Custom internal code for DB errors
)

func NewBackendService(db *gorm.DB, storage storage.Provider, pub messaging.Publisher, chunkTargetBytes int64) *BackendService {
	if err := storage.CreateBucket(context.Background(), uploadBucket); err != nil {
		slog.Error("error creating upload bucket", "error", err)
		panic("failed to create upload bucket")
	}

	return &BackendService{db: db, storage: storage, publisher: pub, chunkTargetBytes: chunkTargetBytes}
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
		r.Post("/{report_id}/stop", RestHandler(s.StopReport))
		r.Delete("/{report_id}", RestHandler(s.DeleteReport))
		r.Get("/{report_id}/groups/{group_id}", RestHandler(s.GetReportGroup))
		r.Get("/{report_id}/entities", RestHandler(s.GetReportEntities))
		r.Get("/{report_id}/search", RestHandler(s.ReportSearch))
		r.Get("/{report_id}/objects", RestHandler(s.GetReportPreviews))
	})

	r.Route("/uploads", func(r chi.Router) {
		r.Post("/", RestHandler(s.UploadFiles))
	})

	r.Route("/metrics", func(r chi.Router) {
		r.Get("/", RestHandler(s.GetInferenceMetrics))
		r.Get("/throughput", RestHandler(s.GetThroughputMetrics))
	})

	r.Route("/validate-group-definition", func(r chi.Router) {
		r.Get("/", RestHandler(s.ValidateGroupDefinition))
	})
}

func (s *BackendService) ListModels(r *http.Request) (any, error) {
	ctx := r.Context()
	var models []database.Model
	if err := s.db.WithContext(ctx).Find(&models).Error; err != nil {
		slog.Error("error getting models", "error", err)
		return nil, CodedErrorf(ErrCodeDB, "error retrieving model records")
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
		return nil, CodedErrorf(ErrCodeDB, "error retrieving model record")
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
			return CodedErrorf(ErrCodeDB, "error retrieving base model record")
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
			return CodedErrorf(ErrCodeDB, "failed to create model entry")
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
	if err := s.db.WithContext(ctx).Preload("Model").Preload("Groups").Where("deleted = ?", false).Find(&reports).Error; err != nil {
		slog.Error("error getting reports", "error", err)
		return nil, CodedErrorf(ErrCodeDB, "error retrieving report records")
	}

	return convertReports(reports), nil
}

func (s *BackendService) CreateReport(r *http.Request) (any, error) {
	req, err := ParseRequest[api.CreateReportRequest](r)
	if err != nil {
		return nil, err
	}

	if req.ModelId == uuid.Nil {
		var model database.Model
		result := s.db.WithContext(r.Context()).Limit(1).Find(&model, "name = ?", "basic")
		if result.Error != nil {
			slog.Error("error getting default model", "error", result.Error)
			return nil, CodedErrorf(ErrCodeDB, "error getting default model")
		}

		if result.RowsAffected != 0 {
			req.ModelId = model.Id
		} else {
			return nil, CodedErrorf(http.StatusUnprocessableEntity, "the following fields are required: ModelId")
		}
	}

	var (
		s3Endpoint     = req.S3Endpoint
		s3Region       = req.S3Region
		sourceS3Bucket = req.SourceS3Bucket
		s3Prefix       = req.SourceS3Prefix
		isUpload       = false
	)

	if req.UploadId != uuid.Nil {
		s3Endpoint = ""
		s3Region = ""
		sourceS3Bucket = uploadBucket
		s3Prefix = req.UploadId.String()
		isUpload = true
	}

	if sourceS3Bucket == "" {
		return nil, CodedErrorf(http.StatusUnprocessableEntity, "the following fields are required: SourceS3Bucket or UploadId")
	}

	if err := validateReportName(req.ReportName); err != nil {
		return nil, err
	}

	for _, tag := range req.Tags {
		if _, ok := req.CustomTags[tag]; ok {
			return nil, CodedErrorf(http.StatusUnprocessableEntity, "tag '%s' cannot be used as a regular and custom tag", tag)
		}
	}

	for tag, pattern := range req.CustomTags {
		if _, err := regexp.Compile(pattern); err != nil {
			return nil, CodedErrorf(http.StatusUnprocessableEntity, "invalid regex pattern '%s' for custom tag '%s': %v", pattern, tag, err)
		}
	}

	report := database.Report{
		Id:             uuid.New(),
		ReportName:     req.ReportName,
		ModelId:        req.ModelId,
		S3Endpoint:     sql.NullString{String: s3Endpoint, Valid: s3Endpoint != ""},
		S3Region:       sql.NullString{String: s3Region, Valid: s3Region != ""},
		SourceS3Bucket: sourceS3Bucket,
		SourceS3Prefix: sql.NullString{String: s3Prefix, Valid: s3Prefix != ""},
		IsUpload:       isUpload,
		CreationTime:   time.Now().UTC(),
	}

	for _, tag := range req.Tags {
		report.Tags = append(report.Tags, database.ReportTag{
			ReportId: report.Id,
			Tag:      tag,
		})
	}

	for tag, pattern := range req.CustomTags {
		report.CustomTags = append(report.CustomTags, database.CustomTag{
			ReportId: report.Id,
			Tag:      tag,
			Pattern:  pattern,
		})
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
		// check if duplicate report name
		var existingReport database.Report
		result := txn.Where("report_name = ?", req.ReportName).Limit(1).Find(&existingReport)
		if result.Error != nil {
			slog.Error("error checking for duplicate report name", "error", result.Error)
			return CodedErrorf(ErrCodeDB, "error checking for duplicate report name")
		} else if result.RowsAffected > 0 {
			return CodedErrorf(http.StatusUnprocessableEntity, "report name '%s' already exists", req.ReportName)
		}

		if err := txn.Preload("Tags").First(&model, "id = ?", req.ModelId).Error; err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return CodedErrorf(http.StatusNotFound, "model not found")
			}
			slog.Error("error getting model", "error", err)
			return CodedErrorf(ErrCodeDB, "error retrieving model record")
		}

		if model.Status != database.ModelTrained {
			return CodedErrorf(http.StatusUnprocessableEntity, "model is not ready: model has status: %s", model.Status)
		}

		modelTags := make(map[string]struct{})
		for _, tag := range model.Tags {
			modelTags[tag.Tag] = struct{}{}
		}

		for _, tag := range req.Tags {
			if _, ok := modelTags[tag]; !ok {
				return CodedErrorf(http.StatusUnprocessableEntity, "model does not support tag '%s', either switch models, or add a custom tag", tag)
			}
		}

		if err := txn.WithContext(ctx).Create(&report).Error; err != nil {
			slog.Error("error creating report entry", "error", err)
			return CodedErrorf(ErrCodeDB, "failed to create report entry")
		}

		if err := txn.WithContext(ctx).Create(&task).Error; err != nil {
			slog.Error("error creating shard data task", "error", err)
			return CodedErrorf(ErrCodeDB, "failed to create shard data task")
		}
		return nil
	}); err != nil {
		return nil, err
	}

	err = s.publisher.PublishShardDataTask(ctx, messaging.ShardDataPayload{ReportId: report.Id})
	if err != nil {
		slog.Error("error queueing shard data task", "error", err)
		_ = database.UpdateShardDataTaskStatus(ctx, s.db, report.Id, database.JobFailed)
		return nil, CodedErrorf(http.StatusInternalServerError, "failed to queue shard data task")
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
	if err := s.db.WithContext(ctx).Preload("Model").Preload("Tags").Preload("CustomTags").Preload("Groups").Preload("ShardDataTask").Preload("Errors").Find(&report, "id = ?", reportId).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, CodedErrorf(http.StatusNotFound, "report not found")
		}
		slog.Error("error getting report", "report_id", reportId, "error", err)
		return nil, CodedErrorf(ErrCodeDB, "error retrieving report data")
	}

	if report.Deleted {
		return nil, CodedErrorf(http.StatusNotFound, "report not found")
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
		slog.Error("error getting inference task statuses", "report_id", reportId, "error", err)
		return nil, CodedErrorf(ErrCodeDB, "error retrieving inference task statuses")
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

func (s *BackendService) DeleteReport(r *http.Request) (any, error) {
	reportId, err := URLParamUUID(r, "report_id")
	if err != nil {
		return nil, err
	}

	ctx := r.Context()

	if err := s.db.WithContext(ctx).Transaction(func(txn *gorm.DB) error {
		result := txn.Model(&database.Report{Id: reportId}).Update("deleted", true)
		if err := result.Error; err != nil {
			slog.Error("error deleting report", "report_id", reportId, "error", err)
			return CodedErrorf(ErrCodeDB, "error deleting report")
		}

		if result.RowsAffected == 0 {
			return CodedErrorf(http.StatusNotFound, "report not found")
		}

		// Delete the report data that takes up the most space
		if err := txn.Delete(&database.ObjectGroup{}, "report_id = ?", reportId).Error; err != nil {
			slog.Error("error deleting report groups", "report_id", reportId, "error", err)
			return CodedErrorf(ErrCodeDB, "error deleting report")
		}

		if err := txn.Delete(&database.ObjectEntity{}, "report_id = ?", reportId).Error; err != nil {
			slog.Error("error deleting report entities", "report_id", reportId, "error", err)
			return CodedErrorf(ErrCodeDB, "error deleting report")
		}

		if err := txn.Delete(&database.ObjectPreview{}, "report_id = ?", reportId).Error; err != nil {
			slog.Error("error deleting report entities", "report_id", reportId, "error", err)
			return CodedErrorf(ErrCodeDB, "error deleting report")
		}

		return nil
	}); err != nil {
		return nil, err
	}

	slog.Info("deleted report", "report_id", reportId)

	return nil, nil
}

func (s *BackendService) StopReport(r *http.Request) (any, error) {
	reportId, err := URLParamUUID(r, "report_id")
	if err != nil {
		return nil, err
	}

	ctx := r.Context()

	if err := s.db.WithContext(ctx).Transaction(func(txn *gorm.DB) error {
		result := txn.Model(&database.Report{Id: reportId}).Update("stopped", true)
		if err := result.Error; err != nil {
			slog.Error("error stopping report", "report_id", reportId, "error", err)
			return CodedErrorf(ErrCodeDB, "error deleting report")
		}

		if result.RowsAffected == 0 {
			return CodedErrorf(http.StatusNotFound, "report not found")
		}
		return nil
	}); err != nil {
		return nil, err
	}

	slog.Info("stopped report", "report_id", reportId)

	return nil, nil
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
			return nil, CodedErrorf(http.StatusNotFound, "report group not found")
		}
		slog.Error("error getting report group", "report_id", reportId, "group_id", groupId, "error", err)
		return nil, CodedErrorf(ErrCodeDB, "error retrieving report group record")
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

	if tags := params["tags"]; len(tags) > 0 {
		query = query.Where("label IN ?", tags)
	}

	var entities []database.ObjectEntity
	if err := query.Find(&entities, "report_id = ?", reportId).Error; err != nil {
		slog.Error("error getting job entities", "report_id", reportId, "error", err)
		return nil, CodedErrorf(ErrCodeDB, "error retrieving report entities")
	}

	return convertEntities(entities), nil
}

func (s *BackendService) GetReportPreviews(r *http.Request) (any, error) {
	reportId, err := URLParamUUID(r, "report_id")
	if err != nil {
		return nil, err
	}

	params := r.URL.Query()
	offset, limit := 0, 100
	if os := params.Get("offset"); os != "" {
		if offset, err = strconv.Atoi(os); err != nil || offset < 0 {
			return nil, CodedErrorf(http.StatusBadRequest, "invalid offset")
		}
	}
	if ls := params.Get("limit"); ls != "" {
		if limit, err = strconv.Atoi(ls); err != nil || limit < 1 || limit > 200 {
			return nil, CodedErrorf(http.StatusBadRequest, "invalid limit")
		}
	}

	ctx := r.Context()
	var previews []database.ObjectPreview
	if err := s.db.WithContext(ctx).
		Where("report_id = ?", reportId).
		Offset(offset).
		Limit(limit).
		Order("object").
		Find(&previews).Error; err != nil {
		slog.Error("error fetching previews", "report_id", reportId, "err", err)
		return nil, CodedErrorf(ErrCodeDB, "error retrieving previews")
	}

	resp := make([]api.ObjectPreviewResponse, len(previews))
	for i, p := range previews {
		var payload struct {
			Tokens []string `json:"tokens"`
			Tags   []string `json:"tags"`
		}
		if err := json.Unmarshal(p.TokenTags, &payload); err != nil {
			slog.Error("unmarshal preview payload", "object", p.Object, "err", err)
		}
		resp[i] = api.ObjectPreviewResponse{
			Object: p.Object,
			Tokens: payload.Tokens,
			Tags:   payload.Tags,
		}
	}

	return resp, nil
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
		return nil, CodedErrorf(http.StatusUnprocessableEntity, "error parsing query: %v", err)
	}

	ctx := r.Context()

	var objects []string
	if err := s.db.WithContext(ctx).Model(&database.ObjectEntity{}).Distinct("object").Where("report_id = ?", reportId).Where(filter).Find(&objects).Error; err != nil {
		slog.Error("error getting job objects by search", "report_id", reportId, "error", err)
		return nil, CodedErrorf(ErrCodeDB, "error retrieving report objects")
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

			if err := s.storage.PutObject(r.Context(), uploadBucket, newFilepath, part); err != nil {
				slog.Error("error uploading file to S3", "error", err)
				return nil, CodedErrorf(http.StatusInternalServerError, "error saving file")
			}
		}
	}

	slog.Info("created upload record", "upload_id", uploadId, "filenames", filenames)

	return api.UploadResponse{Id: uploadId}, nil
}

func (s *BackendService) GetInferenceMetrics(r *http.Request) (any, error) {
	qs := r.URL.Query()

	days := 7
	if ds := qs.Get("days"); ds != "" {
		d, err := strconv.Atoi(ds)
		if err != nil || d < 0 {
			return nil, CodedErrorf(http.StatusBadRequest, "invalid days parameter")
		}
		days = d
	}

	var modelID uuid.UUID
	if ms := qs.Get("model_id"); ms != "" {
		u, err := uuid.Parse(ms)
		if err != nil {
			return nil, CodedErrorf(http.StatusBadRequest, "invalid model_id")
		}
		modelID = u
	}

	since := time.Now().UTC().Add(-time.Duration(days) * 24 * time.Hour)

	type statusMetrics struct {
		Count  int64         `gorm:"column:count"`
		Size   sql.NullInt64 `gorm:"column:size"`
		Tokens sql.NullInt64 `gorm:"column:tokens"`
	}

	var completed, running statusMetrics

	q1 := s.db.Model(&database.InferenceTask{}).
		Select("COUNT(*) AS count, COALESCE(SUM(total_size),0) AS size, COALESCE(SUM(token_count),0) AS tokens").
		Where("inference_tasks.status = ? AND inference_tasks.completion_time >= ?", database.JobCompleted, since)
	if modelID != uuid.Nil {
		q1 = q1.
			Joins("JOIN reports ON reports.id = inference_tasks.report_id").
			Where("reports.model_id = ?", modelID)
	}
	if err := q1.Scan(&completed).Error; err != nil {
		slog.Error("error fetching completed metrics", "error", err)
		return nil, CodedErrorf(ErrCodeDB, "error retrieving metrics")
	}

	q2 := s.db.Model(&database.InferenceTask{}).
		Select("COUNT(*) AS count, COALESCE(SUM(total_size),0) AS size, COALESCE(SUM(token_count),0) AS tokens").
		Where("inference_tasks.status = ? AND inference_tasks.creation_time >= ?", database.JobRunning, since)
	if modelID != uuid.Nil {
		q2 = q2.
			Joins("JOIN reports ON reports.id = inference_tasks.report_id").
			Where("reports.model_id = ?", modelID)
	}
	if err := q2.Scan(&running).Error; err != nil {
		slog.Error("error fetching in-progress metrics", "error", err)
		return nil, CodedErrorf(ErrCodeDB, "error retrieving metrics")
	}

	totalBytes := completed.Size.Int64 + running.Size.Int64
	totalTokens := completed.Tokens.Int64 + running.Tokens.Int64
	dataProcessedMB := float64(totalBytes) / (1024 * 1024)

	return api.InferenceMetricsResponse{
		Completed:       completed.Count,
		InProgress:      running.Count,
		DataProcessedMB: dataProcessedMB,
		TokensProcessed: totalTokens,
	}, nil
}

func (s *BackendService) GetThroughputMetrics(r *http.Request) (any, error) {
	qs := r.URL.Query()

	ms := qs.Get("model_id")
	if ms == "" {
		return nil, CodedErrorf(http.StatusBadRequest, "model_id query param is required")
	}
	modelID, err := uuid.Parse(ms)
	if err != nil {
		return nil, CodedErrorf(http.StatusBadRequest, "invalid model_id")
	}

	var reportID uuid.UUID
	if rs := qs.Get("report_id"); rs != "" {
		rid, err := uuid.Parse(rs)
		if err != nil {
			return nil, CodedErrorf(http.StatusBadRequest, "invalid report_id")
		}
		reportID = rid
	}

	var rows []struct {
		TotalSize      int64        `gorm:"column:total_size"`
		StartTime      sql.NullTime `gorm:"column:start_time"`
		CompletionTime sql.NullTime `gorm:"column:completion_time"`
	}
	q := s.db.Model(&database.InferenceTask{}).
		Joins("JOIN reports ON reports.id = inference_tasks.report_id").
		Where("inference_tasks.status = ?", database.JobCompleted).
		Where("reports.model_id = ?", modelID)
	if reportID != uuid.Nil {
		q = q.Where("inference_tasks.report_id = ?", reportID)
	}
	if err := q.
		Select("inference_tasks.total_size, inference_tasks.start_time, inference_tasks.completion_time").
		Scan(&rows).Error; err != nil {
		slog.Error("error fetching throughput rows", "error", err)
		return nil, CodedErrorf(ErrCodeDB, "error retrieving throughput")
	}

	var totalBytes int64
	var totalSeconds float64
	for _, r := range rows {
		totalBytes += r.TotalSize
		if r.StartTime.Valid && r.CompletionTime.Valid {
			totalSeconds += r.CompletionTime.Time.Sub(r.StartTime.Time).Seconds()
		}
	}

	mb := float64(totalBytes) / (1024.0 * 1024.0)
	var throughputMBPerHour float64
	if totalSeconds > 0 {
		throughputMBPerHour = mb / (totalSeconds / 3600.0)
	}

	return api.ThroughputResponse{
		ModelID:             modelID,
		ReportID:            reportID,
		ThroughputMBPerHour: throughputMBPerHour,
	}, nil
}

func (s *BackendService) ValidateGroupDefinition(r *http.Request) (any, error) {
	groupQuery := r.URL.Query().Get("group_query")
	if groupQuery == "" {
		return nil, CodedErrorf(http.StatusBadRequest, "group query is required")
	}
	if _, err := core.ParseQuery(groupQuery); err != nil {
		return nil, CodedErrorf(http.StatusUnprocessableEntity, "invalid query '%s': %v", groupQuery, err)
	}
	return nil, nil
}
