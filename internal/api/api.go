package api

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime"
	"mime/multipart"
	"ner-backend/internal/core"
	"ner-backend/internal/database"
	"ner-backend/internal/licensing"
	"ner-backend/internal/messaging"
	"ner-backend/internal/storage"
	"regexp"
	"slices"
	"strings"

	"ner-backend/pkg/api"
	"net/http"
	"path/filepath"
	"strconv"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"gorm.io/datatypes"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

type BackendService struct {
	db               *gorm.DB
	storage          storage.ObjectStore
	uploadBucket     string
	publisher        messaging.Publisher
	chunkTargetBytes int64
	licensing        licensing.LicenseVerifier
	enterpriseMode   bool
}

const (
	ErrCodeDB = 1001 // Custom internal code for DB errors
)

func NewBackendService(db *gorm.DB, storage storage.ObjectStore, uploadBucket string, pub messaging.Publisher, chunkTargetBytes int64, licenseVerifier licensing.LicenseVerifier, enterpriseMode bool) *BackendService {

	return &BackendService{db: db, storage: storage, uploadBucket: uploadBucket, publisher: pub, chunkTargetBytes: chunkTargetBytes, licensing: licenseVerifier, enterpriseMode: enterpriseMode}
}

func (s *BackendService) AddRoutes(r chi.Router) {
	r.Get("/health", RestHandler(func(r *http.Request) (any, error) { return nil, nil }))
	r.Route("/models", func(r chi.Router) {
		r.Get("/", RestHandler(s.ListModels))
		r.Get("/{model_id}", RestHandler(s.GetModel))
		r.Post("/{model_id}/finetune", RestHandler(s.FinetuneModel))
		r.Post("/{model_id}/feedback", RestHandler(s.StoreModelFeedback))
		r.Delete("/{model_id}/feedback/{feedback_id}", RestHandler(s.DeleteModelFeedback))
		r.Get("/{model_id}/feedback", RestHandler(s.ListModelFeedback))
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

	r.Route("/validate", func(r chi.Router) {
		r.Get("/group", RestHandler(s.ValidateGroupDefinition))
		r.Get("/s3", RestHandler(s.ValidateS3Access))
	})

	r.Route("/file-name-to-path", func(r chi.Router) {
		r.Post("/{upload_id}", RestHandler(s.StoreFileNameToPath))
		r.Get("/{upload_id}", RestHandler(s.GetFileNameToPath))
	})

	r.Get("/license", RestHandler(s.GetLicense))

	r.Get("/enterprise", RestHandler(s.GetEnterpriseInfo))
}

func (s *BackendService) ListModels(r *http.Request) (any, error) {
	ctx := r.Context()
	var models []database.Model
	if err := s.db.WithContext(ctx).Find(&models).Error; err != nil {
		slog.Error("error getting models", "error", err)
		return nil, CodedErrorf(http.StatusInternalServerError, "error %d: error retrieving model records", ErrCodeDB)
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
		return nil, CodedErrorf(http.StatusInternalServerError, "error %d: error retrieving model record", ErrCodeDB)
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
	if err := validateName(req.Name); err != nil {
		return nil, err
	}

	ctx := r.Context()

	var model database.Model

	if err := s.db.WithContext(ctx).Transaction(func(txn *gorm.DB) error {
		if err := txn.Model(&database.Model{}).Where("name = ?", req.Name).Limit(1).Find(&model).Error; err != nil {
			if !errors.Is(err, gorm.ErrRecordNotFound) {
				slog.Error("error checking for duplicate model name", "error", err, "model_name", req.Name)
				return CodedErrorf(http.StatusInternalServerError, "error %d: error checking for duplicate model name", ErrCodeDB)
			}
		} else if model.Id != uuid.Nil {
			return CodedErrorf(http.StatusUnprocessableEntity, "model name '%s' already exists", req.Name)
		}

		var baseModel database.Model
		if err := txn.Preload("Tags").First(&baseModel, "id = ?", modelId).Error; err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return CodedErrorf(http.StatusNotFound, "base model not found")
			}
			slog.Error("error getting base model", "error", err)
			return CodedErrorf(http.StatusInternalServerError, "error %d: error retrieving base model record", ErrCodeDB)
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
			Tags:         baseModel.Tags,
		}

		if err := txn.WithContext(ctx).Create(&model).Error; err != nil {
			slog.Error("error creating model entry", "error", err)
			return CodedErrorf(http.StatusInternalServerError, "error %d: failed to create model entry", ErrCodeDB)
		}

		return nil
	}); err != nil {
		return nil, err
	}

	samples := make([]api.Sample, 0, len(req.Samples))
	samples = append(samples, req.Samples...)

	_, tokensList, labelsList, err := database.GetFeedbackSamples(ctx, s.db, modelId)
	if err != nil {
		slog.Error("failed to load feedback samples", "model_id", modelId, "error", err)
		return nil, CodedErrorf(http.StatusInternalServerError, "could not load feedback samples: %v", err)
	}

	for i := range tokensList {
		samples = append(samples, api.Sample{
			Tokens: tokensList[i],
			Labels: labelsList[i],
		})
	}

	if len(samples) == 0 {
		slog.Error("no feedback samples found for model", "model_id", modelId)
		return nil, CodedErrorf(http.StatusUnprocessableEntity, "no feedback samples found for model %s", modelId)
	}

	// TODO(anyone): Run some experiment to find optimal value of K based on the number of samples and model being finetuned.
	recordsToGenerate := max(50, len(samples)*3)
	testSplit := 0.1
	recordsPerLlmCall := AutoTuneK(samples, 30, 40)

	payload := messaging.FinetuneTaskPayload{
		ModelId:             model.Id,
		BaseModelId:         model.BaseModelId.UUID,
		Samples:             samples,
		GenerateData:        req.GenerateData,
		RecordsToGenerate:   recordsToGenerate,
		RecordsPerLlmCall:   recordsPerLlmCall,
		TestSplit:           float32(testSplit),
		VerifyGeneratedData: req.VerifyGeneratedData,
	}
	if req.TaskPrompt != nil {
		payload.TaskPrompt = *req.TaskPrompt
	}

	if err := s.publisher.PublishFinetuneTask(ctx, payload); err != nil {
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
		return nil, CodedErrorf(http.StatusInternalServerError, "error %d: error retrieving report records", ErrCodeDB)
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
			return nil, CodedErrorf(http.StatusInternalServerError, "error %d: error getting default model", ErrCodeDB)
		}

		if result.RowsAffected != 0 {
			req.ModelId = model.Id
		} else {
			return nil, CodedErrorf(http.StatusUnprocessableEntity, "the following fields are required: ModelId")
		}
	}

	var isUpload bool = req.StorageType == string(storage.UploadType)
	// No need to check parameter validity if upload since it uses the default app storage
	if !isUpload {
		connectorType, err := storage.ToStorageType(req.StorageType)
		if err != nil {
			return nil, CodedErrorf(http.StatusBadRequest, "invalid storage type: %v", err)
		}

		_, err = storage.NewConnector(r.Context(), connectorType, req.StorageParams)
		if err != nil {
			return nil, CodedErrorf(http.StatusInternalServerError, "error validating connector params: %v", err)
		}
	}

	if err := validateName(req.ReportName); err != nil {
		return nil, err
	}

	for _, tag := range req.Tags {
		if _, ok := req.CustomTags[tag]; ok {
			return nil, CodedErrorf(http.StatusUnprocessableEntity, "tag '%s' cannot be used as a regular and custom tag", tag)
		}
	}

	tagFormat := regexp.MustCompile(`^\w+$`)
	for tag, pattern := range req.CustomTags {
		if !tagFormat.MatchString(tag) {
			return nil, CodedErrorf(http.StatusUnprocessableEntity, "invalid name for custom tag '%s': only alphanumeric characters and underscores are allowed", tag)
		}

		if _, err := regexp.Compile(pattern); err != nil {
			return nil, CodedErrorf(http.StatusUnprocessableEntity, "invalid regex pattern '%s' for custom tag '%s': %v", pattern, tag, err)
		}
	}

	report := database.Report{
		Id:            uuid.New(),
		ReportName:    req.ReportName,
		ModelId:       req.ModelId,
		StorageType:   req.StorageType,
		StorageParams: datatypes.JSON(req.StorageParams),
		CreationTime:  time.Now().UTC(),
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
		result := txn.Where("report_name = ? AND deleted = ?", req.ReportName, false).Limit(1).Find(&existingReport)
		if result.Error != nil {
			slog.Error("error checking for duplicate report name", "error", result.Error)
			return CodedErrorf(http.StatusInternalServerError, "error %d: error checking for duplicate report name", ErrCodeDB)
		} else if result.RowsAffected > 0 {
			return CodedErrorf(http.StatusUnprocessableEntity, "report name '%s' already exists", req.ReportName)
		}

		if err := txn.Preload("Tags").First(&model, "id = ?", req.ModelId).Error; err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return CodedErrorf(http.StatusNotFound, "model not found")
			}
			slog.Error("error getting model", "error", err)
			return CodedErrorf(http.StatusInternalServerError, "error %d: error retrieving model record", ErrCodeDB)
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
			return CodedErrorf(http.StatusInternalServerError, "error %d: failed to create report entry", ErrCodeDB)
		}

		if err := txn.WithContext(ctx).Create(&task).Error; err != nil {
			slog.Error("error creating shard data task", "error", err)
			return CodedErrorf(http.StatusInternalServerError, "error %d: failed to create shard data task", ErrCodeDB)
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

func (s *BackendService) GetInferenceTime(ctx context.Context, reportId uuid.UUID) float64 {
	dbType := s.db.Dialector.Name()

	query := s.db.WithContext(ctx).
		Model(&database.InferenceTask{}).
		Select("MIN(start_time) AS min_start, MAX(completion_time) AS max_end").
		Where("report_id = ? AND start_time IS NOT NULL", reportId)

	var startingTime time.Time

	switch dbType {
	case "sqlite", "sqlite3":
		var infBounds struct {
			MinStart string `gorm:"column:min_start"`
			MaxEnd   string `gorm:"column:max_end"`
		}

		if err := query.Scan(&infBounds).Error; err != nil {
			slog.Error("error fetching time bounds", "err", err)
		}

		const sqliteTimeLayout = "2006-01-02 15:04:05.999999999Z07:00"

		startTime, err := time.Parse(sqliteTimeLayout, infBounds.MinStart)
		if err != nil {
			slog.Error("error parsing min_start", "raw", infBounds.MinStart, "err", err)
		}

		endTime, err := time.Parse(sqliteTimeLayout, infBounds.MaxEnd)
		if err != nil {
			slog.Error("error parsing max_end", "raw", infBounds.MaxEnd, "err", err)
		}

		if !startTime.IsZero() {
			startingTime = startTime
		}

		if !startTime.IsZero() && !endTime.IsZero() {
			return endTime.Sub(startTime).Seconds()
		}

	case "postgres":
		var infBounds struct {
			MinStart sql.NullTime `gorm:"column:min_start"`
			MaxEnd   sql.NullTime `gorm:"column:max_end"`
		}

		if err := query.Scan(&infBounds).Error; err != nil {
			slog.Error("error fetching time bounds", "err", err)
		}

		if infBounds.MinStart.Valid {
			startingTime = infBounds.MinStart.Time
		}

		if infBounds.MinStart.Valid && infBounds.MaxEnd.Valid {
			return infBounds.MaxEnd.Time.Sub(infBounds.MinStart.Time).Seconds()
		}
	}

	if !startingTime.IsZero() {
		return time.Now().UTC().Sub(startingTime).Seconds()
	}

	return 0
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
		return nil, CodedErrorf(http.StatusInternalServerError, "error %d: error retrieving report data", ErrCodeDB)
	}

	if report.Deleted {
		return nil, CodedErrorf(http.StatusNotFound, "report not found")
	}

	type taskStatusCategory struct {
		Status    string
		Count     int
		Total     int
		Completed int
	}

	var statusCategories []taskStatusCategory

	if err := s.db.WithContext(ctx).Model(&database.InferenceTask{}).
		Where("report_id = ?", reportId).
		Select("status, COUNT(*) as count, sum(total_size) as total, SUM(completed_size) as completed").
		Group("status").
		Find(&statusCategories).Error; err != nil {
		slog.Error("error getting inference task statuses", "report_id", reportId, "error", err)
		return nil, CodedErrorf(http.StatusInternalServerError, "error %d: error retrieving inference task statuses", ErrCodeDB)
	}

	apiReport := convertReport(report)
	apiReport.InferenceTaskStatuses = make(map[string]api.TaskStatusCategory)
	for _, category := range statusCategories {
		apiReport.InferenceTaskStatuses[category.Status] = api.TaskStatusCategory{
			TotalTasks:    category.Count,
			TotalSize:     category.Total,
			CompletedSize: category.Completed,
		}
	}

	now := time.Now().UTC()

	apiReport.TotalInferenceTimeSeconds = s.GetInferenceTime(ctx, reportId)

	var shardSecs float64
	if t := report.ShardDataTask; t != nil {
		end := now
		if t.CompletionTime.Valid {
			end = t.CompletionTime.Time
		}
		shardSecs = end.Sub(t.CreationTime).Seconds()
	}
	apiReport.ShardDataTimeSeconds = shardSecs

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
			return CodedErrorf(http.StatusInternalServerError, "error %d: error deleting report", ErrCodeDB)
		}

		if result.RowsAffected == 0 {
			return CodedErrorf(http.StatusNotFound, "report not found")
		}

		// Delete the report data that takes up the most space
		if err := txn.Delete(&database.ObjectGroup{}, "report_id = ?", reportId).Error; err != nil {
			slog.Error("error deleting report groups", "report_id", reportId, "error", err)
			return CodedErrorf(http.StatusInternalServerError, "error %d: error deleting report", ErrCodeDB)
		}

		if err := txn.Delete(&database.ObjectEntity{}, "report_id = ?", reportId).Error; err != nil {
			slog.Error("error deleting report entities", "report_id", reportId, "error", err)
			return CodedErrorf(http.StatusInternalServerError, "error %d: error deleting report", ErrCodeDB)
		}

		if err := txn.Delete(&database.ObjectPreview{}, "report_id = ?", reportId).Error; err != nil {
			slog.Error("error deleting report entities", "report_id", reportId, "error", err)
			return CodedErrorf(http.StatusInternalServerError, "error %d: error deleting report", ErrCodeDB)
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
			return CodedErrorf(http.StatusInternalServerError, "error %d: error deleting report", ErrCodeDB)
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
		return nil, CodedErrorf(http.StatusInternalServerError, "error %d: error retrieving report group record", ErrCodeDB)
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
		return nil, CodedErrorf(http.StatusInternalServerError, "error %d: error retrieving report entities", ErrCodeDB)
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

	tags := params["tags"]
	objectFilter := params.Get("object")

	ctx := r.Context()

	query := s.db.WithContext(ctx).
		Where("report_id = ?", reportId)

	if len(tags) > 0 {
		var matchingObjects []string
		if err := s.db.WithContext(ctx).
			Model(&database.ObjectEntity{}).
			Distinct("object").
			Where("report_id = ? AND label IN ?", reportId, tags).
			Pluck("object", &matchingObjects).Error; err != nil {
			slog.Error("error fetching entities for tag filter", "report_id", reportId, "tags", tags, "error", err)
			return nil, CodedErrorf(http.StatusInternalServerError, "error %d: error applying tag filter", ErrCodeDB)
		}

		if len(matchingObjects) == 0 {
			return []api.ObjectPreviewResponse{}, nil
		}

		query = query.Where("object IN ?", matchingObjects)
	}

	if objectFilter != "" {
		query = query.Where("object = ?", objectFilter)
	}

	query = query.Offset(offset).Limit(limit).Order("object")

	var previews []database.ObjectPreview
	if err := query.Find(&previews).Error; err != nil {
		slog.Error("error fetching previews", "report_id", reportId, "err", err)
		return nil, CodedErrorf(http.StatusInternalServerError, "error %d: error retrieving previews", ErrCodeDB)
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
		return nil, CodedErrorf(http.StatusInternalServerError, "error %d: error retrieving report objects", ErrCodeDB)
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
			if err := s.storage.PutObject(r.Context(), filepath.Join(s.uploadBucket, newFilepath), part); err != nil {
				slog.Error("error uploading file to storage", "error", err)
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

	var completed, running, failed statusMetrics

	// We count distinct report IDs because we are querying the InferenceTask table,
	// where the same report ID can appear multiple times.

	q1 := s.db.Model(&database.InferenceTask{}).
		Select("COUNT(DISTINCT report_id) AS count, COALESCE(SUM(total_size),0) AS size, COALESCE(SUM(token_count),0) AS tokens").
		Where("inference_tasks.status = ? AND inference_tasks.completion_time >= ?", database.JobCompleted, since)
	if modelID != uuid.Nil {
		q1 = q1.
			Joins("JOIN reports ON reports.id = inference_tasks.report_id").
			Where("reports.model_id = ?", modelID)
	}
	if err := q1.Scan(&completed).Error; err != nil {
		slog.Error("error fetching completed metrics", "error", err)
		return nil, CodedErrorf(http.StatusInternalServerError, "error %d: error retrieving metrics", ErrCodeDB)
	}

	q2 := s.db.Model(&database.InferenceTask{}).
		Select("COUNT(DISTINCT report_id) AS count, COALESCE(SUM(total_size),0) AS size, COALESCE(SUM(token_count),0) AS tokens").
		Where("inference_tasks.status = ? AND inference_tasks.creation_time >= ?", database.JobRunning, since)
	if modelID != uuid.Nil {
		q2 = q2.
			Joins("JOIN reports ON reports.id = inference_tasks.report_id").
			Where("reports.model_id = ?", modelID)
	}
	if err := q2.Scan(&running).Error; err != nil {
		slog.Error("error fetching in-progress metrics", "error", err)
		return nil, CodedErrorf(http.StatusInternalServerError, "error %d: error retrieving metrics", ErrCodeDB)
	}

	q3 := s.db.Model(&database.InferenceTask{}).
		Select("COUNT(DISTINCT report_id) AS count, COALESCE(SUM(total_size),0) AS size, COALESCE(SUM(token_count),0) AS tokens").
		Where("inference_tasks.status = ? AND inference_tasks.creation_time >= ?", database.JobFailed, since)
	if modelID != uuid.Nil {
		q3 = q3.
			Joins("JOIN reports ON reports.id = inference_tasks.report_id").
			Where("reports.model_id = ?", modelID)
	}
	if err := q3.Scan(&failed).Error; err != nil {
		slog.Error("error fetching in-progress metrics", "error", err)
		return nil, CodedErrorf(http.StatusInternalServerError, "error %d: error retrieving metrics", ErrCodeDB)
	}

	totalBytes := completed.Size.Int64 + running.Size.Int64 + failed.Size.Int64
	totalTokens := completed.Tokens.Int64 + running.Tokens.Int64 + failed.Tokens.Int64
	dataProcessedMB := float64(totalBytes) / (1024 * 1024)

	return api.InferenceMetricsResponse{
		Completed:       completed.Count,
		Failed:          failed.Count,
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
		Where("inference_tasks.status = ? OR inference_tasks.status = ?", database.JobCompleted, database.JobFailed).
		Where("reports.model_id = ?", modelID)
	if reportID != uuid.Nil {
		q = q.Where("inference_tasks.report_id = ?", reportID)
	}
	if err := q.
		Select("inference_tasks.total_size, inference_tasks.start_time, inference_tasks.completion_time").
		Scan(&rows).Error; err != nil {
		slog.Error("error fetching throughput rows", "error", err)
		return nil, CodedErrorf(http.StatusInternalServerError, "error %d: error retrieving throughput", ErrCodeDB)
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
	req, err := ParseRequestQueryParams[api.ValidateGroupDefinitionRequest](r)
	if err != nil {
		return nil, err
	}
	if _, err := core.ParseQuery(req.GroupQuery); err != nil {
		return nil, CodedErrorf(http.StatusUnprocessableEntity, "invalid query '%s': %v", req.GroupQuery, err)
	}
	return nil, nil
}

func (s *BackendService) GetLicense(r *http.Request) (any, error) {
	licenseInfo, err := s.licensing.VerifyLicense(r.Context())
	return api.GetLicenseResponse{
		LicenseInfo:  licenseInfo,
		LicenseError: fmt.Sprintf("%v", err),
	}, nil
}

func (s *BackendService) GetEnterpriseInfo(r *http.Request) (any, error) {
	return api.GetEnterpriseInfoResponse{
		IsEnterpriseMode: s.enterpriseMode,
	}, nil
}

func (s *BackendService) ValidateS3Access(r *http.Request) (any, error) {
	req, err := ParseRequestQueryParams[api.ValidateS3BucketRequest](r)
	if err != nil {
		return nil, err
	}

	_, err = storage.NewS3Connector(
		r.Context(),
		storage.S3ConnectorParams{
			Endpoint: req.S3Endpoint,
			Region:   req.Region,
			Bucket:   req.SourceS3Bucket,
			Prefix:   req.SourceS3Prefix,
		},
	)

	return nil, err
}

func (s *BackendService) StoreFileNameToPath(r *http.Request) (any, error) {
	uploadId, err := URLParamUUID(r, "upload_id")
	if err != nil {
		return nil, err
	}
	req, err := ParseRequest[api.FileNameToPath](r)
	if err != nil {
		return nil, CodedErrorf(http.StatusBadRequest, "invalid request body")
	}
	mappingJson, err := json.Marshal(req.Mapping)
	if err != nil {
		return nil, CodedErrorf(http.StatusBadRequest, "invalid mapping")
	}
	entry := database.FileNameToPath{
		ID:      uploadId,
		Mapping: mappingJson,
	}
	if err := s.db.Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "id"}},
		DoUpdates: clause.AssignmentColumns([]string{"mapping"}),
	}).Create(&entry).Error; err != nil {
		return nil, CodedErrorf(http.StatusInternalServerError, "failed to store path map")
	}
	return nil, nil
}

func (s *BackendService) GetFileNameToPath(r *http.Request) (any, error) {
	uploadId, err := URLParamUUID(r, "upload_id")
	if err != nil {
		return nil, err
	}
	var entry database.FileNameToPath
	if err := s.db.First(&entry, "id = ?", uploadId).Error; err != nil {
		return nil, CodedErrorf(http.StatusNotFound, "not found")
	}
	var mapping map[string]string
	if err := json.Unmarshal(entry.Mapping, &mapping); err != nil {
		return nil, CodedErrorf(http.StatusInternalServerError, "invalid mapping data")
	}
	return api.FileNameToPath{
		Mapping: mapping,
	}, nil
}

func (s *BackendService) StoreModelFeedback(r *http.Request) (any, error) {
	modelId, err := URLParamUUID(r, "model_id")
	if err != nil {
		return nil, err
	}

	ctx := r.Context()
	var m database.Model
	if err := s.db.WithContext(ctx).Preload("Tags").First(&m, "id = ?", modelId).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, CodedErrorf(http.StatusNotFound, "model not found")
		}
		return nil, CodedErrorf(http.StatusInternalServerError, "error %d: could not retrieve model", ErrCodeDB)
	}

	req, err := ParseRequest[api.FeedbackRequest](r)
	if err != nil {
		return nil, CodedErrorf(http.StatusBadRequest, "invalid request body")
	}
	if len(req.Tokens) == 0 || len(req.Labels) == 0 || len(req.Tokens) != len(req.Labels) {
		return nil, CodedErrorf(http.StatusUnprocessableEntity, "tokens and labels must both be non‐empty and of equal length")
	}

	modelTagsName := make([]string, len(m.Tags))
	for i, tag := range m.Tags {
		modelTagsName[i] = tag.Tag
	}

	unsupportedTags := make([]string, 0, len(req.Labels))
	for _, feedbackTag := range req.Labels {
		if !slices.Contains(modelTagsName, feedbackTag) {
			unsupportedTags = append(unsupportedTags, feedbackTag)
		}
	}
	if len(unsupportedTags) > 0 {
		return nil, CodedErrorf(http.StatusUnprocessableEntity, "model does not support tags: %s", strings.Join(unsupportedTags, ", "))
	}

	if err := database.SaveFeedbackSample(ctx, s.db, modelId, req.Tokens, req.Labels); err != nil {
		return nil, CodedErrorf(http.StatusInternalServerError, "failed to save feedback: %v", err)
	}

	return nil, nil
}

func (s *BackendService) DeleteModelFeedback(r *http.Request) (any, error) {
	modelId, err := URLParamUUID(r, "model_id")
	if err != nil {
		return nil, err
	}
	feedbackId, err := URLParamUUID(r, "feedback_id")
	if err != nil {
		return nil, err
	}
	if err := s.db.Delete(&database.FeedbackSample{}, "id = ? AND model_id = ?", feedbackId, modelId).Error; err != nil {
		return nil, CodedErrorf(http.StatusInternalServerError, "failed to delete feedback: %v", err)
	}
	return nil, nil
}

func (s *BackendService) ListModelFeedback(r *http.Request) (any, error) {
	modelId, err := URLParamUUID(r, "model_id")
	if err != nil {
		return nil, err
	}

	ctx := r.Context()
	var m database.Model
	if err := s.db.WithContext(ctx).First(&m, "id = ?", modelId).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, CodedErrorf(http.StatusNotFound, "model not found")
		}
		return nil, CodedErrorf(http.StatusInternalServerError, "error %d: could not retrieve model", ErrCodeDB)
	}

	ids, tokensList, labelsList, err := database.GetFeedbackSamples(ctx, s.db, modelId)
	if err != nil {
		return nil, CodedErrorf(http.StatusInternalServerError, "could not fetch feedback: %v", err)
	}

	out := make([]api.FeedbackResponse, len(tokensList))
	for i := range tokensList {
		out[i] = api.FeedbackResponse{
			Id:     ids[i],
			Tokens: tokensList[i],
			Labels: labelsList[i],
		}
	}
	return out, nil
}
