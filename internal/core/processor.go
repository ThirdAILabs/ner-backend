package core

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"ner-backend/internal/core/datagen"
	"ner-backend/internal/core/types"
	"ner-backend/internal/database"
	"ner-backend/internal/licensing"
	"ner-backend/internal/messaging"
	"ner-backend/internal/storage"
	"ner-backend/pkg/api"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"
	"gorm.io/datatypes"
	"gorm.io/gorm"
)

type TaskProcessor struct {
	db        *gorm.DB
	storage   storage.ObjectStore
	publisher messaging.Publisher
	reciever  messaging.Reciever
	
	licensing licensing.LicenseVerifier
	
	localModelDir string
	modelBucket   string
	uploadBucket  string
	modelLoaders  map[ModelType]ModelLoader

}

const bytesPerMB = 1024 * 1024

var ExcludedTags = map[string]struct{}{
	"GENDER":             {},
	"SEXUAL_ORIENTATION": {},
	"ETHNICITY":          {},
	"SERVICE_CODE":       {},
}

func NewTaskProcessor(db *gorm.DB, storage storage.ObjectStore, publisher messaging.Publisher, reciever messaging.Reciever, licenseVerifier licensing.LicenseVerifier, localModelDir string, modelBucket string, uploadBucket string, modelLoaders map[ModelType]ModelLoader) *TaskProcessor {
	return &TaskProcessor{
		db:                      db,
		storage:                 storage,
		publisher:               publisher,
		reciever:                reciever,
		licensing:               licenseVerifier,
		localModelDir:           localModelDir,
		modelBucket:             modelBucket,
		uploadBucket:            uploadBucket,
		modelLoaders:            modelLoaders,
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
		var payload messaging.InferenceTaskPayload
		if err = json.Unmarshal(task.Payload(), &payload); err != nil {
			slog.Error("error unmarshalling inference task", "error", err)
			if err := task.Reject(); err != nil { // Discard malformed message
				slog.Error("error rejecting message from queue", "error", err)
			}
			return
		}
		err = proc.processInferenceTask(ctx, payload)

	case messaging.ShardDataQueue:
		var payload messaging.ShardDataPayload
		if err = json.Unmarshal(task.Payload(), &payload); err != nil {
			slog.Error("error unmarshalling shard data task", "error", err)
			if err := task.Reject(); err != nil { // Discard malformed message
				slog.Error("error rejecting message from queue", "error", err)
			}
			return
		}
		err = proc.processShardDataTask(ctx, payload) // Call new handler

	case messaging.FinetuneQueue:
		var payload messaging.FinetuneTaskPayload
		if err = json.Unmarshal(task.Payload(), &payload); err != nil {
			slog.Error("error unmarshalling finetuning task", "error", err)
			if err := task.Reject(); err != nil { // Discard malformed message
				slog.Error("error rejecting message from queue", "error", err)
			}
			return
		}
		err = proc.processFinetuneTask(ctx, payload)

	default:
		slog.Error("received unknown task type", "queue", task.Type())
		if err := task.Reject(); err != nil { // reject unknown message type
			slog.Error("error rejecting message from queue", "error", err)
		}
		return
	}

	if err != nil {
		slog.Error("error processing task", "queue", task.Type(), "error", err)
		if err := task.Nack(); err != nil {
			slog.Error("error reporting processing failure on message from queue", "error", err)
		}
	} else {
		slog.Info("successfully processed task", "queue", task.Type())
		if err := task.Ack(); err != nil {
			slog.Error("error acknowledging message from queue", "error", err)
		}
	}
}

func (proc *TaskProcessor) updateFileCount(reportId uuid.UUID, success bool) error {
	var column string
	if success {
		column = "succeeded_file_count"
	} else {
		column = "failed_file_count"
	}

	if err := proc.db.
		Model(&database.Report{}).
		Where("id = ?", reportId).
		UpdateColumn(column, gorm.Expr(column+" + ?", 1)).
		Error; err != nil {
		slog.Error("could not increment file count", "report_id", reportId, "column", column, "error", err)
		return fmt.Errorf("could not increment file count: %w", err)
	}

	return nil
}

func (proc *TaskProcessor) getConnector(ctx context.Context, report database.Report) (storage.Connector, error) {
	// Custom connector initialization logic for uploads. It is a special case because it has to be consistent with
	// the storage used by the backend service.
	if report.StorageType == string(storage.UploadType) {
		var uploadParams storage.UploadParams
		if err := json.Unmarshal(report.StorageParams, &uploadParams); err != nil {
			return nil, fmt.Errorf("error unmarshalling storage params: %w", err)
		}
		slog.Info("Get upload connector", "upload bucket", proc.uploadBucket)
		return proc.storage.GetUploadConnector(ctx, proc.uploadBucket, uploadParams)
	}
	connectorType, err := storage.ToStorageType(report.StorageType)
	if err != nil {
		return nil, fmt.Errorf("invalid storage type: %v", err)
	}
	return storage.NewConnector(ctx, connectorType, report.StorageParams)

}

func (proc *TaskProcessor) processInferenceTask(ctx context.Context, payload messaging.InferenceTaskPayload) error {

	reportId := payload.ReportId
	taskId := payload.TaskId

	slog.Info("processing inference task", "report_id", reportId, "task_id", payload.TaskId)

	var task database.InferenceTask
	if err := proc.db.Preload("Report").Preload("Report.Model").Preload("Report.Tags").Preload("Report.CustomTags").Preload("Report.Groups").First(&task, "report_id = ? AND task_id = ?", reportId, payload.TaskId).Error; err != nil {
		slog.Error("error fetching inference task", "report_id", reportId, "task_id", payload.TaskId, "error", err)
		return fmt.Errorf("error getting inference task: %w", err)
	}

	if task.Report.Stopped || task.Report.Deleted {
		slog.Info("report stopped, skipping inference task", "report_id", reportId, "task_id", payload.TaskId)
		return nil
	}

	slog.Info("processing inference task", "report_id", reportId, "task_id", payload.TaskId)
	if err := proc.db.
		Model(&database.InferenceTask{}).
		Where("report_id = ? AND task_id = ?", reportId, taskId).
		Updates(map[string]interface{}{
			"status":     database.JobRunning,
			"start_time": time.Now().UTC(),
		}).Error; err != nil {
		slog.Error("error marking task as running", "error", err)
	}

	if _, err := proc.licensing.VerifyLicense(ctx); err != nil {
		slog.Error("license verification failed", "error", err)
		database.UpdateInferenceTaskStatus(ctx, proc.db, reportId, payload.TaskId, database.JobFailed) //nolint:errcheck
		database.SaveReportError(ctx, proc.db, reportId, fmt.Sprintf("license verification failed: %s", err.Error()))
		return err
	}

	tags := make(map[string]struct{})
	for _, tag := range task.Report.Tags {
		tags[tag.Tag] = struct{}{}
	}

	customTags := make(map[string]string)
	for _, tag := range task.Report.CustomTags {
		customTags[tag.Tag] = tag.Pattern
	}

	groupToQuery := map[uuid.UUID]string{}
	for _, group := range task.Report.Groups {
		groupToQuery[group.Id] = group.Query
	}

	connector, err := proc.getConnector(ctx, *task.Report)
	if err != nil {
		return fmt.Errorf("error initializing connector for inference task: %w", err)
	}

	totalTokens, workerErr := proc.runInferenceOnBucket(ctx, taskId, task.ReportId, connector, task.StorageParams, task.Report.Model.Id, ParseModelType(task.Report.Model.Type), tags, customTags, groupToQuery)

	if err := proc.db.
		Model(&database.InferenceTask{}).
		Where("report_id = ? AND task_id = ?", reportId, taskId).
		Update("token_count", totalTokens).
		Error; err != nil {
		slog.Error("unable to update token_count", "report_id", reportId, "task_id", taskId, "error", err)
	}

	if workerErr != nil {
		slog.Error("error running inference task", "report_id", reportId, "task_id", payload.TaskId, "error", workerErr)
		database.UpdateInferenceTaskStatus(ctx, proc.db, reportId, payload.TaskId, database.JobFailed) // nolint:errcheck
		database.SaveReportError(ctx, proc.db, reportId, workerErr.Error())
		return fmt.Errorf("error running inference task: %w", workerErr)
	}

	if err := database.UpdateInferenceTaskStatus(ctx, proc.db, reportId, payload.TaskId, database.JobCompleted); err != nil {
		return fmt.Errorf("error updating inference task status to complete: %w", err)
	}

	slog.Info("inference task completed successfully", "report_id", reportId, "task_id", payload.TaskId)

	return nil
}

func (proc *TaskProcessor) updateInferenceTagCount(reportId uuid.UUID, tagCount map[string]uint64, isCustomTag bool) error {
	var model any
	if isCustomTag {
		model = &database.CustomTag{}
	} else {
		model = &database.ReportTag{}
	}

	for tag, count := range tagCount {
		if err := proc.db.Model(model).Where("report_id = ? AND tag = ?", reportId, tag).Update("count", gorm.Expr("count + ?", count)).Error; err != nil {
			slog.Error("error updating tag count", "report_id", reportId, "tag", tag, "is_custom_tag", isCustomTag, "error", err)
			return fmt.Errorf("error updating tag count: %w", err)
		}
	}

	return nil
}

func (proc *TaskProcessor) runInferenceOnBucket(
	ctx context.Context,
	taskId int,
	reportId uuid.UUID,
	connector storage.Connector,
	taskParams []byte,
	modelId uuid.UUID,
	modelType ModelType,
	tags map[string]struct{},
	customTags map[string]string,
	groupToQuery map[uuid.UUID]string,
) (int64, error) {
	var totalTokens int64
	model, err := proc.loadModel(ctx, modelId, modelType)
	if err != nil {
		return 0, err
	}
	defer model.Release()

	groupToFilter := make(map[uuid.UUID]Filter)
	for groupId, query := range groupToQuery {
		filter, err := ParseQuery(query)
		if err != nil {
			return 0, fmt.Errorf("error loading model: %w", err)
		}
		groupToFilter[groupId] = filter
	}

	customTagsRe := make(map[string]*regexp.Regexp)
	for tag, pat := range customTags {
		re, err := regexp.Compile(pat)
		if err != nil {
			return 0, fmt.Errorf("error compiling regex for tag %s: %w", tag, err)
		}
		customTagsRe[tag] = re
	}

	queue, err := connector.IterTaskChunks(ctx, taskParams)
	if err != nil {
		slog.Error("error iterating over task chunks", "error", err)
		return 0, err
	}

	objectErrorCnt := 0
	totalObjectCnt := 0

	for object := range queue {
		if object.Error != nil {
			slog.Error("error getting object stream", "object", object.Name, "error", object.Error, "task_params", string(taskParams))
			objectErrorCnt++
			if err := proc.updateFileCount(reportId, false); err != nil {
				return totalTokens, err
			}
			continue
		}

		result, err := proc.runInferenceOnObject(reportId, object.Chunks, model, tags, customTagsRe, groupToFilter, object.Name)

		if err != nil {
			slog.Error("error processing object", "object", object.Name, "error", err)
			objectErrorCnt++
			if err := proc.updateFileCount(reportId, false); err != nil {
				return totalTokens, err
			}
			continue
		}

		if err := proc.db.CreateInBatches(&result.Entities, 100).Error; err != nil {
			slog.Error("error saving entities to database", "object", object.Name, "error", err)
			objectErrorCnt++
			if err := proc.updateFileCount(reportId, false); err != nil {
				return totalTokens, err
			}
			continue
		}

		if err := proc.db.CreateInBatches(result.Groups, 100).Error; err != nil {
			slog.Error("error saving groups to database", "object", object.Name, "error", err)
			objectErrorCnt++
			if err := proc.updateFileCount(reportId, false); err != nil {
				return totalTokens, err
			}
			continue
		}

		if err := proc.updateInferenceTagCount(reportId, result.TagCount, false); err != nil {
			slog.Error("error updating tag count", "object", object.Name, "error", err)
			objectErrorCnt++
			if err := proc.updateFileCount(reportId, false); err != nil {
				return totalTokens, err
			}
			continue
		}

		if err := proc.updateInferenceTagCount(reportId, result.CustomTagCount, true); err != nil {
			slog.Error("error updating custom tag count", "object", object.Name, "error", err)
			objectErrorCnt++
			if err := proc.updateFileCount(reportId, false); err != nil {
				return totalTokens, err
			}
			continue
		}

		if err := proc.updateFileCount(reportId, true); err != nil {
			return totalTokens, err
		}

		totalTokens += result.TotalTokens

		if err := proc.db.Model(&database.InferenceTask{}).
			Where("report_id = ? AND task_id = ?", reportId, taskId).
			Update("completed_size", gorm.Expr("completed_size + ?", result.TotalSize)).Error; err != nil {
			slog.Error("could not update completed size in InferenceTask", "error", err)
			return totalTokens, err
		}

		totalObjectCnt++
	}

	if objectErrorCnt > 0 {
		return totalTokens, fmt.Errorf("errors while processing %d/%d objects", objectErrorCnt, totalObjectCnt)
	}

	return totalTokens, nil
}

func (proc *TaskProcessor) getModelDir(modelId uuid.UUID) string {
	return filepath.Join(proc.localModelDir, modelId.String())
}

func (proc *TaskProcessor) loadModel(ctx context.Context, modelId uuid.UUID, modelType ModelType) (Model, error) {

	var localDir string

	if IsStatelessModel(modelType) {
		localDir = ""
	} else {
		localDir = proc.getModelDir(modelId)

		if _, err := os.Stat(localDir); os.IsNotExist(err) {
			slog.Info("model not found locally, downloading from S3", "modelId", modelId)

			if err := proc.storage.DownloadDir(ctx, proc.modelBucket, modelId.String(), localDir, false); err != nil {
				return nil, fmt.Errorf("failed to download model from S3: %w", err)
			}
		}
	}

	model, err := proc.modelLoaders[modelType](localDir)
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}

	return model, nil
}

func (proc *TaskProcessor) createObjectPreview(
	reportId uuid.UUID,
	object string,
	previewText string,
	model Model,
	customTags map[string]*regexp.Regexp,
) error {
	if proc.db == nil {
		return nil
	}

	spans, err := model.Predict(previewText)
	if err != nil {
		return fmt.Errorf("preview inference error: %w", err)
	}
	spans = FilterEntities(previewText, spans)

	for tag, re := range customTags {
		for _, idx := range re.FindAllStringIndex(previewText, -1) {
			start, end := idx[0], idx[1]
			spans = append(spans, types.Entity{
				Label:    tag,
				Text:     previewText[start:end],
				Start:    start,
				End:      end,
				LContext: strings.ToValidUTF8(previewText[max(0, start-20):start], ""),
				RContext: strings.ToValidUTF8(previewText[end:min(len(previewText), end+20)], ""),
			})
		}
	}

	// converting spans to a map for coalescing
	spanEntityMap := make(map[string][]types.Entity)
	for _, span := range spans {
		spanEntityMap[span.Label] = append(spanEntityMap[span.Label], span)
	}

	coalescedSpans := coalesceEntities(spanEntityMap)
	// coalescedSpans will be already sorted by start position

	var (
		tokens []string
		tags   []string
		cursor = 0
		length = len(previewText)
	)
	for _, e := range coalescedSpans {
		if _, exists := ExcludedTags[e.Label]; exists {
			continue
		}

		if e.Start > cursor {
			tokens = append(tokens, previewText[cursor:e.Start])
			tags = append(tags, "O")
		}
		end := min(e.End, length)
		tokens = append(tokens, previewText[e.Start:end])
		tags = append(tags, e.Label)
		cursor = end
	}
	if cursor < length {
		tokens = append(tokens, previewText[cursor:])
		tags = append(tags, "O")
	}

	payload := struct {
		Tokens []string `json:"tokens"`
		Tags   []string `json:"tags"`
	}{
		Tokens: tokens,
		Tags:   tags,
	}
	b, _ := json.Marshal(payload)

	return proc.db.Create(&database.ObjectPreview{
		ReportId:  reportId,
		Object:    object,
		TokenTags: datatypes.JSON(b),
	}).Error
}

func coalesceEntities(labelToEntities map[string][]types.Entity) []types.Entity {
	maxEntityGap := 1 // Assuming this gap is less that any entity.Rcontext length
	flattenedEntities := make([]types.Entity, 0, len(labelToEntities))
	for _, ents := range labelToEntities {
		flattenedEntities = append(flattenedEntities, ents...)
	}
	if len(flattenedEntities) == 0 {
		return nil
	}

	sort.Slice(flattenedEntities, func(i, j int) bool {
		return flattenedEntities[i].Start < flattenedEntities[j].Start
	})

	coalescedEntities := make([]types.Entity, 0, len(flattenedEntities))
	currentEnt := flattenedEntities[0]

	for i := 1; i < len(flattenedEntities); i++ {
		nextEnt := flattenedEntities[i]

		// Merge only if they are adjacent (at most maxEntityGap) and share the same label.
		if currentEnt.Label == nextEnt.Label && nextEnt.Start >= currentEnt.End && nextEnt.Start-currentEnt.End <= maxEntityGap {
			// Extend currentEnt to include nextEnt
			currentEnt.Text += currentEnt.RContext[:nextEnt.Start-currentEnt.End] + nextEnt.Text
			currentEnt.End = nextEnt.End
			currentEnt.RContext = nextEnt.RContext
		} else {
			// Different label or non-adjacent: flush currentEnt and start a new one
			coalescedEntities = append(coalescedEntities, currentEnt)
			currentEnt = nextEnt
		}
	}

	coalescedEntities = append(coalescedEntities, currentEnt)
	return coalescedEntities
}

type InferenceResult struct {
	TotalTokens    int64
	TotalSize      int64
	Entities       []database.ObjectEntity
	Groups         []database.ObjectGroup
	TagCount       map[string]uint64
	CustomTagCount map[string]uint64
}

func (proc *TaskProcessor) runInferenceOnObject(
	reportId uuid.UUID,
	chunks <-chan storage.Chunk,
	model Model,
	tags map[string]struct{},
	customTags map[string]*regexp.Regexp,
	groupFilter map[uuid.UUID]Filter,
	object string,
) (InferenceResult, error) {
	result := InferenceResult{
		TagCount:       make(map[string]uint64),
		CustomTagCount: make(map[string]uint64),
	}

	labelToEntities := make(map[string][]types.Entity)

	const previewLimit = 1000
	previewTokens := make([]string, 0, previewLimit)

	for chunk := range chunks {
		if chunk.Error != nil {
			return result, fmt.Errorf("error parsing document: %w", chunk.Error)
		}

		result.TotalSize += chunk.RawSize

		start := time.Now()
		chunkEntities, err := model.Predict(chunk.Text)
		chunkEntities = FilterEntities(chunk.Text, chunkEntities)
		duration := time.Since(start)
		sizeMB := float64(chunk.RawSize) / float64(bytesPerMB)
		slog.Info("processed chunk",
			"chunk_size_mb", fmt.Sprintf("%.2f", sizeMB),
			"duration", duration,
		)
		if err != nil {
			return result, fmt.Errorf("error running model inference: %w", err)
		}

		for _, entity := range chunkEntities {
			if _, exists := ExcludedTags[entity.Label]; exists {
				continue
			}
			if _, ok := tags[entity.Label]; ok {
				entity.Start += chunk.Offset
				entity.End += chunk.Offset
				labelToEntities[entity.Label] = append(labelToEntities[entity.Label], entity)
			}
		}

		for tag, re := range customTags {
			matches := re.FindAllStringIndex(chunk.Text, -1)
			for _, match := range matches {
				start, end := match[0], match[1]
				labelToEntities[tag] = append(labelToEntities[tag], types.Entity{
					Label:    tag,
					Text:     chunk.Text[start:end],
					Start:    start + chunk.Offset,
					End:      end + chunk.Offset,
					LContext: strings.ToValidUTF8(chunk.Text[max(0, start-20):start], ""),
					RContext: strings.ToValidUTF8(chunk.Text[end:min(len(chunk.Text), end+20)], ""),
				})
			}
		}

		if len(previewTokens) < previewLimit {
			toks := strings.Fields(chunk.Text)
			need := previewLimit - len(previewTokens)
			if len(toks) >= need {
				previewTokens = append(previewTokens, toks[:need]...)
			} else {
				previewTokens = append(previewTokens, toks...)
			}
		}

		toks := strings.Fields(chunk.Text)
		result.TotalTokens += int64(len(toks))
	}

	previewText := strings.Join(previewTokens, " ")
	if err := proc.createObjectPreview(reportId, object, previewText, model, customTags); err != nil {
		slog.Error("saving ObjectPreview failed", "object", object, "err", err)
	}

	groups := make([]database.ObjectGroup, 0)
	for groupId, filter := range groupFilter {
		if filter.Matches(labelToEntities) {
			groups = append(groups, database.ObjectGroup{
				ReportId: reportId,
				GroupId:  groupId,
				Object:   object,
			})
		}
	}

	coalescedEntities := coalesceEntities(labelToEntities)

	allEntities := make([]database.ObjectEntity, len(coalescedEntities))
	for i, entity := range coalescedEntities {
		allEntities[i] = database.ObjectEntity{
			ReportId: reportId,
			Label:    entity.Label,
			Text:     entity.Text,
			Start:    entity.Start,
			End:      entity.End,
			Object:   object,
			LContext: entity.LContext,
			RContext: entity.RContext,
		}

		if _, exists := customTags[entity.Label]; exists {
			result.CustomTagCount[entity.Label]++
		} else {
			result.TagCount[entity.Label]++
		}
	}

	result.Entities = allEntities
	result.Groups = groups

	return result, nil
}

func (proc *TaskProcessor) processShardDataTask(ctx context.Context, payload messaging.ShardDataPayload) error {
	reportId := payload.ReportId

	slog.Info("processing shard data task", "report_id", reportId)

	var task database.ShardDataTask
	if err := proc.db.Preload("Report").First(&task, "report_id = ?", reportId).Error; err != nil {
		slog.Error("error fetching shard data task", "report_id", reportId, "error", err)
		return fmt.Errorf("error getting shard data task: %w", err)
	}

	if task.Report.Stopped || task.Report.Deleted {
		slog.Info("report stopped, skipping shard data task", "report_id", reportId)
		return nil
	}

	database.UpdateShardDataTaskStatus(ctx, proc.db, reportId, database.JobRunning) //nolint:errcheck

	if _, err := proc.licensing.VerifyLicense(ctx); err != nil {
		slog.Error("license verification failed", "error", err)
		database.UpdateShardDataTaskStatus(ctx, proc.db, reportId, database.JobFailed) //nolint:errcheck
		database.SaveReportError(ctx, proc.db, reportId, fmt.Sprintf("license verification failed: %s", err.Error()))
		return err
	}

	slog.Info("Handling generate tasks", "jobId", task.ReportId, "storageParams", string(task.Report.StorageParams))

	targetBytes := task.ChunkTargetBytes
	if targetBytes <= 0 {
		targetBytes = 10 * 1024 * 1024 * 1024 // Default 10GB if not set or invalid
		slog.Info("Using default chunk target size", "targetBytes", targetBytes, "jobId", reportId)
	}

	createInferenceTask := func(ctx context.Context, taskId int, taskMetadata storage.InferenceTask) error {
		slog.Info("Creating inference task", "report_id", reportId, "task_id", taskId, "chunk_size", taskMetadata.TotalSize)

		task := database.InferenceTask{
			ReportId:     reportId,
			TaskId:       taskId,
			Status:       database.JobQueued,
			CreationTime: time.Now().UTC(),
			StorageParams: taskMetadata.Params,
			TotalSize:    taskMetadata.TotalSize,
		}

		inferencePayload := messaging.InferenceTaskPayload{
			ReportId: task.ReportId, TaskId: task.TaskId,
		}

		if err := proc.db.WithContext(ctx).Create(&task).Error; err != nil {
			slog.Error("error saving inference task to db", "report_id", task.ReportId, "task_id", task.TaskId, "error", err)
			database.UpdateShardDataTaskStatus(ctx, proc.db, reportId, database.JobFailed) //nolint:errcheck
			return fmt.Errorf("error saving inference task to db: %w", err)
		}

		if err := proc.publisher.PublishInferenceTask(ctx, inferencePayload); err != nil {
			slog.Error("Handler: Failed to publish inference task", "report_id", reportId, "task_id", taskId, "error", err)
			database.UpdateShardDataTaskStatus(ctx, proc.db, reportId, database.JobFailed) //nolint:errcheck
			return fmt.Errorf("failed to publish inference task %d: %w", taskId, err)
		}

		slog.Info("Created inference task", "report_id", reportId, "task_id", taskId, "chunk_size", taskMetadata.TotalSize)

		return nil
	}

	connector, err := proc.getConnector(ctx, *task.Report)
	if err != nil {
		return fmt.Errorf("error initializing connector for inference task: %w", err)
	}

	inferenceTasks, totalObjects, err := connector.CreateInferenceTasks(ctx, targetBytes)
	if err != nil {
		return fmt.Errorf("error creating inference tasks: %w", err)
	}

	taskId := 0
	for _, task := range inferenceTasks {
		if err := createInferenceTask(ctx, taskId, task); err != nil {
			return err
		}
		taskId++
	}

	if err := proc.db.
		Model(&database.Report{}).
		Where("id = ?", reportId).
		UpdateColumn("total_file_count", totalObjects).
		Error; err != nil {
		slog.Warn("failed to update total_file_count", "report_id", reportId, "totalObjects", totalObjects, "error", err)
	}

	if err := database.UpdateShardDataTaskStatus(ctx, proc.db, reportId, database.JobCompleted); err != nil {
		return fmt.Errorf("failed to update job final status: %w", err)
	}

	slog.Info("Finished generating inference task chunks", "n_tasks", taskId, "report_id", reportId)

	return nil
}

func (proc *TaskProcessor) getModel(ctx context.Context, modelId uuid.UUID) (database.Model, error) {
	var model database.Model
	if err := proc.db.WithContext(ctx).First(&model, "id = ?", modelId).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			slog.Error("model not found", "model_id", modelId)
			return database.Model{}, fmt.Errorf("model not found: %w", err)
		}
		slog.Error("error getting model", "model_id", modelId, "error", err)
		return database.Model{}, fmt.Errorf("error getting model: %w", err)
	}
	return model, nil
}

func extractTagNames(infos []api.TagInfo) []string {
	out := make([]string, len(infos))
	for i, t := range infos {
		out[i] = t.Name
	}
	return out
}

func (proc *TaskProcessor) processFinetuneTask(ctx context.Context, payload messaging.FinetuneTaskPayload) error {
	database.UpdateModelStatus(ctx, proc.db, payload.ModelId, database.ModelTraining) //nolint:errcheck

	slog.Info("processing finetune task", "model_id", payload.ModelId, "base_model_id", payload.BaseModelId)

	baseModel, err := proc.getModel(ctx, payload.BaseModelId)
	if err != nil {
		database.UpdateModelStatus(ctx, proc.db, payload.ModelId, database.ModelFailed) //nolint:errcheck
		slog.Error("error getting base model", "base_model_id", payload.BaseModelId, "model_id", payload.ModelId, "error", err)
		return err
	}

	model, err := proc.loadModel(ctx, payload.BaseModelId, ParseModelType(baseModel.Type))
	if err != nil {
		database.UpdateModelStatus(ctx, proc.db, payload.ModelId, database.ModelFailed) //nolint:errcheck
		slog.Error("error loading base model", "base_model_id", payload.BaseModelId, "model_id", payload.ModelId, "error", err)
		return fmt.Errorf("error loading base model: %w", err)
	}
	defer model.Release()

	slog.Info("base model loaded for finetuning", "model_id", payload.ModelId, "base_model_id", payload.BaseModelId)

	localDir := proc.getModelDir(payload.ModelId)
	if err := os.MkdirAll(localDir, os.ModePerm); err != nil {
		database.UpdateModelStatus(ctx, proc.db, payload.ModelId, database.ModelFailed) //nolint:errcheck
		slog.Error("error creating local model directory", "model_id", payload.ModelId, "error", err)
		return fmt.Errorf("error creating local model directory: %w", err)
	}

	allSamples := payload.Samples
	if payload.GenerateData {
		opts := datagen.DatagenOpts{
			Tags:               extractTagNames(payload.Tags),
			Samples:            payload.Samples,
			NumValuesPerTag:    payload.NumValuesPerTag,
			RecordsToGenerate:  payload.RecordsToGenerate,
			RecordsPerTemplate: payload.RecordsPerTemplate,
			TestSplit:          payload.TestSplit,
		}
		train, test, err := datagen.GenerateData(opts)
		if err != nil {
			database.UpdateModelStatus(ctx, proc.db, payload.ModelId, database.ModelFailed) //nolint
			return fmt.Errorf("datagen error: %w", err)
		}
		allSamples = append(allSamples, train...)
		allSamples = append(allSamples, test...)
	}

	if err := model.FinetuneAndSave(payload.TaskPrompt, payload.Tags, allSamples, localDir); err != nil {
		database.UpdateModelStatus(ctx, proc.db, payload.ModelId, database.ModelFailed) //nolint:errcheck
		slog.Error("error finetuning model", "base_model_id", payload.BaseModelId, "model_id", payload.ModelId, "error", err)
		return fmt.Errorf("error finetuning model: %w", err)
	}

	slog.Info("finetuning completed", "model_id", payload.ModelId, "base_model_id", payload.BaseModelId)

	if err := proc.storage.UploadDir(ctx, proc.modelBucket, payload.ModelId.String(), localDir); err != nil {
		database.UpdateModelStatus(ctx, proc.db, payload.ModelId, database.ModelFailed) //nolint:errcheck
		slog.Error("error uploading finetuned model to S3", "model_id", payload.ModelId, "error", err)
		return fmt.Errorf("error uploading model to S3: %w", err)
	}

	slog.Info("finetuned model uploaded", "model_id", payload.ModelId, "base_model_id", payload.BaseModelId)

	if err := database.UpdateModelStatus(ctx, proc.db, payload.ModelId, database.ModelTrained); err != nil {
		return fmt.Errorf("error updating model status after finetuning: %w", err)
	}

	slog.Info("finetuning completed", "model_id", payload.ModelId, "base_model_id", payload.BaseModelId)

	return nil
}
