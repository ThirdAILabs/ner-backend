package core

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"ner-backend/internal/core/types"
	"ner-backend/internal/database"
	"ner-backend/internal/licensing"
	"ner-backend/internal/messaging"
	"ner-backend/internal/storage"
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
	storage   storage.Provider
	publisher messaging.Publisher
	reciever  messaging.Reciever

	licensing licensing.LicenseVerifier

	localModelDir string
	modelBucket   string
	modelLoaders  map[string]ModelLoader
}

const bytesPerMB = 1024 * 1024

var ExcludedTags = map[string]struct{}{
	"GENDER":             {},
	"SEXUAL_ORIENTATION": {},
	"ETHNICITY":          {},
	"SERVICE_CODE":       {},
}

func NewTaskProcessor(db *gorm.DB, storage storage.Provider, publisher messaging.Publisher, reciever messaging.Reciever, licenseVerifier licensing.LicenseVerifier, localModelDir string, modelBucket string, modelLoaders map[string]ModelLoader) *TaskProcessor {
	return &TaskProcessor{
		db:            db,
		storage:       storage,
		publisher:     publisher,
		reciever:      reciever,
		licensing:     licenseVerifier,
		localModelDir: localModelDir,
		modelBucket:   modelBucket,
		modelLoaders:  modelLoaders,
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

func (proc *TaskProcessor) getStorageClient(report *database.Report) (storage.Provider, error) {
	if report.IsUpload {
		return proc.storage, nil
	}

	// If we are not using the internal storage provider, then we assume allow the client
	// to load credentials from the environment, either environment variables or IAM roles, etc.
	s3Client, err := storage.NewS3Provider(storage.S3ProviderConfig{
		S3EndpointURL: report.S3Endpoint.String,
		S3Region:      report.S3Region.String,
	})
	if err != nil {
		slog.Error("error connecting to S3", "s3_endpoint", report.S3Endpoint.String, "region", report.S3Region.String, "error", err)
		return nil, fmt.Errorf("error connecting to S3: %w", err)
	}
	return s3Client, nil
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

	if err := proc.licensing.VerifyLicense(ctx); err != nil {
		slog.Error("license verification failed", "error", err)
		database.UpdateInferenceTaskStatus(ctx, proc.db, reportId, payload.TaskId, database.JobFailed) //nolint:errcheck
		database.SaveReportError(ctx, proc.db, reportId, fmt.Sprintf("license verification failed: %s", err.Error()))
		return err
	}

	s3Objects := strings.Split(task.SourceS3Keys, ";")

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

	storage, err := proc.getStorageClient(task.Report)
	if err != nil {
		return err
	}

	totalTokens, workerErr := proc.runInferenceOnBucket(ctx, taskId, task.ReportId, storage, task.Report.Model.Id, task.Report.Model.Type, tags, customTags, groupToQuery, task.Report.SourceS3Bucket, s3Objects)

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

type objectChunkStream struct {
	object string
	chunks <-chan ParsedChunk
	err    error
}

func (proc *TaskProcessor) runInferenceOnBucket(
	ctx context.Context,
	taskId int,
	reportId uuid.UUID,
	storage storage.Provider,
	modelId uuid.UUID,
	modelType string,
	tags map[string]struct{},
	customTags map[string]string,
	groupToQuery map[uuid.UUID]string,
	bucket string,
	objects []string,
) (int64, error) {
	var totalTokens int64
	parser := NewDefaultParser()

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

	queue := make(chan objectChunkStream, 1)

	go func() {
		defer close(queue)
		for _, object := range objects {
			objectStream, err := storage.GetObjectStream(bucket, object)
			if err != nil {
				queue <- objectChunkStream{object: object, chunks: nil, err: err}
				continue
			}

			chunks := parser.Parse(object, objectStream)
			queue <- objectChunkStream{object: object, chunks: chunks, err: nil}
		}
	}()

	objectErrorCnt := 0

	for object := range queue {
		if object.err != nil {
			slog.Error("error getting object stream", "bucket", bucket, "object", object.object, "error", err)
			objectErrorCnt++
			if err := proc.updateFileCount(reportId, false); err != nil {
				return totalTokens, err
			}
			continue
		}

		result, err := proc.runInferenceOnObject(reportId, object.chunks, model, tags, customTagsRe, groupToFilter, object.object)

		if err != nil {
			slog.Error("error processing object", "object", object.object, "error", err)
			objectErrorCnt++
			if err := proc.updateFileCount(reportId, false); err != nil {
				return totalTokens, err
			}
			continue
		}

		if err := proc.db.CreateInBatches(&result.Entities, 100).Error; err != nil {
			slog.Error("error saving entities to database", "object", object.object, "error", err)
			objectErrorCnt++
			if err := proc.updateFileCount(reportId, false); err != nil {
				return totalTokens, err
			}
			continue
		}

		if err := proc.db.CreateInBatches(result.Groups, 100).Error; err != nil {
			slog.Error("error saving groups to database", "object", object.object, "error", err)
			objectErrorCnt++
			if err := proc.updateFileCount(reportId, false); err != nil {
				return totalTokens, err
			}
			continue
		}

		if err := proc.updateInferenceTagCount(reportId, result.TagCount, false); err != nil {
			slog.Error("error updating tag count", "object", object.object, "error", err)
			objectErrorCnt++
			if err := proc.updateFileCount(reportId, false); err != nil {
				return totalTokens, err
			}
			continue
		}

		if err := proc.updateInferenceTagCount(reportId, result.CustomTagCount, true); err != nil {
			slog.Error("error updating custom tag count", "object", object.object, "error", err)
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
	}

	if objectErrorCnt > 0 {
		return totalTokens, fmt.Errorf("errors while processing %d/%d objects", objectErrorCnt, len(objects))
	}

	return totalTokens, nil
}

func (proc *TaskProcessor) getModelDir(modelId uuid.UUID) string {
	return filepath.Join(proc.localModelDir, modelId.String())
}

func (proc *TaskProcessor) loadModel(ctx context.Context, modelId uuid.UUID, modelType string) (Model, error) {

	var localDir string

	if IsStatelessModel(modelType) {
		localDir = ""
	} else {
		localDir = proc.getModelDir(modelId)

		if _, err := os.Stat(localDir); os.IsNotExist(err) {
			slog.Info("model not found locally, downloading from S3", "modelId", modelId)

			if err := proc.storage.DownloadDir(ctx, proc.modelBucket, modelId.String(), localDir); err != nil {
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
) error {
	if proc.db == nil {
		return nil
	}

	spans, err := model.Predict(previewText)
	if err != nil {
		return fmt.Errorf("preview inference error: %w", err)
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
	maxEntityGap := 3
	flattenedEntities := make([]types.Entity, 0)
	coalescedEntities := make([]types.Entity, 0)

	for _, ents := range labelToEntities {
		flattenedEntities = append(flattenedEntities, ents...)
	}

	sort.Slice(flattenedEntities, func(i, j int) bool {
		return flattenedEntities[i].Start < flattenedEntities[j].Start
	})

	currentEnt := flattenedEntities[0]
	for i := 1; i < len(flattenedEntities); i++ {
		if currentEnt.Label != flattenedEntities[i].Label {
			// current entity does not match the next one
			coalescedEntities = append(coalescedEntities, currentEnt)
			currentEnt = flattenedEntities[i]
			continue
		}

		if currentEnt.End >= flattenedEntities[i].Start {
			if currentEnt.End < flattenedEntities[i].End {
				// Entities partially overlap
				currentEnt.End = flattenedEntities[i].End
				currentEnt.Text += flattenedEntities[i].Text[currentEnt.End-flattenedEntities[i].Start:]
				currentEnt.RContext = flattenedEntities[i].RContext
			}
		} else if flattenedEntities[i].Start-currentEnt.End <= min(maxEntityGap, len(currentEnt.RContext)) {
			betweenText := currentEnt.RContext[:flattenedEntities[i].Start-currentEnt.End]
			currentEnt.End = flattenedEntities[i].End
			currentEnt.Text += betweenText + flattenedEntities[i].Text
			currentEnt.RContext = flattenedEntities[i].RContext
		} else {
			// Entities are not adjacent
			coalescedEntities = append(coalescedEntities, currentEnt)
			currentEnt = flattenedEntities[i]
		}
	}
	// last entry
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
	chunks <-chan ParsedChunk,
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
	if err := proc.createObjectPreview(reportId, object, previewText, model); err != nil {
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

	if err := proc.licensing.VerifyLicense(ctx); err != nil {
		slog.Error("license verification failed", "error", err)
		database.UpdateShardDataTaskStatus(ctx, proc.db, reportId, database.JobFailed) //nolint:errcheck
		database.SaveReportError(ctx, proc.db, reportId, fmt.Sprintf("license verification failed: %s", err.Error()))
		return err
	}

	slog.Info("Handling generate tasks", "jobId", task.ReportId, "sourceBucket", task.Report.SourceS3Bucket, "sourcePrefix", task.Report.SourceS3Prefix)

	targetBytes := task.ChunkTargetBytes
	if targetBytes <= 0 {
		targetBytes = 10 * 1024 * 1024 * 1024 // Default 10GB if not set or invalid
		slog.Info("Using default chunk target size", "targetBytes", targetBytes, "jobId", reportId)
	}

	createInferenceTask := func(ctx context.Context, taskId int, chunkKeys []string, chunkSize int64) error {
		slog.Info("Creating inference task", "report_id", reportId, "task_id", taskId, "chunk_size", chunkSize)

		task := database.InferenceTask{
			ReportId:     reportId,
			TaskId:       taskId,
			Status:       database.JobQueued,
			CreationTime: time.Now().UTC(),
			SourceS3Keys: strings.Join(chunkKeys, ";"),
			TotalSize:    chunkSize,
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

		slog.Info("Created inference task", "report_id", reportId, "task_id", taskId, "chunk_size", chunkSize)

		return nil
	}

	storage, err := proc.getStorageClient(task.Report)
	if err != nil {
		return err
	}

	var currentChunkKeys []string
	var currentChunkSize int64 = 0
	var taskId int = 0
	var totalFiles int = 0

	for obj, err := range storage.IterObjects(ctx, task.Report.SourceS3Bucket, task.Report.SourceS3Prefix.String) {
		if err != nil {
			slog.Error("error iterating over S3 objects", "bucket", task.Report.SourceS3Bucket, "prefix", task.Report.SourceS3Prefix.String, "error", err)
			database.UpdateShardDataTaskStatus(ctx, proc.db, reportId, database.JobFailed) //nolint:errcheck
			return fmt.Errorf("error iterating over S3 objects: %w", err)
		}

		totalFiles++

		if currentChunkSize+obj.Size > targetBytes && len(currentChunkKeys) > 0 {
			if err := createInferenceTask(ctx, taskId, currentChunkKeys, currentChunkSize); err != nil {
				return err
			}
			currentChunkKeys = []string{}
			currentChunkSize = 0
			taskId++
		}

		currentChunkKeys = append(currentChunkKeys, obj.Name)
		currentChunkSize += obj.Size
	}

	if len(currentChunkKeys) > 0 {
		if err := createInferenceTask(ctx, taskId, currentChunkKeys, currentChunkSize); err != nil {
			return err
		}
		taskId++
	}

	if err := proc.db.
		Model(&database.Report{}).
		Where("id = ?", reportId).
		UpdateColumn("total_file_count", totalFiles).
		Error; err != nil {
		slog.Warn("failed to update total_file_count", "report_id", reportId, "totalFiles", totalFiles, "error", err)
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

func (proc *TaskProcessor) processFinetuneTask(ctx context.Context, payload messaging.FinetuneTaskPayload) error {
	database.UpdateModelStatus(ctx, proc.db, payload.ModelId, database.ModelTraining) //nolint:errcheck

	baseModel, err := proc.getModel(ctx, payload.BaseModelId)
	if err != nil {
		database.UpdateModelStatus(ctx, proc.db, payload.ModelId, database.ModelFailed) //nolint:errcheck
		slog.Error("error getting base model", "base_model_id", payload.BaseModelId, "model_id", payload.ModelId, "error", err)
		return err
	}

	model, err := proc.loadModel(ctx, payload.BaseModelId, baseModel.Type)
	if err != nil {
		database.UpdateModelStatus(ctx, proc.db, payload.ModelId, database.ModelFailed) //nolint:errcheck
		slog.Error("error loading base model", "base_model_id", payload.BaseModelId, "model_id", payload.ModelId, "error", err)
		return fmt.Errorf("error loading base model: %w", err)
	}
	defer model.Release()

	if err := model.Finetune(payload.TaskPrompt, payload.Tags, payload.Samples); err != nil {
		database.UpdateModelStatus(ctx, proc.db, payload.ModelId, database.ModelFailed) //nolint:errcheck
		slog.Error("error finetuning model", "base_model_id", payload.BaseModelId, "model_id", payload.ModelId, "error", err)
		return fmt.Errorf("error finetuning model: %w", err)
	}

	localDir := proc.getModelDir(payload.ModelId)
	if err := os.MkdirAll(localDir, os.ModePerm); err != nil {
		database.UpdateModelStatus(ctx, proc.db, payload.ModelId, database.ModelFailed) //nolint:errcheck
		slog.Error("error creating local model directory", "model_id", payload.ModelId, "error", err)
		return fmt.Errorf("error creating local model directory: %w", err)
	}

	if err := model.Save(localDir); err != nil {
		database.UpdateModelStatus(ctx, proc.db, payload.ModelId, database.ModelFailed) //nolint:errcheck
		slog.Error("error saving finetuned model locally", "model_id", payload.ModelId, "error", err)
		return fmt.Errorf("error saving finetuned model: %w", err)
	}

	if err := proc.storage.UploadDir(ctx, proc.modelBucket, payload.ModelId.String(), localDir); err != nil {
		database.UpdateModelStatus(ctx, proc.db, payload.ModelId, database.ModelFailed) //nolint:errcheck
		slog.Error("error uploading finetuned model to S3", "model_id", payload.ModelId, "error", err)
		return fmt.Errorf("error uploading model to S3: %w", err)
	}

	if err := database.UpdateModelStatus(ctx, proc.db, payload.ModelId, database.ModelTrained); err != nil {
		return fmt.Errorf("error updating model status after finetuning: %w", err)
	}

	return nil
}
