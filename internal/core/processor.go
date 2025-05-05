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
	"ner-backend/internal/s3"
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
	s3Client  *s3.Client
	publisher messaging.Publisher
	reciever  messaging.Reciever

	licensing licensing.LicenseVerifier

	localModelDir string
	modelBucket   string
}

func NewTaskProcessor(db *gorm.DB, s3client *s3.Client, publisher messaging.Publisher, reciever messaging.Reciever, licenseVerifier licensing.LicenseVerifier, localModelDir string, modelBucket string) *TaskProcessor {
	return &TaskProcessor{
		db:            db,
		s3Client:      s3client,
		publisher:     publisher,
		reciever:      reciever,
		licensing:     licenseVerifier,
		localModelDir: localModelDir,
		modelBucket:   modelBucket,
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

func (proc *TaskProcessor) processInferenceTask(ctx context.Context, payload messaging.InferenceTaskPayload) error {
	reportId := payload.ReportId

	slog.Info("processing inference task", "report_id", reportId, "task_id", payload.TaskId)
	database.UpdateInferenceTaskStatus(ctx, proc.db, reportId, payload.TaskId, database.JobRunning) //nolint:errcheck

	// if err := proc.licensing.VerifyLicense(ctx); err != nil {
	// 	slog.Error("license verification failed", "error", err)
	// 	database.UpdateInferenceTaskStatus(ctx, proc.db, reportId, payload.TaskId, database.JobFailed) //nolint:errcheck
	// 	database.SaveReportError(ctx, proc.db, reportId, fmt.Sprintf("license verification failed: %s", err.Error()))
	// 	return err
	// }

	var task database.InferenceTask
	if err := proc.db.Preload("Report").Preload("Report.Model").Preload("Report.Tags").Preload("Report.CustomTags").Preload("Report.Groups").First(&task, "report_id = ? AND task_id = ?", reportId, payload.TaskId).Error; err != nil {
		slog.Error("error fetching inference task", "report_id", reportId, "task_id", payload.TaskId, "error", err)
		return fmt.Errorf("error getting inference task: %w", err)
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

	workerErr := proc.runInferenceOnBucket(task.ReportId, task.Report.Model.Id, task.Report.Model.Type, tags, customTags, groupToQuery, task.SourceS3Bucket, s3Objects)
	if workerErr != nil {
		slog.Error("error running inference task", "report_id", reportId, "task_id", payload.TaskId, "error", workerErr)
		database.UpdateInferenceTaskStatus(ctx, proc.db, reportId, payload.TaskId, database.JobFailed) // nolint:errcheck
		return fmt.Errorf("error running inference task: %w", workerErr)
	}

	if err := database.UpdateInferenceTaskStatus(ctx, proc.db, reportId, payload.TaskId, database.JobCompleted); err != nil {
		return fmt.Errorf("error updating inference task status to complete: %w", err)
	}

	slog.Info("inference task completed successfully", "report_id", reportId, "task_id", payload.TaskId)

	return nil
}

func (proc *TaskProcessor) runInferenceOnBucket(
	reportId uuid.UUID,
	modelId uuid.UUID,
	modelType string,
	tags map[string]struct{},
	customTags map[string]string,
	groupToQuery map[uuid.UUID]string,
	bucket string,
	objects []string,
) error {
	parser := NewDefaultParser()

	model, err := proc.loadModel(modelId, modelType)
	if err != nil {
		return err
	}
	defer model.Release()

	groupToFilter := make(map[uuid.UUID]Filter)
	for groupId, query := range groupToQuery {
		filter, err := ParseQuery(query)
		if err != nil {
			return fmt.Errorf("error loading model: %w", err)
		}
		groupToFilter[groupId] = filter
	}

	customTagsRe := make(map[string]*regexp.Regexp)
	for tag, pat := range customTags {
		re, err := regexp.Compile(pat)
		if err != nil {
			return fmt.Errorf("error compiling regex for tag %s: %w", tag, err)
		}
		customTagsRe[tag] = re
	}

	objectErrorCnt := 0

	for _, object := range objects {
		entities, groups, err := proc.runInferenceOnObject(reportId, parser, model, tags, customTagsRe, groupToFilter, bucket, object)
		fmt.Println("entities", entities)
		fmt.Println("groups", groups)
		if err != nil {
			slog.Error("error processing object", "object", object, "error", err)
			objectErrorCnt++
			continue
		}

		if err := proc.db.CreateInBatches(&entities, 100).Error; err != nil {
			slog.Error("error saving entities to database", "object", object, "error", err)
			objectErrorCnt++
			continue
		}

		if err := proc.db.CreateInBatches(groups, 100).Error; err != nil {
			slog.Error("error saving groups to database", "object", object, "error", err)
			objectErrorCnt++
			continue
		}
	}

	if objectErrorCnt > 0 {
		return fmt.Errorf("errors while processing %d/%d objects", objectErrorCnt, len(objects))
	}

	return nil
}

func (proc *TaskProcessor) getlocalModelDir(modelId uuid.UUID) string {
	return filepath.Join(proc.localModelDir, modelId.String())
}

func (proc *TaskProcessor) localModelPath(modelId uuid.UUID) string {
	return filepath.Join(proc.localModelDir, modelId.String(), "model.bin")
}

func (proc *TaskProcessor) loadModel(modelId uuid.UUID, modelType string) (Model, error) {

	var localPath string

	if IsStatelessModel(modelType) {
		localPath = ""
	} else {
		localPath = proc.localModelPath(modelId)

		// Check if the model file exists locally
		if _, err := os.Stat(localPath); os.IsNotExist(err) {
			slog.Info("model not found locally, downloading from S3", "modelId", modelId)

			modelObjectKey := filepath.Join(modelId.String(), "model.bin")
			if modelType == "python_combined_ner_model" {
				modelObjectKey = filepath.Join(modelType)
			}
			if modelType == "ensemble" {
				modelObjectKey = filepath.Join("python_ensemble_ner_model")
				localPath = proc.getlocalModelDir(modelId)
			}

			if err := proc.s3Client.DownloadFileOrFolder(context.TODO(), proc.modelBucket, modelObjectKey, localPath); err != nil {
				return nil, fmt.Errorf("failed to download model from S3: %w", err)
			}
		}
	}

	model, err := LoadModel(modelType, localPath)
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

	sort.Slice(spans, func(i, j int) bool {
		return spans[i].Start < spans[j].Start
	})

	var (
		tokens []string
		tags   []string
		cursor = 0
		length = len(previewText)
	)
	for _, e := range spans {
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

func (proc *TaskProcessor) runInferenceOnObject(
	reportId uuid.UUID,
	parser Parser,
	model Model,
	tags map[string]struct{},
	customTags map[string]*regexp.Regexp,
	groupFilter map[uuid.UUID]Filter,
	bucket string,
	object string) (
	[]database.ObjectEntity, []database.ObjectGroup, error) {

	chunks := parser.Parse(object, proc.s3Client.DownloadFileStream(bucket, object))

	labelToEntities := make(map[string][]types.Entity)

	const previewLimit = 1000

	previewTokens := make([]string, 0, previewLimit)

	fmt.Println("chunks", chunks)
	for chunk := range chunks {
		if chunk.Error != nil {
			return nil, nil, fmt.Errorf("error parsing document: %w", chunk.Error)
		}

		chunkEntities, err := model.Predict(chunk.Text)
		if err != nil {
			return nil, nil, fmt.Errorf("error running model inference: %w", err)
		}

		fmt.Println("tags", tags)

		for _, entity := range chunkEntities {
			if _, ok := tags[entity.Label]; ok {
				entity.Start += chunk.Offset
				entity.End += chunk.Offset
				labelToEntities[entity.Label] = append(labelToEntities[entity.Label], entity)
			} else {
				fmt.Println("label not found in tags: ", entity.Label)
			}
		}
		fmt.Println("chunkEntities after adding offset", chunkEntities)
		fmt.Println("labelToEntities after tag filter", labelToEntities)

		for tag, re := range customTags {
			matches := re.FindAllStringIndex(chunk.Text, -1)
			for _, match := range matches {
				start, end := match[0], match[1]
				labelToEntities[tag] = append(labelToEntities[tag], types.Entity{
					Label:    tag,
					Text:     chunk.Text[start:end],
					Start:    start + chunk.Offset,
					End:      end + chunk.Offset,
					LContext: chunk.Text[max(0, start-20):start],
					RContext: chunk.Text[end:min(len(chunk.Text), end+20)],
				})
			}
		}
		fmt.Println("labelToEntities after custom tag filter", labelToEntities)

		if len(previewTokens) < previewLimit {
			toks := strings.Fields(chunk.Text)
			need := previewLimit - len(previewTokens)
			if len(toks) >= need {
				previewTokens = append(previewTokens, toks[:need]...)
			} else {
				previewTokens = append(previewTokens, toks...)
			}
		}
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
	fmt.Println("labelToEntities", labelToEntities)

	allEntities := make([]database.ObjectEntity, 0)
	for _, entities := range labelToEntities {
		for _, entity := range entities {
			allEntities = append(allEntities, database.ObjectEntity{
				ReportId: reportId,
				Label:    entity.Label,
				Text:     entity.Text,
				Start:    entity.Start,
				End:      entity.End,
				Object:   object,
				LContext: entity.LContext,
				RContext: entity.RContext,
			})
		}
	}
	fmt.Println("allEntities", allEntities)

	return allEntities, groups, nil
}

func (proc *TaskProcessor) processShardDataTask(ctx context.Context, payload messaging.ShardDataPayload) error {
	reportId := payload.ReportId

	database.UpdateShardDataTaskStatus(ctx, proc.db, reportId, database.JobRunning) //nolint:errcheck

	// if err := proc.licensing.VerifyLicense(ctx); err != nil {
	// 	slog.Error("license verification failed", "error", err)
	// 	database.UpdateShardDataTaskStatus(ctx, proc.db, reportId, database.JobFailed) //nolint:errcheck
	// 	database.SaveReportError(ctx, proc.db, reportId, fmt.Sprintf("license verification failed: %s", err.Error()))
	// 	return err
	// }

	var task database.ShardDataTask
	if err := proc.db.Preload("Report").First(&task, "report_id = ?", reportId).Error; err != nil {
		slog.Error("error fetching shard data task", "report_id", reportId, "error", err)
		return fmt.Errorf("error getting shard data task: %w", err)
	}

	slog.Info("Handling generate tasks", "jobId", task.ReportId, "sourceBucket", task.Report.SourceS3Bucket, "sourcePrefix", task.Report.SourceS3Prefix)

	targetBytes := task.ChunkTargetBytes
	if targetBytes <= 0 {
		targetBytes = 10 * 1024 * 1024 * 1024 // Default 10GB if not set or invalid
		slog.Info("Using default chunk target size", "targetBytes", targetBytes, "jobId", reportId)
	}

	var taskIndex int = 0

	// Define the callback function to process each chunk
	processChunkCallback := func(ctx context.Context, chunkKeys []string, chunkSize int64) error {
		taskIndex++ // Increment count *after* successful publishing
		slog.Info("Handler: Processing chunk", "chunkIndex", taskIndex, "jobId", reportId, "chunkSize", chunkSize, "keyCount", len(chunkKeys))

		task := database.InferenceTask{
			ReportId:     reportId,
			TaskId:       taskIndex,
			Status:       database.JobQueued,
			CreationTime: time.Now().UTC(),

			SourceS3Bucket: task.Report.SourceS3Bucket,
			SourceS3Keys:   strings.Join(chunkKeys, ";"),
			TotalSize:      chunkSize,
		}

		inferencePayload := messaging.InferenceTaskPayload{
			ReportId: task.ReportId, TaskId: task.TaskId,
		}

		if err := proc.publisher.PublishInferenceTask(ctx, inferencePayload); err != nil {
			// Use slog.Error for failures
			slog.Error("Handler: Failed to publish inference chunk", "report_id", reportId, "task_id", taskIndex, "error", err)
			// Return the error so the helper function knows processing failed
			return fmt.Errorf("failed to publish inference chunk %d: %w", taskIndex, err)
		}

		if err := proc.db.WithContext(ctx).Create(&task).Error; err != nil {
			slog.Error("error saving inference task to db", "report_id", task.ReportId, "task_id", task.TaskId, "error", err)
			return fmt.Errorf("error saving inference task to db: %w", err)
		}

		return nil
	}

	processedChunks, err := proc.s3Client.ListAndChunkS3Objects(
		ctx,
		task.Report.SourceS3Bucket,
		task.Report.SourceS3Prefix.String,
		targetBytes,
		reportId,
		processChunkCallback,
	)

	if err != nil {
		slog.Error("Failed during S3 processing/chunk publishing", "report_id", reportId, "error", err)
		database.UpdateShardDataTaskStatus(ctx, proc.db, reportId, database.JobFailed) //nolint:errcheck
		return fmt.Errorf("failed during task generation for job %s: %w", reportId, err)
	}

	if processedChunks != taskIndex {
		slog.Warn("Mismatch between processed chunks and tasks queued", "processedChunks", processedChunks, "n_tasks", taskIndex, "report_id", reportId)
	}

	slog.Info("Finished generating inference task chunks", "n_tasks", taskIndex, "report_id", reportId)

	if err := database.UpdateShardDataTaskStatus(ctx, proc.db, reportId, database.JobCompleted); err != nil {
		return fmt.Errorf("failed to update job final status: %w", err)
	}

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

	model, err := proc.loadModel(payload.BaseModelId, baseModel.Type)
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

	localPath := proc.localModelPath(payload.ModelId)
	if err := os.MkdirAll(filepath.Dir(localPath), 0755); err != nil {
		database.UpdateModelStatus(ctx, proc.db, payload.ModelId, database.ModelFailed) //nolint:errcheck
		slog.Error("error creating local model directory", "model_id", payload.ModelId, "error", err)
		return fmt.Errorf("error creating local model directory: %w", err)
	}

	if err := model.Save(localPath); err != nil {
		database.UpdateModelStatus(ctx, proc.db, payload.ModelId, database.ModelFailed) //nolint:errcheck
		slog.Error("error saving finetuned model locally", "model_id", payload.ModelId, "error", err)
		return fmt.Errorf("error saving finetuned model: %w", err)
	}

	if _, err := proc.s3Client.UploadFile(ctx, localPath, proc.modelBucket, filepath.Join(payload.ModelId.String(), "model.bin")); err != nil {
		database.UpdateModelStatus(ctx, proc.db, payload.ModelId, database.ModelFailed) //nolint:errcheck
		slog.Error("error uploading finetuned model to S3", "model_id", payload.ModelId, "error", err)
		return fmt.Errorf("error uploading model to S3: %w", err)
	}

	if err := database.UpdateModelStatus(ctx, proc.db, payload.ModelId, database.ModelTrained); err != nil {
		return fmt.Errorf("error updating model status after finetuning: %w", err)
	}

	return nil
}
