package main

import (
	"errors"
	"fmt"
	"log"
	"ner_backend/internal/database" // Adjust import path
	"ner_backend/internal/messaging"
	"ner_backend/internal/s3"
	"ner_backend/pkg/models"
	"net/http"
	"strings"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/render"
	"github.com/google/uuid"
)

type APIHandler struct {
	db        *database.Queries
	publisher *messaging.TaskPublisher
	s3Client  *s3.Client
}

func NewAPIHandler(db *database.Queries, pub *messaging.TaskPublisher, s3c *s3.Client) *APIHandler {
	return &APIHandler{db: db, publisher: pub, s3Client: s3c}
}

// --- Helper Function ---
func renderError(w http.ResponseWriter, r *http.Request, statusCode int, err error, details string) {
	log.Printf("Error: %v - Details: %s", err, details)
	render.Status(r, statusCode)
	render.JSON(w, r, map[string]string{"error": err.Error(), "details": details})
}

// --- Handlers ---

func (h *APIHandler) HealthCheck(w http.ResponseWriter, r *http.Request) {
	// Could add checks for DB, RabbitMQ connectivity here
	render.JSON(w, r, map[string]string{"status": "ok"})
}

func (h *APIHandler) SubmitTrainingJob(w http.ResponseWriter, r *http.Request) {
	var req models.TrainRequest
	if err := render.DecodeJSON(r.Body, &req); err != nil {
		renderError(w, r, http.StatusBadRequest, err, "Invalid request body")
		return
	}

	if req.SourceS3Path == "" || !strings.HasPrefix(req.SourceS3Path, "s3://") {
		renderError(w, r, http.StatusBadRequest, errors.New("invalid source_s3_path"), "source_s3_path is required and must start with s3://")
		return
	}

	modelID := uuid.NewString()
	ctx := r.Context() // Use request context

	_, err := h.db.CreateModel(ctx, modelID, req.SourceS3Path)
	if err != nil {
		renderError(w, r, http.StatusInternalServerError, err, "Failed to create model entry in database")
		return
	}

	payload := models.TrainTaskPayload{
		ModelID:          modelID,
		SourceS3PathTags: req.SourceS3Path,
	}

	// Publish task in background? For now, do it directly.
	err = h.publisher.PublishTrainTask(ctx, payload)
	if err != nil {
		// Attempt to rollback DB state or mark as failed? Difficult without transactions.
		// For now, log error and return failure to client.
		log.Printf("CRITICAL: Failed to publish training task for model %s after DB entry created: %v", modelID, err)
		renderError(w, r, http.StatusInternalServerError, err, "Failed to queue training task")
		return
	}

	log.Printf("Submitted training job for model %s", modelID)
	render.Status(r, http.StatusAccepted)
	render.JSON(w, r, models.TrainSubmitResponse{Message: "Training job submitted", ModelID: modelID})
}

func (h *APIHandler) GetModelStatus(w http.ResponseWriter, r *http.Request) {
	modelID := chi.URLParam(r, "modelID")
	if modelID == "" {
		renderError(w, r, http.StatusBadRequest, errors.New("missing modelID"), "Model ID is required in URL path")
		return
	}

	ctx := r.Context()
	model, err := h.db.GetModel(ctx, modelID)
	if err != nil {
		renderError(w, r, http.StatusInternalServerError, err, "Failed to query database")
		return
	}
	if model == nil {
		renderError(w, r, http.StatusNotFound, errors.New("not found"), "Model not found")
		return
	}

	render.JSON(w, r, model) // Render the pkg/models.Model struct directly
}

func (h *APIHandler) SubmitInferenceJob(w http.ResponseWriter, r *http.Request) {
	var req models.InferenceRequest
	if err := render.DecodeJSON(r.Body, &req); err != nil {
		renderError(w, r, http.StatusBadRequest, err, "Invalid request body")
		return
	}

	if req.ModelID == "" || req.SourceS3Bucket == "" || req.DestS3Bucket == "" {
		renderError(w, r, http.StatusBadRequest, errors.New("missing required fields"), "model_id, source_s3_bucket, and dest_s3_bucket are required")
		return
	}

	ctx := r.Context()

	// 1. Check model status
	model, err := h.db.GetModel(ctx, req.ModelID)
	if err != nil {
		renderError(w, r, http.StatusInternalServerError, err, "Failed to check model status")
		return
	}
	if model == nil {
		renderError(w, r, http.StatusNotFound, errors.New("model not found"), fmt.Sprintf("Model %s not found", req.ModelID))
		return
	}
	if model.Status != models.StatusTrained {
		renderError(w, r, http.StatusBadRequest, errors.New("model not ready"), fmt.Sprintf("Model %s is not ready (status: %s)", req.ModelID, model.Status))
		return
	}
	if !model.ModelArtifactPath.Valid || model.ModelArtifactPath.String == "" {
		renderError(w, r, http.StatusInternalServerError, errors.New("model artifact path missing"), fmt.Sprintf("Model %s is trained but artifact path is missing", req.ModelID))
		return
	}

	// 2. Create inference job entry
	jobID := uuid.NewString()
	_, err = h.db.CreateInferenceJob(ctx, jobID, req.ModelID, req.SourceS3Bucket, req.SourceS3Prefix, req.DestS3Bucket)
	if err != nil {
		renderError(w, r, http.StatusInternalServerError, err, "Failed to create inference job entry")
		return
	}

	// 3. List files (Consider doing this asynchronously for large buckets)
	var filesToProcess []string
	var listErr error
	filesToProcess, listErr = h.s3Client.ListFiles(ctx, req.SourceS3Bucket, req.SourceS3Prefix)
	if listErr != nil {
		// Mark job as failed
		_ = h.db.UpdateInferenceJobStatus(ctx, jobID, models.JobFailed)
		renderError(w, r, http.StatusInternalServerError, listErr, "Failed to list source S3 location")
		return
	}

	taskCount := len(filesToProcess)
	if taskCount == 0 {
		_ = h.db.UpdateInferenceJobStatus(ctx, jobID, models.JobCompleted)
		log.Printf("No files found for inference job %s", jobID)
		render.Status(r, http.StatusOK) // OK, not Accepted, as job is immediately complete
		render.JSON(w, r, models.InferenceSubmitResponse{
			Message:   "No files found to process in specified location",
			JobID:     jobID,
			TaskCount: 0,
		})
		return
	}

	// 4. Submit tasks (Can do this in a goroutine?)
	log.Printf("Queueing %d inference tasks for job %s...", taskCount, jobID)
	publishErrors := false
	for _, s3Key := range filesToProcess {
		payload := models.InferenceTaskPayload{
			JobID:             jobID,
			ModelID:           req.ModelID,
			ModelArtifactPath: model.ModelArtifactPath.String,
			SourceS3Bucket:    req.SourceS3Bucket,
			SourceS3Key:       s3Key,
			DestS3Bucket:      req.DestS3Bucket,
		}
		// Use context from request, maybe add timeout?
		if err := h.publisher.PublishInferenceTask(ctx, payload); err != nil {
			log.Printf("ERROR: Failed to publish inference task for key %s, job %s: %v", s3Key, jobID, err)
			publishErrors = true
			// Decide whether to stop or continue queueing other tasks
			// break // Option: stop queueing on first error
		}
	}

	// 5. Update job status
	finalStatus := models.JobRunning
	if publishErrors {
		// Decide final status if some tasks failed to publish
		// Could mark as FAILED, or RUNNING_WITH_ERRORS if we had such a state
		log.Printf("WARNING: Some tasks failed to publish for job %s", jobID)
		// Keeping status as RUNNING for now, assuming some tasks might have queued
	}

	err = h.db.UpdateInferenceJobStatus(ctx, jobID, finalStatus)
	if err != nil {
		// Log error, but tasks might already be running
		log.Printf("ERROR: Failed to update job %s status to %s after queueing: %v", jobID, finalStatus, err)
	}

	log.Printf("Submitted inference job %s with %d tasks (Publish errors: %v)", jobID, taskCount, publishErrors)
	render.Status(r, http.StatusAccepted)
	render.JSON(w, r, models.InferenceSubmitResponse{
		Message:   fmt.Sprintf("Inference job submitted with %d tasks", taskCount),
		JobID:     jobID,
		TaskCount: taskCount,
	})
}

func (h *APIHandler) GetInferenceJobStatus(w http.ResponseWriter, r *http.Request) {
	jobID := chi.URLParam(r, "jobID")
	if jobID == "" {
		renderError(w, r, http.StatusBadRequest, errors.New("missing jobID"), "Job ID is required in URL path")
		return
	}

	ctx := r.Context()
	job, err := h.db.GetInferenceJob(ctx, jobID)
	if err != nil {
		renderError(w, r, http.StatusInternalServerError, err, "Failed to query database")
		return
	}
	if job == nil {
		renderError(w, r, http.StatusNotFound, errors.New("not found"), "Inference job not found")
		return
	}

	// Optional: Add task counts here by querying DB if you track individual tasks
	// job.TaskCount = ... // This requires adding task tracking

	render.JSON(w, r, job)
}
