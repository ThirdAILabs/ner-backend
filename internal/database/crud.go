package database

import (
	"context"
	"database/sql"
	"fmt"
	"log"

	"ner_backend/pkg/models" // Adjust import path

	"github.com/jackc/pgx/v5/pgxpool"
)

type Queries struct {
	pool *pgxpool.Pool
}

func NewQueries(pool *pgxpool.Pool) *Queries {
	return &Queries{pool: pool}
}

func (q *Queries) CreateModel(ctx context.Context, modelID, sourceDataPath string) (*models.Model, error) {
	query := `
        INSERT INTO models (model_id, source_data_path, status, creation_time)
        VALUES ($1, $2, $3, NOW())
        RETURNING model_id, status, source_data_path, model_artifact_path, creation_time, completion_time`

	row := q.pool.QueryRow(ctx, query, modelID, sql.NullString{String: sourceDataPath, Valid: sourceDataPath != ""}, models.StatusPending)

	var m models.Model
	err := row.Scan(&m.ModelID, &m.Status, &m.SourceDataPath, &m.ModelArtifactPath, &m.CreationTime, &m.CompletionTime)
	if err != nil {
		return nil, fmt.Errorf("error creating model: %w", err)
	}
	return &m, nil
}

func (q *Queries) GetModel(ctx context.Context, modelID string) (*models.Model, error) {
	query := `SELECT model_id, status, source_data_path, model_artifact_path, creation_time, completion_time FROM models WHERE model_id = $1`
	row := q.pool.QueryRow(ctx, query, modelID)

	var m models.Model
	err := row.Scan(&m.ModelID, &m.Status, &m.SourceDataPath, &m.ModelArtifactPath, &m.CreationTime, &m.CompletionTime)
	if err != nil {
		if err == sql.ErrNoRows || err.Error() == "no rows in result set" { // pgx might return different error text
			return nil, nil // Not found
		}
		return nil, fmt.Errorf("error getting model %s: %w", modelID, err)
	}
	return &m, nil
}

func (q *Queries) UpdateModelStatus(ctx context.Context, modelID string, status models.ModelStatus, artifactPath string) error {
	query := `
        UPDATE models
        SET status = $2::VARCHAR(50),
            model_artifact_path = CASE WHEN $3 <> '' THEN $3 ELSE model_artifact_path END,
            completion_time = CASE WHEN $2 = $4 OR $2 = $5 THEN NOW() ELSE completion_time END
        WHERE model_id = $1`

	_, err := q.pool.Exec(ctx, query, modelID, status, artifactPath, models.StatusTrained, models.StatusFailed)
	if err != nil {
		return fmt.Errorf("error updating model %s status to %s: %w", modelID, status, err)
	}
	log.Printf("Updated model %s status to %s", modelID, status)
	return nil
}

func (q *Queries) CreateInferenceJob(ctx context.Context, jobID, modelID, sourceBucket, sourcePrefix, destBucket string) (*models.InferenceJob, error) {
	resultPrefix := fmt.Sprintf("s3://%s/results/%s/", destBucket, jobID)
	query := `
        INSERT INTO inference_jobs (job_id, model_id, source_s3_bucket, source_s3_prefix, dest_s3_bucket, result_s3_prefix, status, creation_time)
        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
        RETURNING job_id, model_id, source_s3_bucket, source_s3_prefix, dest_s3_bucket, result_s3_prefix, status, creation_time, completion_time`

	row := q.pool.QueryRow(ctx, query,
		jobID, modelID, sourceBucket, sql.NullString{String: sourcePrefix, Valid: sourcePrefix != ""},
		destBucket, sql.NullString{String: resultPrefix, Valid: resultPrefix != ""}, models.JobPending)

	var job models.InferenceJob
	// Ensure scan order matches RETURNING clause
	err := row.Scan(&job.JobID, &job.ModelID, &job.SourceS3Bucket, &job.SourceS3Prefix, &job.DestS3Bucket, &job.ResultS3Prefix, &job.Status, &job.CreationTime, &job.CompletionTime)
	if err != nil {
		return nil, fmt.Errorf("error creating inference job: %w", err)
	}
	// Manually set fields not returned by DB insert if needed
	job.DestS3Bucket = destBucket
	return &job, nil
}

func (q *Queries) GetInferenceJob(ctx context.Context, jobID string) (*models.InferenceJob, error) {
	query := `SELECT job_id, model_id, source_s3_bucket, source_s3_prefix, dest_s3_bucket, result_s3_prefix, status, creation_time, completion_time FROM inference_jobs WHERE job_id = $1`
	row := q.pool.QueryRow(ctx, query, jobID)

	var job models.InferenceJob
	err := row.Scan(&job.JobID, &job.ModelID, &job.SourceS3Bucket, &job.SourceS3Prefix, &job.DestS3Bucket, &job.ResultS3Prefix, &job.Status, &job.CreationTime, &job.CompletionTime)

	if err != nil {
		if err == sql.ErrNoRows || err.Error() == "no rows in result set" {
			return nil, nil // Not found
		}
		return nil, fmt.Errorf("error getting inference job %s: %w", jobID, err)
	}
	return &job, nil
}

func (q *Queries) UpdateInferenceJobStatus(ctx context.Context, jobID string, status models.JobStatus) error {
	query := `
        UPDATE inference_jobs
        SET status = $2,
            completion_time = CASE WHEN $2 = $3 OR $2 = $4 THEN NOW() ELSE completion_time END
        WHERE job_id = $1`

	_, err := q.pool.Exec(ctx, query, jobID, status, models.JobCompleted, models.JobFailed)
	if err != nil {
		return fmt.Errorf("error updating job %s status to %s: %w", jobID, status, err)
	}
	log.Printf("Updated job %s status to %s", jobID, status)
	return nil
}
