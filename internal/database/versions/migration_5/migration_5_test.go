package migration_5

import (
	"database/sql"
	"encoding/json"
	"testing"
	"time"

	"ner-backend/internal/storage"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
)

type OldReport struct {
	Id         uuid.UUID `gorm:"type:uuid;primaryKey"`
	ReportName string    `gorm:"not null"`
	ModelId    uuid.UUID `gorm:"type:uuid"`
	Deleted    bool      `gorm:"default:false"`
	Stopped    bool      `gorm:"default:false"`

	// Old source fields
	S3Endpoint     sql.NullString
	S3Region       sql.NullString
	SourceS3Bucket string
	SourceS3Prefix sql.NullString
	IsUpload       bool

	CreationTime       time.Time
	SucceededFileCount int `gorm:"default:0"`
	FailedFileCount    int `gorm:"default:0"`
	TotalFileCount     int `gorm:"default:0"`
}

// Override the default name, which is "old_reports" (plural snake case of struct name)
func (OldReport) TableName() string {
	return "reports"
}

type OldInferenceTask struct {
	ReportId uuid.UUID `gorm:"type:uuid;primaryKey"`
	TaskId   int       `gorm:"primaryKey"`
	Status   string    `gorm:"size:20;not null"`
	CreationTime time.Time
	StartTime sql.NullTime
	CompletionTime sql.NullTime

	// Old source field
	SourceS3Keys string

	TotalSize  int64
	TokenCount int64 `gorm:"not null;default:0"`
}

// Override the default name, which is "old_inference_tasks" (plural snake case of struct name)
func (OldInferenceTask) TableName() string {
	return "inference_tasks"
}

func setupTestDB(t *testing.T) *gorm.DB {
	db, err := gorm.Open(sqlite.Open(":memory:"), &gorm.Config{})
	require.NoError(t, err)

	err = db.AutoMigrate(&OldReport{}, &OldInferenceTask{})
	require.NoError(t, err)

	return db
}

func TestMigration_Reports(t *testing.T) {
	db := setupTestDB(t)

	// Create test data with old schema
	reportID := uuid.New()
	modelID := uuid.New()

	s3Report := OldReport{
		Id:              reportID,
		ReportName:      "test-s3-report",
		ModelId:         modelID,
		S3Endpoint:      sql.NullString{String: "https://s3.amazonaws.com", Valid: true},
		S3Region:        sql.NullString{String: "us-east-1", Valid: true},
		SourceS3Bucket:  "test-bucket",
		SourceS3Prefix:  sql.NullString{String: "test/prefix", Valid: true},
		IsUpload:        false,
		CreationTime:    time.Now(),
	}

	err := db.Create(&s3Report).Error
	require.NoError(t, err)

	err = Migration(db)
	require.NoError(t, err)

	// Verify the transformation
	var result struct {
		StorageType   string
		StorageParams string
	}

	err = db.Raw("SELECT storage_type, storage_params FROM reports WHERE id = ?", reportID).Scan(&result).Error
	require.NoError(t, err)

	assert.Equal(t, storage.S3ConnectorType, result.StorageType)

	var params storage.S3ConnectorParams
	err = json.Unmarshal([]byte(result.StorageParams), &params)
	require.NoError(t, err)

	assert.Equal(t, "https://s3.amazonaws.com", params.Endpoint)
	assert.Equal(t, "us-east-1", params.Region)
	assert.Equal(t, "test-bucket", params.Bucket)
	assert.Equal(t, "test/prefix", params.Prefix)
}

func TestMigration_InferenceTasks(t *testing.T) {
	db := setupTestDB(t)

	// Create test report and inference task
	reportID := uuid.New()
	modelID := uuid.New()

	report := OldReport{
		Id:              reportID,
		ReportName:      "test-report",
		ModelId:         modelID,
		IsUpload:        false, // S3 case
		CreationTime:    time.Now(),
	}

	err := db.Create(&report).Error
	require.NoError(t, err)

	task := OldInferenceTask{
		ReportId:     reportID,
		TaskId:       1,
		SourceS3Keys: "key1;key2;key3",
		Status:       "QUEUED",
		CreationTime: time.Now(),
		TotalSize:    1000,
		TokenCount:   100,
	}

	err = db.Create(&task).Error
	require.NoError(t, err)

	err = Migration(db)
	require.NoError(t, err)

	// Verify the transformation
	var result struct {
		StorageParams string
	}

	err = db.Raw("SELECT storage_params FROM inference_tasks WHERE report_id = ? AND task_id = ?", reportID, 1).Scan(&result).Error
	require.NoError(t, err)

	var params storage.S3ConnectorTaskParams
	err = json.Unmarshal([]byte(result.StorageParams), &params)
	require.NoError(t, err)

	assert.Equal(t, []string{"key1", "key2", "key3"}, params.ChunkKeys)
}

func TestMigration_ColumnRemoval(t *testing.T) {
	db := setupTestDB(t)

	// Create test data
	reportID := uuid.New()
	modelID := uuid.New()

	report := OldReport{
		Id:              reportID,
		ReportName:      "test-report",
		ModelId:         modelID,
		S3Endpoint:      sql.NullString{String: "https://s3.amazonaws.com", Valid: true},
		S3Region:        sql.NullString{String: "us-east-1", Valid: true},
		SourceS3Bucket:  "test-bucket",
		SourceS3Prefix:  sql.NullString{String: "test/prefix", Valid: true},
		IsUpload:        false,
		CreationTime:    time.Now(),
	}

	err := db.Create(&report).Error
	require.NoError(t, err)

	err = Migration(db)
	require.NoError(t, err)

	// Verify old columns are removed
	var columnExists bool
	
	err = db.Raw("SELECT COUNT(*) > 0 FROM pragma_table_info('reports') WHERE name = 's3_endpoint'").Scan(&columnExists).Error
	require.NoError(t, err)
	assert.False(t, columnExists, "s3_endpoint column should be removed")

	err = db.Raw("SELECT COUNT(*) > 0 FROM pragma_table_info('inference_tasks') WHERE name = 'source_s3_keys'").Scan(&columnExists).Error
	require.NoError(t, err)
	assert.False(t, columnExists, "source_s3_keys column should be removed")
} 