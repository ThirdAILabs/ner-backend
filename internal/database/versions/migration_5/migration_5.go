package migration_5

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	m0 "ner-backend/internal/database/versions/migration_0"

	"ner-backend/internal/storage"

	"github.com/google/uuid"
	"gorm.io/datatypes"
	"gorm.io/gorm"
)

type InferenceTask struct {
	ReportId uuid.UUID `gorm:"type:uuid;primaryKey"`
	TaskId   int       `gorm:"primaryKey"`
	Report   *Report   `gorm:"foreignKey:ReportId;constraint:OnDelete:CASCADE"`

	Status         string `gorm:"size:20;not null"`
	CreationTime   time.Time
	StartTime      sql.NullTime
	CompletionTime sql.NullTime

	// Old source field
	SourceS3Keys string

	// New source field. Default value only for migration.
	SourceParams datatypes.JSON `gorm:"type:jsonb;not null;default:'{}'"`

	TotalSize    int64
	TokenCount   int64 `gorm:"not null;default:0"`
}

type Report struct {
	Id         uuid.UUID `gorm:"type:uuid;primaryKey"`
	ReportName string    `gorm:"not null"`

	ModelId uuid.UUID `gorm:"type:uuid"`
	Model   *m0.Model `gorm:"foreignKey:ModelId"`

	Deleted bool `gorm:"default:false"`
	Stopped bool `gorm:"default:false"`

	// Old source fields
	S3Endpoint     sql.NullString
	S3Region       sql.NullString
	SourceS3Bucket string
	SourceS3Prefix sql.NullString
	IsUpload       bool
	
	// New source fields. Default value only for migration.
	SourceType     string `gorm:"size:20;not null;default:''"`
	SourceParams   datatypes.JSON `gorm:"type:jsonb;not null;default:'{}'"`

	CreationTime       time.Time
	SucceededFileCount int `gorm:"default:0"`
	FailedFileCount    int `gorm:"default:0"`
	TotalFileCount     int `gorm:"default:0"`

	Tags       []m0.ReportTag `gorm:"foreignKey:ReportId;constraint:OnDelete:CASCADE"`
	CustomTags []m0.CustomTag `gorm:"foreignKey:ReportId;constraint:OnDelete:CASCADE"`

	Groups []m0.Group `gorm:"foreignKey:ReportId;constraint:OnDelete:CASCADE"`

	ShardDataTask  *m0.ShardDataTask  `gorm:"foreignKey:ReportId;constraint:OnDelete:CASCADE"`
	InferenceTasks []InferenceTask `gorm:"foreignKey:ReportId;constraint:OnDelete:CASCADE"`

	Errors []m0.ReportError `gorm:"foreignKey:ReportId;constraint:OnDelete:CASCADE"`
}

func Migration(db *gorm.DB) error {
	// Add new columns
	if err := db.Migrator().AddColumn(&Report{}, "source_type"); err != nil {
		return fmt.Errorf("error adding SourceType column: %w", err)
	}
	
	if err := db.Migrator().AddColumn(&Report{}, "source_params"); err != nil {
		return fmt.Errorf("error adding SourceParams column: %w", err)
	}

	if err := db.Migrator().AddColumn(&InferenceTask{}, "source_params"); err != nil {
		return fmt.Errorf("error adding SourceParams column: %w", err)
	}

	// Perform custom data transformation logic
	if err := transformReports(db); err != nil {
		return fmt.Errorf("error transforming reports: %w", err)
	}

	if err := transformInferenceTasks(db); err != nil {
		return fmt.Errorf("error transforming inference tasks: %w", err)
	}
	
	// Remove old columns
	if err := db.Migrator().DropColumn(&Report{}, "s3_endpoint"); err != nil {
		return fmt.Errorf("error dropping S3Endpoint column: %w", err)
	}

	if err := db.Migrator().DropColumn(&Report{}, "s3_region"); err != nil {
		return fmt.Errorf("error dropping S3Region column: %w", err)
	}
	
	if err := db.Migrator().DropColumn(&Report{}, "source_s3_bucket"); err != nil {
		return fmt.Errorf("error dropping SourceS3Bucket column: %w", err)
	}

	if err := db.Migrator().DropColumn(&Report{}, "source_s3_prefix"); err != nil {
		return fmt.Errorf("error dropping SourceS3Prefix column: %w", err)
	}
	
	if err := db.Migrator().DropColumn(&Report{}, "is_upload"); err != nil {
		return fmt.Errorf("error dropping IsUpload column: %w", err)
	}

	if err := db.Migrator().DropColumn(&InferenceTask{}, "source_s3_keys"); err != nil {
		return fmt.Errorf("error dropping SourceS3Keys column: %w", err)
	}

	return nil
}

func transformReports(db *gorm.DB) error {
	var reports []struct {
		Id              uuid.UUID
		S3Endpoint      sql.NullString
		S3Region        sql.NullString
		SourceS3Bucket  string
		SourceS3Prefix  sql.NullString
		IsUpload        bool
	}

	if err := db.Table("reports").Select("id, s3_endpoint, s3_region, source_s3_bucket, source_s3_prefix, is_upload").Find(&reports).Error; err != nil {
		return fmt.Errorf("error fetching reports: %w", err)
	}

	for _, report := range reports {
		var sourceType string
		var sourceParams interface{}

		if report.IsUpload {
			sourceType = storage.LocalConnectorType
			params := storage.LocalConnectorParams{
				Bucket:   report.SourceS3Bucket,
				UploadId: report.SourceS3Prefix.String,
			}
			sourceParams = params
		} else {
			sourceType = storage.S3ConnectorType
			params := storage.S3ConnectorParams{
				Endpoint: report.S3Endpoint.String,
				Region:   report.S3Region.String,
				Bucket:   report.SourceS3Bucket,
				Prefix:   report.SourceS3Prefix.String,
			}
			sourceParams = params
		}

		paramsJSON, err := json.Marshal(sourceParams)
		if err != nil {
			return fmt.Errorf("error marshaling params for report %s: %w", report.Id, err)
		}

		if err := db.Exec("UPDATE reports SET source_type = ?, source_params = ? WHERE id = ?", 
			sourceType, string(paramsJSON), report.Id).Error; err != nil {
			return fmt.Errorf("error updating report %s: %w", report.Id, err)
		}
	}

	return nil
}

func transformInferenceTasks(db *gorm.DB) error {
	var tasks []struct {
		ReportId      uuid.UUID
		TaskId        int
		SourceS3Keys  string
		IsUpload      bool  // From associated report
	}

	if err := db.Table("inference_tasks").
		Select("inference_tasks.report_id, inference_tasks.task_id, inference_tasks.source_s3_keys, reports.is_upload").
		Joins("JOIN reports ON inference_tasks.report_id = reports.id").
		Find(&tasks).Error; err != nil {
		return fmt.Errorf("error fetching inference tasks: %w", err)
	}

	for _, task := range tasks {
		var params []byte
		var err error

		if task.IsUpload {
			params, err = json.Marshal(storage.LocalConnectorTaskParams{
				ChunkKeys: strings.Split(task.SourceS3Keys, ";"),
			})
		} else {
			params, err = json.Marshal(storage.S3ConnectorTaskParams{
				ChunkKeys: strings.Split(task.SourceS3Keys, ";"),
			})
		}
		if err != nil {
			return fmt.Errorf("error marshaling params for task %d in report %s: %w", task.TaskId, task.ReportId, err)
		}

		if err := db.Exec("UPDATE inference_tasks SET source_params = ? WHERE report_id = ? AND task_id = ?", 
			string(params), task.ReportId, task.TaskId).Error; err != nil {
			return fmt.Errorf("error updating inference task %d in report %s: %w", task.TaskId, task.ReportId, err)
		}
	}

	return nil
} 