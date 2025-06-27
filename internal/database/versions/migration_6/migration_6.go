package migration_6

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

var defaultStorageProvider = storage.S3ConnectorType

func SetDefaultStorageProvider(provider string) error {
	connectorType, err := storage.ToConnectorType(provider)
	if err != nil {
		return fmt.Errorf("invalid storage type: %v", err)
	}
	defaultStorageProvider = connectorType
	return nil
}

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
	StorageParams datatypes.JSON `gorm:"type:jsonb;not null;default:'{}'"`

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
	
	// New source fields. Default value only for migration.
	StorageType     string `gorm:"size:20;not null;default:''"`
	StorageParams   datatypes.JSON `gorm:"type:jsonb;not null;default:'{}'"`
	
	IsUpload        bool
	
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
	if err := db.Migrator().AddColumn(&Report{}, "storage_type"); err != nil {
		return fmt.Errorf("error adding StorageType column: %w", err)
	}
	
	if err := db.Migrator().AddColumn(&Report{}, "storage_params"); err != nil {
		return fmt.Errorf("error adding StorageParams column: %w", err)
	}

	if err := db.Migrator().AddColumn(&InferenceTask{}, "storage_params"); err != nil {
		return fmt.Errorf("error adding StorageParams column: %w", err)
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
		return fmt.Errorf("error dropping Region column: %w", err)
	}
	
	if err := db.Migrator().DropColumn(&Report{}, "source_s3_bucket"); err != nil {
		return fmt.Errorf("error dropping SourceS3Bucket column: %w", err)
	}

	if err := db.Migrator().DropColumn(&Report{}, "source_s3_prefix"); err != nil {
		return fmt.Errorf("error dropping SourceS3Prefix column: %w", err)
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
	}

	if err := db.Table("reports").Select("id, s3_endpoint, s3_region, source_s3_bucket, source_s3_prefix, is_upload").Find(&reports).Error; err != nil {
		return fmt.Errorf("error fetching reports: %w", err)
	}

	for _, report := range reports {
		// We default to the configured storage provider because there is no way to tell if
		// the old system used local or s3 storage; even if isUpload is true,
		// files could have been uploaded to s3.
		storageType := defaultStorageProvider
		
		var paramsJSON []byte
		var err error
		
		if storageType == storage.LocalConnectorType {
			storageParams := storage.LocalConnectorParams{
				Bucket:  report.SourceS3Bucket,
				Prefix:  report.SourceS3Prefix.String,
			}
			paramsJSON, err = json.Marshal(storageParams)
		} else {
			storageParams := storage.S3ConnectorParams{
				Endpoint: report.S3Endpoint.String,
				Region:   report.S3Region.String,
				Bucket:   report.SourceS3Bucket,
				Prefix:   report.SourceS3Prefix.String,
			}
			paramsJSON, err = json.Marshal(storageParams)
		}
		
		if err != nil {
			return fmt.Errorf("error marshaling params for report %s: %w", report.Id, err)
		}

		if err := db.Exec("UPDATE reports SET storage_type = ?, storage_params = ? WHERE id = ?", 
			storageType, string(paramsJSON), report.Id).Error; err != nil {
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

		// We default to S3 connector because there is no way to tell if
		// the old system used local or s3 storage; even if isUpload is true,
		// files could have been uploaded to s3.
		params, err = json.Marshal(storage.S3ConnectorTaskParams{
			ChunkKeys: strings.Split(task.SourceS3Keys, ";"),
		})
		if err != nil {
			return fmt.Errorf("error marshaling params for task %d in report %s: %w", task.TaskId, task.ReportId, err)
		}

		if err := db.Exec("UPDATE inference_tasks SET storage_params = ? WHERE report_id = ? AND task_id = ?", 
			string(params), task.ReportId, task.TaskId).Error; err != nil {
			return fmt.Errorf("error updating inference task %d in report %s: %w", task.TaskId, task.ReportId, err)
		}
	}

	return nil
} 