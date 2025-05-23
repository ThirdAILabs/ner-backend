package versions

import (
	"database/sql"
	"fmt"
	"time"

	"github.com/google/uuid"
	"gorm.io/datatypes"
	"gorm.io/gorm"
)

type ObjectGroup struct {
	ReportId uuid.UUID `gorm:"type:uuid;primaryKey"`
	Object   string    `gorm:"primaryKey"`
	GroupId  uuid.UUID `gorm:"type:uuid;primaryKey"`
}

type ModelTag struct {
	ModelId uuid.UUID `gorm:"type:uuid;primaryKey"`
	Tag     string    `gorm:"primaryKey"`
}

type ReportTag struct {
	ReportId uuid.UUID `gorm:"type:uuid;primaryKey"`
	Tag      string    `gorm:"primaryKey"`
	Count    uint64    `gorm:"default:0"`
}

type CustomTag struct {
	ReportId uuid.UUID `gorm:"type:uuid;primaryKey"`
	Tag      string    `gorm:"primaryKey"`
	Pattern  string
	Count    uint64 `gorm:"default:0"`
}

type Group struct {
	Id       uuid.UUID `gorm:"type:uuid;primaryKey"`
	Name     string
	ReportId uuid.UUID `gorm:"type:uuid"`
	Query    string

	Objects []ObjectGroup `gorm:"foreignKey:GroupId;constraint:OnDelete:CASCADE"`
}

type ReportError struct {
	ReportId  uuid.UUID `gorm:"type:uuid;primaryKey"`
	ErrorId   uuid.UUID `gorm:"type:uuid;primaryKey"`
	Error     string
	Timestamp time.Time
}

type Model struct {
	Id uuid.UUID `gorm:"type:uuid;primaryKey"`

	BaseModelId uuid.NullUUID `gorm:"type:uuid"`
	BaseModel   *Model        `gorm:"foreignKey:BaseModelId"`

	Name           string
	Type           string `gorm:"size:20;not null"`
	Status         string `gorm:"size:20;not null"`
	CreationTime   time.Time
	CompletionTime sql.NullTime

	Tags []ModelTag `gorm:"foreignKey:ModelId;constraint:OnDelete:CASCADE"`
}

const (
	JobQueued    string = "QUEUED"
	JobRunning   string = "RUNNING"
	JobCompleted string = "COMPLETED"
	JobFailed    string = "FAILED"
)

type ShardDataTask struct {
	ReportId uuid.UUID `gorm:"type:uuid;primaryKey"`
	Report   *Report   `gorm:"foreignKey:ReportId;constraint:OnDelete:CASCADE"`

	Status         string `gorm:"size:20;not null"`
	CreationTime   time.Time
	CompletionTime sql.NullTime

	ChunkTargetBytes int64
}

type InferenceTask struct {
	ReportId uuid.UUID `gorm:"type:uuid;primaryKey"`
	TaskId   int       `gorm:"primaryKey"`
	Report   *Report   `gorm:"foreignKey:ReportId;constraint:OnDelete:CASCADE"`

	Status         string `gorm:"size:20;not null"`
	CreationTime   time.Time
	StartTime      sql.NullTime
	CompletionTime sql.NullTime

	SourceS3Keys string
	TotalSize    int64
	TokenCount   int64 `gorm:"not null;default:0"`
}

type Report struct {
	Id         uuid.UUID `gorm:"type:uuid;primaryKey"`
	ReportName string    `gorm:"not null"`

	ModelId uuid.UUID `gorm:"type:uuid"`
	Model   *Model    `gorm:"foreignKey:ModelId"`

	Deleted bool `gorm:"default:false"`
	Stopped bool `gorm:"default:false"`

	S3Endpoint     sql.NullString
	S3Region       sql.NullString
	SourceS3Bucket string
	SourceS3Prefix sql.NullString
	IsUpload       bool

	CreationTime       time.Time
	SucceededFileCount int `gorm:"default:0"`
	TotalFileCount     int `gorm:"default:0"`
	FailedFileCount    int `gorm:"default:0"`

	Tags       []ReportTag `gorm:"foreignKey:ReportId;constraint:OnDelete:CASCADE"`
	CustomTags []CustomTag `gorm:"foreignKey:ReportId;constraint:OnDelete:CASCADE"`

	Groups []Group `gorm:"foreignKey:ReportId;constraint:OnDelete:CASCADE"`

	ShardDataTask  *ShardDataTask  `gorm:"foreignKey:ReportId;constraint:OnDelete:CASCADE"`
	InferenceTasks []InferenceTask `gorm:"foreignKey:ReportId;constraint:OnDelete:CASCADE"`

	Errors []ReportError `gorm:"foreignKey:ReportId;constraint:OnDelete:CASCADE"`
}

type ObjectEntity struct {
	ReportId uuid.UUID `gorm:"type:uuid;primaryKey"`
	Object   string    `gorm:"primaryKey"`
	Start    int       `gorm:"primaryKey"`
	End      int       `gorm:"primaryKey"`
	Label    string
	Text     string
	LContext string
	RContext string
}

type ObjectPreview struct {
	ReportId  uuid.UUID      `gorm:"type:uuid;primaryKey"`
	Object    string         `gorm:"primaryKey;size:255"`
	TokenTags datatypes.JSON `gorm:"type:jsonb;not null"` // [{"token":"…","tag":"…"},…]
}

func Migration0(db *gorm.DB) error {
	err := db.AutoMigrate(
		&Model{}, &ModelTag{}, &Report{}, &ReportTag{}, &CustomTag{}, &ShardDataTask{}, &InferenceTask{}, &Group{}, &ObjectGroup{}, &ObjectEntity{}, &ReportError{}, &ObjectPreview{},
	)
	if err != nil {
		return fmt.Errorf("initial migration failed: %w", err)
	}
	return nil
}
