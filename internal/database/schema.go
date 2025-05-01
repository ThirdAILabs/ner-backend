package database

import (
	"database/sql"
	"time"

	"github.com/google/uuid"
	"gorm.io/datatypes"
)

const (
	ModelQueued   string = "QUEUED"
	ModelTraining string = "TRAINING"
	ModelTrained  string = "TRAINED"
	ModelFailed   string = "FAILED"
)

type Model struct {
	Id uuid.UUID `gorm:"type:uuid;primaryKey"`

	BaseModelId uuid.NullUUID `gorm:"type:uuid"`
	BaseModel   *Model        `gorm:"foreignKey:BaseModelId"`

	Name           string
	Type           string `gorm:"size:20;not null"`
	Status         string `gorm:"size:20;not null"`
	CreationTime   time.Time
	CompletionTime sql.NullTime
}

const (
	JobQueued    string = "QUEUED"
	JobRunning   string = "RUNNING"
	JobCompleted string = "COMPLETED"
	JobFailed    string = "FAILED"
)

type Report struct {
	Id uuid.UUID `gorm:"type:uuid;primaryKey"`

	ModelId uuid.UUID `gorm:"type:uuid"`
	Model   *Model    `gorm:"foreignKey:ModelId"`

	SourceS3Bucket string
	SourceS3Prefix sql.NullString

	CreationTime time.Time

	Groups []Group `gorm:"foreignKey:ReportId;constraint:OnDelete:CASCADE"`

	ShardDataTask  *ShardDataTask  `gorm:"foreignKey:ReportId;constraint:OnDelete:CASCADE"`
	InferenceTasks []InferenceTask `gorm:"foreignKey:ReportId;constraint:OnDelete:CASCADE"`
}

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
	CompletionTime sql.NullTime

	SourceS3Bucket string
	SourceS3Keys   string
	TotalSize      int64
}

type Group struct {
	Id       uuid.UUID `gorm:"type:uuid;primaryKey"`
	Name     string
	ReportId uuid.UUID `gorm:"type:uuid"`
	Query    string

	Objects []ObjectGroup `gorm:"foreignKey:GroupId;constraint:OnDelete:CASCADE"`
}

type ObjectGroup struct {
	ReportId uuid.UUID `gorm:"type:uuid;primaryKey"`
	Object   string    `gorm:"primaryKey"`
	GroupId  uuid.UUID `gorm:"type:uuid;primaryKey"`
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
	Preview   string         `gorm:"type:text"`           // first ~1 000 tokens
	TokenTags datatypes.JSON `gorm:"type:jsonb;not null"` // [{"token":"…","tag":"…"},…]
}
