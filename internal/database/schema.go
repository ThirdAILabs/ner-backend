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

	Name           string `gorm:"unique;not null"`
	Type           string `gorm:"size:20;not null"`
	Status         string `gorm:"size:20;not null"`
	CreationTime   time.Time
	CompletionTime sql.NullTime

	Tags []ModelTag `gorm:"foreignKey:ModelId;constraint:OnDelete:CASCADE"`
}

type ModelTag struct {
	ModelId     uuid.UUID      `gorm:"type:uuid;primaryKey"`
	Tag         string         `gorm:"primaryKey"`
	Description string         `gorm:"default:''"`
	Examples    datatypes.JSON `gorm:"type:jsonb;default:'[]'"`
	Contexts    datatypes.JSON `gorm:"type:jsonb;default:'[]'"`
}

const (
	JobQueued    string = "QUEUED"
	JobRunning   string = "RUNNING"
	JobCompleted string = "COMPLETED"
	JobFailed    string = "FAILED"
	JobAborted   string = "ABORTED"
)

type Report struct {
	Id         uuid.UUID `gorm:"type:uuid;primaryKey"`
	ReportName string    `gorm:"not null"`

	ModelId uuid.UUID `gorm:"type:uuid"`
	Model   *Model    `gorm:"foreignKey:ModelId"`

	Deleted bool `gorm:"default:false"`
	Stopped bool `gorm:"default:false"`

	StorageType   string         `gorm:"size:20;not null"`
	StorageParams datatypes.JSON `gorm:"type:jsonb;not null"`

	CreationTime       time.Time
	SucceededFileCount int `gorm:"default:0"`
	FailedFileCount    int `gorm:"default:0"`
	SkippedFileCount   int `gorm:"default:0"`
	TotalFileCount     int `gorm:"default:0"`

	Tags       []ReportTag `gorm:"foreignKey:ReportId;constraint:OnDelete:CASCADE"`
	CustomTags []CustomTag `gorm:"foreignKey:ReportId;constraint:OnDelete:CASCADE"`

	Groups []Group `gorm:"foreignKey:ReportId;constraint:OnDelete:CASCADE"`

	ShardDataTask  *ShardDataTask  `gorm:"foreignKey:ReportId;constraint:OnDelete:CASCADE"`
	InferenceTasks []InferenceTask `gorm:"foreignKey:ReportId;constraint:OnDelete:CASCADE"`

	Errors []ReportError `gorm:"foreignKey:ReportId;constraint:OnDelete:CASCADE"`
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

	StorageParams datatypes.JSON `gorm:"type:jsonb;not null"`

	TotalSize     int64
	CompletedSize int64 `gorm:"default:0"`
	TokenCount    int64 `gorm:"not null;default:0"`
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
	TokenTags datatypes.JSON `gorm:"type:jsonb;not null"` // [{"token":"…","tag":"…"},…]
}

type ReportError struct {
	ReportId  uuid.UUID `gorm:"type:uuid;primaryKey"`
	ErrorId   uuid.UUID `gorm:"type:uuid;primaryKey"`
	Error     string
	Timestamp time.Time
}

type ChatSession struct {
	ID          uuid.UUID      `gorm:"type:uuid;primaryKey"`
	Title       string         `gorm:"not null"`
	TagMetadata datatypes.JSON `gorm:"type:jsonb"`
}

type ChatHistory struct {
	ID          uint      `gorm:"primary_key"`
	SessionID   uuid.UUID `gorm:"index"`
	MessageType string    // 'user' or 'ai'
	Content     string
	Timestamp   time.Time      `gorm:"autoCreateTime"`
	Metadata    datatypes.JSON `gorm:"type:jsonb"` // {"key": "value"}
}

type FileNameToPath struct {
	ID      uuid.UUID      `gorm:"type:uuid;primaryKey"`
	Mapping datatypes.JSON `gorm:"type:jsonb"`
}

type FeedbackSample struct {
	ID           uuid.UUID      `gorm:"type:uuid;primaryKey"`
	ModelId      uuid.UUID      `gorm:"type:uuid;index;not null"`
	Tokens       datatypes.JSON `gorm:"type:jsonb;not null"` // JSON‐encoded []string
	Labels       datatypes.JSON `gorm:"type:jsonb;not null"` // JSON‐encoded []string
	CreationTime time.Time      `gorm:"autoCreateTime"`
}
