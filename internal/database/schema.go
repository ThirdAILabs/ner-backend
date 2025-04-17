package database

import (
	"database/sql"
	"time"

	"github.com/google/uuid"
)

const (
	ModelQueued   string = "QUEUED"
	ModelTraining string = "TRAINING"
	ModelTrained  string = "TRAINED"
	ModelFailed   string = "FAILED"
)

type Model struct {
	Id                uuid.UUID `gorm:"type:uuid;primaryKey"`
	Name              string
	Type              string `gorm:"size:20;not null"`
	Status            string `gorm:"size:20;not null"`
	ModelArtifactPath string `gorm:"not null"`
	CreationTime      time.Time
	CompletionTime    sql.NullTime
}

const (
	JobQueued    string = "QUEUED"
	JobRunning   string = "RUNNING"
	JobCompleted string = "COMPLETED"
	JobFailed    string = "FAILED"
)

type GenerateInferenceTasksTask struct {
	Id      uuid.UUID `gorm:"type:uuid;primaryKey"`
	ModelId uuid.UUID `gorm:"type:uuid"`

	SourceS3Bucket string
	SourceS3Prefix sql.NullString
	DestS3Bucket   string

	Status         string `gorm:"size:20;not null"`
	CreationTime   time.Time
	CompletionTime sql.NullTime
}

type InferenceTask struct {
	Id      uuid.UUID `gorm:"type:uuid;primaryKey"`
	ModelId uuid.UUID `gorm:"type:uuid"`

	SourceS3Bucket string
	SourceS3Prefix sql.NullString
	DestS3Bucket   string

	Status         string `gorm:"size:20;not null"`
	CreationTime   time.Time
	CompletionTime sql.NullTime
}

type TaggedEntity struct {
	InferenceJobId uuid.UUID `gorm:"type:uuid;primaryKey"`
	Object         string    `gorm:"primaryKey"`
	Index          uint      `gorm:"primaryKey"`
	Tag            string
	Value          string
}
