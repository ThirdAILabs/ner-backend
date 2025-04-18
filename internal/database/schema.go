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

type InferenceJob struct {
	Id      uuid.UUID `gorm:"type:uuid;primaryKey"`
	ModelId uuid.UUID `gorm:"type:uuid"`

	SourceS3Bucket string
	SourceS3Prefix sql.NullString
	DestS3Bucket   string

	Status         string `gorm:"size:20;not null"`
	CreationTime   time.Time
	CompletionTime sql.NullTime

	Groups []Group `gorm:"foreignKey:InferenceJobId;constraint:OnDelete:CASCADE"`
}

type Group struct {
	Id             uuid.UUID `gorm:"type:uuid;primaryKey"`
	Name           string
	InferenceJobId uuid.UUID `gorm:"type:uuid"`
	Query          string

	Objects []ObjectGroup `gorm:"foreignKey:InferenceJobId;constraint:OnDelete:CASCADE"`
}

type ObjectGroup struct {
	InferenceJobId uuid.UUID `gorm:"type:uuid;primaryKey"`
	Object         string    `gorm:"primaryKey"`
	GroupId        uuid.UUID `gorm:"type:uuid;primaryKey"`
}

type ObjectEntity struct {
	InferenceJobId uuid.UUID `gorm:"type:uuid;primaryKey"`
	Object         string    `gorm:"primaryKey"`
	Start          int       `gorm:"primaryKey"`
	End            int       `gorm:"primaryKey"`
	Label          string
	Text           string
}
