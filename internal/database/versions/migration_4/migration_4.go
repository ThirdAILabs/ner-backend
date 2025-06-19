package migration_4

import (
	"fmt"
	"time"

	"github.com/google/uuid"
	"gorm.io/datatypes"
	"gorm.io/gorm"
)

type FeedbackSample struct {
	ID           uuid.UUID      `gorm:"type:uuid;primaryKey"`
	ModelId      uuid.UUID      `gorm:"type:uuid;index;not null"`
	Tokens       datatypes.JSON `gorm:"type:jsonb;not null"` // JSON‐encoded []string
	Labels       datatypes.JSON `gorm:"type:jsonb;not null"` // JSON‐encoded []string
	CreationTime time.Time      `gorm:"autoCreateTime"`
}

func Migration(db *gorm.DB) error {
	if err := db.AutoMigrate(&FeedbackSample{}); err != nil {
		return fmt.Errorf("Migration3 failed: %w", err)
	}
	return nil
}

func Rollback(db *gorm.DB) error {
	if err := db.Migrator().DropTable(&FeedbackSample{}); err != nil {
		return fmt.Errorf("Rollback3 failed: %w", err)
	}
	return nil
}
