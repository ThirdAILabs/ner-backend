// This migration is a repair migration to fix inconsistencies introduced by migration 1 in a previous release.
// It adds and populates the completed_size column for InferenceTask.
package migration_4

import (
	"fmt"

	"gorm.io/gorm"
)

type InferenceTask struct {
	CompletedSize int64 `gorm:"default:0"`
}

func Migration(db *gorm.DB) error {
	if err := db.Migrator().AddColumn(&InferenceTask{}, "completed_size"); err != nil {
		return fmt.Errorf("error adding CompletedSize column: %w", err)
	}

	if err := db.Model(&InferenceTask{}).
		Where("completed_size IS NULL").
		Update("completed_size", 0).Error; err != nil {
		return fmt.Errorf("Migration4 failed - error setting default value for CompletedSize: %w", err)
	}

	return nil
}

func Rollback(db *gorm.DB) error {
	if err := db.Migrator().DropColumn(&InferenceTask{}, "CompletedSize"); err != nil {
		return fmt.Errorf("Rollback4 failed - error dropping CompletedSize column: %w", err)
	}

	return nil
}
