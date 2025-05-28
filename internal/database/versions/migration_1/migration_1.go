package migration_1

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
		return fmt.Errorf("error setting default value for CompletedSize: %w", err)
	}

	return nil
}

func Rollback(db *gorm.DB) error {
	if err := db.Migrator().DropColumn(&InferenceTask{}, "CompletedSize"); err != nil {
		return fmt.Errorf("error dropping CompletedSize column: %w", err)
	}

	return nil
}
