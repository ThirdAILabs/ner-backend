package versions

import (
	"fmt"

	"gorm.io/gorm"
)

func Migration1(db *gorm.DB) error {
	if err := db.Migrator().AddColumn(&InferenceTask{}, "CompletedSize"); err != nil {
		return fmt.Errorf("error adding CompletedSize column: %w", err)
	}

	if err := db.Model(&InferenceTask{}).
		Where("CompletedSize IS NULL").
		Update("CompletedSize", 0).Error; err != nil {
		return fmt.Errorf("error setting default value for CompletedSize: %w", err)
	}

	return nil
}

func RollbackMigration1(db *gorm.DB) error {
	if err := db.Migrator().DropColumn(&InferenceTask{}, "CompletedSize"); err != nil {
		return fmt.Errorf("error dropping CompletedSize column: %w", err)
	}

	return nil
}
