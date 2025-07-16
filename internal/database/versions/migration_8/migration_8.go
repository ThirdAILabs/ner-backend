package migration_8

import (
	"fmt"

	"gorm.io/gorm"
)

type Report struct {
	SkippedFileCount int `gorm:"default:0"`
}

func Migration(db *gorm.DB) error {
	if err := db.Migrator().AddColumn(&Report{}, "skipped_file_count"); err != nil {
		return fmt.Errorf("error adding skipped_file_count column: %w", err)
	}
	return nil
}

func Rollback(db *gorm.DB) error {
	if err := db.Migrator().DropColumn(&Report{}, "skipped_file_count"); err != nil {
		return fmt.Errorf("error dropping skipped_file_count column: %w", err)
	}
	return nil
}
