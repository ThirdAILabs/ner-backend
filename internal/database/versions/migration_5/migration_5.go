package migration_5

import (
	"fmt"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

type Model struct {
	Id   uuid.UUID `gorm:"type:uuid;primaryKey"`
	Name string    `gorm:"unique;not null"`
}

func Migration(db *gorm.DB) error {
	if err := db.Migrator().AlterColumn(&Model{}, "name"); err != nil {
		return fmt.Errorf("Migration4 failed: %w", err)
	}
	return nil
}

func Rollback(db *gorm.DB) error {
	if err := db.Exec("ALTER TABLE models DROP CONSTRAINT IF EXISTS models_name_key").Error; err != nil {
		return fmt.Errorf("Rollback4 failed: %w", err)
	}
	return nil
}
