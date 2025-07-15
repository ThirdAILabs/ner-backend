package migration_3

import (
	"fmt"

	"github.com/google/uuid"
	"gorm.io/datatypes"
	"gorm.io/gorm"
)

type FileNameToPath struct {
	ID      uuid.UUID      `gorm:"type:uuid;primaryKey"`
	Mapping datatypes.JSON `gorm:"type:jsonb"`
}

func Migration(db *gorm.DB) error {
	if err := db.AutoMigrate(&FileNameToPath{}); err != nil {
		return fmt.Errorf("Migration3 failed: %w", err)
	}
	return nil
}

func Rollback(db *gorm.DB) error {
	if err := db.Migrator().DropTable(&FileNameToPath{}); err != nil {
		return fmt.Errorf("Rollback3 failed: %w", err)
	}
	return nil
}
