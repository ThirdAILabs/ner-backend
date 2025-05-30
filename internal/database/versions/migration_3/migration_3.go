package migration_3

import (
	"fmt"

	"gorm.io/gorm"
)

// Redefine UploadFilePath here to avoid import cycle
type UploadFilePath struct {
	FileIdentifier string `gorm:"primaryKey;size:300"`
	FullPath string
}

func Migration(db *gorm.DB) error {
	if err := db.AutoMigrate(&UploadFilePath{}); err != nil {
		return fmt.Errorf("Migration3 failed: %w", err)
	}
	return nil
}

func Rollback(db *gorm.DB) error {
	if err := db.Migrator().DropTable(&UploadFilePath{}); err != nil {
		return fmt.Errorf("Rollback3 failed: %w", err)
	}
	return nil
} 