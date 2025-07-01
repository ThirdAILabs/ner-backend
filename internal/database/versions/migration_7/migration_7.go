package migration_7

import (
	"fmt"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

type ChatSession struct {
	ExtensionSessionId uuid.NullUUID `gorm:"type:uuid"`
}

func Migration(db *gorm.DB) error {
	if err := db.Migrator().AddColumn(&ChatSession{}, "extension_session_id"); err != nil {
		return fmt.Errorf("error adding ExtensionSessionId column: %w", err)
	}

	if err := db.Model(&ChatSession{}).
		Where("extension_session_id IS NULL").
		Update("extension_session_id", uuid.NullUUID{}).Error; err != nil {
		return fmt.Errorf("error setting default value for ExtensionSessionId: %w", err)
	}

	return nil
}

func Rollback(db *gorm.DB) error {
	if err := db.Migrator().DropColumn(&ChatSession{}, "extension_session_id"); err != nil {
		return fmt.Errorf("error dropping ExtensionSessionId column: %w", err)
	}

	return nil
}
