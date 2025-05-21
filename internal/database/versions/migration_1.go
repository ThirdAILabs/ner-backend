// versions/migration_1.go
package versions

import (
	"fmt"
	"time"

	"gorm.io/datatypes"
	"gorm.io/gorm"
)

// ChatHistory mirrors your database.ChatHistory schema.
type ChatHistory struct {
	ID          uint   `gorm:"primaryKey"`
	SessionID   string `gorm:"index"`
	MessageType string
	Content     string
	Timestamp   time.Time      `gorm:"autoCreateTime"`
	Metadata    datatypes.JSON `gorm:"type:jsonb"` // {"key":"value"}
}

func Migration1(db *gorm.DB) error {
	if err := db.AutoMigrate(&ChatHistory{}); err != nil {
		return fmt.Errorf("Migration1 failed: %w", err)
	}
	return nil
}

func Rollback1(db *gorm.DB) error {
	if err := db.Migrator().DropTable(&ChatHistory{}); err != nil {
		return fmt.Errorf("Rollback1 failed: %w", err)
	}
	return nil
}
