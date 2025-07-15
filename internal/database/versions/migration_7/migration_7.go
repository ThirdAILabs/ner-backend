package migration_7

import (
	"encoding/json"
	"fmt"
	"ner-backend/internal/core/types"

	"github.com/google/uuid"
	"gorm.io/datatypes"
	"gorm.io/gorm"
)

type ModelTag struct {
	ModelId     uuid.UUID      `gorm:"type:uuid;primaryKey"`
	Tag         string         `gorm:"primaryKey"`
	Description string         `gorm:"default:''"`
	Examples    datatypes.JSON `gorm:"type:jsonb;default:'[]'"`
	Contexts    datatypes.JSON `gorm:"type:jsonb;default:'[]'"`
}

func Migration(db *gorm.DB) error {
	// Add new fields to ModelTag
	if err := db.AutoMigrate(&ModelTag{}); err != nil {
		return err
	}

	for _, tagInfo := range types.CommonModelTags {
		bExamples, err := json.Marshal(tagInfo.Examples)
		if err != nil {
			return fmt.Errorf("Migration failed: could not marshal examples for tag %s: %w", tagInfo.Name, err)
		}
		bContexts, err := json.Marshal(tagInfo.Contexts)
		if err != nil {
			return fmt.Errorf("Migration failed: could not marshal contexts for tag %s: %w", tagInfo.Name, err)
		}
		result := db.Model(&ModelTag{}).
			Where("tag = ?", tagInfo.Name).
			Updates(map[string]interface{}{
				"description": tagInfo.Desc,
				"examples":    bExamples,
				"contexts":    bContexts,
			})

		if result.Error != nil {
			return result.Error
		}
	}

	return nil
}
func Rollback(db *gorm.DB) error {
	if err := db.Migrator().DropColumn(&ModelTag{}, "description"); err != nil {
		return err
	}
	if err := db.Migrator().DropColumn(&ModelTag{}, "examples"); err != nil {
		return err
	}
	if err := db.Migrator().DropColumn(&ModelTag{}, "contexts"); err != nil {
		return err
	}

	return nil
}
