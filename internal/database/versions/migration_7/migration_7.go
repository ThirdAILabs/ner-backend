package migration_7

import (
	"ner-backend/internal/core/types"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

type ModelTag struct {
	ModelId     uuid.UUID `gorm:"type:uuid;primaryKey"`
	Tag         string    `gorm:"primaryKey"`
	Description string    `gorm:"size:1023;default:''"`
	Examples    []string  `gorm:"type:text[];default:array[]::text[]"`
	Contexts    []string  `gorm:"type:text[];default:array[]::text[]"`
}

func Migration(db *gorm.DB) error {
	// Add new fields to ModelTag
	if err := db.Migrator().AddColumn(&ModelTag{}, "description"); err != nil {
		return err
	}
	if err := db.Migrator().AddColumn(&ModelTag{}, "examples"); err != nil {
		return err
	}
	if err := db.Migrator().AddColumn(&ModelTag{}, "contexts"); err != nil {
		return err
	}

	for _, tagInfo := range types.CommonModelTags {
		result := db.Model(&ModelTag{}).
			Where("tag = ?", tagInfo.Name).
			Updates(map[string]interface{}{
				"description": tagInfo.Desc,
				"examples":    tagInfo.Examples,
				"contexts":    tagInfo.Contexts,
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
