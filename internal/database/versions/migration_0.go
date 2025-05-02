package database

import "gorm.io/gorm"

func Migration0(db *gorm.DB) error {
	type Tag struct {
		Name string `gorm:"primaryKey"`
	}

	type Model struct {
		Tags []Tag `gorm:"foreignKey:Name"`
	}

	if err := db.AutoMigrate(&Tag{}, &Model{}); err != nil {
		return err
	}
	return nil
}

func Rollback0(db *gorm.DB) error {
	if err := db.Migrator().DropTable("tags"); err != nil {
		return err
	}
	if err := db.Migrator().DropColumn("models", "tags"); err != nil {
		return err
	}
	return nil
}
